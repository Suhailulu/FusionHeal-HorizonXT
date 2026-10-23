from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from fusionheal.evidence import EvidenceEngine
from fusionheal.reconstruction import ReconstructionEngine, generate_candidate
from fusionheal.constraints import ProcessConstraintEngine
from fusionheal.confidence import ConfidenceEngine
from fusionheal.decisions import DecisionEngine, decide
from fusionheal.audit import create_audit_record, write_audit
from fusionheal.pipeline import run_pipeline


def make_test_df(n: int = 15) -> pd.DataFrame:
    """Helper to generate a clean synthetic dataframe for intelligence engine testing."""
    return pd.DataFrame({
        "source_record_id": [f"ROW-{i:08d}" for i in range(n)],
        "timestamp_original": [f"2026-10-23 10:{i:02d}:00" for i in range(n)],
        "timestamp_parsed": pd.date_range("2026-10-23 10:00:00", periods=n, freq="min"),
        "temperature": [65.0 + (i % 3) * 0.2 for i in range(n)],
        "pressure": [15.0 + (i % 2) * 0.1 for i in range(n)],
    })


# 1. Evidence Record Creation & Types
def test_evidence_record_creation():
    df = make_test_df(12)
    engine = EvidenceEngine({
        "sensor_limits": {"temperature": {"min": 0, "max": 100}},
        "detection": {"min_history_points": 5},
    })
    issue = {
        "issue_id": "ISS-TEMP-00000005",
        "sensor_id": "temperature",
        "row_position": 5,
        "fault_type": "missing",
        "original_value": None,
    }
    candidate = {"candidate_id": "CAND-01", "candidate_value": 65.2, "method": "linear_interpolation"}

    evidence = engine.collect_evidence(df, issue, candidate)
    assert len(evidence) >= 3

    types = {e["evidence_type"] for e in evidence}
    assert "temporal" in types
    assert "historical" in types
    assert "process_rule" in types

    for e in evidence:
        assert "evidence_id" in e
        assert e["issue_id"] == "ISS-TEMP-00000005"
        assert e["direction"] in ("supporting", "contradicting", "neutral")
        assert "explanation" in e


# 2. Supporting vs Contradicting Evidence
def test_supporting_and_contradicting_evidence():
    df = make_test_df(15)
    engine = EvidenceEngine({
        "sensor_limits": {"temperature": {"min": 0, "max": 100}},
        "detection": {"min_history_points": 5},
    })
    issue = {
        "issue_id": "ISS-TEMP-00000006",
        "sensor_id": "temperature",
        "row_position": 6,
        "fault_type": "missing",
    }

    # Supporting candidate (near local median ~65.2)
    good_cand = {"candidate_id": "CAND-GOOD", "candidate_value": 65.2}
    ev_good = engine.collect_evidence(df, issue, good_cand)
    good_agg = EvidenceEngine.aggregate_evidence(ev_good)
    assert good_agg["supporting_count"] > 0
    assert not good_agg["has_contradictions"]

    # Contradicting candidate (far from normal distribution ~500.0)
    bad_cand = {"candidate_id": "CAND-BAD", "candidate_value": 500.0}
    ev_bad = engine.collect_evidence(df, issue, bad_cand)
    bad_agg = EvidenceEngine.aggregate_evidence(ev_bad)
    assert bad_agg["contradicting_count"] > 0
    assert bad_agg["has_contradictions"]


# 3. Missing and Insufficient Evidence
def test_missing_and_insufficient_evidence():
    # Very short dataset with only 3 samples
    df_short = make_test_df(3)
    engine = EvidenceEngine({"detection": {"min_history_points": 5}})
    issue = {"issue_id": "ISS-01", "sensor_id": "temperature", "row_position": 1, "fault_type": "missing"}
    cand = {"candidate_id": "CAND-01", "candidate_value": 65.0}

    evidence = engine.collect_evidence(df_short, issue, cand)
    hist_ev = next((e for e in evidence if e["evidence_type"] == "historical"), None)
    assert hist_ev is not None
    assert hist_ev["direction"] == "neutral"
    assert "Insufficient historical data" in hist_ev["explanation"]


# 4. Cross-sensor unconfigured is marked unavailable
def test_cross_sensor_unconfigured_evidence_marked_unavailable():
    df = make_test_df(10)
    engine = EvidenceEngine({"detection": {"cross_sensor": {"rules": []}}})
    issue = {"issue_id": "ISS-01", "sensor_id": "temperature", "row_position": 3, "fault_type": "missing"}
    cand = {"candidate_id": "CAND-01", "candidate_value": 65.0}

    evidence = engine.collect_evidence(df, issue, cand)
    cross_ev = next((e for e in evidence if e["evidence_type"] == "cross_sensor"), None)
    assert cross_ev is not None
    assert cross_ev["direction"] == "neutral"
    assert "UNAVAILABLE" in cross_ev["explanation"]
    assert cross_ev["metadata"]["status"] == "unavailable"


# 5. Multiple Reconstruction Candidates
def test_multiple_reconstruction_candidates():
    df = make_test_df(15)
    df.loc[5, "temperature"] = np.nan

    reconstruction_engine = ReconstructionEngine({
        "detection": {
            "cross_sensor": {
                "rules": [
                    {
                        "rule_id": "temp_press_rel",
                        "sensor_a": "temperature",
                        "sensor_b": "pressure",
                        "nominal_offset": 50.0,
                    }
                ]
            }
        }
    })
    issue = {
        "issue_id": "ISS-TEMP-00000005",
        "sensor_id": "temperature",
        "row_position": 5,
        "fault_type": "missing",
        "original_value": None,
    }

    candidates = reconstruction_engine.generate_candidates(df, issue)
    # Should generate linear_interpolation, local_median, and cross_sensor_estimate!
    assert len(candidates) >= 2
    methods = {c["method"] for c in candidates}
    assert "linear_interpolation" in methods
    assert "local_median" in methods
    assert "cross_sensor_estimate" in methods

    for c in candidates:
        assert "assumptions" in c
        assert "diagnostics" in c
        assert "limitations" in c


# 6. Valid and Invalid Interpolation Contexts
def test_interpolation_gap_limits():
    df = make_test_df(20)
    # Create large missing gap of 8 consecutive points (indices 5 to 12)
    for i in range(5, 13):
        df.loc[i, "temperature"] = np.nan

    reconstruction_engine = ReconstructionEngine({"reconstruction": {"max_interpolation_gap": 5}})
    issue = {
        "issue_id": "ISS-TEMP-00000008",
        "sensor_id": "temperature",
        "row_position": 8,
        "fault_type": "missing",
    }
    candidates = reconstruction_engine.generate_candidates(df, issue)
    methods = [c["method"] for c in candidates]
    # Linear interpolation should NOT be offered across an 8-sample gap when max is 5
    assert "linear_interpolation" not in methods


# 7. Hard Constraint Failures block acceptance
def test_hard_constraint_failure_blocks_acceptance():
    df = make_test_df(10)
    constraint_engine = ProcessConstraintEngine({
        "constraints": [
            {
                "rule_id": "hard_max_limit",
                "applicable_sensor": "temperature",
                "severity": "hard",
                "rule_type": "range",
                "min": 0,
                "max": 100,
            }
        ]
    })
    confidence_engine = ConfidenceEngine()
    decision_engine = DecisionEngine()

    issue = {"issue_id": "ISS-01", "sensor_id": "temperature", "row_position": 4, "fault_type": "outlier"}
    candidate = {"candidate_id": "CAND-BAD", "candidate_value": 250.0, "method": "linear_interpolation"}

    constraints = constraint_engine.evaluate_constraints(df, issue, candidate)
    assert any(c["severity"] == "hard" and c["status"] == "fail" for c in constraints)

    confidence = confidence_engine.assess(candidate, [], constraints)
    assert confidence["score"] == 0.0

    decision = decision_engine.decide(issue, candidate, constraints, [], confidence)
    assert decision["decision"] == "REJECT"
    assert "violates applicable hard process constraint" in decision["reason"]


# 8. Soft Constraint Warnings generate REVIEW
def test_soft_constraint_warning_requires_review():
    df = make_test_df(10)
    # Preceding point at row 3 is 65.0
    # Candidate at row 4 with 85.0 -> rate of change is 20.0 (exceeds soft max 10.0)
    constraint_engine = ProcessConstraintEngine({
        "constraints": [
            {
                "rule_id": "soft_thermal_roc",
                "applicable_sensor": "temperature",
                "severity": "soft",
                "rule_type": "rate_of_change",
                "max_rate_of_change": 10.0,
            }
        ]
    })
    confidence_engine = ConfidenceEngine()
    decision_engine = DecisionEngine()

    issue = {"issue_id": "ISS-02", "sensor_id": "temperature", "row_position": 4, "fault_type": "spike"}
    candidate = {"candidate_id": "CAND-02", "candidate_value": 85.0, "method": "linear_interpolation"}

    constraints = constraint_engine.evaluate_constraints(df, issue, candidate)
    soft_warn = next((c for c in constraints if c["severity"] == "soft"), None)
    assert soft_warn is not None
    assert soft_warn["status"] == "warning"

    decision = decision_engine.decide(issue, candidate, constraints, [], confidence_engine.assess(candidate, [], constraints))
    # Soft constraint warning must defer to human review, not auto-accept!
    assert decision["decision"] == "REVIEW"
    assert "Soft process constraint warning present" in decision["reason"]


# 9. Unknown Constraint Status prevents acceptance
def test_unknown_constraint_status():
    df = make_test_df(10)
    # Cross sensor difference rule with a paired sensor that does not exist in df
    constraint_engine = ProcessConstraintEngine({
        "constraints": [
            {
                "rule_id": "unknown_paired_rule",
                "applicable_sensor": "temperature",
                "paired_sensor": "non_existent_sensor",
                "severity": "hard",
                "rule_type": "cross_sensor_difference",
                "max_difference": 5.0,
            }
        ]
    })
    issue = {"issue_id": "ISS-03", "sensor_id": "temperature", "row_position": 3, "fault_type": "missing"}
    candidate = {"candidate_id": "CAND-03", "candidate_value": 65.0, "method": "linear_interpolation"}

    constraints = constraint_engine.evaluate_constraints(df, issue, candidate)
    unknown_rule = next(c for c in constraints if c["rule_id"] == "unknown_paired_rule")
    assert unknown_rule["status"] == "unknown"

    decision_engine = DecisionEngine()
    decision = decision_engine.decide(issue, candidate, constraints, [], ConfidenceEngine().assess(candidate, [], constraints))
    assert decision["decision"] == "REVIEW"
    assert "could not be verified" in decision["reason"]


# 10. Decision Outcomes: ACCEPT, REVIEW, REJECT, NO_ACTION
def test_all_four_decision_outcomes():
    decision_engine = DecisionEngine({
        "decision": {
            "auto_accept_interpolation": True,
            "min_accept_confidence": 0.60,
        }
    })

    # 1. ACCEPT
    iss_acc = {"issue_id": "ISS-A", "sensor_id": "temperature", "row_position": 2, "fault_type": "missing"}
    cand_acc = {"candidate_id": "C-A", "candidate_value": 65.0, "method": "linear_interpolation"}
    c_pass = [{"rule_id": "R1", "severity": "hard", "status": "pass"}]
    conf_good = {"score": 0.85, "score_type": "heuristic"}
    dec_acc = decision_engine.decide(iss_acc, cand_acc, c_pass, [], conf_good)
    assert dec_acc["decision"] == "ACCEPT"

    # 2. REJECT (Hard constraint fail)
    c_fail = [{"rule_id": "R1", "severity": "hard", "status": "fail", "explanation": "Breached max bounds."}]
    dec_rej = decision_engine.decide(iss_acc, cand_acc, c_fail, [], {"score": 0.0})
    assert dec_rej["decision"] == "REJECT"

    # 3. REVIEW (No candidate)
    dec_rev = decision_engine.decide(iss_acc, None, [], [], None)
    assert dec_rev["decision"] == "REVIEW"

    # 4. NO_ACTION (Timestamp gap - observational integrity)
    iss_ts = {"issue_id": "ISS-TS", "sensor_id": "timestamp", "row_position": 4, "fault_type": "timestamp_gap"}
    dec_no_act = decision_engine.decide(iss_ts, None, [], [], None)
    assert dec_no_act["decision"] == "NO_ACTION"


# 11. Heuristic Score Transparency
def test_confidence_assessment_transparency():
    confidence_engine = ConfidenceEngine()
    candidate = {"candidate_id": "CAND-01", "candidate_value": 65.0, "method": "linear_interpolation"}
    constraints = [{"rule_id": "R1", "severity": "hard", "status": "pass"}]
    evidence = [{"evidence_id": "E1", "evidence_type": "temporal", "direction": "supporting", "explanation": "Stable."}]

    assessment = confidence_engine.assess(candidate, evidence, constraints)
    assert assessment["score_type"] == "heuristic"
    assert assessment["calibration_status"] == "uncalibrated"
    assert 0.0 < assessment["score"] < 1.0
    assert any("heuristic" in lim.lower() for lim in assessment["limitations"])


# 12. Audit Record Completeness
def test_audit_record_completeness(tmp_path):
    issue = {"issue_id": "ISS-100", "source_record_id": "ROW-00000001", "sensor_id": "temperature", "row_position": 1, "timestamp": "2026-10-23 10:01:00", "fault_type": "missing", "original_value": None, "detector": "null_check"}
    candidate = {"candidate_id": "CAND-100", "candidate_value": 65.5, "method": "linear_interpolation", "assumptions": ["Linear change."]}
    decision = {"decision": "ACCEPT", "reason": "Passed hard constraints."}
    evidence = [{"evidence_id": "E1", "evidence_type": "temporal", "direction": "supporting"}]
    constraints = [{"rule_id": "R1", "severity": "hard", "status": "pass"}]
    confidence = {"score": 0.8, "score_type": "heuristic"}

    audit_entry = create_audit_record(
        run_id="RUN-TEST-001",
        issue=issue,
        candidate=candidate,
        decision=decision,
        evidence_records=evidence,
        constraint_results=constraints,
        confidence_assessment=confidence,
        config={"timestamp_column": "timestamp"},
    )

    assert audit_entry["run_id"] == "RUN-TEST-001"
    assert audit_entry["issue_id"] == "ISS-100"
    assert audit_entry["decision"] == "ACCEPT"
    assert "config_hash" in audit_entry
    assert "algorithm_version" in audit_entry
    assert "audit_timestamp_utc" in audit_entry

    out_file = tmp_path / "audit.jsonl"
    write_audit([audit_entry], out_file)
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1


# 13. Data Preservation
def test_data_preservation_through_intelligence_engine():
    df = make_test_df(10)
    df.loc[3, "temperature"] = np.nan
    df_copy = df.copy(deep=True)

    cfg = {"sensor_limits": {"temperature": {"min": 0, "max": 100}}}
    engine = EvidenceEngine(cfg)
    reconstruction_engine = ReconstructionEngine(cfg)
    constraint_engine = ProcessConstraintEngine(cfg)

    issue = {"issue_id": "ISS-01", "sensor_id": "temperature", "row_position": 3, "fault_type": "missing"}
    cands = reconstruction_engine.generate_candidates(df, issue)
    assert len(cands) > 0

    ev = engine.collect_evidence(df, issue, cands[0])
    assert len(ev) > 0

    res = constraint_engine.evaluate_constraints(df, issue, cands[0])
    assert len(res) > 0

    # Ensure source dataframe was never mutated during evidence collection or candidate generation
    pd.testing.assert_frame_equal(df, df_copy)
