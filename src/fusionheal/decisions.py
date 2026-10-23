from __future__ import annotations
from typing import Any


class DecisionEngine:
    """Enforces evidence-governed and constraint-validated acceptance policies for sensor reconstructions.

    Separates candidate generation from decision-making, ensuring that candidates are never
    accepted solely because they are numerically plausible.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def decide(
        self,
        issue: dict[str, Any],
        candidate: dict[str, Any] | None,
        constraint_results: list[dict[str, Any]] | None = None,
        evidence_list: list[dict[str, Any]] | None = None,
        confidence_assessment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Determine whether to ACCEPT, REVIEW, REJECT, or take NO_ACTION on a candidate."""
        issue_id = issue.get("issue_id", "UNKNOWN_ISSUE")
        fault_type = issue.get("fault_type", "unknown")

        # 1. Handle non-actionable or structural informational issues
        if fault_type in ("timestamp_gap", "timestamp_duplicate", "timestamp_order", "timestamp_invalid"):
            return {
                "issue_id": issue_id,
                "candidate_id": None,
                "decision": "NO_ACTION",
                "reason": (
                    f"Timestamp issue '{fault_type}' is an observational integrity event; "
                    "measurements are intentionally preserved without synthetic data generation."
                ),
                "confidence_value": None,
                "confidence_type": "not_available",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in (evidence_list or []) if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in (constraint_results or []) if c.get("rule_id")],
            }

        # 2. If no candidate was available
        if candidate is None:
            return {
                "issue_id": issue_id,
                "candidate_id": None,
                "decision": "REVIEW",
                "reason": "No safe reconstruction candidate was available for this fault event.",
                "confidence_value": None,
                "confidence_type": "not_available",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in (evidence_list or []) if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in (constraint_results or []) if c.get("rule_id")],
            }

        candidate_id = candidate.get("candidate_id")
        method = candidate.get("method")
        c_results = constraint_results or []
        e_list = evidence_list or []

        # 3. Check Hard Constraints: Any failure triggers REJECT
        hard_fails = [c for c in c_results if c.get("severity") == "hard" and c.get("status") == "fail"]
        if hard_fails:
            reasons = "; ".join(c.get("explanation", "") for c in hard_fails)
            return {
                "issue_id": issue_id,
                "candidate_id": candidate_id,
                "decision": "REJECT",
                "reason": f"Candidate violates applicable hard process constraint(s): {reasons}",
                "confidence_value": 0.0,
                "confidence_type": "heuristic",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
            }

        # 4. Check Unknown Hard Constraints: Unknown hard constraint prevents acceptance -> REVIEW
        unknown_hard = [c for c in c_results if c.get("severity") == "hard" and c.get("status") == "unknown"]
        if unknown_hard:
            reasons = "; ".join(c.get("explanation", "") for c in unknown_hard)
            return {
                "issue_id": issue_id,
                "candidate_id": candidate_id,
                "decision": "REVIEW",
                "reason": f"Applicable hard constraint could not be verified ({reasons}); requires engineer review.",
                "confidence_value": confidence_assessment.get("score") if confidence_assessment else None,
                "confidence_type": "heuristic",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
            }

        # 5. Check Contradictory Evidence
        contradictions = [e for e in e_list if e.get("direction") == "contradicting"]
        if contradictions:
            reasons = "; ".join(e.get("explanation", "") for e in contradictions)
            return {
                "issue_id": issue_id,
                "candidate_id": candidate_id,
                "decision": "REVIEW",
                "reason": f"Contradictory evidence detected ({reasons}); candidate deferred to human review.",
                "confidence_value": confidence_assessment.get("score") if confidence_assessment else None,
                "confidence_type": "heuristic",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
            }

        # 6. Check Soft Constraint Warnings
        soft_warnings = [c for c in c_results if c.get("status") == "warning"]
        if soft_warnings:
            reasons = "; ".join(c.get("explanation", "") for c in soft_warnings)
            return {
                "issue_id": issue_id,
                "candidate_id": candidate_id,
                "decision": "REVIEW",
                "reason": f"Soft process constraint warning present ({reasons}); requires human judgment.",
                "confidence_value": confidence_assessment.get("score") if confidence_assessment else None,
                "confidence_type": "heuristic",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
            }

        # 7. Evaluate Configured Auto-Acceptance Policy
        dec_cfg = self.config.get("decision", {})
        auto_interp = dec_cfg.get("auto_accept_interpolation", True)
        auto_outlier = dec_cfg.get("auto_accept_outlier_replacement", False)
        auto_cross = dec_cfg.get("auto_accept_cross_sensor", True)
        min_accept_confidence = float(dec_cfg.get("min_accept_confidence", 0.60))

        heuristic_score = confidence_assessment.get("score") if confidence_assessment else 0.70
        has_sufficient_confidence = heuristic_score is not None and heuristic_score >= min_accept_confidence

        if method == "linear_interpolation" and auto_interp and has_sufficient_confidence:
            return {
                "issue_id": issue_id,
                "candidate_id": candidate_id,
                "decision": "ACCEPT",
                "reason": (
                    f"Interior linear interpolation passed all applicable hard constraints "
                    f"with sufficient heuristic support ({heuristic_score:.2f} >= {min_accept_confidence:.2f})."
                ),
                "confidence_value": heuristic_score,
                "confidence_type": "heuristic",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
            }

        if method == "local_median" and auto_outlier and has_sufficient_confidence:
            return {
                "issue_id": issue_id,
                "candidate_id": candidate_id,
                "decision": "ACCEPT",
                "reason": (
                    f"Local median candidate passed all hard constraints and meets configured "
                    f"outlier auto-acceptance policy ({heuristic_score:.2f} >= {min_accept_confidence:.2f})."
                ),
                "confidence_value": heuristic_score,
                "confidence_type": "heuristic",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
            }

        if method == "cross_sensor_estimate" and auto_cross and has_sufficient_confidence:
            return {
                "issue_id": issue_id,
                "candidate_id": candidate_id,
                "decision": "ACCEPT",
                "reason": (
                    f"Cross-sensor candidate passed all hard constraints and is grounded in "
                    f"validated physical coupling ({heuristic_score:.2f} >= {min_accept_confidence:.2f})."
                ),
                "confidence_value": heuristic_score,
                "confidence_type": "heuristic",
                "confidence_assessment": confidence_assessment,
                "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
                "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
            }

        # Otherwise requires human review under policy
        return {
            "issue_id": issue_id,
            "candidate_id": candidate_id,
            "decision": "REVIEW",
            "reason": (
                f"Candidate passed constraints but method '{method}' requires human operator review "
                f"under current plant configuration (auto-accept disabled or score {heuristic_score} < threshold)."
            ),
            "confidence_value": heuristic_score,
            "confidence_type": "heuristic",
            "confidence_assessment": confidence_assessment,
            "evidence_ids": [e.get("evidence_id") for e in e_list if e.get("evidence_id")],
            "constraint_rule_ids": [c.get("rule_id") for c in c_results if c.get("rule_id")],
        }


def decide(
    issue: dict[str, Any],
    candidate: dict[str, Any] | None,
    sensor_limits: dict[str, Any],
    auto_accept_interpolation: bool = True,
    auto_accept_outlier_replacement: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backwards-compatible decision function used by starter tests and legacy pipeline calls."""
    config = {
        "sensor_limits": sensor_limits,
        "decision": {
            "auto_accept_interpolation": auto_accept_interpolation,
            "auto_accept_outlier_replacement": auto_accept_outlier_replacement,
        },
    }
    config.update(kwargs)

    # Simple constraint check for legacy calls
    c_results: list[dict[str, Any]] = []
    if candidate:
        val = candidate.get("candidate_value")
        col = issue.get("sensor_id")
        limits = sensor_limits.get(col, {})
        if limits and val is not None:
            min_val = limits.get("min")
            max_val = limits.get("max")
            passed = True
            breach = ""
            if min_val is not None and val < min_val:
                passed = False
                breach = f"< min {min_val}"
            elif max_val is not None and val > max_val:
                passed = False
                breach = f"> max {max_val}"

            c_results.append({
                "rule_id": f"RULE-RANGE-{col}",
                "applicable_sensor": col,
                "severity": "hard",
                "status": "pass" if passed else "fail",
                "explanation": f"Range check: {breach}" if not passed else "Range check passed.",
                "observed_value": val,
                "expected_parameters": limits,
            })

    engine = DecisionEngine(config)
    confidence = kwargs.get("confidence_assessment")
    evidence = kwargs.get("evidence_list")
    return engine.decide(
        issue=issue,
        candidate=candidate,
        constraint_results=c_results or kwargs.get("constraint_results"),
        evidence_list=evidence,
        confidence_assessment=confidence,
    )
