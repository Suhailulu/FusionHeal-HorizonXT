from __future__ import annotations
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
import pandas as pd
import yaml

from .ingestion import load_dataset
from .detection import detect_issues
from .evidence import EvidenceEngine
from .reconstruction import ReconstructionEngine, generate_candidate
from .constraints import ProcessConstraintEngine
from .confidence import ConfidenceEngine
from .decisions import DecisionEngine, decide
from .audit import create_audit_record, write_audit


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    # Catch common configuration typo early instead of failing deep in the pipeline.
    window = config.get("detection", {}).get("rolling_window", 9)
    if not isinstance(window, int) or window < 3:
        raise ValueError("detection.rolling_window must be an integer >= 3.")
    return config


def run_pipeline(input_path: str | Path, config_path: str | Path, output_dir: str | Path) -> dict:
    config = load_config(config_path)
    timestamp_col = config.get("timestamp_column", "timestamp")
    df = load_dataset(input_path, timestamp_col)

    reserved = {"source_record_id", "timestamp_original", "timestamp_parsed"}
    sensor_columns = [
        col for col in df.columns
        if col not in reserved and col != timestamp_col
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    if not sensor_columns:
        sensor_columns = [
            col for col in df.columns
            if col not in reserved and col != timestamp_col
            and pd.to_numeric(df[col], errors="coerce").notna().any()
        ]
    if not sensor_columns:
        raise ValueError("No numeric sensor columns found in input CSV.")

    # 1. Fault Detection
    detection_cfg = config.get("detection", {})
    limits = config.get("sensor_limits", {})
    issues = detect_issues(
        df, sensor_columns, limits,
        rolling_window=detection_cfg.get("rolling_window", 9),
        mad_threshold=float(detection_cfg.get("mad_threshold", 5.0)),
        min_history_points=int(detection_cfg.get("min_history_points", 5)),
        config=config,
    )

    # 2. Instantiate Intelligence Engines
    evidence_engine = EvidenceEngine(config)
    reconstruction_engine = ReconstructionEngine(config)
    constraint_engine = ProcessConstraintEngine(config)
    confidence_engine = ConfidenceEngine(config)
    decision_engine = DecisionEngine(config)

    run_id = f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    # Index rows with detected faults to avoid using them as clean neighbors in temporal evidence
    flagged_positions: set[int] = {
        int(iss["row_position"]) for iss in issues if iss.get("row_position") is not None
    }

    all_candidates: list[dict] = []
    decisions: list[dict] = []
    audit_records: list[dict] = []
    processed = df.copy()

    for issue in issues:
        # Generate candidates (can generate multiple viable options)
        candidates = reconstruction_engine.generate_candidates(df, issue)
        primary_candidate = candidates[0] if candidates else None

        if primary_candidate:
            all_candidates.append(primary_candidate)

        # Collect evidence across temporal, historical, cross-sensor, and process rules
        evidence_list = evidence_engine.collect_evidence(
            df, issue, candidate=primary_candidate, flagged_positions=flagged_positions
        )

        # Evaluate process constraints (hard vs soft)
        constraint_results = (
            constraint_engine.evaluate_constraints(df, issue, primary_candidate)
            if primary_candidate else []
        )

        # Assess uncalibrated heuristic confidence
        confidence_assessment = confidence_engine.assess(
            primary_candidate, evidence_list, constraint_results
        )

        # Apply multi-policy decision engine
        decision = decision_engine.decide(
            issue=issue,
            candidate=primary_candidate,
            constraint_results=constraint_results,
            evidence_list=evidence_list,
            confidence_assessment=confidence_assessment,
        )
        decisions.append(decision)

        # Construct comprehensive audit record
        audit_entry = create_audit_record(
            run_id=run_id,
            issue=issue,
            candidate=primary_candidate,
            decision=decision,
            evidence_records=evidence_list,
            constraint_results=constraint_results,
            confidence_assessment=confidence_assessment,
            config=config,
        )
        audit_records.append(audit_entry)

        # Only apply reconstruction when decision is strictly ACCEPT
        if primary_candidate and decision["decision"] == "ACCEPT":
            pos = issue.get("row_position")
            sensor = issue.get("sensor_id")
            if pos is not None and sensor in processed.columns:
                processed.loc[int(pos), sensor] = primary_candidate["candidate_value"]

    # Write output artifacts
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output / "processed_data.csv", index=False)
    write_audit(audit_records, output / "audit.jsonl")

    summary = {
        "run_id": run_id,
        "input_rows": int(len(df)),
        "sensor_columns": sensor_columns,
        "issues_detected": len(issues),
        "candidates_generated": len(all_candidates),
        "accepted": sum(d["decision"] == "ACCEPT" for d in decisions),
        "review": sum(d["decision"] == "REVIEW" for d in decisions),
        "rejected": sum(d["decision"] == "REJECT" for d in decisions),
        "no_action": sum(d["decision"] == "NO_ACTION" for d in decisions),
        "fault_counts": pd.Series([i["fault_type"] for i in issues]).value_counts().to_dict(),
        "outputs": ["processed_data.csv", "audit.jsonl", "summary.json"],
        "prototype_notice": "Heuristic prototype; configured limits and acceptance policy require domain validation.",
    }

    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    return summary
