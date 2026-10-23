from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import yaml
from .ingestion import load_dataset
from .detection import detect_issues
from .reconstruction import generate_candidate
from .decisions import decide
from .audit import write_audit

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
        # Numeric strings are supported if at least one non-null value parses.
        sensor_columns = [
            col for col in df.columns
            if col not in reserved and col != timestamp_col
            and pd.to_numeric(df[col], errors="coerce").notna().any()
        ]
    if not sensor_columns:
        raise ValueError("No numeric sensor columns found in input CSV.")
    detection_cfg = config.get("detection", {})
    limits = config.get("sensor_limits", {})
    issues = detect_issues(
        df, sensor_columns, limits,
        rolling_window=detection_cfg.get("rolling_window", 9),
        mad_threshold=float(detection_cfg.get("mad_threshold", 5.0)),
        min_history_points=int(detection_cfg.get("min_history_points", 5)),
        config=config,
    )
    candidates, decisions, audit_records = [], [], []
    processed = df.copy()
    for issue in issues:
        candidate = generate_candidate(df, issue)
        decision = decide(
            issue, candidate, limits,
            auto_accept_interpolation=config.get("decision", {}).get("auto_accept_interpolation", True),
            auto_accept_outlier_replacement=config.get("decision", {}).get("auto_accept_outlier_replacement", False),
        )
        if candidate:
            candidates.append(candidate)
        decisions.append(decision)
        audit_records.append({"issue": issue, "candidate": candidate, "decision": decision})
        if candidate and decision["decision"] == "ACCEPT":
            pos = issue.get("row_position")
            sensor = issue.get("sensor_id")
            if pos is not None and sensor in processed.columns:
                processed.loc[int(pos), sensor] = candidate["candidate_value"]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output / "processed_data.csv", index=False)
    write_audit(audit_records, output / "audit.jsonl")
    summary = {
        "input_rows": int(len(df)),
        "sensor_columns": sensor_columns,
        "issues_detected": len(issues),
        "candidates_generated": len(candidates),
        "accepted": sum(d["decision"] == "ACCEPT" for d in decisions),
        "review": sum(d["decision"] == "REVIEW" for d in decisions),
        "rejected": sum(d["decision"] == "REJECT" for d in decisions),
        "fault_counts": pd.Series([i["fault_type"] for i in issues]).value_counts().to_dict(),
        "outputs": ["processed_data.csv", "audit.jsonl", "summary.json"],
        "prototype_notice": "Heuristic prototype; configured limits and acceptance policy require domain validation."
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary
