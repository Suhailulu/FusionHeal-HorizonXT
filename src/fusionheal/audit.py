from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

ALGORITHM_VERSION = "0.2.0-intelligence-engine"


def compute_config_hash(config: dict[str, Any]) -> str:
    """Produce deterministic SHA-256 hash of active pipeline configuration."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def create_audit_record(
    run_id: str,
    issue: dict[str, Any],
    candidate: dict[str, Any] | None,
    decision: dict[str, Any],
    evidence_records: list[dict[str, Any]] | None = None,
    constraint_results: list[dict[str, Any]] | None = None,
    confidence_assessment: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a comprehensive, tamper-evident audit entry for a single pipeline fault resolution."""
    cfg = config or {}
    score = (
        confidence_assessment.get("score")
        if confidence_assessment
        else decision.get("confidence_value")
    )
    score_type = (
        confidence_assessment.get("score_type", "heuristic")
        if confidence_assessment
        else decision.get("confidence_type", "heuristic")
    )

    return {
        "run_id": str(run_id),
        "issue_id": str(issue.get("issue_id", "")),
        "source_record_id": str(issue.get("source_record_id", "")),
        "sensor_id": str(issue.get("sensor_id", "")),
        "row_position": issue.get("row_position"),
        "timestamp": str(issue.get("timestamp", "")),
        "original_value": issue.get("original_value"),
        "fault_type": str(issue.get("fault_type", "")),
        "detector": str(issue.get("detector", "")),
        "candidate_id": candidate.get("candidate_id") if candidate else None,
        "candidate_value": candidate.get("candidate_value") if candidate else None,
        "candidate_method": candidate.get("method") if candidate else None,
        "candidate_assumptions": candidate.get("assumptions", []) if candidate else [],
        "evidence_records": evidence_records or [],
        "constraint_results": constraint_results or [],
        "heuristic_score": float(score) if score is not None else None,
        "score_type": str(score_type),
        "decision": str(decision.get("decision", "REVIEW")),
        "decision_rationale": str(decision.get("reason", "")),
        "config_hash": compute_config_hash(cfg),
        "algorithm_version": ALGORITHM_VERSION,
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_audit(records: list[dict[str, Any]], path: str | Path, append: bool = False) -> None:
    """Write structured audit records to disk in newline-delimited JSON (JSONL) format."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with target_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            # Handle both legacy payload records {"issue": ..., "candidate": ...} and enriched audit records
            if "run_id" in record:
                payload = record
            else:
                payload = {
                    "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "record": record,
                }
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
