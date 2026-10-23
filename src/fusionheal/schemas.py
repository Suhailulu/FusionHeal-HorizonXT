from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

DecisionType = Literal["ACCEPT", "REVIEW", "REJECT", "NO_ACTION"]
EvidenceType = Literal["temporal", "historical", "cross_sensor", "process_rule"]
EvidenceDirection = Literal["supporting", "contradicting", "neutral"]
ConstraintSeverity = Literal["hard", "soft"]
ConstraintStatus = Literal["pass", "fail", "warning", "unknown"]


class FaultRecord(BaseModel):
    issue_id: str
    source_record_id: str
    sensor_id: str
    timestamp: str
    fault_type: str
    detector: str
    detection_score: float | None = None
    reason: str
    row_position: int | None = None
    original_value: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRecord(BaseModel):
    evidence_id: str
    issue_id: str
    evidence_type: EvidenceType
    direction: EvidenceDirection
    explanation: str
    source_sensor: str | None = None
    source_record_id: str | None = None
    strength: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReconstructionCandidate(BaseModel):
    candidate_id: str
    issue_id: str
    method: str
    original_value: float | None = None
    candidate_value: float
    evidence_references: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class ConstraintResult(BaseModel):
    rule_id: str
    applicable_sensor: str
    severity: ConstraintSeverity
    status: ConstraintStatus
    explanation: str
    observed_value: float | None = None
    expected_parameters: dict[str, Any] = Field(default_factory=dict)


class ConfidenceAssessment(BaseModel):
    score: float | None
    score_type: Literal["heuristic", "not_available"] = "heuristic"
    supporting_factors: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    calibration_status: Literal["uncalibrated", "calibrated"] = "uncalibrated"


class DecisionRecord(BaseModel):
    issue_id: str
    candidate_id: str | None = None
    decision: DecisionType
    reason: str
    confidence_value: float | None = None
    confidence_type: Literal["not_available", "heuristic", "calibrated_probability"] = "heuristic"
    confidence_assessment: ConfidenceAssessment | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    constraint_rule_ids: list[str] = Field(default_factory=list)


class AuditRecord(BaseModel):
    run_id: str
    issue_id: str
    source_record_id: str
    sensor_id: str
    timestamp: str
    original_value: Any = None
    candidate_value: float | None = None
    candidate_method: str | None = None
    evidence_records: list[dict[str, Any]] = Field(default_factory=list)
    constraint_results: list[dict[str, Any]] = Field(default_factory=list)
    heuristic_score: float | None = None
    score_type: str = "heuristic"
    decision: DecisionType
    decision_rationale: str
    config_hash: str
    algorithm_version: str
    audit_timestamp_utc: str
