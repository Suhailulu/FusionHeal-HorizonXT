from typing import Any, Literal
from pydantic import BaseModel, Field

DecisionType = Literal["ACCEPT", "REVIEW", "REJECT", "NO_ACTION"]

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

class ReconstructionCandidate(BaseModel):
    candidate_id: str
    issue_id: str
    method: str
    original_value: float | None
    candidate_value: float
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

class DecisionRecord(BaseModel):
    issue_id: str
    candidate_id: str | None
    decision: DecisionType
    reason: str
    confidence_value: float | None = None
    confidence_type: Literal["not_available", "heuristic", "calibrated_probability"] = "not_available"
