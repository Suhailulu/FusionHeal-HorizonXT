from __future__ import annotations
from typing import Any, List, Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator
import math


SourceType = Literal["simulated", "physical"]
QualityStatus = Literal["VALID", "SUSPECT", "MISSING", "INVALID"]
DecisionStatus = Literal["PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"]


class TelemetryReadingInput(BaseModel):
    reading_id: str = Field(..., description="Unique reading identifier (UUID or stable serial ID)", min_length=1)
    asset_id: str = Field(..., description="Industrial asset identifier (e.g. boiler_01, pump_01)", min_length=1)
    sensor_id: str = Field(..., description="Sensor identifier (e.g. temperature_01, vibration_01)", min_length=1)
    event_timestamp: Optional[datetime] = Field(None, description="ISO 8601 observation timestamp normalized to UTC")
    # For backward compatibility if callers pass 'timestamp'
    timestamp: Optional[datetime] = Field(None, description="Alias/fallback for event_timestamp")
    sequence_number: int = Field(..., ge=0, description="Monotonically increasing sequence number per sensor stream")
    value: Optional[float] = Field(None, description="Numeric sensor measurement or null for explicit missing reading")
    unit: str = Field(..., description="Measurement unit (e.g. C, m/s2)", min_length=1)
    source_type: str = Field("simulated", description="Data origin vocabulary: 'simulated' or 'physical'")

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean not in ("simulated", "physical"):
            raise ValueError(f"source_type must be 'simulated' or 'physical', got '{v}'.")
        return v_clean

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Sensor reading value must be a finite number or null (NaN and Infinity rejected).")
        return v

    @model_validator(mode="after")
    def validate_and_normalize_timestamps(self) -> TelemetryReadingInput:
        raw_ts = self.event_timestamp or self.timestamp
        if raw_ts is None:
            raise ValueError("Either 'event_timestamp' or 'timestamp' must be provided.")

        if raw_ts.tzinfo is None:
            normalized = raw_ts.replace(tzinfo=timezone.utc)
        else:
            normalized = raw_ts.astimezone(timezone.utc)

        self.event_timestamp = normalized
        self.timestamp = normalized
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "reading_id": "RD-000001",
                "asset_id": "boiler_01",
                "sensor_id": "temperature_01",
                "event_timestamp": "2026-10-27T04:18:30.000Z",
                "sequence_number": 1001,
                "value": 72.5,
                "unit": "C",
                "source_type": "simulated"
            }
        }
    }


class TelemetryBatchInput(BaseModel):
    readings: List[TelemetryReadingInput] = Field(..., min_length=1)


class TelemetryProcessingResult(BaseModel):
    event_id: str
    reading_id: Optional[str] = None
    asset_id: str
    sensor_id: str
    event_timestamp: datetime
    received_at: Optional[datetime] = None
    processed_at: datetime
    issue_type: Optional[str] = None
    quality_status: str = "VALID"  # VALID, SUSPECT, MISSING, INVALID
    decision_status: str = "PRESERVE"  # PRESERVE, REPAIR, REVIEW, UNRESOLVED
    original_value: Optional[float] = None
    candidate_value: Optional[float] = None
    final_value: Optional[float] = None
    unit: str
    validation_status: str
    confidence: Optional[float] = None
    reason: str
    human_review_required: bool = False
    model_version: str = "unavailable"
    source_type: str = "simulated"

    # Backward compatibility attributes
    status: str = "PRESERVE"
    decision_reason: str = ""
    issues_detected: List[str] = Field(default_factory=list)
    audit_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    inference_latency_ms: Optional[float] = None

    @model_validator(mode="after")
    def sync_compat_fields(self) -> TelemetryProcessingResult:
        if not self.decision_reason and self.reason:
            self.decision_reason = self.reason
        if not self.reason and self.decision_reason:
            self.reason = self.decision_reason
        if not self.timestamp:
            self.timestamp = self.event_timestamp
        if not self.status:
            self.status = self.decision_status
        if self.decision_status == "REVIEW":
            self.human_review_required = True
        return self


class TelemetryBatchResponse(BaseModel):
    total_received: int
    processed: int
    results: List[TelemetryProcessingResult]


class ReviewDecisionInput(BaseModel):
    new_status: Literal["PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"] = Field(
        ..., description="New official decision status"
    )
    actor: str = Field("operator", description="Operator identifier or username", min_length=1)
    reason: str = Field(..., min_length=3, description="Technical rationale for review decision")
    accepted_value: Optional[float] = Field(None, description="Approved replacement value if transitioning to REPAIR")


class OverviewResponse(BaseModel):
    total_readings_received: int
    readings_preserved: int
    repairs_accepted: int
    readings_under_review: int
    unresolved_events: int
    issues_detected: int
    fault_distribution: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    database: str
    environment: str
    timestamp: datetime
