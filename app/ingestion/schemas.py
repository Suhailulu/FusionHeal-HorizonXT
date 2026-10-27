from __future__ import annotations
from typing import Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import math


class TelemetryReadingInput(BaseModel):
    reading_id: str = Field(..., description="Unique reading identifier", min_length=1)
    asset_id: str = Field(..., description="Asset identifier (e.g. boiler_01)", min_length=1)
    sensor_id: str = Field(..., description="Sensor identifier (e.g. temperature_01)", min_length=1)
    timestamp: datetime = Field(..., description="ISO 8601 observation timestamp")
    sequence_number: int = Field(..., ge=0, description="Monotonically increasing sequence number per sensor")
    value: Optional[float] = Field(None, description="Numeric sensor value or null for explicit missing reading")
    unit: str = Field(..., description="Measurement unit (e.g. C, m/s2)", min_length=1)
    source_type: str = Field("simulated", description="Data origin: simulated or physical")

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Sensor reading value must be a finite number or null.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "reading_id": "read-b1-temp-0001",
                "asset_id": "boiler_01",
                "sensor_id": "temperature_01",
                "timestamp": "2026-10-27T08:00:00Z",
                "sequence_number": 1,
                "value": 72.4,
                "unit": "C",
                "source_type": "simulated"
            }
        }
    }


class TelemetryBatchInput(BaseModel):
    readings: List[TelemetryReadingInput] = Field(..., min_length=1)


class TelemetryProcessingResult(BaseModel):
    reading_id: str
    asset_id: str
    sensor_id: str
    timestamp: datetime
    original_value: Optional[float]
    final_value: Optional[float]
    status: str  # PRESERVE, REPAIR, UNRESOLVED
    issues_detected: List[str]
    decision_reason: str
    audit_id: str


class TelemetryBatchResponse(BaseModel):
    total_received: int
    processed: int
    results: List[TelemetryProcessingResult]


class OverviewResponse(BaseModel):
    total_readings_received: int
    readings_preserved: int
    repairs_accepted: int
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
