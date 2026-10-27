from __future__ import annotations
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.storage.database import get_db
from app.storage.repository import TelemetryRepository

router = APIRouter(prefix="/api", tags=["Readings"])


class ReadingResponse(BaseModel):
    id: int
    reading_id: str
    asset_id: str
    sensor_id: str
    event_timestamp: datetime
    timestamp: datetime
    received_at: Optional[datetime] = None
    ingested_at: datetime
    processed_at: Optional[datetime] = None
    sequence_number: int
    value: Optional[float]
    is_missing: bool
    quality_status: str = "VALID"
    unit: str
    source_type: str
    validation_status: str

    model_config = {"from_attributes": True}


@router.get("/readings", response_model=List[ReadingResponse], summary="Query preserved raw sensor readings")
def get_readings(
    asset_id: Optional[str] = Query(None, description="Filter by asset ID (e.g. boiler_01)"),
    sensor_id: Optional[str] = Query(None, description="Filter by sensor ID (e.g. temperature_01)"),
    start_time: Optional[datetime] = Query(None, description="Filter by start timestamp (ISO 8601)"),
    end_time: Optional[datetime] = Query(None, description="Filter by end timestamp (ISO 8601)"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Records offset"),
    db: Session = Depends(get_db),
):
    """Retrieve immutable original readings preserved before any pipeline repair or rejection."""
    repo = TelemetryRepository(db)
    readings = repo.get_readings(
        asset_id=asset_id,
        sensor_id=sensor_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return readings
