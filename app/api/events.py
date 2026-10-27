from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.storage.database import get_db
from app.storage.repository import TelemetryRepository

router = APIRouter(prefix="/api", tags=["Quality Events"])


class QualityEventResponse(BaseModel):
    id: int
    event_id: str
    reading_id: Optional[str]
    asset_id: str
    sensor_id: str
    timestamp: datetime
    end_timestamp: Optional[datetime]
    issue_type: str
    detection_rule: str
    observed_evidence: Dict[str, Any]
    severity: str
    decision_status: str
    detected_at: datetime

    model_config = {"from_attributes": True}


def parse_event(evt) -> QualityEventResponse:
    evidence = {}
    try:
        evidence = json.loads(evt.observed_evidence) if evt.observed_evidence else {}
    except Exception:
        evidence = {"raw": evt.observed_evidence}

    return QualityEventResponse(
        id=evt.id,
        event_id=evt.event_id,
        reading_id=evt.reading_id,
        asset_id=evt.asset_id,
        sensor_id=evt.sensor_id,
        timestamp=evt.timestamp,
        end_timestamp=evt.end_timestamp,
        issue_type=evt.issue_type,
        detection_rule=evt.detection_rule,
        observed_evidence=evidence,
        severity=evt.severity,
        decision_status=evt.decision_status,
        detected_at=evt.detected_at,
    )


@router.get("/events", response_model=List[QualityEventResponse], summary="List detected data-quality events")
def list_events(
    issue_type: Optional[str] = Query(None, description="Filter by issue type (e.g. SUDDEN_SPIKE, FROZEN_SENSOR)"),
    asset_id: Optional[str] = Query(None, description="Filter by asset ID (e.g. boiler_01)"),
    sensor_id: Optional[str] = Query(None, description="Filter by sensor ID (e.g. temperature_01)"),
    decision_status: Optional[str] = Query(None, description="Filter by decision status (PRESERVE, REPAIR, UNRESOLVED)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve detected industrial data quality anomalies and physical rule violations."""
    repo = TelemetryRepository(db)
    events = repo.get_events(
        issue_type=issue_type,
        asset_id=asset_id,
        sensor_id=sensor_id,
        decision_status=decision_status,
        limit=limit,
        offset=offset,
    )
    return [parse_event(e) for e in events]


@router.get("/events/{event_id}", response_model=QualityEventResponse, summary="Get full details of a single event")
def get_event(event_id: str, db: Session = Depends(get_db)):
    """Retrieve full technical explanation, observed evidence, and resolution for a specific event."""
    repo = TelemetryRepository(db)
    evt = repo.get_event_by_id(event_id)
    if not evt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality event with ID '{event_id}' not found.",
        )
    return parse_event(evt)
