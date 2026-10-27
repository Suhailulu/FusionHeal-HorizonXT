from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.config import get_settings
from app.storage.database import get_db
from app.storage.repository import TelemetryRepository
from app.audit.logger import AuditService

router = APIRouter(prefix="/api", tags=["Audit & Provenance"])
settings = get_settings()


class AuditLogResponse(BaseModel):
    id: int
    audit_id: str
    reading_id: Optional[str]
    event_id: Optional[str]
    asset_id: str
    sensor_id: str
    timestamp: datetime
    original_value: Optional[float]
    candidate_value: Optional[float]
    accepted_value: Optional[float]
    decision_status: str
    validation_result: str
    decision_reason: str
    model_version: Optional[str]
    rule_config_version: str
    config_hash: str
    record_hash: str
    prev_record_hash: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/audit", response_model=List[AuditLogResponse], summary="Retrieve append-oriented audit log")
def get_audit(
    asset_id: Optional[str] = Query(None, description="Filter by asset ID"),
    sensor_id: Optional[str] = Query(None, description="Filter by sensor ID"),
    decision_status: Optional[str] = Query(None, description="Filter by decision status (PRESERVE, REPAIR, UNRESOLVED)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve immutable, cryptographically chained audit history for every decision."""
    repo = TelemetryRepository(db)
    return repo.get_audit_logs(
        asset_id=asset_id,
        sensor_id=sensor_id,
        decision_status=decision_status,
        limit=limit,
        offset=offset,
    )


@router.get("/audit/verify", summary="Verify cryptographic audit hash chain integrity")
def verify_audit_chain(db: Session = Depends(get_db)):
    """Verifies that the entire historical SHA-256 hash chain is untampered and mathematically valid."""
    repo = TelemetryRepository(db)
    service = AuditService(repo, settings)
    return service.verify_integrity()
