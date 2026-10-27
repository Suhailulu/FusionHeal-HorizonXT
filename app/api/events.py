import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.config import get_settings
from app.storage.database import get_db
from app.storage.models import RawReading, AcceptedReading, AuditLog, QualityEvent
from app.storage.repository import TelemetryRepository
from app.ingestion.schemas import ReviewDecisionInput

router = APIRouter(prefix="/api", tags=["Quality Events"])


class QualityEventResponse(BaseModel):
    id: int
    event_id: str
    reading_id: Optional[str] = None
    asset_id: str
    sensor_id: str
    event_timestamp: datetime
    timestamp: datetime
    received_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    issue_type: str
    quality_status: str = "SUSPECT"
    decision_status: str
    original_value: Optional[float] = None
    candidate_value: Optional[float] = None
    final_value: Optional[float] = None
    unit: str = "C"
    validation_status: str = "VALID"
    reason: Optional[str] = None
    human_review_required: bool = False
    model_version: str = "unavailable"
    source_type: str = "simulated"

    # Compatibility & investigation fields
    detection_rule: str
    observed_evidence: Dict[str, Any]
    severity: str
    detected_at: datetime
    explanation: Optional[str] = None
    relevant_observations: Optional[Dict[str, Any]] = None
    candidate_reconstruction: Optional[Dict[str, Any]] = None
    accepted_output: Optional[Dict[str, Any]] = None
    audit_references: List[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}


def parse_event(evt: QualityEvent, db: Optional[Session] = None) -> QualityEventResponse:
    evidence = {}
    try:
        evidence = json.loads(evt.observed_evidence) if evt.observed_evidence else {}
    except Exception:
        evidence = {"raw": evt.observed_evidence}

    explanation = evt.reason or evidence.get("reason") or f"Data-quality issue '{evt.issue_type}' detected by rule '{evt.detection_rule}'."
    relevant_obs = None
    candidate_rec = None
    accepted_out = None
    audit_refs = []

    if db and evt.reading_id:
        raw = db.query(RawReading).filter(RawReading.reading_id == evt.reading_id).first()
        if raw:
            relevant_obs = {
                "reading_id": raw.reading_id,
                "value": raw.value,
                "unit": raw.unit,
                "timestamp": raw.timestamp.isoformat() if raw.timestamp else None,
                "event_timestamp": raw.event_timestamp.isoformat() if raw.event_timestamp else None,
                "validation_status": raw.validation_status,
                "quality_status": raw.quality_status,
            }

        acc = db.query(AcceptedReading).filter(AcceptedReading.reading_id == evt.reading_id).first()
        if acc:
            accepted_out = {
                "status": acc.status,
                "decision_status": acc.decision_status,
                "original_value": acc.original_value,
                "accepted_value": acc.accepted_value,
                "final_value": acc.final_value,
                "repair_method": acc.repair_method,
            }
            if acc.candidate_value is not None:
                candidate_rec = {
                    "candidate_value": acc.candidate_value,
                    "method": acc.repair_method,
                }

        audits = db.query(AuditLog).filter(
            (AuditLog.event_id == evt.event_id) | (AuditLog.reading_id == evt.reading_id)
        ).all()
        audit_refs = [a.audit_id for a in audits]

    return QualityEventResponse(
        id=evt.id,
        event_id=evt.event_id,
        reading_id=evt.reading_id,
        asset_id=evt.asset_id,
        sensor_id=evt.sensor_id,
        timestamp=evt.timestamp,
        event_timestamp=evt.event_timestamp or evt.timestamp,
        received_at=evt.received_at,
        processed_at=evt.processed_at,
        end_timestamp=evt.end_timestamp,
        issue_type=evt.issue_type,
        quality_status=evt.quality_status or "SUSPECT",
        decision_status=evt.decision_status,
        original_value=evt.original_value,
        candidate_value=evt.candidate_value,
        final_value=evt.final_value,
        unit=evt.unit or "C",
        validation_status=evt.validation_status or "VALID",
        reason=evt.reason or explanation,
        human_review_required=bool(evt.human_review_required or evt.decision_status == "REVIEW"),
        model_version=evt.model_version or "unavailable",
        source_type=evt.source_type or "simulated",
        detection_rule=evt.detection_rule,
        observed_evidence=evidence,
        severity=evt.severity,
        detected_at=evt.detected_at,
        explanation=explanation,
        relevant_observations=relevant_obs,
        candidate_reconstruction=candidate_rec,
        accepted_output=accepted_out,
        audit_references=audit_refs,
    )


@router.get("/events", response_model=List[QualityEventResponse], summary="List detected data-quality events")
def list_events(
    issue_type: Optional[str] = Query(None, description="Filter by issue type (e.g. SUDDEN_SPIKE, FROZEN_SENSOR)"),
    asset_id: Optional[str] = Query(None, description="Filter by asset ID (e.g. boiler_01)"),
    sensor_id: Optional[str] = Query(None, description="Filter by sensor ID (e.g. temperature_01)"),
    decision_status: Optional[str] = Query(None, description="Filter by decision status (PRESERVE, REPAIR, REVIEW, UNRESOLVED)"),
    start_time: Optional[datetime] = Query(None, description="Filter by start timestamp (ISO 8601)"),
    end_time: Optional[datetime] = Query(None, description="Filter by end timestamp (ISO 8601)"),
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
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
    return [parse_event(e, db=db) for e in events]


@router.get("/reviews", response_model=List[QualityEventResponse], summary="Pending review queue")
def list_pending_reviews(
    asset_id: Optional[str] = Query(None, description="Filter by asset ID"),
    sensor_id: Optional[str] = Query(None, description="Filter by sensor ID"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve pending review queue events requiring human operator assessment."""
    repo = TelemetryRepository(db)
    events = repo.get_pending_reviews(
        asset_id=asset_id,
        sensor_id=sensor_id,
        limit=limit,
        offset=offset,
    )
    return [parse_event(e, db=db) for e in events]


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
    return parse_event(evt, db=db)


def _process_review_decision(event_id: str, decision_input: ReviewDecisionInput, db: Session) -> QualityEventResponse:
    repo = TelemetryRepository(db)
    evt = repo.get_event_by_id(event_id)
    if not evt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality event with ID '{event_id}' not found.",
        )

    # Transition Guard: Only events currently under REVIEW can be transitioned
    if evt.decision_status != "REVIEW":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Event '{event_id}' is currently '{evt.decision_status}'. Only events in 'REVIEW' status can be transitioned.",
        )

    settings = get_settings()
    now_utc = datetime.now(timezone.utc)
    prev_status = evt.decision_status
    new_status = decision_input.new_status
    accepted_val: Optional[float] = None

    # Handle REPAIR transition and validate acceptance policy
    if new_status == "REPAIR":
        if decision_input.accepted_value is not None:
            accepted_val = decision_input.accepted_value
        elif evt.candidate_value is not None:
            accepted_val = evt.candidate_value
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="A valid accepted_value must be provided or exist on the event when transitioning to REPAIR.",
            )

        # Validate human-approved repair against physical bounds
        constraint = settings.get_constraint(evt.asset_id, evt.sensor_id)
        if constraint:
            if accepted_val < constraint.min_value or accepted_val > constraint.max_value:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Accepted repair value {accepted_val} violates physical bounds "
                        f"[{constraint.min_value}, {constraint.max_value}] {constraint.unit} for {evt.sensor_id}."
                    ),
                )

    # Apply State Updates to QualityEvent
    evt.decision_status = new_status
    evt.human_review_required = (new_status == "REVIEW")
    evt.processed_at = now_utc
    evt.reason = f"Operator review [{decision_input.actor}]: {decision_input.reason}"

    if new_status == "REPAIR":
        evt.final_value = accepted_val
        evt.validation_status = "PASS"
    elif new_status in ("PRESERVE", "UNRESOLVED"):
        evt.final_value = evt.original_value
        evt.validation_status = "PASS" if new_status == "PRESERVE" else "FAIL"

    # Update corresponding AcceptedReading if reading exists
    if evt.reading_id:
        acc = db.query(AcceptedReading).filter(AcceptedReading.reading_id == evt.reading_id).first()
        if acc:
            acc.status = new_status
            acc.decision_status = new_status
            acc.human_review_required = (new_status == "REVIEW")
            acc.processed_at = now_utc
            acc.reason = evt.reason
            if new_status == "REPAIR":
                acc.accepted_value = accepted_val
                acc.final_value = accepted_val
                acc.validation_status = "PASS"
            elif new_status in ("PRESERVE", "UNRESOLVED"):
                acc.final_value = evt.original_value
                acc.accepted_value = evt.original_value if new_status == "PRESERVE" else None

    # Record auditable history log with cryptographic chain hash
    prev_hash = repo.get_latest_audit_hash()
    audit_id = f"AUD-REV-{uuid.uuid4().hex[:8]}"

    payload_for_hash = {
        "audit_id": audit_id,
        "action": "REVIEW_TRANSITION",
        "actor": decision_input.actor,
        "prev_status": prev_status,
        "new_status": new_status,
        "reading_id": evt.reading_id,
        "event_id": evt.event_id,
        "asset_id": evt.asset_id,
        "sensor_id": evt.sensor_id,
        "original_value": evt.original_value,
        "accepted_value": accepted_val if new_status == "REPAIR" else evt.original_value,
        "reason": decision_input.reason,
        "timestamp": now_utc.isoformat(),
        "prev_record_hash": prev_hash,
    }
    rec_hash = hashlib.sha256(json.dumps(payload_for_hash, sort_keys=True).encode("utf-8")).hexdigest()

    audit_entry = AuditLog(
        audit_id=audit_id,
        reading_id=evt.reading_id,
        event_id=evt.event_id,
        asset_id=evt.asset_id,
        sensor_id=evt.sensor_id,
        action="REVIEW_TRANSITION",
        actor=decision_input.actor,
        prev_status=prev_status,
        new_status=new_status,
        timestamp=now_utc,
        event_timestamp=evt.event_timestamp or evt.timestamp,
        original_value=evt.original_value,
        candidate_value=evt.candidate_value,
        accepted_value=accepted_val if new_status == "REPAIR" else evt.original_value,
        quality_status=evt.quality_status,
        decision_status=new_status,
        validation_result="PASS" if new_status != "UNRESOLVED" else "FAIL",
        reason=decision_input.reason,
        decision_reason=f"Status changed from {prev_status} to {new_status} by {decision_input.actor}: {decision_input.reason}",
        model_version=evt.model_version or "unavailable",
        rule_config_version=settings.rule_config_version,
        config_hash="REVIEW_ACTION",
        record_hash=rec_hash,
        prev_record_hash=prev_hash,
        created_at=now_utc,
        processed_at=now_utc,
    )
    repo.save_audit_log(audit_entry)

    db.commit()
    db.refresh(evt)
    return parse_event(evt, db=db)


@router.post("/events/{event_id}/review", response_model=QualityEventResponse, summary="Submit review decision for a pending event")
def submit_event_review(event_id: str, decision_input: ReviewDecisionInput, db: Session = Depends(get_db)):
    """Record operator review outcome for a pending review event and commit auditable transition."""
    return _process_review_decision(event_id, decision_input, db)


@router.post("/reviews/{event_id}/decision", response_model=QualityEventResponse, summary="Submit review decision (alias)")
def submit_review_decision_alias(event_id: str, decision_input: ReviewDecisionInput, db: Session = Depends(get_db)):
    """Alias for review decision submission."""
    return _process_review_decision(event_id, decision_input, db)
