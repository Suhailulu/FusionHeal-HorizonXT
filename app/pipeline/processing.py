from __future__ import annotations
import json
from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput, TelemetryProcessingResult
from app.ingestion.validator import TelemetryValidator
from app.storage.models import RawReading, QualityEvent, AcceptedReading
from app.storage.repository import TelemetryRepository
from app.pipeline.detection import QualityDetectionEngine
from app.pipeline.reconstruction import ReconstructionModel, PlaceholderReconstructionAdapter
from app.pipeline.decision_engine import DecisionEngine
from app.audit.logger import AuditService


class PipelineProcessor:
    """Core orchestration pipeline for FusionHeal."""

    def __init__(
        self,
        settings: Settings,
        model: Optional[ReconstructionModel] = None,
    ):
        self.settings = settings
        self.validator = TelemetryValidator(settings)
        self.detector = QualityDetectionEngine(settings)
        self.decision_engine = DecisionEngine(settings)
        self.model = model or PlaceholderReconstructionAdapter()

    def reset_state(self) -> None:
        """Reset internal channel state (useful for test isolation or pipeline restarts)."""
        self.detector.reset()

    def process_reading(
        self,
        reading: TelemetryReadingInput,
        db: Session,
    ) -> TelemetryProcessingResult:
        repo = TelemetryRepository(db)
        audit_service = AuditService(repo, self.settings)

        # ---------------------------------------------------------------------
        # 1. State lookup for sequence and ordering validation
        # ---------------------------------------------------------------------
        channel_state = self.detector.get_channel_state(reading.asset_id, reading.sensor_id)
        val_outcome = self.validator.validate_reading(
            reading=reading,
            last_timestamp=channel_state.last_timestamp,
            last_sequence=channel_state.last_sequence,
        )

        # ---------------------------------------------------------------------
        # 2. Raw Data Preservation (Mandatory: Before any repair or rejection!)
        # ---------------------------------------------------------------------
        # Check if already exists for idempotency
        existing = repo.get_raw_reading_by_id(reading.reading_id)
        if existing:
            # Idempotent response for duplicate submission
            return TelemetryProcessingResult(
                reading_id=reading.reading_id,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                timestamp=reading.timestamp,
                original_value=existing.value,
                final_value=existing.value,
                status="PRESERVE",
                issues_detected=[],
                decision_reason="Duplicate reading ID ignored; previous raw observation preserved.",
                audit_id="IDEMPOTENT",
            )

        raw_reading = RawReading(
            reading_id=reading.reading_id,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            timestamp=reading.timestamp,
            sequence_number=reading.sequence_number,
            value=reading.value,
            is_missing=(reading.value is None),
            unit=reading.unit,
            source_type=reading.source_type,
            raw_payload=json.dumps(reading.model_dump(), default=str),
            validation_status=val_outcome.status,
            validation_error="; ".join(val_outcome.errors) if val_outcome.errors else None,
        )
        repo.save_raw_reading(raw_reading)

        # ---------------------------------------------------------------------
        # 3. Data Quality Detection
        # ---------------------------------------------------------------------
        issues = self.detector.detect_issues(reading)
        # Also include any validator errors as invalid/corrupted issues
        if not val_outcome.is_valid:
            from app.pipeline.detection import QualityDetectionEvent
            import uuid
            for err in val_outcome.errors:
                issues.append(
                    QualityDetectionEvent(
                        event_id=f"EVT-VALERR-{uuid.uuid4().hex[:8]}",
                        asset_id=reading.asset_id,
                        sensor_id=reading.sensor_id,
                        timestamp=reading.timestamp,
                        issue_type="INVALID_CORRUPTED",
                        detection_rule="SchemaAndConstraintValidatorRule",
                        observed_evidence={"error": err, "validation_status": val_outcome.status},
                        severity="HIGH",
                        reading_id=reading.reading_id,
                    )
                )

        # Persist detected events to database
        primary_event_id: Optional[str] = None
        for issue in issues:
            primary_event_id = issue.event_id
            evt_model = QualityEvent(
                event_id=issue.event_id,
                reading_id=reading.reading_id,
                asset_id=issue.asset_id,
                sensor_id=issue.sensor_id,
                timestamp=issue.timestamp,
                end_timestamp=issue.end_timestamp,
                issue_type=issue.issue_type,
                detection_rule=issue.detection_rule,
                observed_evidence=json.dumps(issue.observed_evidence),
                severity=issue.severity,
                decision_status="UNRESOLVED",  # updated after decision
            )
            repo.save_quality_event(evt_model)

        # ---------------------------------------------------------------------
        # 4. ML Inference Interface
        # ---------------------------------------------------------------------
        # Prepare sensor window context for model
        history_window = [
            {"timestamp": ts.isoformat(), "value": val}
            for ts, val in channel_state.history
        ]
        prediction = self.model.predict(
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            sensor_window=history_window,
        )

        # ---------------------------------------------------------------------
        # 5. Decision Engine
        # ---------------------------------------------------------------------
        decision = self.decision_engine.decide(
            reading=reading,
            issues=issues,
            prediction=prediction,
        )

        # Update event decision statuses if events occurred
        for issue in issues:
            saved_evt = repo.get_event_by_id(issue.event_id)
            if saved_evt:
                saved_evt.decision_status = decision.status
                db.commit()

        # ---------------------------------------------------------------------
        # 6. Accepted Output Storage (Strictly separated from raw telemetry!)
        # ---------------------------------------------------------------------
        accepted = AcceptedReading(
            reading_id=reading.reading_id,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            timestamp=reading.timestamp,
            original_value=decision.original_value,
            accepted_value=decision.accepted_value,
            status=decision.status,
            repair_method=decision.repair_method,
        )
        repo.save_accepted_reading(accepted)

        # ---------------------------------------------------------------------
        # 7. Audit & Provenance Logging
        # ---------------------------------------------------------------------
        audit_record = audit_service.log_decision(
            reading=reading,
            decision=decision,
            event_id=primary_event_id,
        )

        return TelemetryProcessingResult(
            reading_id=reading.reading_id,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            timestamp=reading.timestamp,
            original_value=decision.original_value,
            final_value=decision.accepted_value,
            status=decision.status,
            issues_detected=[i.issue_type for i in issues],
            decision_reason=decision.decision_reason,
            audit_id=audit_record.audit_id,
        )

    def process_batch(
        self,
        readings: List[TelemetryReadingInput],
        db: Session,
    ) -> List[TelemetryProcessingResult]:
        results: List[TelemetryProcessingResult] = []
        for r in readings:
            res = self.process_reading(r, db)
            results.append(res)
        return results
