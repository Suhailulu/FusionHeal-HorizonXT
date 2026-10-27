from __future__ import annotations
import json
import uuid
from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput, TelemetryProcessingResult
from app.ingestion.validator import TelemetryValidator
from app.storage.models import RawReading, QualityEvent, AcceptedReading
from app.storage.repository import TelemetryRepository
from app.pipeline.detection import QualityDetectionEngine, QualityDetectionEvent
from app.pipeline.reconstruction import (
    ReconstructionModel,
    PlaceholderReconstructionAdapter,
    SpatialTemporalTransformerAdapter,
)
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
        self.model = model or SpatialTemporalTransformerAdapter(
            model_path=settings.transformer_model_path,
            device=settings.transformer_device,
        )

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
        received_at = datetime.now(timezone.utc)
        channel_state = self.detector.get_channel_state(reading.asset_id, reading.sensor_id)
        val_outcome = self.validator.validate_reading(
            reading=reading,
            last_timestamp=channel_state.last_timestamp,
            last_sequence=channel_state.last_sequence,
        )

        event_ts = reading.event_timestamp or reading.timestamp
        if event_ts.tzinfo is None:
            event_ts = event_ts.replace(tzinfo=timezone.utc)
        else:
            event_ts = event_ts.astimezone(timezone.utc)

        # ---------------------------------------------------------------------
        # 2. Raw Data Preservation (Mandatory: Before any repair or rejection!)
        # ---------------------------------------------------------------------
        existing = repo.get_raw_reading_by_id(reading.reading_id)
        if existing:
            processed_at = datetime.now(timezone.utc)
            # Idempotent response for duplicate submission
            return TelemetryProcessingResult(
                event_id=f"FH-IDEMP-{reading.reading_id}",
                reading_id=reading.reading_id,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                event_timestamp=existing.event_timestamp or existing.timestamp,
                received_at=received_at,
                processed_at=processed_at,
                issue_type=None,
                quality_status=existing.quality_status,
                decision_status="PRESERVE",
                status="PRESERVE",
                human_review_required=False,
                original_value=existing.value,
                candidate_value=None,
                final_value=existing.value,
                unit=reading.unit,
                validation_status="PASS",
                confidence=None,
                reason="Duplicate reading ID ignored; previous raw observation preserved.",
                decision_reason="Duplicate reading ID ignored; previous raw observation preserved.",
                model_version="unavailable",
                source_type=reading.source_type,
                issues_detected=[],
                audit_id="IDEMPOTENT",
            )

        # ---------------------------------------------------------------------
        # 3. Data Quality Detection
        # ---------------------------------------------------------------------
        issues = self.detector.detect_issues(reading)
        if not val_outcome.is_valid:
            for err in val_outcome.errors:
                issues.append(
                    QualityDetectionEvent(
                        event_id=f"EVT-VALERR-{uuid.uuid4().hex[:8]}",
                        asset_id=reading.asset_id,
                        sensor_id=reading.sensor_id,
                        timestamp=event_ts,
                        event_timestamp=event_ts,
                        issue_type="INVALID_CORRUPTED",
                        detection_rule="SchemaAndConstraintValidatorRule",
                        observed_evidence={"error": err, "validation_status": val_outcome.status},
                        severity="HIGH",
                        reading_id=reading.reading_id,
                    )
                )

        # Determine quality status (VALID, SUSPECT, MISSING, INVALID)
        if not val_outcome.is_valid:
            quality_status = "INVALID"
        elif any(i.issue_type == "MISSING_DATA" for i in issues):
            quality_status = "MISSING"
        elif any(i.issue_type == "INVALID_CORRUPTED" for i in issues):
            quality_status = "INVALID"
        elif issues:
            quality_status = "SUSPECT"
        else:
            quality_status = "VALID"

        raw_reading = RawReading(
            reading_id=reading.reading_id,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            timestamp=event_ts,
            event_timestamp=event_ts,
            ingested_at=received_at,
            received_at=received_at,
            processed_at=received_at,
            sequence_number=reading.sequence_number,
            value=reading.value,
            is_missing=(reading.value is None),
            quality_status=quality_status,
            unit=reading.unit,
            source_type=reading.source_type,
            raw_payload=json.dumps(reading.model_dump(mode="json"), default=str),
            validation_status=val_outcome.status,
            validation_error="; ".join(val_outcome.errors) if val_outcome.errors else None,
        )
        repo.save_raw_reading(raw_reading)

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
                event_timestamp=issue.event_timestamp or issue.timestamp,
                end_timestamp=issue.end_timestamp,
                received_at=received_at,
                processed_at=received_at,
                issue_type=issue.issue_type,
                quality_status=quality_status,
                detection_rule=issue.detection_rule,
                observed_evidence=json.dumps(issue.observed_evidence),
                severity=issue.severity,
                decision_status="UNRESOLVED",  # updated after decision
                human_review_required=False,
                original_value=reading.value,
                candidate_value=None,
                final_value=reading.value,
                unit=reading.unit,
                source_type=reading.source_type,
            )
            repo.save_quality_event(evt_model)

        # ---------------------------------------------------------------------
        # 4. ML Inference Interface
        # ---------------------------------------------------------------------
        history_window = [
            {"timestamp": ts.isoformat(), "value": val}
            for ts, val in channel_state.history
        ]
        prediction = self.model.safe_predict(
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            sensor_window=history_window,
            target_timestamp=event_ts,
        )

        # ---------------------------------------------------------------------
        # 5. Decision Engine
        # ---------------------------------------------------------------------
        decision = self.decision_engine.decide(
            reading=reading,
            issues=issues,
            prediction=prediction,
            previous_value=channel_state.last_valid_value,
        )

        processed_at = datetime.now(timezone.utc)

        # Update event decision statuses & metadata
        for issue in issues:
            saved_evt = repo.get_event_by_id(issue.event_id)
            if saved_evt:
                saved_evt.decision_status = decision.decision_status
                saved_evt.human_review_required = decision.human_review_required
                saved_evt.candidate_value = decision.candidate_value
                saved_evt.final_value = decision.final_value
                saved_evt.original_value = decision.original_value
                saved_evt.reason = decision.decision_reason
                saved_evt.validation_status = decision.validation_result
                saved_evt.model_version = decision.model_version or "unavailable"
                saved_evt.processed_at = processed_at
                db.commit()

        # ---------------------------------------------------------------------
        # 6. Accepted Output Storage (Strictly separated from raw telemetry!)
        # ---------------------------------------------------------------------
        chosen_event_id = primary_event_id or f"FH-{reading.reading_id}"
        issue_type_primary = issues[0].issue_type if issues else None

        accepted = AcceptedReading(
            event_id=chosen_event_id,
            reading_id=reading.reading_id,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            timestamp=event_ts,
            event_timestamp=event_ts,
            received_at=received_at,
            processed_at=processed_at,
            issue_type=issue_type_primary,
            quality_status=quality_status,
            status=decision.status,
            decision_status=decision.decision_status,
            human_review_required=decision.human_review_required,
            original_value=decision.original_value,
            candidate_value=decision.candidate_value,
            final_value=decision.final_value,
            accepted_value=decision.accepted_value,
            unit=reading.unit,
            validation_status=decision.validation_result,
            confidence=decision.confidence_score,
            reason=decision.decision_reason,
            model_version=decision.model_version or "unavailable",
            source_type=reading.source_type,
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
            event_id=chosen_event_id,
            reading_id=reading.reading_id,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            event_timestamp=event_ts,
            timestamp=event_ts,
            received_at=received_at,
            processed_at=processed_at,
            issue_type=issue_type_primary,
            quality_status=quality_status,
            decision_status=decision.decision_status,
            status=decision.status,
            human_review_required=decision.human_review_required,
            original_value=decision.original_value,
            candidate_value=decision.candidate_value,
            final_value=decision.final_value,
            unit=reading.unit,
            validation_status=decision.validation_result,
            confidence=decision.confidence_score,
            reason=decision.decision_reason,
            decision_reason=decision.decision_reason,
            model_version=decision.model_version or "unavailable",
            source_type=reading.source_type,
            issues_detected=[i.issue_type for i in issues],
            audit_id=audit_record.audit_id,
            inference_latency_ms=prediction.inference_latency_ms,
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
