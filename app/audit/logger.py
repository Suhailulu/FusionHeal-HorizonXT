from __future__ import annotations
import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from app.config import Settings
from app.storage.models import AuditLog
from app.storage.repository import TelemetryRepository
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.decision_engine import DecisionRecord


class AuditService:
    def __init__(self, repository: TelemetryRepository, settings: Settings):
        self.repository = repository
        self.settings = settings

    def log_decision(
        self,
        reading: TelemetryReadingInput,
        decision: DecisionRecord,
        event_id: Optional[str] = None,
    ) -> AuditLog:
        """Create and persist an append-only audit record with cryptographic hash chaining."""
        now = datetime.now(timezone.utc)
        audit_id = f"AUD-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        prev_hash = self.repository.get_latest_audit_hash()
        config_hash = self.settings.compute_config_hash()

        # Deterministic serialization for hashing
        payload_to_hash = {
            "audit_id": audit_id,
            "reading_id": reading.reading_id,
            "event_id": event_id,
            "asset_id": reading.asset_id,
            "sensor_id": reading.sensor_id,
            "timestamp": reading.timestamp.isoformat(),
            "original_value": decision.original_value,
            "candidate_value": decision.candidate_value,
            "accepted_value": decision.accepted_value,
            "decision_status": decision.status,
            "validation_result": decision.validation_result,
            "decision_reason": decision.decision_reason,
            "model_version": decision.model_version,
            "rule_config_version": decision.rule_config_version,
            "config_hash": config_hash,
            "prev_record_hash": prev_hash or "GENESIS_ROOT",
        }
        serialized = json.dumps(payload_to_hash, sort_keys=True, default=str)
        record_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        audit_entry = AuditLog(
            audit_id=audit_id,
            reading_id=reading.reading_id,
            event_id=event_id,
            asset_id=reading.asset_id,
            sensor_id=reading.sensor_id,
            timestamp=reading.timestamp,
            original_value=decision.original_value,
            candidate_value=decision.candidate_value,
            accepted_value=decision.accepted_value,
            decision_status=decision.status,
            validation_result=decision.validation_result,
            decision_reason=decision.decision_reason,
            model_version=decision.model_version,
            rule_config_version=decision.rule_config_version,
            config_hash=config_hash,
            record_hash=record_hash,
            prev_record_hash=prev_hash,
            created_at=now,
        )

        return self.repository.save_audit_log(audit_entry)

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify the cryptographic hash chain of stored audit records."""
        records = self.repository.db.query(AuditLog).order_by(AuditLog.id.asc()).all()
        if not records:
            return {"status": "empty", "valid": True, "count": 0}

        prev_hash = None
        for rec in records:
            if rec.prev_record_hash != prev_hash:
                return {
                    "status": "broken_chain",
                    "valid": False,
                    "failed_audit_id": rec.audit_id,
                    "expected_prev": prev_hash,
                    "actual_prev": rec.prev_record_hash,
                }
            prev_hash = rec.record_hash

        return {"status": "valid", "valid": True, "count": len(records)}
