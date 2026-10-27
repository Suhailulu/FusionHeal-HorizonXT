from __future__ import annotations
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.storage.database import Base
from app.storage.models import AuditLog
from app.storage.repository import TelemetryRepository
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.decision_engine import DecisionRecord
from app.audit.logger import AuditService

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def init_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_audit_record_creation_and_tamper_detection():
    db = TestingSessionLocal()
    repo = TelemetryRepository(db)
    settings = Settings()
    audit_service = AuditService(repo, settings)

    # 1. Log three sequential decisions
    reading1 = TelemetryReadingInput(
        reading_id="aud-1",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        sequence_number=1,
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    dec1 = DecisionRecord(
        status="PRESERVE",
        original_value=72.0,
        candidate_value=None,
        accepted_value=72.0,
        repair_method=None,
        decision_reason="Clean observation",
        validation_result="PASS",
        policy_version="1.0.0",
        rule_config_version="2026.1",
    )
    r1 = audit_service.log_decision(reading1, dec1)
    assert r1.prev_record_hash is None

    reading2 = TelemetryReadingInput(
        reading_id="aud-2",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        sequence_number=2,
        value=None,
        unit="C",
        source_type="simulated",
    )
    dec2 = DecisionRecord(
        status="UNRESOLVED",
        original_value=None,
        candidate_value=None,
        accepted_value=None,
        repair_method=None,
        decision_reason="Missing value",
        validation_result="FAIL",
        policy_version="1.0.0",
        rule_config_version="2026.1",
    )
    r2 = audit_service.log_decision(reading2, dec2, event_id="EVT-002")
    assert r2.prev_record_hash == r1.record_hash

    # 2. Verify integrity passes
    ver_clean = audit_service.verify_integrity()
    assert ver_clean["valid"] is True
    assert ver_clean["count"] == 2

    # 3. Simulate unauthorized tampering with historical record
    first_record = db.query(AuditLog).filter(AuditLog.id == 1).first()
    first_record.record_hash = "faked_tampered_hash_value"
    db.commit()

    # 4. Verify integrity detects tamper
    ver_tampered = audit_service.verify_integrity()
    assert ver_tampered["valid"] is False
    assert ver_tampered["status"] == "broken_chain"

    db.close()
