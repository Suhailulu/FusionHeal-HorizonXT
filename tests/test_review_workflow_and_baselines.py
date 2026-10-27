from __future__ import annotations
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.config import get_settings
from app.storage.database import get_db
from app.storage.models import RawReading, QualityEvent, AcceptedReading, AuditLog
from app.storage.repository import TelemetryRepository
from app.audit.logger import AuditService
from app.evaluation.metrics import run_reproducible_benchmark, EvaluationSuite
from app.simulator.generator import SyntheticSensorGenerator
from app.simulator.fault_injector import FaultInjector

client = TestClient(app)


def test_four_official_decision_statuses():
    """Verify that PRESERVE, REPAIR, REVIEW, and UNRESOLVED are all valid official statuses."""
    from app.ingestion.schemas import TelemetryProcessingResult, ReviewDecisionInput

    for status_name in ("PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"):
        res = TelemetryProcessingResult(
            event_id=f"EVT-TEST-{status_name}",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            event_timestamp=datetime.now(timezone.utc),
            processed_at=datetime.now(timezone.utc),
            decision_status=status_name,
            original_value=72.0,
            final_value=72.0,
            unit="C",
            validation_status="PASS",
            reason=f"Testing {status_name}",
        )
        assert res.decision_status == status_name
        if status_name == "REVIEW":
            assert res.human_review_required is True


def test_review_queue_and_transitions(db_session=None):
    """Test retrieving pending reviews and executing transitions to PRESERVE, REPAIR, UNRESOLVED, REVIEW."""
    from conftest import TestingSessionLocal
    db: Session = TestingSessionLocal()
    repo = TelemetryRepository(db)

    now = datetime.now(timezone.utc)
    # Create an initial event in REVIEW status
    evt = QualityEvent(
        event_id="EVT-REV-001",
        reading_id="RD-REV-001",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=now,
        event_timestamp=now,
        issue_type="SUDDEN_SPIKE",
        quality_status="SUSPECT",
        decision_status="REVIEW",
        human_review_required=True,
        original_value=120.0,
        candidate_value=74.5,
        final_value=120.0,  # REVIEW keeps original value intact
        unit="C",
        validation_status="PASS",
        reason="Borderline model reconstruction confidence; operator assessment needed.",
        model_version="test-model-1.0",
        source_type="simulated",
        detection_rule="SpikeDetector",
        observed_evidence='{"spike_val": 120.0}',
        severity="MEDIUM",
    )
    repo.save_quality_event(evt)

    acc = AcceptedReading(
        event_id="EVT-REV-001",
        reading_id="RD-REV-001",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=now,
        event_timestamp=now,
        received_at=now,
        processed_at=now,
        issue_type="SUDDEN_SPIKE",
        quality_status="SUSPECT",
        status="REVIEW",
        decision_status="REVIEW",
        human_review_required=True,
        original_value=120.0,
        candidate_value=74.5,
        final_value=120.0,
        unit="C",
        validation_status="PASS",
        reason="Pending operator evaluation",
    )
    repo.save_accepted_reading(acc)

    # 1. Query pending review queue
    resp = client.get("/api/reviews")
    assert resp.status_code == 200
    reviews = resp.json()
    assert len(reviews) >= 1
    found = [r for r in reviews if r["event_id"] == "EVT-REV-001"]
    assert len(found) == 1
    assert found[0]["human_review_required"] is True
    assert found[0]["decision_status"] == "REVIEW"

    # 2. Transition: REVIEW -> REVIEW (further investigation required)
    rev_payload = {
        "new_status": "REVIEW",
        "actor": "operator_jane",
        "reason": "Requested field inspection of thermocouple probe wiring.",
    }
    resp = client.post("/api/events/EVT-REV-001/review", json=rev_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision_status"] == "REVIEW"
    assert data["human_review_required"] is True

    # 3. Transition: REVIEW -> REPAIR with physical bounds validation
    # Test invalid out-of-bounds repair (Boiler max is 150 C, attempt 250 C)
    bad_repair = {
        "new_status": "REPAIR",
        "actor": "operator_jane",
        "reason": "Attempting gross invalid replacement",
        "accepted_value": 250.0,
    }
    resp = client.post("/api/events/EVT-REV-001/review", json=bad_repair)
    assert resp.status_code == 422
    assert "violates physical bounds" in resp.json()["detail"]

    # Test valid repair (74.0 C within bounds)
    good_repair = {
        "new_status": "REPAIR",
        "actor": "operator_jane",
        "reason": "Thermocouple probe checked, verified true process temperature was 74.0 C.",
        "accepted_value": 74.0,
    }
    resp = client.post("/api/events/EVT-REV-001/review", json=good_repair)
    assert resp.status_code == 200
    repaired_data = resp.json()
    assert repaired_data["decision_status"] == "REPAIR"
    assert repaired_data["final_value"] == 74.0
    assert repaired_data["original_value"] == 120.0
    assert repaired_data["human_review_required"] is False

    # 4. Attempting transition on already resolved event should be rejected
    dup_transition = {
        "new_status": "PRESERVE",
        "actor": "operator_jane",
        "reason": "Attempting transition on non-review event",
    }
    resp = client.post("/api/events/EVT-REV-001/review", json=dup_transition)
    assert resp.status_code == 400
    assert "Only events in 'REVIEW' status can be transitioned" in resp.json()["detail"]


def test_transition_to_preserve_and_unresolved():
    """Test transitions from REVIEW to PRESERVE and UNRESOLVED."""
    from conftest import TestingSessionLocal
    db: Session = TestingSessionLocal()
    repo = TelemetryRepository(db)
    now = datetime.now(timezone.utc)

    # Event 2 for PRESERVE transition
    evt2 = QualityEvent(
        event_id="EVT-REV-002",
        reading_id="RD-REV-002",
        asset_id="pump_01",
        sensor_id="vibration_01",
        timestamp=now,
        event_timestamp=now,
        issue_type="FROZEN_SENSOR",
        decision_status="REVIEW",
        human_review_required=True,
        original_value=2.8,
        final_value=2.8,
        unit="m/s2",
        validation_status="PASS",
        detection_rule="FrozenDetector",
        observed_evidence="{}",
        severity="LOW",
    )
    repo.save_quality_event(evt2)

    resp = client.post(
        "/api/events/EVT-REV-002/review",
        json={
            "new_status": "PRESERVE",
            "actor": "operator_bob",
            "reason": "Pump operates in steady state; constant vibration reading is genuine machine behavior.",
        },
    )
    assert resp.status_code == 200
    p_data = resp.json()
    assert p_data["decision_status"] == "PRESERVE"
    assert p_data["final_value"] == 2.8
    assert p_data["original_value"] == 2.8
    assert p_data["human_review_required"] is False

    # Event 3 for UNRESOLVED transition
    evt3 = QualityEvent(
        event_id="EVT-REV-003",
        reading_id="RD-REV-003",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=now,
        event_timestamp=now,
        issue_type="INVALID_CORRUPTED",
        decision_status="REVIEW",
        human_review_required=True,
        original_value=350.0,
        final_value=350.0,
        unit="C",
        validation_status="FAIL",
        detection_rule="RangeValidator",
        observed_evidence="{}",
        severity="HIGH",
    )
    repo.save_quality_event(evt3)

    resp = client.post(
        "/api/events/EVT-REV-003/review",
        json={
            "new_status": "UNRESOLVED",
            "actor": "operator_bob",
            "reason": "Severe signal corruption without nearby context. Unresolvable at gateway layer.",
        },
    )
    assert resp.status_code == 200
    u_data = resp.json()
    assert u_data["decision_status"] == "UNRESOLVED"
    assert u_data["final_value"] == 350.0  # Original kept intact
    assert u_data["original_value"] == 350.0
    assert u_data["human_review_required"] is False


def test_review_transition_audit_chain_integrity():
    """Verify that review transitions generate cryptographic audit logs and maintain unbroken hash chains."""
    from conftest import TestingSessionLocal
    db: Session = TestingSessionLocal()
    repo = TelemetryRepository(db)
    settings = get_settings()
    audit_svc = AuditService(repo, settings)

    now = datetime.now(timezone.utc)
    evt = QualityEvent(
        event_id="EVT-AUD-CHAIN",
        reading_id="RD-AUD-CHAIN",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=now,
        event_timestamp=now,
        issue_type="SUDDEN_SPIKE",
        decision_status="REVIEW",
        human_review_required=True,
        original_value=115.0,
        candidate_value=72.0,
        final_value=115.0,
        unit="C",
        validation_status="PASS",
        detection_rule="SpikeDetector",
        observed_evidence="{}",
        severity="MEDIUM",
    )
    repo.save_quality_event(evt)

    # Perform review transition via API
    resp = client.post(
        "/api/events/EVT-AUD-CHAIN/review",
        json={
            "new_status": "REPAIR",
            "actor": "lead_engineer",
            "reason": "Transient electrical noise confirmed. Reconstructed with physical bounds.",
            "accepted_value": 72.5,
        },
    )
    assert resp.status_code == 200

    # Verify audit integrity
    verification = audit_svc.verify_integrity()
    assert verification["valid"] is True
    assert verification["count"] >= 1

    # Verify audit record contents
    audits = repo.get_audit_logs(limit=5)
    transition_logs = [a for a in audits if a.action == "REVIEW_TRANSITION"]
    assert len(transition_logs) >= 1
    t_log = transition_logs[0]
    assert t_log.actor == "lead_engineer"
    assert t_log.prev_status == "REVIEW"
    assert t_log.new_status == "REPAIR"
    assert t_log.accepted_value == 72.5
    assert t_log.record_hash is not None


def test_reproducible_benchmark_and_3way_baselines():
    """Verify the reproducible benchmark evaluation framework and calculation of all 3 baselines."""
    from conftest import TestingSessionLocal
    db: Session = TestingSessionLocal()

    # Run reproducible benchmark
    report = run_reproducible_benchmark(seed=42, sample_count=40, db=db)

    assert report.run_id.startswith("EVAL-")
    assert report.total_eval_points > 0
    assert report.f1_score is not None
    assert report.precision is not None
    assert report.recall is not None

    # Verify 3-way baseline comparison
    b_comp = report.baseline_comparison
    assert "baseline_1_no_repair" in b_comp
    assert "baseline_2_linear_interpolation" in b_comp
    assert "baseline_3_fusionheal" in b_comp

    b1 = b_comp["baseline_1_no_repair"]
    b2 = b_comp["baseline_2_linear_interpolation"]
    b3 = b_comp["baseline_3_fusionheal"]

    assert b1["name"] == "Baseline 1: No-Repair Baseline"
    assert b2["name"] == "Baseline 2: Linear Interpolation"
    assert b3["name"] == "Baseline 3: FusionHeal Pipeline"

    # FusionHeal pipeline should report valid preserved percentage and decisions
    assert "valid_preserved_percentage" in b3
    assert "decision_distribution" in b3
    for st in ("PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"):
        assert st in b3["decision_distribution"]

    # Verify API metrics endpoint returns baseline comparison
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    m_data = resp.json()
    assert m_data["available"] is True
    assert m_data["baseline_comparison"] is not None
    assert "baseline_1_no_repair" in m_data["baseline_comparison"]
    assert "baseline_2_linear_interpolation" in m_data["baseline_comparison"]
    assert "baseline_3_fusionheal" in m_data["baseline_comparison"]
