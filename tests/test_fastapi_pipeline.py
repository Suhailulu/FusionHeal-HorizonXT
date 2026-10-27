from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import math
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.config import Settings, get_settings
from app.storage.database import Base, get_db
from app.storage.models import RawReading, AcceptedReading, QualityEvent, AuditLog
from app.storage.repository import TelemetryRepository
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.processing import PipelineProcessor
from app.pipeline.reconstruction import ReconstructionModel, ModelPrediction
from app.simulator.generator import SyntheticSensorGenerator
from app.simulator.fault_injector import FaultInjector
from app.evaluation.metrics import EvaluationSuite

from conftest import TestingSessionLocal
client = TestClient(app)



# -----------------------------------------------------------------------------
# Test 1: Service Health Endpoint
# -----------------------------------------------------------------------------
def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "FusionHeal"
    assert data["database"] == "connected"


# -----------------------------------------------------------------------------
# Test 2: Ingest Single Clean Reading -> PRESERVE & Raw Preservation
# -----------------------------------------------------------------------------
def test_ingest_single_reading_preserve():
    payload = {
        "reading_id": "test-read-001",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 72.5,
        "unit": "C",
        "source_type": "simulated",
    }
    resp = client.post("/api/telemetry", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    assert data["reading_id"] == "test-read-001"
    assert data["status"] == "PRESERVE"
    assert data["original_value"] == 72.5
    assert data["final_value"] == 72.5
    assert len(data["issues_detected"]) == 0
    assert "conforms to physical plausibility" in data["decision_reason"]

    # Verify raw storage preserved original observation
    readings_resp = client.get("/api/readings")
    assert readings_resp.status_code == 200
    raw_list = readings_resp.json()
    assert len(raw_list) == 1
    assert raw_list[0]["reading_id"] == "test-read-001"
    assert raw_list[0]["value"] == 72.5
    assert raw_list[0]["validation_status"] == "VALID"


# -----------------------------------------------------------------------------
# Test 3: Idempotent Handling of Duplicate Reading ID
# -----------------------------------------------------------------------------
def test_idempotent_duplicate_reading():
    payload = {
        "reading_id": "dup-read-001",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 70.0,
        "unit": "C",
        "source_type": "simulated",
    }
    # First ingest
    r1 = client.post("/api/telemetry", json=payload)
    assert r1.status_code == 201

    # Second ingest with identical reading_id
    r2 = client.post("/api/telemetry", json=payload)
    assert r2.status_code == 201
    assert r2.json()["audit_id"] == "IDEMPOTENT"

    # Only 1 raw reading in DB
    readings = client.get("/api/readings").json()
    assert len(readings) == 1


# -----------------------------------------------------------------------------
# Test 4: Missing Value -> UNRESOLVED (No ML Fabrication)
# -----------------------------------------------------------------------------
def test_missing_value_detection_unresolved():
    payload = {
        "reading_id": "missing-val-001",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": None,  # explicit null
        "unit": "C",
        "source_type": "simulated",
    }
    resp = client.post("/api/telemetry", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    assert data["status"] == "UNRESOLVED"
    assert data["original_value"] is None
    assert data["final_value"] is None
    assert "MISSING_DATA" in data["issues_detected"]
    assert "Refusing to fabricate replacement" in data["decision_reason"]

    # Verify event stored in /api/events
    evt_resp = client.get("/api/events")
    assert evt_resp.status_code == 200
    events = evt_resp.json()
    assert len(events) >= 1
    assert any(e["issue_type"] == "MISSING_DATA" for e in events)


# -----------------------------------------------------------------------------
# Test 5: Sudden Spike -> Flagged & UNRESOLVED without connected Transformer
# -----------------------------------------------------------------------------
def test_sudden_spike_detection():
    # 1. First reading: normal
    client.post("/api/telemetry", json={
        "reading_id": "spike-init",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 72.0,
        "unit": "C",
        "source_type": "simulated",
    })

    # 2. Spike reading: +50 degrees in 1 second (exceeds max ROC of 5.0 C/s)
    resp = client.post("/api/telemetry", json={
        "reading_id": "spike-event",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:01Z",
        "sequence_number": 2,
        "value": 122.0,
        "unit": "C",
        "source_type": "simulated",
    })
    assert resp.status_code == 201
    data = resp.json()

    assert data["status"] == "UNRESOLVED"
    assert "SUDDEN_SPIKE" in data["issues_detected"]

    # Raw value preserved as 122.0, not overwritten!
    readings = client.get("/api/readings").json()
    spike_raw = next(r for r in readings if r["reading_id"] == "spike-event")
    assert spike_raw["value"] == 122.0


# -----------------------------------------------------------------------------
# Test 6: Frozen Sensor -> Consecutive Identical Value Detection
# -----------------------------------------------------------------------------
def test_frozen_sensor_detection():
    for i in range(1, 6):
        client.post("/api/telemetry", json={
            "reading_id": f"frozen-step-{i}",
            "asset_id": "pump_01",
            "sensor_id": "vibration_01",
            "timestamp": f"2026-10-27T08:00:0{i}Z",
            "sequence_number": i,
            "value": 3.1415,
            "unit": "m/s2",
            "source_type": "simulated",
        })

    # 5th identical reading should trigger FROZEN_SENSOR
    events = client.get("/api/events?issue_type=FROZEN_SENSOR").json()
    assert len(events) >= 1
    assert events[0]["sensor_id"] == "vibration_01"
    assert events[0]["issue_type"] == "FROZEN_SENSOR"


# -----------------------------------------------------------------------------
# Test 7: Communication Gap Detection
# -----------------------------------------------------------------------------
def test_communication_gap_detection():
    # 1. Normal reading
    client.post("/api/telemetry", json={
        "reading_id": "gap-before",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 71.0,
        "unit": "C",
        "source_type": "simulated",
    })

    # 2. Telemetry arriving 10 seconds later with skipped sequence numbers
    resp = client.post("/api/telemetry", json={
        "reading_id": "gap-after",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:10Z",
        "sequence_number": 10,
        "value": 71.5,
        "unit": "C",
        "source_type": "simulated",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "COMMUNICATION_GAP" in data["issues_detected"]


# -----------------------------------------------------------------------------
# Test 8: Physical Plausibility Violation (Out of Range)
# -----------------------------------------------------------------------------
def test_physical_plausibility_violation():
    # Boiler max limit is 150 C
    resp = client.post("/api/telemetry", json={
        "reading_id": "range-breach",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:00Z",
        "sequence_number": 1,
        "value": 240.0,
        "unit": "C",
        "source_type": "simulated",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "UNRESOLVED"
    assert "INVALID_CORRUPTED" in data["issues_detected"]


# -----------------------------------------------------------------------------
# Test 9: Connected Model Adapter -> Verified REPAIR vs Constraint Block
# -----------------------------------------------------------------------------
class MockTransformerModel(ReconstructionModel):
    def __init__(self, proposed_val: float):
        self.proposed_val = proposed_val

    def predict(self, asset_id: str, sensor_id: str, sensor_window):
        return ModelPrediction(
            available=True,
            candidate_value=self.proposed_val,
            uncertainty_lower=self.proposed_val - 1.0,
            uncertainty_upper=self.proposed_val + 1.0,
            confidence_score=0.92,
            model_name="MockSpatialTemporalTransformer",
            model_version="1.0.0",
            reconstruction_method="saits_attention",
            notes="Mock model output for testing.",
        )


def test_connected_model_repair_success():
    """Verify that when a model is connected and proposes a valid candidate, REPAIR is accepted."""
    settings = get_settings()
    mock_model = MockTransformerModel(proposed_val=73.5)
    processor = PipelineProcessor(settings, model=mock_model)

    db = TestingSessionLocal()
    # 1. Establish normal baseline
    processor.process_reading(
        TelemetryReadingInput(
            reading_id="model-test-01",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            timestamp=datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc),
            sequence_number=1,
            value=72.0,
            unit="C",
            source_type="simulated",
        ),
        db,
    )

    # 2. Inject missing value -> Model proposes 73.5 -> passes constraints -> REPAIR
    result = processor.process_reading(
        TelemetryReadingInput(
            reading_id="model-test-02",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            timestamp=datetime(2026, 10, 27, 8, 0, 1, tzinfo=timezone.utc),
            sequence_number=2,
            value=None,  # missing
            unit="C",
            source_type="simulated",
        ),
        db,
    )
    db.close()

    assert result.status == "REPAIR"
    assert result.original_value is None
    assert result.final_value == 73.5
    assert "satisfies all physical constraints" in result.decision_reason


def test_connected_model_rejected_by_constraints():
    """Verify that if a model proposes an invalid value, constraints CANNOT be overridden!"""
    settings = get_settings()
    # Model proposes 250 C (exceeds boiler max 150 C)
    mock_model = MockTransformerModel(proposed_val=250.0)
    processor = PipelineProcessor(settings, model=mock_model)

    db = TestingSessionLocal()
    result = processor.process_reading(
        TelemetryReadingInput(
            reading_id="model-test-03",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            timestamp=datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc),
            sequence_number=1,
            value=None,
            unit="C",
            source_type="simulated",
        ),
        db,
    )
    db.close()

    assert result.status == "UNRESOLVED"
    assert result.final_value is None
    assert "failed physical bounds" in result.decision_reason
    assert "cannot be overridden" in result.decision_reason


def test_connected_model_review_on_borderline_confidence():
    """Verify that when a model proposes a candidate with borderline confidence (e.g. 0.65), it routes to REVIEW."""
    settings = get_settings()

    class BorderlineTransformerModel(ReconstructionModel):
        def predict(self, asset_id: str, sensor_id: str, sensor_window):
            return ModelPrediction(
                available=True,
                candidate_value=73.5,
                confidence_score=0.65,  # 0.40 <= conf < 0.80 -> REVIEW
                model_name="MockTransformer",
                model_version="1.0.0",
                notes="Borderline confidence prediction",
            )

    processor = PipelineProcessor(settings, model=BorderlineTransformerModel())
    db = TestingSessionLocal()
    result = processor.process_reading(
        TelemetryReadingInput(
            reading_id="review-test-01",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            timestamp=datetime(2026, 10, 27, 8, 0, 1, tzinfo=timezone.utc),
            sequence_number=1,
            value=None,
            unit="C",
            source_type="simulated",
        ),
        db,
    )
    db.close()

    assert result.status == "REVIEW"
    assert result.original_value is None
    assert result.final_value is None
    assert "below automatic acceptance threshold" in result.decision_reason


# -----------------------------------------------------------------------------
# Test 10: Cryptographic Audit Hash Chaining & Verification
# -----------------------------------------------------------------------------
def test_audit_hash_chain():
    # Ingest 3 readings
    for i in range(1, 4):
        client.post("/api/telemetry", json={
            "reading_id": f"audit-read-{i}",
            "asset_id": "boiler_01",
            "sensor_id": "temperature_01",
            "timestamp": f"2026-10-27T08:00:0{i}Z",
            "sequence_number": i,
            "value": 70.0 + i,
            "unit": "C",
            "source_type": "simulated",
        })

    # Verify audit endpoint
    audit_resp = client.get("/api/audit")
    assert audit_resp.status_code == 200
    logs = audit_resp.json()
    assert len(logs) == 3

    # Verify cryptographic chain verification endpoint
    verify_resp = client.get("/api/audit/verify")
    assert verify_resp.status_code == 200
    vdata = verify_resp.json()
    assert vdata["valid"] is True
    assert vdata["status"] == "valid"
    assert vdata["count"] == 3


# -----------------------------------------------------------------------------
# Test 11: Overview API & Counting Rules
# -----------------------------------------------------------------------------
def test_overview_endpoint():
    # 1. Clean reading -> PRESERVE
    client.post("/api/telemetry", json={
        "reading_id": "ov-01",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:01Z",
        "sequence_number": 1,
        "value": 71.0,
        "unit": "C",
        "source_type": "simulated",
    })
    # 2. Missing reading -> UNRESOLVED
    client.post("/api/telemetry", json={
        "reading_id": "ov-02",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "timestamp": "2026-10-27T08:00:02Z",
        "sequence_number": 2,
        "value": None,
        "unit": "C",
        "source_type": "simulated",
    })

    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_readings_received"] == 2
    assert data["readings_preserved"] == 1
    assert data["unresolved_events"] == 1
    assert data["repairs_accepted"] == 0
    assert data["readings_under_review"] == 0
    assert data["issues_detected"] >= 1


# -----------------------------------------------------------------------------
# Test 12: Simulator & Fault Injector Ground Truth Tracking
# -----------------------------------------------------------------------------
def test_simulator_and_fault_injection():
    gen = SyntheticSensorGenerator(random_seed=123)
    injector = FaultInjector()

    # Boiler normal
    b_rd, clean_temp = gen.generate_boiler_temperature()
    assert 60.0 <= clean_temp <= 85.0
    assert b_rd.asset_id == "boiler_01"
    assert b_rd.unit == "C"

    # Pump normal
    p_rd, clean_vib = gen.generate_pump_vibration()
    assert 1.0 <= clean_vib <= 6.0
    assert p_rd.asset_id == "pump_01"
    assert p_rd.unit == "m/s2"

    # Inject spike
    spiked_rd, gt = injector.inject_fault(b_rd, clean_temp, "spike")
    assert spiked_rd.value > clean_temp + 30.0
    assert gt.fault_type == "SUDDEN_SPIKE"
    assert gt.clean_ground_truth_value == clean_temp

    # Verify ground truth log
    assert len(injector.ground_truth_log) == 1


# -----------------------------------------------------------------------------
# Test 13: Evaluation Suite
# -----------------------------------------------------------------------------
def test_evaluation_suite():
    gen = SyntheticSensorGenerator(random_seed=99)
    injector = FaultInjector()

    rd, clean_v = gen.generate_boiler_temperature()
    spiked_rd, gt = injector.inject_fault(rd, clean_v, "spike")

    processed = [{
        "reading_id": spiked_rd.reading_id,
        "status": "UNRESOLVED",
        "issues_detected": ["SUDDEN_SPIKE"],
        "final_value": None,
        "original_value": spiked_rd.value,
    }]

    report = EvaluationSuite.evaluate(
        scenario_name="test_spike_evaluation",
        ground_truth_records=[gt],
        processed_results=processed,
        start_time_perf=1.0,
        end_time_perf=1.05,
    )
    assert report.tp == 1
    assert report.fp == 0
    assert report.fn == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1_score == 1.0
