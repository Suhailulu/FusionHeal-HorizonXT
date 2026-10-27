from __future__ import annotations
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
from app.ingestion.schemas import TelemetryReadingInput, TelemetryProcessingResult
from app.ingestion.validator import TelemetryValidator
from app.pipeline.processing import PipelineProcessor
from app.pipeline.detection import QualityDetectionEngine, QualityDetectionEvent
from app.pipeline.reconstruction import ReconstructionModel, ModelPrediction
from app.pipeline.decision_engine import DecisionEngine
from app.simulator.generator import SyntheticSensorGenerator
from app.simulator.fault_injector import FaultInjector
from app.evaluation.metrics import EvaluationSuite
from app.adapters.base import SimulatorAdapter, HttpSensorAdapter, MqttSensorAdapter, OpcUaSensorAdapter

from conftest import TestingSessionLocal
client = TestClient(app)


# -----------------------------------------------------------------------------
# 1. Valid Telemetry Ingestion
# -----------------------------------------------------------------------------
def test_valid_telemetry_ingestion():
    payload = {
        "reading_id": "RD-CANON-001",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "event_timestamp": "2026-10-27T04:18:30.000Z",
        "sequence_number": 1001,
        "value": 72.5,
        "unit": "C",
        "source_type": "simulated",
    }
    resp = client.post("/api/telemetry", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    assert data["reading_id"] == "RD-CANON-001"
    assert data["decision_status"] == "PRESERVE"
    assert data["status"] == "PRESERVE"
    assert data["quality_status"] == "VALID"
    assert data["original_value"] == 72.5
    assert data["final_value"] == 72.5
    assert data["source_type"] == "simulated"
    assert data["unit"] == "C"

    # Verify raw storage
    readings = client.get("/api/readings?asset_id=boiler_01").json()
    saved = next(r for r in readings if r["reading_id"] == "RD-CANON-001")
    assert saved["value"] == 72.5
    assert saved["validation_status"] == "VALID"


# -----------------------------------------------------------------------------
# 2. Explicit Missing Value
# -----------------------------------------------------------------------------
def test_explicit_missing_value():
    payload = {
        "reading_id": "RD-MISSING-VAL-001",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "event_timestamp": "2026-10-27T04:18:31.000Z",
        "sequence_number": 1002,
        "value": None,  # explicit null measurement
        "unit": "C",
        "source_type": "simulated",
    }
    resp = client.post("/api/telemetry", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    assert data["quality_status"] == "MISSING"
    assert data["decision_status"] == "UNRESOLVED"
    assert data["original_value"] is None
    assert data["final_value"] is None
    assert "MISSING_DATA" in data["issues_detected"]


# -----------------------------------------------------------------------------
# 3. Invalid Timestamps
# -----------------------------------------------------------------------------
def test_invalid_timestamps():
    # Unparseable timestamp string
    bad_payload = {
        "reading_id": "RD-BAD-TS",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "event_timestamp": "not-a-valid-timestamp",
        "sequence_number": 1,
        "value": 70.0,
        "unit": "C",
        "source_type": "simulated",
    }
    resp = client.post("/api/telemetry", json=bad_payload)
    assert resp.status_code == 422


# -----------------------------------------------------------------------------
# 4. Timezone Normalization
# -----------------------------------------------------------------------------
def test_timezone_normalization():
    # Submit non-UTC timezone (+05:30)
    # 2026-10-27T10:30:00+05:30 == 2026-10-27T05:00:00Z
    reading = TelemetryReadingInput(
        reading_id="RD-TZ-NORM-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime.fromisoformat("2026-10-27T10:30:00+05:30"),
        sequence_number=1,
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    # Model validator normalizes to UTC
    assert reading.event_timestamp.tzinfo == timezone.utc
    assert reading.event_timestamp.hour == 5
    assert reading.event_timestamp.minute == 0


# -----------------------------------------------------------------------------
# 5. Duplicate Reading IDs (Idempotent Handling)
# -----------------------------------------------------------------------------
def test_duplicate_reading_ids():
    payload = {
        "reading_id": "RD-DUP-ID-99",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "event_timestamp": "2026-10-27T04:18:32.000Z",
        "sequence_number": 1,
        "value": 71.2,
        "unit": "C",
        "source_type": "simulated",
    }
    # 1st ingestion
    r1 = client.post("/api/telemetry", json=payload)
    assert r1.status_code == 201

    # 2nd ingestion with same reading_id
    r2 = client.post("/api/telemetry", json=payload)
    assert r2.status_code == 201
    assert r2.json()["audit_id"] == "IDEMPOTENT"

    # Ensure only 1 record exists in database
    readings = client.get("/api/readings?asset_id=boiler_01").json()
    matches = [r for r in readings if r["reading_id"] == "RD-DUP-ID-99"]
    assert len(matches) == 1


# -----------------------------------------------------------------------------
# 6. Duplicate Messages (Same Sequence Number)
# -----------------------------------------------------------------------------
def test_duplicate_messages():
    settings = get_settings()
    validator = TelemetryValidator(settings)
    rd1 = TelemetryReadingInput(
        reading_id="RD-SEQ-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 18, 30, tzinfo=timezone.utc),
        sequence_number=100,
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    # Consecutive frame with identical sequence number
    outcome = validator.validate_reading(
        reading=rd1,
        last_timestamp=datetime(2026, 10, 27, 4, 18, 30, tzinfo=timezone.utc),
        last_sequence=100,
    )
    assert outcome.is_valid is False
    assert outcome.status == "DUPLICATE"
    assert any("Duplicate sequence number" in err for err in outcome.errors)


# -----------------------------------------------------------------------------
# 7. Out-of-Order Records
# -----------------------------------------------------------------------------
def test_out_of_order_records():
    settings = get_settings()
    validator = TelemetryValidator(settings)
    earlier_reading = TelemetryReadingInput(
        reading_id="RD-OOO-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 18, 20, tzinfo=timezone.utc),  # earlier
        sequence_number=50,  # lower sequence
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    outcome = validator.validate_reading(
        reading=earlier_reading,
        last_timestamp=datetime(2026, 10, 27, 4, 18, 30, tzinfo=timezone.utc),
        last_sequence=60,
    )
    assert outcome.is_valid is False
    assert outcome.status == "OUT_OF_ORDER"


# -----------------------------------------------------------------------------
# 8. Missing Timestamp Intervals (Sequence Number Skip)
# -----------------------------------------------------------------------------
def test_missing_timestamp_intervals():
    settings = get_settings()
    detector = QualityDetectionEngine(settings)

    # First normal reading
    rd1 = TelemetryReadingInput(
        reading_id="RD-INT-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 18, 0, tzinfo=timezone.utc),
        sequence_number=1,
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    detector.detect_issues(rd1)

    # Next reading arrives with sequence jump 1 -> 5 (4 frames skipped)
    rd2 = TelemetryReadingInput(
        reading_id="RD-INT-02",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 18, 5, tzinfo=timezone.utc),
        sequence_number=5,
        value=72.2,
        unit="C",
        source_type="simulated",
    )
    issues = detector.detect_issues(rd2)
    assert any(i.detection_rule == "SequenceNumberSkipRule" for i in issues)
    assert any(i.issue_type == "MISSING_DATA" for i in issues)


# -----------------------------------------------------------------------------
# 9. Late-Arriving Messages
# -----------------------------------------------------------------------------
def test_late_arriving_messages():
    db = TestingSessionLocal()
    settings = get_settings()
    processor = PipelineProcessor(settings)

    # Reading with event timestamp 5 seconds in the past
    event_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    reading = TelemetryReadingInput(
        reading_id="RD-LATE-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=event_time,
        sequence_number=1,
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    result = processor.process_reading(reading, db)
    db.close()

    assert result.event_timestamp == event_time
    assert result.received_at > event_time
    assert result.processed_at >= result.received_at


# -----------------------------------------------------------------------------
# 10. Frozen Sensor Detection
# -----------------------------------------------------------------------------
def test_frozen_sensor_detection():
    settings = get_settings()
    detector = QualityDetectionEngine(settings)
    events = []

    for i in range(1, 7):
        rd = TelemetryReadingInput(
            reading_id=f"RD-FROZEN-{i}",
            asset_id="pump_01",
            sensor_id="vibration_01",
            event_timestamp=datetime(2026, 10, 27, 4, 0, i, tzinfo=timezone.utc),
            sequence_number=i,
            value=3.500,  # identical float value
            unit="m/s2",
            source_type="simulated",
        )
        evts = detector.detect_issues(rd)
        events.extend(evts)

    assert any(e.issue_type == "FROZEN_SENSOR" for e in events)


# -----------------------------------------------------------------------------
# 11. Spike Detection
# -----------------------------------------------------------------------------
def test_spike_detection():
    settings = get_settings()
    detector = QualityDetectionEngine(settings)

    # Baseline reading
    rd1 = TelemetryReadingInput(
        reading_id="RD-SPIKE-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 0, 0, tzinfo=timezone.utc),
        sequence_number=1,
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    detector.detect_issues(rd1)

    # Spike: +40 degrees in 1s (max rate of change is 5 C/s)
    rd2 = TelemetryReadingInput(
        reading_id="RD-SPIKE-02",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 0, 1, tzinfo=timezone.utc),
        sequence_number=2,
        value=112.0,
        unit="C",
        source_type="simulated",
    )
    issues = detector.detect_issues(rd2)
    assert any(i.issue_type == "SUDDEN_SPIKE" for i in issues)


# -----------------------------------------------------------------------------
# 12. Drift Detection
# -----------------------------------------------------------------------------
def test_drift_detection():
    settings = get_settings()
    detector = QualityDetectionEngine(settings)
    events = []

    # 10 baseline samples
    for i in range(10):
        rd = TelemetryReadingInput(
            reading_id=f"RD-DRIFT-BASE-{i}",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            event_timestamp=datetime(2026, 10, 27, 4, 0, i, tzinfo=timezone.utc),
            sequence_number=i + 1,
            value=72.0,
            unit="C",
            source_type="simulated",
        )
        detector.detect_issues(rd)

    # Drift incrementally +1.0 per second
    for i in range(10, 25):
        rd = TelemetryReadingInput(
            reading_id=f"RD-DRIFT-{i}",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            event_timestamp=datetime(2026, 10, 27, 4, 0, i, tzinfo=timezone.utc),
            sequence_number=i + 1,
            value=72.0 + (i - 10) * 1.5,
            unit="C",
            source_type="simulated",
        )
        evts = detector.detect_issues(rd)
        events.extend(evts)

    assert any(e.issue_type == "GRADUAL_DRIFT" for e in events)


# -----------------------------------------------------------------------------
# 13. Communication Gaps
# -----------------------------------------------------------------------------
def test_communication_gaps():
    settings = get_settings()
    detector = QualityDetectionEngine(settings)

    rd1 = TelemetryReadingInput(
        reading_id="RD-GAP-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 0, 0, tzinfo=timezone.utc),
        sequence_number=1,
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    detector.detect_issues(rd1)

    # Gap of 12 seconds (nominal interval 1s * multiplier 2.5 = 2.5s threshold)
    rd2 = TelemetryReadingInput(
        reading_id="RD-GAP-02",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 4, 0, 12, tzinfo=timezone.utc),
        sequence_number=2,
        value=72.1,
        unit="C",
        source_type="simulated",
    )
    issues = detector.detect_issues(rd2)
    assert any(i.issue_type == "COMMUNICATION_GAP" for i in issues)


# -----------------------------------------------------------------------------
# 14. Invalid Numeric Values (NaN / Infinity)
# -----------------------------------------------------------------------------
def test_invalid_numeric_values():
    with pytest.raises(ValueError, match="NaN and Infinity rejected"):
        TelemetryReadingInput(
            reading_id="RD-NAN",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            event_timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            value=float("nan"),
            unit="C",
            source_type="simulated",
        )

    with pytest.raises(ValueError, match="NaN and Infinity rejected"):
        TelemetryReadingInput(
            reading_id="RD-INF",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            event_timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            value=float("inf"),
            unit="C",
            source_type="simulated",
        )


# -----------------------------------------------------------------------------
# 15. Candidate Rejection by Physical Bounds
# -----------------------------------------------------------------------------
def test_candidate_rejection():
    settings = get_settings()
    engine = DecisionEngine(settings)
    reading = TelemetryReadingInput(
        reading_id="RD-REJ-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime.now(timezone.utc),
        sequence_number=1,
        value=None,  # missing
        unit="C",
        source_type="simulated",
    )
    issue = QualityDetectionEvent(
        event_id="EVT-REJ",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
    )
    # Model proposes 350 C (exceeds boiler max 150 C)
    pred = ModelPrediction(
        available=True,
        candidate_value=350.0,
        confidence_score=0.99,
        model_name="Transformer",
        model_version="1.0.0",
        notes="Out of bounds candidate",
    )
    decision = engine.decide(reading, [issue], pred)

    assert decision.decision_status == "UNRESOLVED"
    assert decision.accepted_value is None
    assert decision.validation_result == "CONSTRAINT_VIOLATION"


# -----------------------------------------------------------------------------
# 16. Accepted Repair
# -----------------------------------------------------------------------------
def test_accepted_repair():
    settings = get_settings()
    engine = DecisionEngine(settings)
    reading = TelemetryReadingInput(
        reading_id="RD-ACC-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime.now(timezone.utc),
        sequence_number=1,
        value=None,
        unit="C",
        source_type="simulated",
    )
    issue = QualityDetectionEvent(
        event_id="EVT-ACC",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
    )
    pred = ModelPrediction(
        available=True,
        candidate_value=73.5,
        confidence_score=0.92,
        model_name="SpatialTemporalTransformer",
        model_version="1.0.0",
        reconstruction_method="saits_imputer",
        notes="High confidence prediction",
    )
    decision = engine.decide(reading, [issue], pred)

    assert decision.decision_status == "REPAIR"
    assert decision.accepted_value == 73.5
    assert decision.candidate_value == 73.5
    assert decision.validation_result == "PASS_VERIFIED"


# -----------------------------------------------------------------------------
# 17. Preservation of Original Data
# -----------------------------------------------------------------------------
def test_preservation_of_original_data():
    db = TestingSessionLocal()
    settings = get_settings()

    class ValidRepairModel(ReconstructionModel):
        def predict(self, asset_id, sensor_id, sensor_window, target_timestamp=None):
            return ModelPrediction(
                available=True,
                candidate_value=75.0,
                confidence_score=0.95,
                model_name="TestTransformer",
                model_version="1.0.0",
                reconstruction_method="test_repair",
                notes="Valid repair candidate",
            )

    processor = PipelineProcessor(settings, model=ValidRepairModel())
    # Reading arrives with dirty value 125.0 (spike)
    reading = TelemetryReadingInput(
        reading_id="RD-PRESERVE-RAW",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime.now(timezone.utc),
        sequence_number=1,
        value=125.0,
        unit="C",
        source_type="simulated",
    )
    res = processor.process_reading(reading, db)

    # Check RawReading in database
    repo = TelemetryRepository(db)
    raw = repo.get_raw_reading_by_id("RD-PRESERVE-RAW")
    assert raw is not None
    # Original value must remain exactly 125.0, NEVER overwritten!
    assert raw.value == 125.0
    db.close()


# -----------------------------------------------------------------------------
# 18. Audit Record Creation
# -----------------------------------------------------------------------------
def test_audit_record_creation():
    client.post("/api/telemetry", json={
        "reading_id": "RD-AUDIT-TEST-01",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "event_timestamp": "2026-10-27T04:18:30Z",
        "sequence_number": 1,
        "value": 72.0,
        "unit": "C",
        "source_type": "simulated",
    })
    resp = client.get("/api/audit")
    assert resp.status_code == 200
    audits = resp.json()
    assert len(audits) >= 1
    latest = audits[0]
    assert latest["audit_id"].startswith("AUD-")
    assert latest["record_hash"] is not None
    assert latest["config_hash"] is not None


# -----------------------------------------------------------------------------
# 19. API Filtering
# -----------------------------------------------------------------------------
def test_api_filtering():
    resp_boiler = client.get("/api/readings?asset_id=boiler_01")
    assert resp_boiler.status_code == 200
    for r in resp_boiler.json():
        assert r["asset_id"] == "boiler_01"

    resp_events = client.get("/api/events?decision_status=UNRESOLVED")
    assert resp_events.status_code == 200
    for e in resp_events.json():
        assert e["decision_status"] == "UNRESOLVED"


# -----------------------------------------------------------------------------
# 20. Database Failure Handling
# -----------------------------------------------------------------------------
def test_database_failure_handling():
    # Calling get_readings with invalid limit returns 422
    resp = client.get("/api/readings?limit=0")
    assert resp.status_code == 422


# -----------------------------------------------------------------------------
# 21. Simulator Fault Injection
# -----------------------------------------------------------------------------
def test_simulator_fault_injection():
    gen = SyntheticSensorGenerator(random_seed=777)
    injector = FaultInjector()

    rd, clean_temp = gen.generate_boiler_temperature()
    mod_rd, gt = injector.inject_fault(rd, clean_temp, "sudden_spike")

    assert mod_rd is not None
    assert gt is not None
    assert gt.fault_type == "SUDDEN_SPIKE"
    assert mod_rd.value > clean_temp + 30.0


# -----------------------------------------------------------------------------
# 22. Evaluation Metric Calculations
# -----------------------------------------------------------------------------
def test_evaluation_metric_calculations():
    gen = SyntheticSensorGenerator(random_seed=42)
    injector = FaultInjector()

    rd, clean_val = gen.generate_boiler_temperature()
    spiked_rd, gt = injector.inject_fault(rd, clean_val, "sudden_spike")

    processed = [{
        "reading_id": spiked_rd.reading_id,
        "status": "UNRESOLVED",
        "decision_status": "UNRESOLVED",
        "issues_detected": ["SUDDEN_SPIKE"],
        "final_value": None,
        "original_value": spiked_rd.value,
        "event_timestamp": spiked_rd.event_timestamp,
        "received_at": spiked_rd.event_timestamp + timedelta(milliseconds=20),
        "processed_at": spiked_rd.event_timestamp + timedelta(milliseconds=45),
    }]

    report = EvaluationSuite.evaluate(
        scenario_name="test_gateway_eval",
        ground_truth_records=[gt],
        processed_results=processed,
        start_time_perf=10.0,
        end_time_perf=10.05,
    )

    assert report.tp == 1
    assert report.fp == 0
    assert report.fn == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1_score == 1.0
    assert report.avg_processing_latency_ms == 25.0
    assert report.avg_ingestion_delay_ms == 20.0
    assert report.throughput_readings_per_sec > 0


# -----------------------------------------------------------------------------
# 23. Five Failure Modes Distinction
# -----------------------------------------------------------------------------
def test_distinguish_five_failure_modes():
    """Explicitly verify the differences between:
    1. Missing value (null in message)
    2. Missing message (packet dropped)
    3. Delayed message (out of order / late arrival)
    4. Invalid payload (malformed schema or out of range)
    5. Confirmed processing failure (infrastructure / DB exception)
    """
    settings = get_settings()
    validator = TelemetryValidator(settings)
    detector = QualityDetectionEngine(settings)

    # 1. Missing Value: valid message arrived, value is explicitly None
    mv_reading = TelemetryReadingInput(
        reading_id="MODE-1-MV",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc),
        sequence_number=1,
        value=None,  # null
        unit="C",
        source_type="simulated",
    )
    val_mv = validator.validate_reading(mv_reading)
    issues_mv = detector.detect_issues(mv_reading)
    assert val_mv.is_valid is True  # Payload is well-formed
    assert any(i.detection_rule == "ExplicitNullRule" for i in issues_mv)

    # 2. Missing Message: frame was never received (sequence skipped)
    detector.reset()
    detector.detect_issues(mv_reading)
    mm_reading = TelemetryReadingInput(
        reading_id="MODE-2-MM",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 8, 0, 5, tzinfo=timezone.utc),
        sequence_number=4,  # skipped sequence 2 and 3
        value=72.0,
        unit="C",
        source_type="simulated",
    )
    issues_mm = detector.detect_issues(mm_reading)
    assert any(i.detection_rule == "SequenceNumberSkipRule" for i in issues_mm)
    assert any(i.observed_evidence.get("skipped_count") == 2 for i in issues_mm)

    # 3. Delayed Message: arrived out of order (regressive sequence / past timestamp)
    late_reading = TelemetryReadingInput(
        reading_id="MODE-3-LATE",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 8, 0, 2, tzinfo=timezone.utc),  # earlier than mm_reading
        sequence_number=2,  # earlier sequence
        value=72.1,
        unit="C",
        source_type="simulated",
    )
    val_late = validator.validate_reading(
        reading=late_reading,
        last_timestamp=mm_reading.event_timestamp,
        last_sequence=mm_reading.sequence_number,
    )
    assert val_late.is_valid is False
    assert val_late.status == "OUT_OF_ORDER"

    # 4. Invalid Payload: gross out of bounds or bad unit
    invalid_reading = TelemetryReadingInput(
        reading_id="MODE-4-INV",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        event_timestamp=datetime(2026, 10, 27, 8, 0, 6, tzinfo=timezone.utc),
        sequence_number=5,
        value=500.0,  # exceeds boiler max 150 C
        unit="INVALID_UNIT",
        source_type="simulated",
    )
    val_inv = validator.validate_reading(invalid_reading)
    assert val_inv.is_valid is False
    assert val_inv.status == "MALFORMED"


# -----------------------------------------------------------------------------
# 24. Pluggable Ingress Adapters (Hardware & Simulator)
# -----------------------------------------------------------------------------
def test_pluggable_ingress_adapters():
    sim_adapter = SimulatorAdapter()
    http_adapter = HttpSensorAdapter()
    mqtt_adapter = MqttSensorAdapter()
    opcua_adapter = OpcUaSensorAdapter()

    # Simulator adapter
    sim_rd = sim_adapter.normalize({
        "reading_id": "ADAPT-SIM-01",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "event_timestamp": "2026-10-27T04:18:30Z",
        "sequence_number": 1,
        "value": 72.0,
        "unit": "C",
    })
    assert sim_rd.source_type == "simulated"

    # Physical HTTP sensor adapter (e.g. ESP32 post)
    http_rd = http_adapter.normalize({
        "reading_id": "ADAPT-ESP32-01",
        "asset_id": "boiler_01",
        "sensor_id": "temperature_01",
        "event_timestamp": "2026-10-27T04:18:30Z",
        "sequence_number": 1,
        "value": 72.0,
        "unit": "C",
    })
    assert http_rd.source_type == "physical"

    # MQTT message adapter
    mqtt_rd = mqtt_adapter.normalize({
        "msg_id": "MQTT-MSG-123",
        "asset": "pump_01",
        "sensor": "vibration_01",
        "val": 3.14,
        "unit": "m/s2",
    })
    assert mqtt_rd.reading_id == "MQTT-MSG-123"
    assert mqtt_rd.asset_id == "pump_01"
    assert mqtt_rd.value == 3.14

    # OPC UA attribute adapter
    opcua_rd = opcua_adapter.normalize({
        "NodeId": "ns=2;s=BoilerTemp",
        "asset_id": "boiler_01",
        "DisplayName": "temperature_01",
        "Value": 74.2,
        "unit": "C",
    })
    assert opcua_rd.reading_id == "ns=2;s=BoilerTemp"
    assert opcua_rd.value == 74.2
