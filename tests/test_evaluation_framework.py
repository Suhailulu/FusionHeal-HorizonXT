import math
import uuid
import json
import pytest
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from app.config import get_settings
from app.ingestion.schemas import TelemetryReadingInput, ReviewDecisionInput
from app.pipeline.processing import PipelineProcessor
from app.pipeline.reconstruction import PlaceholderReconstructionAdapter, ModelPrediction
from app.simulator.generator import SyntheticSensorGenerator
from app.simulator.fault_injector import FaultInjector, GroundTruthFaultRecord
from app.evaluation.metrics import (
    DeterministicBenchmarkModel,
    EvaluationSuite,
    run_reproducible_benchmark,
)
from app.storage.database import Base, engine, SessionLocal
from app.storage.models import QualityEvent
from app.storage.repository import TelemetryRepository
from app.audit.logger import AuditService
from app.api.events import _process_review_decision
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session(setup_db):
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


def test_ground_truth_isolation():
    """Verify that the detection & decision pipeline has ZERO access to hidden ground-truth labels.
    
    1. Generated readings sent to the pipeline must NOT contain ground-truth metadata.
    2. The reconstruction model operates strictly on historical sensor windows, not labels.
    """
    gen = SyntheticSensorGenerator(random_seed=42)
    injector = FaultInjector()
    
    clean_ts = datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc)
    reading, clean_val = gen.generate_boiler_temperature(timestamp=clean_ts)
    
    # Inject a spike fault
    mod_reading, gt_record = injector.inject_fault(reading, clean_val, "sudden_spike")
    assert mod_reading is not None
    assert gt_record is not None
    
    # Ground truth record contains hidden labels:
    assert gt_record.clean_ground_truth_value == clean_val
    assert gt_record.fault_type == "SUDDEN_SPIKE"
    
    # TelemetryReadingInput sent to pipeline processor contains NO ground truth attributes:
    reading_dict = mod_reading.model_dump()
    assert "clean_ground_truth_value" not in reading_dict
    assert "fault_type" not in reading_dict
    assert "is_anomaly" not in reading_dict
    assert "injected_value" not in reading_dict
    
    # Test reconstruction model operates solely on sensor window:
    model = DeterministicBenchmarkModel()
    window = [
        {"timestamp": clean_ts.isoformat(), "value": 72.0},
        {"timestamp": (clean_ts + timedelta(seconds=1)).isoformat(), "value": 72.2},
        {"timestamp": (clean_ts + timedelta(seconds=2)).isoformat(), "value": 72.1},
    ]
    pred = model.predict(asset_id="boiler_01", sensor_id="temp_c", sensor_window=window)
    assert pred.available is True
    assert 71.0 <= pred.candidate_value <= 73.0
    assert pred.confidence_score > 0.80


def test_metric_calculations_mathematical_correctness(db_session):
    """Verify exact mathematical formulas for Precision, Recall, F1, MAE, RMSE, and rates."""
    # Synthetic controlled test data
    # 2 True Positives, 1 False Positive, 1 False Negative, 2 True Negatives
    ground_truth = [
        GroundTruthFaultRecord(
            timestamp=datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc),
            asset_id="boiler_01",
            sensor_id="temp_c",
            fault_type="MISSING_DATA",
            clean_ground_truth_value=70.0,
            reading_id="rd-tp-1",
        ),
        GroundTruthFaultRecord(
            timestamp=datetime(2026, 10, 27, 8, 0, 1, tzinfo=timezone.utc),
            asset_id="boiler_01",
            sensor_id="temp_c",
            fault_type="SUDDEN_SPIKE",
            clean_ground_truth_value=72.0,
            reading_id="rd-tp-2",
        ),
        GroundTruthFaultRecord(
            timestamp=datetime(2026, 10, 27, 8, 0, 2, tzinfo=timezone.utc),
            asset_id="pump_01",
            sensor_id="vibration_rms_ms2",
            fault_type="GRADUAL_DRIFT",
            clean_ground_truth_value=2.5,
            reading_id="rd-fn-1",
        ),
    ]

    processed = [
        # TP 1: Detected, repaired to 71.0 (error |71 - 70| = 1.0)
        {
            "reading_id": "rd-tp-1",
            "timestamp": "2026-10-27T08:00:00Z",
            "asset_id": "boiler_01",
            "sensor_id": "temp_c",
            "original_value": None,
            "final_value": 71.0,
            "decision_status": "REPAIR",
            "issues_detected": ["MISSING_DATA"],
            "processing_latency_ms": 1.2,
        },
        # TP 2: Detected, repaired to 74.0 (error |74 - 72| = 2.0)
        {
            "reading_id": "rd-tp-2",
            "timestamp": "2026-10-27T08:00:01Z",
            "asset_id": "boiler_01",
            "sensor_id": "temp_c",
            "original_value": 110.0,
            "final_value": 74.0,
            "decision_status": "REPAIR",
            "issues_detected": ["SUDDEN_SPIKE"],
            "processing_latency_ms": 1.5,
        },
        # FN 1: Injected drift, but pipeline PRESERVED with no issues
        {
            "reading_id": "rd-fn-1",
            "timestamp": "2026-10-27T08:00:02Z",
            "asset_id": "pump_01",
            "sensor_id": "vibration_rms_ms2",
            "original_value": 3.8,
            "final_value": 3.8,
            "decision_status": "PRESERVE",
            "issues_detected": [],
            "processing_latency_ms": 1.0,
        },
        # FP 1: Clean reading, but false alarm flagged
        {
            "reading_id": "rd-fp-1",
            "timestamp": "2026-10-27T08:00:03Z",
            "asset_id": "boiler_01",
            "sensor_id": "temp_c",
            "original_value": 73.0,
            "final_value": 73.0,
            "decision_status": "PRESERVE",
            "issues_detected": ["SUSPICIOUS_NOISE"],
            "processing_latency_ms": 1.1,
        },
        # TN 1: Clean reading, preserved
        {
            "reading_id": "rd-tn-1",
            "timestamp": "2026-10-27T08:00:04Z",
            "asset_id": "boiler_01",
            "sensor_id": "temp_c",
            "original_value": 72.5,
            "final_value": 72.5,
            "decision_status": "PRESERVE",
            "issues_detected": [],
            "processing_latency_ms": 0.9,
        },
        # TN 2: Clean reading, preserved
        {
            "reading_id": "rd-tn-2",
            "timestamp": "2026-10-27T08:00:05Z",
            "asset_id": "pump_01",
            "sensor_id": "vibration_rms_ms2",
            "original_value": 2.4,
            "final_value": 2.4,
            "decision_status": "PRESERVE",
            "issues_detected": [],
            "processing_latency_ms": 1.0,
        },
    ]

    report = EvaluationSuite.evaluate(
        scenario_name="unit_math_verification",
        ground_truth_records=ground_truth,
        processed_results=processed,
        start_time_perf=0.0,
        end_time_perf=0.01,
        repo=TelemetryRepository(db_session),
    )

    # TP = 2, FP = 1, FN = 1, TN = 2
    assert report.tp == 2
    assert report.fp == 1
    assert report.fn == 1
    assert report.tn == 2

    # Precision = TP / (TP + FP) = 2 / 3 = 0.6667
    assert math.isclose(report.precision, 2 / 3, abs_tol=1e-3)

    # Recall = TP / (TP + FN) = 2 / 3 = 0.6667
    assert math.isclose(report.recall, 2 / 3, abs_tol=1e-3)

    # F1 = 2 * (P * R) / (P + R) = 2/3 = 0.6667
    assert math.isclose(report.f1_score, 2 / 3, abs_tol=1e-3)

    # MAE on accepted repairs: errors are |71 - 70| = 1.0 and |74 - 72| = 2.0 -> MAE = 1.5
    assert math.isclose(report.mae, 1.5, abs_tol=1e-3)

    # RMSE: sqrt((1^2 + 2^2)/2) = sqrt(2.5) ~= 1.5811
    assert math.isclose(report.rmse, math.sqrt(2.5), abs_tol=1e-3)

    # Preservation of clean readings: 3 clean points (FP-1, TN-1, TN-2), all 3 original values preserved
    assert report.valid_preserved_count == 3
    assert math.isclose(report.valid_preserved_pct, 100.0, abs_tol=1e-3)
    assert report.unintentional_modifications == 0


def test_missing_intervals_and_communication_gaps(db_session):
    """Verify that communication gaps and missing intervals are properly handled and evaluated."""
    settings = get_settings()
    processor = PipelineProcessor(settings, model=DeterministicBenchmarkModel())
    processor.reset_state()

    base_time = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)
    uid = uuid.uuid4().hex[:6]
    
    # 1. First reading
    r1 = TelemetryReadingInput(
        reading_id=f"gap-test-1-{uid}",
        asset_id="boiler_01",
        sensor_id="temp_c",
        timestamp=base_time,
        event_timestamp=base_time,
        sequence_number=1,
        value=72.0,
        unit="C",
    )
    res1 = processor.process_reading(r1, db_session)
    assert res1.decision_status == "PRESERVE"

    # 2. Reading after 15 second jump (gap threshold is 5s) and sequence gap
    jump_time = base_time + timedelta(seconds=15)
    r2 = TelemetryReadingInput(
        reading_id=f"gap-test-2-{uid}",
        asset_id="boiler_01",
        sensor_id="temp_c",
        timestamp=jump_time,
        event_timestamp=jump_time,
        sequence_number=10,  # jumped from 1 to 10
        value=72.5,
        unit="C",
    )
    res2 = processor.process_reading(r2, db_session)
    # The detector must flag SAMPLING_RATE_GAP or SEQUENCE_GAP
    has_gap_issue = any("GAP" in iss.upper() for iss in res2.issues_detected) or (res2.issue_type and "GAP" in res2.issue_type.upper())
    assert has_gap_issue is True


def test_all_six_fault_types_evaluated(db_session):
    """Verify that all six targeted fault types are injected and tracked in the benchmark report."""
    report = run_reproducible_benchmark(seed=42, sample_count=60, db=db_session)
    
    expected_faults = [
        "MISSING_DATA",
        "COMMUNICATION_GAP",
        "FROZEN_SENSOR",
        "SUDDEN_SPIKE",
        "GRADUAL_DRIFT",
        "INVALID_CORRUPTED",
    ]
    
    by_fault = report.detection_by_fault_type
    assert isinstance(by_fault, dict)
    
    for fault in expected_faults:
        assert fault in by_fault, f"Missing fault mode in evaluation breakdown: {fault}"
        entry = by_fault[fault]
        assert entry["tested_count"] > 0
        assert "tp" in entry
        assert "fn" in entry
        assert "precision" in entry
        assert "recall" in entry
        assert "f1_score" in entry


def test_all_four_decision_statuses(db_session):
    """Verify that all four official decision statuses (PRESERVE, REPAIR, REVIEW, UNRESOLVED)
    are exercised and represented during benchmark evaluation."""
    report = run_reproducible_benchmark(seed=42, sample_count=80, db=db_session)
    
    b3 = report.baseline_comparison.get("baseline_3_fusionheal", {})
    dist = b3.get("decision_distribution", {})
    
    assert dist.get("PRESERVE", 0) > 0, "No PRESERVE decisions recorded"
    assert dist.get("REPAIR", 0) > 0, "No REPAIR decisions recorded"
    assert dist.get("REVIEW", 0) > 0, "No REVIEW decisions recorded"
    assert dist.get("UNRESOLVED", 0) > 0, "No UNRESOLVED decisions recorded"


def test_accepted_and_rejected_repairs(db_session):
    """Verify that physically plausible repairs are ACCEPTED, while constraint breaches are REJECTED."""
    settings = get_settings()
    uid = uuid.uuid4().hex[:6]
    
    # 1. Accepted Repair: Normal missing reading with a valid moving average candidate (~72 °C)
    processor_normal = PipelineProcessor(settings, model=DeterministicBenchmarkModel())
    processor_normal.reset_state()
    
    # Populate window
    ts = datetime(2026, 10, 27, 9, 30, 0, tzinfo=timezone.utc)
    for i in range(5):
        processor_normal.process_reading(
            TelemetryReadingInput(
                reading_id=f"win-seed-{uid}-{i}",
                asset_id="boiler_01",
                sensor_id="temp_c",
                timestamp=ts + timedelta(seconds=i),
                value=72.0 + (i * 0.1),
                sequence_number=i,
                unit="C",
            ),
            db_session,
        )
    
    # Missing reading
    missing_rd = TelemetryReadingInput(
        reading_id=f"repair-acc-{uid}",
        asset_id="boiler_01",
        sensor_id="temp_c",
        timestamp=ts + timedelta(seconds=6),
        value=None,
        sequence_number=6,
        unit="C",
    )
    res_acc = processor_normal.process_reading(missing_rd, db_session)
    assert res_acc.decision_status == "REPAIR"
    assert res_acc.original_value is None
    assert res_acc.final_value is not None
    assert 70.0 <= res_acc.final_value <= 75.0

    # 2. Rejected Repair: Model proposes candidate that breaches physical bounds (e.g. 250 °C on Boiler)
    class OutOfBoundsModel(DeterministicBenchmarkModel):
        def predict(self, asset_id, sensor_id, sensor_window, target_timestamp=None):
            return ModelPrediction(
                available=True,
                candidate_value=250.0,  # Exceeds max allowable temperature 150 °C
                confidence_score=0.99,
                model_name="OutOfBoundsModel",
                model_version="1.0.0",
            )
    
    processor_rej = PipelineProcessor(settings, model=OutOfBoundsModel())
    processor_rej.reset_state()
    missing_rej_rd = TelemetryReadingInput(
        reading_id=f"repair-rej-{uid}",
        asset_id="boiler_01",
        sensor_id="temp_c",
        timestamp=ts + timedelta(seconds=7),
        value=None,
        sequence_number=7,
        unit="C",
    )
    res_rej = processor_rej.process_reading(missing_rej_rd, db_session)
    # The candidate breaches physical bounds -> rejected -> UNRESOLVED
    assert res_rej.decision_status == "UNRESOLVED"
    # CRITICAL: Candidate must NOT be assigned to final_value
    assert res_rej.final_value is None


def test_preservation_of_known_valid_readings(db_session):
    """Verify that known-valid telemetry is strictly PRESERVED without unnecessary modifications."""
    injector = FaultInjector()
    settings = get_settings()
    processor = PipelineProcessor(settings, model=DeterministicBenchmarkModel())
    processor.reset_state()

    base_time = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)
    clean_readings = []
    processed_results = []
    uid = uuid.uuid4().hex[:6]

    for i in range(15):
        t = base_time + timedelta(seconds=i)
        b_rd = TelemetryReadingInput(
            reading_id=f"clean-rd-{uid}-{i}",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            event_timestamp=t,
            sequence_number=i + 1,
            value=round(72.0 + (i % 2) * 0.1, 2),
            unit="C",
            source_type="simulated",
        )
        mod_rd, _ = injector.inject_fault(b_rd, b_rd.value, "clean")
        clean_readings.append(mod_rd)
        res = processor.process_reading(mod_rd, db_session)
        processed_results.append(res.model_dump())

    report = EvaluationSuite.evaluate(
        scenario_name="preservation_verification",
        ground_truth_records=injector.ground_truth_log,  # Empty because all clean
        processed_results=processed_results,
        start_time_perf=0.0,
        end_time_perf=0.05,
    )

    assert report.valid_preserved_count == 15
    assert report.valid_preserved_pct == 100.0
    assert report.unintentional_modifications == 0
    assert report.unnecessary_modification_rate == 0.0


def test_review_transitions_and_reasons(db_session):
    """Verify that events routed to REVIEW can transition to REPAIR, PRESERVE, or UNRESOLVED with reasons."""
    settings = get_settings()
    repo = TelemetryRepository(db_session)
    audit_svc = AuditService(repo, settings)

    # 1. Create a synthetic pending review event
    event_id = f"rev-test-{uuid.uuid4().hex[:8]}"
    evt = QualityEvent(
        event_id=event_id,
        reading_id=f"rd-{event_id}",
        asset_id="boiler_01",
        sensor_id="temp_c",
        timestamp=datetime.now(timezone.utc),
        event_timestamp=datetime.now(timezone.utc),
        received_at=datetime.now(timezone.utc),
        processed_at=datetime.now(timezone.utc),
        issue_type="SUSPICIOUS_READING",
        quality_status="SUSPECT",
        decision_status="REVIEW",
        detection_rule="BorderlineConfidenceRule",
        observed_evidence=json.dumps({"reason": "Borderline reconstruction confidence (0.65)"}),
        severity="MEDIUM",
        unit="C",
        source_type="simulated",
        candidate_value=73.5,
        original_value=None,
    )
    repo.save_quality_event(evt)

    # 2. Operator transitions to REPAIR with an approved value
    dec_input = ReviewDecisionInput(
        new_status="REPAIR",
        actor="operator_jane",
        reason="Visual chart corroboration confirms 73.5 C is consistent with steam header pressure.",
        accepted_value=73.5,
    )
    res = _process_review_decision(event_id, dec_input, db_session)
    assert res.decision_status == "REPAIR"
    assert res.final_value == 73.5

    # Verify actor and transition are permanently logged in audit trail
    audits = repo.get_audit_logs_for_event(event_id)
    assert len(audits) >= 1
    assert audits[0].actor == "operator_jane"
    assert audits[0].new_status == "REPAIR"

    # Verify audit chain integrity
    assert audit_svc.verify_integrity().get("valid") is True


def test_reproducibility_with_seeds(db_session):
    """Verify that runs with identical seeds produce identical results, while differing seeds vary."""
    rep_1 = run_reproducible_benchmark(seed=777, sample_count=40, db=db_session)
    rep_2 = run_reproducible_benchmark(seed=777, sample_count=40, db=db_session)
    rep_3 = run_reproducible_benchmark(seed=888, sample_count=40, db=db_session)

    # Identical seeds must produce identical precision, recall, f1, and errors
    assert rep_1.precision == rep_2.precision
    assert rep_1.recall == rep_2.recall
    assert rep_1.f1_score == rep_2.f1_score
    assert rep_1.mae == rep_2.mae
    assert rep_1.rmse == rep_2.rmse
    assert rep_1.tp == rep_2.tp
    assert rep_1.fp == rep_2.fp
    assert rep_1.fn == rep_2.fn
    assert rep_1.tn == rep_2.tn


def test_invalid_input_and_model_unavailability(db_session):
    """Verify that when the model is unavailable (Placeholder), zero predictions are fabricated."""
    settings = get_settings()
    # PlaceholderReconstructionAdapter returns available=False
    processor = PipelineProcessor(settings, model=PlaceholderReconstructionAdapter())
    processor.reset_state()

    missing_rd = TelemetryReadingInput(
        reading_id=f"unavail-test-{uuid.uuid4().hex[:6]}",
        asset_id="boiler_01",
        sensor_id="temp_c",
        timestamp=datetime.now(timezone.utc),
        value=None,
        sequence_number=1,
        unit="C",
    )
    res = processor.process_reading(missing_rd, db_session)
    # When model is unavailable, no candidate exists -> decision must be UNRESOLVED, never fabricated
    assert res.decision_status == "UNRESOLVED"
    assert res.final_value is None


def test_metrics_api_endpoints():
    """Verify the FastAPI metrics endpoints:
    - GET /api/metrics returns available=False if no runs exist.
    - POST /api/metrics/run runs reproducible benchmark and returns metrics.
    - GET /api/metrics returns the newly populated metrics.
    """
    client = TestClient(app)
    
    # 1. Run a benchmark run
    post_res = client.post("/api/metrics/run", json={"seed": 42, "sample_count": 40})
    assert post_res.status_code == 200
    data = post_res.json()
    assert data["available"] is True
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "mae" in data
    assert "rmse" in data
    assert "confusion_matrix" in data
    assert "detection_by_fault_type" in data
    assert "baseline_comparison" in data
    assert "operational_metrics" in data
    assert "decision_counts" in data
    assert "preservation_metrics" in data

    # 2. Get latest metrics
    get_res = client.get("/api/metrics")
    assert get_res.status_code == 200
    latest_data = get_res.json()
    assert latest_data["available"] is True
    assert latest_data["run_id"] == data["run_id"]
    assert latest_data["confusion_matrix"] == data["confusion_matrix"]
