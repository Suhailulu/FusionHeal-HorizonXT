from __future__ import annotations
import math
import time
import os
import tempfile
import json
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional

from app.config import get_settings
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.processing import PipelineProcessor
from app.pipeline.reconstruction import (
    ModelPrediction,
    PlaceholderReconstructionAdapter,
    SpatialTemporalTransformerAdapter,
)
from app.pipeline.decision_engine import DecisionEngine
from app.pipeline.detection import QualityDetectionEngine, QualityDetectionEvent
from app.storage.database import Base, engine, SessionLocal
from app.storage.repository import TelemetryRepository


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


# -----------------------------------------------------------------------------
# 1. Model Loading Tests
# -----------------------------------------------------------------------------

def test_adapter_loading_missing_file():
    """Verify that adapter handles missing model file gracefully without crashing."""
    adapter = SpatialTemporalTransformerAdapter(model_path="models/non_existent_weights.onnx")
    assert adapter.model_loaded is False
    assert adapter.model_version == "unavailable"
    assert "not found" in adapter.load_error.lower()


def test_adapter_loading_unsupported_format(tmp_path):
    """Verify that adapter identifies unsupported model extensions."""
    dummy_file = tmp_path / "model.unknown"
    dummy_file.write_text("fake binary")
    adapter = SpatialTemporalTransformerAdapter(model_path=str(dummy_file))
    assert adapter.model_loaded is False
    assert "unsupported" in adapter.load_error.lower()


def test_adapter_loading_with_custom_forward_fn():
    """Verify that adapter accepts a custom forward function for testing and inference."""
    def dummy_forward(norm: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Return simulated predictions
        return norm, None

    adapter = SpatialTemporalTransformerAdapter(custom_forward_fn=dummy_forward, version="custom-v1.0")
    assert adapter.model_loaded is True
    assert adapter.framework == "custom"
    assert adapter.model_version == "custom-v1.0"
    assert adapter.load_error is None


def test_adapter_loading_numpy_checkpoint(tmp_path):
    """Verify loading from a NumPy / JSON checkpoint."""
    ckpt_file = tmp_path / "weights.json"
    ckpt_file.write_text(json.dumps({"model_type": "transformer_saits", "layers": 2}))
    adapter = SpatialTemporalTransformerAdapter(model_path=str(ckpt_file), version="saits-np-1.0")
    assert adapter.model_loaded is True
    assert adapter.framework == "numpy_checkpoint"
    assert adapter.model_version == "saits-np-1.0"


# -----------------------------------------------------------------------------
# 2. Input Preprocessing & Masking Tests
# -----------------------------------------------------------------------------

def test_preprocessing_sequence_normalization_and_masking():
    """Verify Z-score normalization, sequence length alignment, and missingness mask construction."""
    adapter = SpatialTemporalTransformerAdapter(window_size=8)
    
    # boiler_01:temperature_01 has mean=72.0, std=3.5
    raw_window = [
        {"timestamp": "2026-10-27T08:00:00Z", "value": 72.0},  # z = 0.0, mask = 1
        {"timestamp": "2026-10-27T08:00:01Z", "value": 75.5},  # z = +1.0, mask = 1
        {"timestamp": "2026-10-27T08:00:02Z", "value": None},  # missing, z = 0.0, mask = 0
        {"timestamp": "2026-10-27T08:00:03Z", "value": 68.5},  # z = -1.0, mask = 1
    ]
    
    pre = adapter.preprocess_window("boiler_01", "temperature_01", raw_window)
    norm = pre["normalized_tensor"]
    mask = pre["missing_mask"]
    
    assert norm.shape == (1, 8, 1)
    assert mask.shape == (1, 8, 1)
    assert pre["window_size"] == 8
    assert pre["raw_length"] == 4
    # Because window was 4 and window_size was 8, first 4 positions are left-padded with 0
    assert pre["observed_count"] == 3
    assert pre["missing_count"] == 5
    
    # Check that padded entries have mask 0
    assert mask[0, 0, 0] == 0.0
    assert mask[0, 3, 0] == 0.0
    # Check that position 4 is the first observation (72.0 -> z = 0.0, mask = 1.0)
    assert mask[0, 4, 0] == 1.0
    assert abs(norm[0, 4, 0] - 0.0) < 1e-3
    # Position 5 is 75.5 -> z = 1.0, mask = 1.0
    assert mask[0, 5, 0] == 1.0
    assert abs(norm[0, 5, 0] - 1.0) < 1e-3
    # Position 6 is None -> z = 0.0, mask = 0.0
    assert mask[0, 6, 0] == 0.0
    assert abs(norm[0, 6, 0] - 0.0) < 1e-3
    # Position 7 is 68.5 -> z = -1.0, mask = 1.0
    assert mask[0, 7, 0] == 1.0
    assert abs(norm[0, 7, 0] - (-1.0)) < 1e-3


def test_preprocessing_window_truncation():
    """Verify that windows longer than window_size are truncated to the most recent elements."""
    adapter = SpatialTemporalTransformerAdapter(window_size=4)
    raw_window = [{"value": float(i)} for i in range(10)]
    pre = adapter.preprocess_window("boiler_01", "temperature_01", raw_window)
    assert pre["normalized_tensor"].shape == (1, 4, 1)
    assert pre["raw_length"] == 10
    assert pre["observed_count"] == 4


# -----------------------------------------------------------------------------
# 3. Missing or Malformed Inputs Tests
# -----------------------------------------------------------------------------

def test_preprocessing_empty_window_error():
    """Verify that an empty observation window raises ValueError and predict returns ERROR."""
    def dummy_forward(norm, mask):
        return norm, None
    adapter = SpatialTemporalTransformerAdapter(custom_forward_fn=dummy_forward)
    
    pred = adapter.predict(
        asset_id="boiler_01",
        sensor_id="temperature_01",
        sensor_window=[],
    )
    assert pred.available is False
    assert pred.inference_status == "ERROR"
    assert "empty" in pred.notes.lower()
    assert pred.inference_latency_ms is not None
    assert pred.inference_latency_ms >= 0.0


def test_preprocessing_malformed_non_numeric_values():
    """Verify that non-numeric, corrupted, and NaN values are cleanly masked as missing."""
    adapter = SpatialTemporalTransformerAdapter(window_size=4)
    raw_window = [
        {"value": "corrupted_text"},
        {"value": float("nan")},
        {"value": 72.0},
        {"value": None},
    ]
    pre = adapter.preprocess_window("boiler_01", "temperature_01", raw_window)
    mask = pre["missing_mask"]
    # Only position 2 (value 72.0) is valid
    assert mask[0, 0, 0] == 0.0
    assert mask[0, 1, 0] == 0.0
    assert mask[0, 2, 0] == 1.0
    assert mask[0, 3, 0] == 0.0
    assert pre["observed_count"] == 1


# -----------------------------------------------------------------------------
# 4. Model Unavailability & Failure Tests
# -----------------------------------------------------------------------------

def test_model_unavailability_does_not_fabricate_predictions():
    """When model is unavailable, adapter returns available=False, UNAVAILABLE status, and records latency."""
    adapter = SpatialTemporalTransformerAdapter(model_path="models/non_existent.onnx")
    window = [{"value": 72.0}, {"value": 72.5}]
    pred = adapter.predict("boiler_01", "temperature_01", window)
    
    assert pred.available is False
    assert pred.candidate_value is None
    assert pred.confidence_score is None
    assert pred.inference_status == "UNAVAILABLE"
    assert pred.model_version == "unavailable"
    assert pred.inference_latency_ms is not None
    assert pred.inference_latency_ms >= 0.0
    assert "unavailable" in pred.notes.lower()


def test_model_forward_exception_handling():
    """When model forward execution raises an unexpected error, adapter safely traps it."""
    def broken_forward(norm, mask):
        raise RuntimeError("GPU out of memory or numerical NaN explosion")

    adapter = SpatialTemporalTransformerAdapter(custom_forward_fn=broken_forward)
    window = [{"value": 72.0}]
    pred = adapter.predict("boiler_01", "temperature_01", window)
    
    assert pred.available is False
    assert pred.candidate_value is None
    assert pred.inference_status == "ERROR"
    assert "numerical nan explosion" in pred.notes.lower() or "runtimeerror" in pred.notes.lower()


# -----------------------------------------------------------------------------
# 5. Inference Output Validation & Latency Logging Tests
# -----------------------------------------------------------------------------

def test_inference_output_denormalization_and_confidence():
    """Verify that predictions are denormalized to engineering units with calibrated confidence."""
    # Custom forward returning normalized point z = 0.5 (for boiler: 72.0 + 0.5 * 3.5 = 73.75 C)
    def fixed_forward(norm, mask):
        W = norm.shape[1]
        out = np.zeros_like(norm)
        out[0, -1, 0] = 0.5
        unc = np.full_like(norm, 0.05)  # small uncertainty
        return out, unc

    adapter = SpatialTemporalTransformerAdapter(
        custom_forward_fn=fixed_forward,
        version="saits-v1.2.0-test",
    )
    window = [{"value": 72.0}, {"value": 72.1}]
    pred = adapter.predict("boiler_01", "temperature_01", window)
    
    assert pred.available is True
    assert pred.inference_status == "SUCCESS"
    assert pred.model_version == "saits-v1.2.0-test"
    # Denormalized candidate: 72.0 + 0.5 * 3.5 = 73.75
    assert abs(pred.candidate_value - 73.75) < 1e-2
    assert pred.confidence_score is not None
    assert pred.confidence_score > 0.80
    assert pred.uncertainty_lower < pred.candidate_value < pred.uncertainty_upper
    assert pred.inference_latency_ms is not None
    assert pred.inference_latency_ms >= 0.0


# -----------------------------------------------------------------------------
# 6. Candidate Validation & Decision Engine Integration
# -----------------------------------------------------------------------------

def test_candidate_validation_hard_physical_bound_rejection():
    """Decision engine rejects candidate violating physical plausibility limits and marks UNRESOLVED."""
    settings = get_settings()
    engine = DecisionEngine(settings)
    
    # Candidate of 220.0 C breaches boiler max bound of 150.0 C
    cand_pred = ModelPrediction(
        available=True,
        candidate_value=220.0,
        confidence_score=0.99,
        model_name="SpatialTemporalTransformer",
        model_version="saits-v1.2.0",
        notes="Candidate violating physics",
    )
    reading = TelemetryReadingInput(
        reading_id="rd-bound-breach",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        value=None,  # missing value fault
        unit="C",
        sequence_number=101,
        timestamp=datetime.now(timezone.utc),
    )
    issue = QualityDetectionEvent(
        event_id="EVT-1",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=reading.timestamp,
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
        severity="HIGH",
    )
    
    decision = engine.decide(reading, [issue], cand_pred)
    assert decision.decision_status == "UNRESOLVED"
    assert decision.status == "UNRESOLVED"
    assert decision.accepted_value is None
    assert decision.final_value is None  # original is None
    assert decision.candidate_value == 220.0  # retained separately
    assert "failed physical bounds" in decision.decision_reason.lower()


def test_candidate_borderline_confidence_routes_to_review():
    """Candidate with borderline confidence (0.40 <= conf < 0.80) routes to REVIEW with human flag."""
    settings = get_settings()
    engine = DecisionEngine(settings)
    
    cand_pred = ModelPrediction(
        available=True,
        candidate_value=72.5,
        confidence_score=0.65,  # Borderline confidence
        model_name="SpatialTemporalTransformer",
        model_version="saits-v1.2.0",
        notes="Borderline confidence candidate",
    )
    reading = TelemetryReadingInput(
        reading_id="rd-conf-borderline",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        value=None,
        unit="C",
        sequence_number=102,
        timestamp=datetime.now(timezone.utc),
    )
    issue = QualityDetectionEvent(
        event_id="EVT-2",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=reading.timestamp,
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
        severity="HIGH",
    )
    
    decision = engine.decide(reading, [issue], cand_pred)
    assert decision.decision_status == "REVIEW"
    assert decision.human_review_required is True
    assert decision.accepted_value is None
    assert decision.candidate_value == 72.5


def test_candidate_valid_confidence_accepted_for_repair():
    """Candidate satisfying physical limits and confidence >= 0.80 is accepted as REPAIR."""
    settings = get_settings()
    engine = DecisionEngine(settings)
    
    cand_pred = ModelPrediction(
        available=True,
        candidate_value=72.3,
        confidence_score=0.91,
        model_name="SpatialTemporalTransformer",
        model_version="saits-v1.2.0",
        notes="High confidence valid candidate",
    )
    reading = TelemetryReadingInput(
        reading_id="rd-repair-valid",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        value=None,
        unit="C",
        sequence_number=103,
        timestamp=datetime.now(timezone.utc),
    )
    issue = QualityDetectionEvent(
        event_id="EVT-3",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=reading.timestamp,
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
        severity="HIGH",
    )
    
    decision = engine.decide(reading, [issue], cand_pred)
    assert decision.decision_status == "REPAIR"
    assert decision.human_review_required is False
    assert decision.candidate_value == 72.3
    assert decision.accepted_value == 72.3
    assert decision.final_value == 72.3
    assert decision.original_value is None


# -----------------------------------------------------------------------------
# 7. Original Value Preservation & Ingestion Safety
# -----------------------------------------------------------------------------

def test_original_value_preservation_across_all_statuses(db_session):
    """Verify that raw readings and original values are preserved without corruption."""
    settings = get_settings()
    
    # Custom adapter providing valid predictions
    def valid_forward(norm, mask):
        out = np.zeros_like(norm)
        out[0, -1, 0] = 0.0  # predicts mean: 72.0 C
        return out, None

    model = SpatialTemporalTransformerAdapter(custom_forward_fn=valid_forward)
    processor = PipelineProcessor(settings, model=model)
    processor.reset_state()
    
    # 1. Clean reading -> PRESERVE (Must NOT replace original with candidate)
    import uuid
    rd_pres_id = f"rd-pres-{uuid.uuid4().hex[:8]}"
    rd_spike_id = f"rd-spike-{uuid.uuid4().hex[:8]}"

    clean_ts = datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc)
    clean_rd = TelemetryReadingInput(
        reading_id=rd_pres_id,
        asset_id="boiler_01",
        sensor_id="temperature_01",
        value=74.2,
        unit="C",
        sequence_number=1,
        timestamp=clean_ts,
    )
    res_clean = processor.process_reading(clean_rd, db_session)
    assert res_clean.decision_status == "PRESERVE"
    assert res_clean.original_value == 74.2
    assert res_clean.final_value == 74.2
    assert res_clean.candidate_value is None
    
    # 2. Spike reading -> candidate generated, but original spike retained separately in raw & decision
    spike_rd = TelemetryReadingInput(
        reading_id=rd_spike_id,
        asset_id="boiler_01",
        sensor_id="temperature_01",
        value=130.0,  # Sudden spike
        unit="C",
        sequence_number=2,
        timestamp=clean_ts + timedelta(seconds=1),
    )
    res_spike = processor.process_reading(spike_rd, db_session)
    # The original value remains 130.0
    assert res_spike.original_value == 130.0
    repo = TelemetryRepository(db_session)
    raw_saved = repo.get_raw_reading_by_id(rd_spike_id)
    assert raw_saved is not None
    assert raw_saved.value == 130.0
    assert res_spike.inference_latency_ms is not None
    assert res_spike.inference_latency_ms >= 0.0
