from __future__ import annotations
from datetime import datetime, timezone
import pytest
from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.detection import QualityDetectionEvent
from app.pipeline.decision_engine import DecisionEngine
from app.pipeline.reconstruction import ModelPrediction


@pytest.fixture
def test_settings():
    return Settings(
        policy_version="1.0.0-test",
        rule_config_version="2026.1-test",
        auto_accept_repairs=True,
    )


def make_reading(val: float | None = 72.0) -> TelemetryReadingInput:
    return TelemetryReadingInput(
        reading_id="dec-test-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        sequence_number=1,
        value=val,
        unit="C",
        source_type="simulated",
    )


def test_decision_preserve_when_clean(test_settings):
    engine = DecisionEngine(test_settings)
    reading = make_reading(72.0)
    decision = engine.decide(reading, issues=[], prediction=None)

    assert decision.status == "PRESERVE"
    assert decision.accepted_value == 72.0
    assert decision.candidate_value is None
    assert decision.validation_result == "PASS"


def test_decision_unresolved_when_model_unavailable(test_settings):
    engine = DecisionEngine(test_settings)
    reading = make_reading(None)
    issue = QualityDetectionEvent(
        event_id="EVT-01",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
    )
    # Model unavailable
    pred = ModelPrediction(
        available=False,
        candidate_value=None,
        model_name="Placeholder",
        model_version="0.0.0",
        notes="Model disconnected",
    )
    decision = engine.decide(reading, issues=[issue], prediction=pred)

    assert decision.status == "UNRESOLVED"
    assert decision.accepted_value is None
    assert "Refusing to fabricate replacement" in decision.decision_reason


def test_decision_unresolved_when_candidate_violates_bounds(test_settings):
    engine = DecisionEngine(test_settings)
    reading = make_reading(120.0)
    issue = QualityDetectionEvent(
        event_id="EVT-02",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="SUDDEN_SPIKE",
        detection_rule="RateOfChangeSpikeRule",
        observed_evidence={},
    )
    # Candidate violates boiler max (150 C)
    pred = ModelPrediction(
        available=True,
        candidate_value=220.0,
        confidence_score=0.99,  # High confidence cannot override physical constraint!
        model_name="TransformerModel",
        model_version="1.0.0",
        notes="High confidence prediction",
    )
    decision = engine.decide(reading, issues=[issue], prediction=pred)

    assert decision.status == "UNRESOLVED"
    assert decision.accepted_value is None
    assert decision.validation_result == "CONSTRAINT_VIOLATION"
    assert "cannot be overridden by ML scores" in decision.decision_reason


def test_decision_repair_when_valid(test_settings):
    engine = DecisionEngine(test_settings)
    reading = make_reading(None)
    issue = QualityDetectionEvent(
        event_id="EVT-03",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
    )
    pred = ModelPrediction(
        available=True,
        candidate_value=73.2,
        confidence_score=0.91,
        model_name="SpatialTemporalTransformer",
        model_version="1.0.0",
        reconstruction_method="saits_imputer",
        notes="Valid candidate",
    )
    decision = engine.decide(reading, issues=[issue], prediction=pred)

    assert decision.status == "REPAIR"
    assert decision.accepted_value == 73.2
    assert decision.candidate_value == 73.2
    assert decision.validation_result == "PASS_VERIFIED"
