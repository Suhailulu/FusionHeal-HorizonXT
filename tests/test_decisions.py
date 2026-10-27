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
        min_auto_accept_confidence=0.80,
        min_review_confidence=0.40,
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
        confidence_score=0.91,  # >= 0.80 -> High confidence
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


# -----------------------------------------------------------------------------
# Tests for REVIEW Status & Rules
# -----------------------------------------------------------------------------

def test_review_rule_r1_policy_gate(test_settings):
    """Rule R1: If plant policy disables auto-accept or sensor requires sign-off -> REVIEW."""
    test_settings.auto_accept_repairs = False
    engine = DecisionEngine(test_settings)
    reading = make_reading(None)
    issue = QualityDetectionEvent(
        event_id="EVT-R1",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
    )
    pred = ModelPrediction(
        available=True,
        candidate_value=72.5,
        confidence_score=0.95,
        model_name="Transformer",
        model_version="1.0.0",
        notes="Valid high confidence candidate",
    )
    decision = engine.decide(reading, issues=[issue], prediction=pred)

    assert decision.status == "REVIEW"
    assert decision.candidate_value == 72.5
    assert decision.accepted_value is None  # Candidate withheld until operator sign-off!
    assert decision.review_rule_triggered == "RULE_R1_POLICY_GATE"
    assert "mandatory plant operator sign-off" in decision.decision_reason


def test_review_rule_r2_borderline_confidence(test_settings):
    """Rule R2: Candidate valid, but confidence score is borderline (0.40 <= conf < 0.80) -> REVIEW."""
    engine = DecisionEngine(test_settings)
    reading = make_reading(None)
    issue = QualityDetectionEvent(
        event_id="EVT-R2",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
    )
    pred = ModelPrediction(
        available=True,
        candidate_value=74.0,
        confidence_score=0.62,  # Between 0.40 and 0.80
        model_name="Transformer",
        model_version="1.0.0",
        notes="Borderline confidence prediction",
    )
    decision = engine.decide(reading, issues=[issue], prediction=pred)

    assert decision.status == "REVIEW"
    assert decision.candidate_value == 74.0
    assert decision.accepted_value is None
    assert decision.review_rule_triggered == "RULE_R2_CONFIDENCE_GATE"
    assert "below automatic acceptance threshold" in decision.decision_reason


def test_review_rule_r3_frozen_sensor_staleness(test_settings):
    """Rule R3: Frozen sensor could be intentional steady-state operation -> REVIEW."""
    engine = DecisionEngine(test_settings)
    reading = make_reading(72.0)
    issue = QualityDetectionEvent(
        event_id="EVT-R3",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="FROZEN_SENSOR",
        detection_rule="ConsecutiveIdenticalValueRule",
        observed_evidence={"repeated_value": 72.0, "consecutive_count": 6},
    )
    pred = ModelPrediction(
        available=True,
        candidate_value=72.2,
        confidence_score=0.88,
        model_name="Transformer",
        model_version="1.0.0",
        notes="Candidate for frozen reading",
    )
    decision = engine.decide(reading, issues=[issue], prediction=pred)

    assert decision.status == "REVIEW"
    assert decision.review_rule_triggered == "RULE_R3_STALENESS_AMBIGUITY"
    assert "Routed to REVIEW to confirm with operator" in decision.decision_reason


def test_review_rule_r4_soft_constraint_warning(test_settings):
    """Rule R4: Candidate approaches dynamic rate-of-change envelope -> REVIEW."""
    engine = DecisionEngine(test_settings)
    reading = make_reading(None)
    issue = QualityDetectionEvent(
        event_id="EVT-R4",
        asset_id="boiler_01",
        sensor_id="temperature_01",
        timestamp=datetime.now(timezone.utc),
        issue_type="MISSING_DATA",
        detection_rule="ExplicitNullRule",
        observed_evidence={},
    )
    # Boiler max rate of change is 5.0 C/s.
    # Previous value was 70.0; candidate is 74.0 -> delta is 4.0 (> 70% of 5.0 = 3.5 C/s)
    pred = ModelPrediction(
        available=True,
        candidate_value=74.0,
        confidence_score=0.85,
        model_name="Transformer",
        model_version="1.0.0",
        notes="Candidate near rate of change limit",
    )
    decision = engine.decide(reading, issues=[issue], prediction=pred, previous_value=70.0)

    assert decision.status == "REVIEW"
    assert decision.review_rule_triggered == "RULE_R4_SOFT_CONSTRAINT"
    assert "soft constraint review" in decision.decision_reason
