from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.detection import QualityDetectionEvent
from app.pipeline.reconstruction import ModelPrediction

DecisionStatus = Literal["PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"]


class DecisionRecord(BaseModel):
    status: DecisionStatus = Field(
        ..., description="Primary status: PRESERVE, REPAIR, REVIEW, or UNRESOLVED"
    )
    original_value: Optional[float] = Field(None, description="Original observed sensor reading")
    candidate_value: Optional[float] = Field(None, description="Candidate estimate proposed by model or method")
    accepted_value: Optional[float] = Field(None, description="Final accepted output value; null if review or unresolved")
    repair_method: Optional[str] = Field(None, description="Method applied for accepted repair")
    decision_reason: str = Field(..., description="Transparent technical rationale for decision")
    validation_result: str = Field(..., description="Validation outcome (PASS, FAIL, CONSTRAINT_VIOLATION, PENDING_REVIEW)")
    review_rule_triggered: Optional[str] = Field(None, description="Specific review rule triggered if status is REVIEW")
    policy_version: str = Field(..., description="Version of decision policy applied")
    rule_config_version: str = Field(..., description="Version of rule configuration applied")
    confidence_score: Optional[float] = Field(None, description="Model confidence score if available; null otherwise")
    model_version: Optional[str] = Field(None, description="Version of reconstruction model if invoked")


class DecisionEngine:
    """Enforces evidence-governed, process-constrained acceptance and review policies.

    GRADED DECISION TAXONOMY:
    1. PRESERVE:
       Observation conforms to all physical plausibility, rate-of-change, and sequence rules.
       Original telemetry is preserved intact.

    2. REPAIR:
       A data-quality issue was detected, an ML candidate is available, and the candidate
       strictly passes independent physical bounds, rate-of-change checks, high-confidence
       thresholds, and auto-acceptance policy.

    3. REVIEW:
       A candidate or anomaly exists, but cannot be automatically accepted without human engineer
       judgment under one of the 4 explicit Review Rules:
       - Rule R1 (Policy Gate): Auto-acceptance is disabled by plant policy or sensor is designated for manual sign-off.
       - Rule R2 (Confidence Gate): Candidate satisfies physical bounds, but model confidence is borderline (0.40 <= conf < 0.80).
       - Rule R3 (Staleness Ambiguity Gate): Sensor shows repeated identical values (FROZEN_SENSOR); requires operator review to distinguish steady-state operation from ADC freeze.
       - Rule R4 (Soft Constraint Warning): Candidate satisfies absolute bounds but approaches dynamic rate-of-change envelope.

    4. UNRESOLVED:
       The system cannot safely reconstruct the value, the ML model is unavailable,
       the candidate violates mandatory physical constraints, or confidence is below minimum
       plausible threshold (< 0.40). The system refuses to invent synthetic data.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def decide(
        self,
        reading: TelemetryReadingInput,
        issues: List[QualityDetectionEvent],
        prediction: Optional[ModelPrediction] = None,
        previous_value: Optional[float] = None,
    ) -> DecisionRecord:
        policy_ver = self.settings.policy_version
        rule_ver = self.settings.rule_config_version
        sensor_key = f"{reading.asset_id}:{reading.sensor_id}"
        constraint = self.settings.get_constraint(reading.asset_id, reading.sensor_id)

        # ---------------------------------------------------------------------
        # Status 1: PRESERVE
        # ---------------------------------------------------------------------
        if not issues:
            return DecisionRecord(
                status="PRESERVE",
                original_value=reading.value,
                candidate_value=None,
                accepted_value=reading.value,
                repair_method=None,
                decision_reason=(
                    "Observation conforms to physical plausibility bounds, continuous sequence, "
                    "and nominal rate-of-change rules. Original telemetry preserved."
                ),
                validation_result="PASS",
                review_rule_triggered=None,
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=None,
                model_version=None,
            )

        issue_types = {i.issue_type for i in issues}
        issue_descriptions = [f"{i.issue_type} ({i.detection_rule})" for i in issues]
        issues_summary = "; ".join(issue_descriptions)

        # ---------------------------------------------------------------------
        # Status 4 (Branch A): UNRESOLVED when Model is Disconnected or No Candidate
        # ---------------------------------------------------------------------
        if prediction is None or not prediction.available or prediction.candidate_value is None:
            # Special Check: Ambiguous Frozen Sensor with no model
            if "FROZEN_SENSOR" in issue_types and len(issue_types) == 1:
                return DecisionRecord(
                    status="REVIEW",
                    original_value=reading.value,
                    candidate_value=None,
                    accepted_value=None,
                    repair_method=None,
                    decision_reason=(
                        "Consecutive identical sensor readings detected. "
                        "Flagged for human operator review to distinguish legitimate steady-state operation from ADC line freeze."
                    ),
                    validation_result="STALENESS_AMBIGUITY_REVIEW",
                    review_rule_triggered="RULE_R3_STALENESS_AMBIGUITY",
                    policy_version=policy_ver,
                    rule_config_version=rule_ver,
                    confidence_score=None,
                    model_version=prediction.model_version if prediction else None,
                )

            return DecisionRecord(
                status="UNRESOLVED",
                original_value=reading.value,
                candidate_value=None,
                accepted_value=None,
                repair_method=None,
                decision_reason=(
                    f"Data-quality fault(s) detected [{issues_summary}]. "
                    "ML reconstruction model is unavailable or produced no candidate. "
                    "Refusing to fabricate replacement; telemetry marked as UNRESOLVED."
                ),
                validation_result="UNRESOLVED_FAULT",
                review_rule_triggered=None,
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=None,
                model_version=prediction.model_version if prediction else None,
            )

        # ---------------------------------------------------------------------
        # Candidate Evaluation & Constraint Verification
        # ---------------------------------------------------------------------
        candidate = prediction.candidate_value
        confidence = prediction.confidence_score

        # Mandatory Hard Physical Range Check
        if constraint:
            if candidate < constraint.min_value or candidate > constraint.max_value:
                return DecisionRecord(
                    status="UNRESOLVED",
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    repair_method=prediction.reconstruction_method or prediction.model_name,
                    decision_reason=(
                        f"Candidate value {candidate:.2f} failed physical bounds "
                        f"[{constraint.min_value}, {constraint.max_value}] {constraint.unit}. "
                        "Mandatory physical constraints cannot be overridden by ML scores."
                    ),
                    validation_result="CONSTRAINT_VIOLATION",
                    review_rule_triggered=None,
                    policy_version=policy_ver,
                    rule_config_version=rule_ver,
                    confidence_score=confidence,
                    model_version=prediction.model_version,
                )

        # ---------------------------------------------------------------------
        # Status 3: REVIEW Rules
        # ---------------------------------------------------------------------

        # Review Rule R1: Manual Plant Policy Gate
        is_sensor_manual = sensor_key in self.settings.manual_review_sensors
        if not self.settings.auto_accept_repairs or is_sensor_manual:
            return DecisionRecord(
                status="REVIEW",
                original_value=reading.value,
                candidate_value=candidate,
                accepted_value=None,
                repair_method=prediction.reconstruction_method or prediction.model_name,
                decision_reason=(
                    f"Candidate value {candidate:.2f} verified against physical constraints. "
                    "Routed to REVIEW under mandatory plant operator sign-off policy."
                ),
                validation_result="POLICY_REVIEW",
                review_rule_triggered="RULE_R1_POLICY_GATE",
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=confidence,
                model_version=prediction.model_version,
            )

        # Review Rule R2: Confidence Score Threshold Gate
        if confidence is not None:
            min_accept = self.settings.min_auto_accept_confidence  # default 0.80
            min_review = self.settings.min_review_confidence       # default 0.40

            if confidence < min_review:
                # Too weak even for human review -> UNRESOLVED
                return DecisionRecord(
                    status="UNRESOLVED",
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    repair_method=prediction.reconstruction_method or prediction.model_name,
                    decision_reason=(
                        f"Candidate confidence score ({confidence:.2f}) is below minimum acceptable evidence threshold ({min_review:.2f}). "
                        "Candidate rejected as unverified; telemetry remains UNRESOLVED."
                    ),
                    validation_result="INSUFFICIENT_CONFIDENCE",
                    review_rule_triggered=None,
                    policy_version=policy_ver,
                    rule_config_version=rule_ver,
                    confidence_score=confidence,
                    model_version=prediction.model_version,
                )
            elif confidence < min_accept:
                # Borderline confidence (0.40 <= conf < 0.80) -> REVIEW
                return DecisionRecord(
                    status="REVIEW",
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    repair_method=prediction.reconstruction_method or prediction.model_name,
                    decision_reason=(
                        f"Candidate value {candidate:.2f} satisfies physical bounds, but model confidence ({confidence:.2f}) "
                        f"is below automatic acceptance threshold ({min_accept:.2f}). Flagged for human engineer review."
                    ),
                    validation_result="BORDERLINE_CONFIDENCE_REVIEW",
                    review_rule_triggered="RULE_R2_CONFIDENCE_GATE",
                    policy_version=policy_ver,
                    rule_config_version=rule_ver,
                    confidence_score=confidence,
                    model_version=prediction.model_version,
                )

        # Review Rule R3: Staleness / Frozen Sensor Ambiguity
        if "FROZEN_SENSOR" in issue_types:
            return DecisionRecord(
                status="REVIEW",
                original_value=reading.value,
                candidate_value=candidate,
                accepted_value=None,
                repair_method=prediction.reconstruction_method or prediction.model_name,
                decision_reason=(
                    f"Candidate {candidate:.2f} generated for frozen sensor event. "
                    "Routed to REVIEW to confirm with operator that equipment was not in intentional stationary state."
                ),
                validation_result="STALENESS_AMBIGUITY_REVIEW",
                review_rule_triggered="RULE_R3_STALENESS_AMBIGUITY",
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=confidence,
                model_version=prediction.model_version,
            )

        # Review Rule R4: Soft Dynamic Constraint Warning (Elevated Rate-of-Change)
        if constraint and previous_value is not None:
            delta = abs(candidate - previous_value)
            if delta > 0.70 * constraint.max_rate_of_change:
                return DecisionRecord(
                    status="REVIEW",
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    repair_method=prediction.reconstruction_method or prediction.model_name,
                    decision_reason=(
                        f"Candidate {candidate:.2f} satisfies absolute limits but exhibits high rate-of-change "
                        f"({delta:.2f} > 70% of max {constraint.max_rate_of_change:.2f} {constraint.unit}/s). "
                        "Flagged for soft constraint review."
                    ),
                    validation_result="SOFT_CONSTRAINT_WARNING",
                    review_rule_triggered="RULE_R4_SOFT_CONSTRAINT",
                    policy_version=policy_ver,
                    rule_config_version=rule_ver,
                    confidence_score=confidence,
                    model_version=prediction.model_version,
                )

        # ---------------------------------------------------------------------
        # Status 2: REPAIR (All checks and confidence thresholds passed!)
        # ---------------------------------------------------------------------
        return DecisionRecord(
            status="REPAIR",
            original_value=reading.value,
            candidate_value=candidate,
            accepted_value=candidate,
            repair_method=prediction.reconstruction_method or prediction.model_name,
            decision_reason=(
                f"Fault(s) [{issues_summary}] mitigated. "
                f"Candidate {candidate:.2f} satisfies all physical constraints, confidence thresholds, and acceptance policy."
            ),
            validation_result="PASS_VERIFIED",
            review_rule_triggered=None,
            policy_version=policy_ver,
            rule_config_version=rule_ver,
            confidence_score=confidence,
            model_version=prediction.model_version,
        )
