from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator
from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.detection import QualityDetectionEvent
from app.pipeline.reconstruction import ModelPrediction

DecisionStatus = Literal["PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"]


class DecisionRecord(BaseModel):
    status: str = Field(
        ..., description="Official decision status: PRESERVE, REPAIR, REVIEW, or UNRESOLVED"
    )
    decision_status: DecisionStatus = Field(
        "PRESERVE", description="Official 4-status industrial decision: PRESERVE, REPAIR, REVIEW, or UNRESOLVED"
    )
    human_review_required: bool = Field(
        False, description="True if this observation requires human engineering assessment"
    )
    original_value: Optional[float] = Field(None, description="Original observed sensor reading")
    candidate_value: Optional[float] = Field(None, description="Candidate estimate proposed by model or method")
    accepted_value: Optional[float] = Field(None, description="Final accepted output value; null if review or unresolved")
    final_value: Optional[float] = Field(None, description="Published output value: original for PRESERVE/REVIEW/UNRESOLVED, candidate for REPAIR")
    repair_method: Optional[str] = Field(None, description="Method applied for accepted repair")
    decision_reason: str = Field(..., description="Transparent technical rationale for decision")
    reason: Optional[str] = Field(None, description="Alias for decision_reason")
    validation_result: str = Field(..., description="Validation outcome (PASS, FAIL, CONSTRAINT_VIOLATION, PENDING_REVIEW)")
    review_rule_triggered: Optional[str] = Field(None, description="Specific review rule triggered if status is REVIEW")
    policy_version: str = Field(..., description="Version of decision policy applied")
    rule_config_version: str = Field(..., description="Version of rule configuration applied")
    confidence_score: Optional[float] = Field(None, description="Model confidence score if available; null otherwise")
    confidence: Optional[float] = Field(None, description="Alias for confidence_score")
    model_version: Optional[str] = Field(None, description="Version of reconstruction model if invoked")
    inference_latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds if model was executed")

    @model_validator(mode="after")
    def sync_decision_fields(self) -> DecisionRecord:
        if not self.reason:
            self.reason = self.decision_reason
        if self.confidence is None and self.confidence_score is not None:
            self.confidence = self.confidence_score

        # Synchronize status and decision_status across all 4 official statuses
        if self.status in ("PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"):
            self.decision_status = self.status
        else:
            self.status = self.decision_status

        # Set human_review_required flag
        self.human_review_required = (self.decision_status == "REVIEW")

        # Guarantee: final_value matches accepted_value for REPAIR, original_value for PRESERVE/REVIEW/UNRESOLVED
        if self.decision_status == "REPAIR":
            if self.final_value is None and self.accepted_value is not None:
                self.final_value = self.accepted_value
            elif self.accepted_value is None and self.final_value is not None:
                self.accepted_value = self.final_value
        elif self.decision_status in ("REVIEW", "UNRESOLVED"):
            # Original data kept intact! Candidate is NEVER placed into final_value or accepted_value
            self.accepted_value = None
            self.final_value = self.original_value
        elif self.decision_status == "PRESERVE":
            self.final_value = self.original_value
            self.accepted_value = self.original_value

        return self


class DecisionEngine:
    """Centralized industrial decision engine enforcing evidence-governed, process-constrained policies.

    THE FOUR OFFICIAL DECISION STATUSES:
    1. PRESERVE:
       No actionable data-quality problem is detected and no modification is justified.
       Original observation is retained unchanged.

    2. REPAIR:
       A data-quality issue was identified, a correction candidate exists, passes independent
       physical bounds & rate-of-change constraints, satisfies confidence policy, and auto-repair is permitted.
       Stores original and accepted values separately.

    3. REVIEW:
       A decision requires additional evidence, human assessment, or further investigation.
       Original telemetry is kept intact, candidate (if any) is preserved separately,
       and the event is marked pending review.

    4. UNRESOLVED:
       The current processing attempt cannot establish a reliable correction and no available
       investigation path is expected to resolve the issue. Original data is kept intact.
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
        latency_ms = prediction.inference_latency_ms if prediction else None

        # ---------------------------------------------------------------------
        # Status 1: PRESERVE (Clean telemetry)
        # ---------------------------------------------------------------------
        if not issues:
            return DecisionRecord(
                status="PRESERVE",
                decision_status="PRESERVE",
                human_review_required=False,
                original_value=reading.value,
                candidate_value=None,
                accepted_value=reading.value,
                final_value=reading.value,
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
                inference_latency_ms=latency_ms,
            )

        issue_types = {i.issue_type for i in issues}
        issue_descriptions = [f"{i.issue_type} ({i.detection_rule})" for i in issues]
        issues_summary = "; ".join(issue_descriptions)

        # ---------------------------------------------------------------------
        # Handling Anomaly Without Model Candidate
        # ---------------------------------------------------------------------
        if prediction is None or not prediction.available or prediction.candidate_value is None:
            # Ambiguous Frozen Sensor with no model -> Requires operator review (Rule R3)
            if "FROZEN_SENSOR" in issue_types and len(issue_types) == 1:
                return DecisionRecord(
                    status="REVIEW",
                    decision_status="REVIEW",
                    human_review_required=True,
                    original_value=reading.value,
                    candidate_value=None,
                    accepted_value=None,
                    final_value=reading.value,
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
                    model_version=prediction.model_version if prediction else "unavailable",
                    inference_latency_ms=latency_ms,
                )

            return DecisionRecord(
                status="UNRESOLVED",
                decision_status="UNRESOLVED",
                human_review_required=False,
                original_value=reading.value,
                candidate_value=None,
                accepted_value=None,
                final_value=reading.value,
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
                model_version=prediction.model_version if prediction else "unavailable",
                inference_latency_ms=latency_ms,
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
                    decision_status="UNRESOLVED",
                    human_review_required=False,
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    final_value=reading.value,
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
                    inference_latency_ms=latency_ms,
                )

        # ---------------------------------------------------------------------
        # Status 3: REVIEW Rules (Pending human assessment)
        # ---------------------------------------------------------------------

        # Review Rule R1: Manual Plant Policy Gate
        is_sensor_manual = sensor_key in self.settings.manual_review_sensors
        if not self.settings.auto_accept_repairs or is_sensor_manual:
            return DecisionRecord(
                status="REVIEW",
                decision_status="REVIEW",
                human_review_required=True,
                original_value=reading.value,
                candidate_value=candidate,
                accepted_value=None,
                final_value=reading.value,
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
                inference_latency_ms=latency_ms,
            )

        # Review Rule R2: Confidence Score Threshold Gate
        if confidence is not None:
            min_accept = self.settings.min_auto_accept_confidence  # default 0.80
            min_review = self.settings.min_review_confidence       # default 0.40

            if confidence < min_review:
                # Too weak even for human review -> UNRESOLVED
                return DecisionRecord(
                    status="UNRESOLVED",
                    decision_status="UNRESOLVED",
                    human_review_required=False,
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    final_value=reading.value,
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
                    inference_latency_ms=latency_ms,
                )
            elif confidence < min_accept:
                # Borderline confidence (0.40 <= conf < 0.80) -> REVIEW
                return DecisionRecord(
                    status="REVIEW",
                    decision_status="REVIEW",
                    human_review_required=True,
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    final_value=reading.value,
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
                    inference_latency_ms=latency_ms,
                )

        # Review Rule R3: Staleness / Frozen Sensor Ambiguity
        if "FROZEN_SENSOR" in issue_types:
            return DecisionRecord(
                status="REVIEW",
                decision_status="REVIEW",
                human_review_required=True,
                original_value=reading.value,
                candidate_value=candidate,
                accepted_value=None,
                final_value=reading.value,
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
                inference_latency_ms=latency_ms,
            )

        # Review Rule R4: Soft Dynamic Constraint Warning (Elevated Rate-of-Change)
        if constraint and previous_value is not None:
            delta = abs(candidate - previous_value)
            if delta > 0.70 * constraint.max_rate_of_change:
                return DecisionRecord(
                    status="REVIEW",
                    decision_status="REVIEW",
                    human_review_required=True,
                    original_value=reading.value,
                    candidate_value=candidate,
                    accepted_value=None,
                    final_value=reading.value,
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
                    inference_latency_ms=latency_ms,
                )

        # ---------------------------------------------------------------------
        # Status 2: REPAIR (All checks and confidence thresholds passed!)
        # ---------------------------------------------------------------------
        return DecisionRecord(
            status="REPAIR",
            decision_status="REPAIR",
            human_review_required=False,
            original_value=reading.value,
            candidate_value=candidate,
            accepted_value=candidate,
            final_value=candidate,
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
            inference_latency_ms=latency_ms,
        )
