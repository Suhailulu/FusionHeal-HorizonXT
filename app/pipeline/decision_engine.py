from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput
from app.pipeline.detection import QualityDetectionEvent
from app.pipeline.reconstruction import ModelPrediction


class DecisionRecord(BaseModel):
    status: str = Field(..., description="Primary status: PRESERVE, REPAIR, or UNRESOLVED")
    original_value: Optional[float] = Field(None, description="Original observed sensor reading")
    candidate_value: Optional[float] = Field(None, description="Candidate estimate proposed by model or method")
    accepted_value: Optional[float] = Field(None, description="Final accepted output value; null if unresolved")
    repair_method: Optional[str] = Field(None, description="Method applied for accepted repair")
    decision_reason: str = Field(..., description="Transparent technical rationale for decision")
    validation_result: str = Field(..., description="Validation outcome (PASS, FAIL, CONSTRAINT_VIOLATION)")
    policy_version: str = Field(..., description="Version of decision policy applied")
    rule_config_version: str = Field(..., description="Version of rule configuration applied")
    confidence_score: Optional[float] = Field(None, description="Model confidence score if available; null otherwise")
    model_version: Optional[str] = Field(None, description="Version of reconstruction model if invoked")


class DecisionEngine:
    """Enforces evidence-based, process-constrained acceptance policies.

    CRITICAL RULES:
    1. Detection and repair are strictly separate.
    2. A detection does not imply an automatic repair.
    3. Model predictions never become accepted output without independent constraint validation.
    4. Mandatory physical constraints cannot be overridden by a confidence score.
    5. When the ML model is not connected or evidence is insufficient, status is UNRESOLVED.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def decide(
        self,
        reading: TelemetryReadingInput,
        issues: List[QualityDetectionEvent],
        prediction: Optional[ModelPrediction] = None,
    ) -> DecisionRecord:
        policy_ver = self.settings.policy_version
        rule_ver = self.settings.rule_config_version
        constraint = self.settings.get_constraint(reading.asset_id, reading.sensor_id)

        # ---------------------------------------------------------
        # Case 1: No Quality Issues Detected -> PRESERVE
        # ---------------------------------------------------------
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
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=None,
                model_version=None,
            )

        # ---------------------------------------------------------
        # Case 2: Issues Detected -> Evaluate Candidate / Model Availability
        # ---------------------------------------------------------
        issue_descriptions = [f"{i.issue_type} ({i.detection_rule})" for i in issues]
        issues_summary = "; ".join(issue_descriptions)

        # If model is unavailable or returned no candidate
        if prediction is None or not prediction.available or prediction.candidate_value is None:
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
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=None,
                model_version=prediction.model_version if prediction else None,
            )

        # ---------------------------------------------------------
        # Case 3: Candidate Available -> Independent Constraint Verification
        # ---------------------------------------------------------
        candidate = prediction.candidate_value

        # Mandatory physical range check
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
                    policy_version=policy_ver,
                    rule_config_version=rule_ver,
                    confidence_score=prediction.confidence_score,
                    model_version=prediction.model_version,
                )

        # Policy acceptance check
        if self.settings.auto_accept_repairs:
            return DecisionRecord(
                status="REPAIR",
                original_value=reading.value,
                candidate_value=candidate,
                accepted_value=candidate,
                repair_method=prediction.reconstruction_method or prediction.model_name,
                decision_reason=(
                    f"Fault(s) [{issues_summary}] mitigated. "
                    f"Candidate {candidate:.2f} satisfies all physical constraints and acceptance policy."
                ),
                validation_result="PASS_VERIFIED",
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=prediction.confidence_score,
                model_version=prediction.model_version,
            )
        else:
            return DecisionRecord(
                status="UNRESOLVED",
                original_value=reading.value,
                candidate_value=candidate,
                accepted_value=None,
                repair_method=prediction.reconstruction_method or prediction.model_name,
                decision_reason="Candidate passed constraints but auto-accept is disabled under active plant policy.",
                validation_result="PENDING_REVIEW",
                policy_version=policy_ver,
                rule_config_version=rule_ver,
                confidence_score=prediction.confidence_score,
                model_version=prediction.model_version,
            )
