from __future__ import annotations
from typing import Any


class ConfidenceEngine:
    """Computes transparent, uncalibrated heuristic reliability scores for proposed sensor corrections.

    CRITICAL ENGINEERING PRINCIPLE:
    This score is an engineering heuristic reflecting evidence alignment and constraint compliance.
    It is NOT a Bayesian probability, statistical confidence interval, or percentage chance of truth.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def assess(
        self,
        candidate: dict[str, Any] | None,
        evidence_list: list[dict[str, Any]],
        constraint_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Produce a structured heuristic reliability assessment for a candidate."""
        if candidate is None:
            return {
                "score": None,
                "score_type": "not_available",
                "supporting_factors": [],
                "limitations": ["No candidate replacement value available."],
                "calibration_status": "uncalibrated",
            }

        # 1. Hard constraint check: Any hard constraint failure drops score to 0.0
        hard_fails = [c for c in constraint_results if c.get("severity") == "hard" and c.get("status") == "fail"]
        if hard_fails:
            reasons = [c.get("explanation", "") for c in hard_fails]
            return {
                "score": 0.0,
                "score_type": "heuristic",
                "supporting_factors": [],
                "limitations": [f"Candidate violated hard process constraint: {r}" for r in reasons],
                "calibration_status": "uncalibrated",
            }

        # 2. Base score derived from candidate method
        method = candidate.get("method", "unknown")
        if method == "linear_interpolation":
            base_score = 0.70
            base_factor = "Linear interpolation over valid local interior interval."
        elif method == "local_median":
            base_score = 0.65
            base_factor = "Local median over robust temporal window."
        elif method == "cross_sensor_estimate":
            base_score = 0.75
            base_factor = "Estimate grounded in configured cross-sensor physical relationship."
        else:
            base_score = 0.40
            base_factor = f"Uncategorized reconstruction method '{method}'."

        supporting_factors: list[str] = [base_factor]
        limitations: list[str] = [
            "Score is a heuristic engineering metric, not a calibrated statistical probability."
        ]

        # 3. Evidence influence
        supporting_ev = [e for e in evidence_list if e.get("direction") == "supporting"]
        contradicting_ev = [e for e in evidence_list if e.get("direction") == "contradicting"]
        unknown_constraints = [c for c in constraint_results if c.get("status") == "unknown"]
        soft_warnings = [c for c in constraint_results if c.get("status") == "warning"]

        # Boost for supporting evidence
        if supporting_ev:
            bonus = min(0.20, len(supporting_ev) * 0.05)
            base_score += bonus
            for e in supporting_ev:
                supporting_factors.append(f"{e.get('evidence_type', '').capitalize()}: {e.get('explanation')}")

        # Penalty for contradictory evidence
        if contradicting_ev:
            penalty = len(contradicting_ev) * 0.25
            base_score -= penalty
            for e in contradicting_ev:
                limitations.append(f"Contradiction from {e.get('evidence_type')}: {e.get('explanation')}")

        # Penalty for soft warnings or unknown constraints
        if soft_warnings:
            base_score -= len(soft_warnings) * 0.15
            for w in soft_warnings:
                limitations.append(f"Soft constraint warning: {w.get('explanation')}")

        if unknown_constraints:
            base_score -= len(unknown_constraints) * 0.10
            for u in unknown_constraints:
                limitations.append(f"Constraint status unknown: {u.get('explanation')}")

        # Clamp between 0.05 and 0.95 (heuristic scores should never claim 100% certainty)
        final_score = float(max(0.05, min(0.95, round(base_score, 3))))

        return {
            "score": final_score,
            "score_type": "heuristic",
            "supporting_factors": supporting_factors,
            "limitations": limitations,
            "calibration_status": "uncalibrated",
        }
