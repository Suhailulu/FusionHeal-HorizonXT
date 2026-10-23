from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


class ProcessConstraintEngine:
    """Evaluates candidate replacement values against configured engineering and process constraints.

    Enforces that failed hard constraints block automatic acceptance, unknown constraints
    are not treated as passed, and soft constraints generate diagnostic warnings.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def evaluate_constraints(
        self,
        df: pd.DataFrame,
        issue: dict[str, Any],
        candidate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Evaluate all applicable process constraints for a given candidate."""
        results: list[dict[str, Any]] = []
        col = issue.get("sensor_id")
        pos = issue.get("row_position")
        val = candidate.get("candidate_value")

        if col is None or val is None:
            return results

        pos = int(pos) if pos is not None else None

        # 1. Configured constraints list
        raw_constraints = self.config.get("constraints", [])
        sensor_limits = self.config.get("sensor_limits", {})

        evaluated_rule_ids: set[str] = set()

        for c in raw_constraints:
            rule_id = c.get("rule_id", "unnamed_rule")
            app_sensor = c.get("applicable_sensor")

            if app_sensor != col:
                continue

            evaluated_rule_ids.add(rule_id)
            res = self._evaluate_single_constraint(c, df, col, pos, val)
            results.append(res)

        # 2. Default Range Constraint from sensor_limits (treated as hard bounds by default)
        if col in sensor_limits and f"{col}_limits" not in evaluated_rule_ids:
            limit = sensor_limits[col]
            min_lim = limit.get("min")
            max_lim = limit.get("max")

            if min_lim is not None or max_lim is not None:
                passed = True
                failed_boundary = None

                if min_lim is not None and val < min_lim:
                    passed = False
                    failed_boundary = f"< minimum {min_lim}"
                elif max_lim is not None and val > max_lim:
                    passed = False
                    failed_boundary = f"> maximum {max_lim}"

                results.append({
                    "rule_id": f"RULE-RANGE-{col}",
                    "applicable_sensor": col,
                    "severity": "hard",
                    "status": "pass" if passed else "fail",
                    "explanation": (
                        f"Candidate value {val:.2f} satisfies configured physical boundaries [{min_lim}, {max_lim}]."
                        if passed
                        else f"Candidate value {val:.2f} breaches hard range boundary ({failed_boundary})."
                    ),
                    "observed_value": float(val),
                    "expected_parameters": {"min": min_lim, "max": max_lim},
                })

        return results

    def _evaluate_single_constraint(
        self,
        c: dict[str, Any],
        df: pd.DataFrame,
        col: str,
        pos: int | None,
        val: float,
    ) -> dict[str, Any]:
        """Evaluate an individual structured constraint rule."""
        rule_id = c.get("rule_id", "rule")
        severity = c.get("severity", "hard").lower()
        rule_type = c.get("rule_type", "range").lower()

        # Rule Type A: Range
        if rule_type == "range":
            min_val = c.get("min")
            max_val = c.get("max")
            if min_val is None and max_val is None:
                return {
                    "rule_id": rule_id,
                    "applicable_sensor": col,
                    "severity": severity,
                    "status": "unknown",
                    "explanation": f"Constraint '{rule_id}' has neither min nor max configured.",
                    "observed_value": float(val),
                    "expected_parameters": {},
                }

            passed = True
            breach_text = ""
            if min_val is not None and val < min_val:
                passed = False
                breach_text = f"below min {min_val}"
            elif max_val is not None and val > max_val:
                passed = False
                breach_text = f"above max {max_val}"

            status = "pass" if passed else ("fail" if severity == "hard" else "warning")
            expl = (
                f"Candidate {val:.2f} is within valid bounds [{min_val}, {max_val}]."
                if passed
                else f"Candidate {val:.2f} is {breach_text}."
            )
            return {
                "rule_id": rule_id,
                "applicable_sensor": col,
                "severity": severity,
                "status": status,
                "explanation": expl,
                "observed_value": float(val),
                "expected_parameters": {"min": min_val, "max": max_val},
            }

        # Rule Type B: Rate of Change (ROC)
        elif rule_type == "rate_of_change":
            max_roc = c.get("max_rate_of_change")
            if max_roc is None or pos is None:
                return {
                    "rule_id": rule_id,
                    "applicable_sensor": col,
                    "severity": severity,
                    "status": "unknown",
                    "explanation": f"Constraint '{rule_id}' could not evaluate rate of change (missing parameter or index).",
                    "observed_value": float(val),
                    "expected_parameters": {"max_rate_of_change": max_roc},
                }

            # Check preceding point
            if pos > 0 and pos < len(df):
                prev_val = pd.to_numeric(df.iloc[pos - 1][col], errors="coerce")
                if np.isfinite(prev_val):
                    delta = abs(val - prev_val)
                    if delta <= max_roc:
                        return {
                            "rule_id": rule_id,
                            "applicable_sensor": col,
                            "severity": severity,
                            "status": "pass",
                            "explanation": f"Rate of change |{val:.2f} - {prev_val:.2f}| = {delta:.2f} <= max {max_roc:.2f}.",
                            "observed_value": float(delta),
                            "expected_parameters": {"max_rate_of_change": max_roc},
                        }
                    else:
                        status = "fail" if severity == "hard" else "warning"
                        return {
                            "rule_id": rule_id,
                            "applicable_sensor": col,
                            "severity": severity,
                            "status": status,
                            "explanation": f"Rate of change {delta:.2f} exceeds allowable max {max_roc:.2f}.",
                            "observed_value": float(delta),
                            "expected_parameters": {"max_rate_of_change": max_roc},
                        }

            return {
                "rule_id": rule_id,
                "applicable_sensor": col,
                "severity": severity,
                "status": "unknown",
                "explanation": f"Preceding reference observation unavailable to evaluate rate of change for '{rule_id}'.",
                "observed_value": float(val),
                "expected_parameters": {"max_rate_of_change": max_roc},
            }

        # Rule Type C: Cross-sensor difference
        elif rule_type == "cross_sensor_difference":
            paired = c.get("paired_sensor")
            max_diff = c.get("max_difference")
            if not paired or max_diff is None or pos is None or paired not in df.columns:
                return {
                    "rule_id": rule_id,
                    "applicable_sensor": col,
                    "severity": severity,
                    "status": "unknown",
                    "explanation": f"Paired sensor '{paired}' unavailable to evaluate constraint '{rule_id}'.",
                    "observed_value": float(val),
                    "expected_parameters": {"paired_sensor": paired, "max_difference": max_diff},
                }

            paired_val = pd.to_numeric(df.iloc[pos][paired], errors="coerce")
            if not np.isfinite(paired_val):
                return {
                    "rule_id": rule_id,
                    "applicable_sensor": col,
                    "severity": severity,
                    "status": "unknown",
                    "explanation": f"Paired sensor '{paired}' reading is missing/invalid at row {pos}.",
                    "observed_value": float(val),
                    "expected_parameters": {"paired_sensor": paired, "max_difference": max_diff},
                }

            diff = abs(val - paired_val)
            if diff <= max_diff:
                return {
                    "rule_id": rule_id,
                    "applicable_sensor": col,
                    "severity": severity,
                    "status": "pass",
                    "explanation": f"Difference |{val:.2f} - {paired_val:.2f}| = {diff:.2f} <= allowable {max_diff:.2f}.",
                    "observed_value": float(diff),
                    "expected_parameters": {"paired_sensor": paired, "max_difference": max_diff},
                }
            else:
                status = "fail" if severity == "hard" else "warning"
                return {
                    "rule_id": rule_id,
                    "applicable_sensor": col,
                    "severity": severity,
                    "status": status,
                    "explanation": f"Cross-sensor difference {diff:.2f} exceeds allowable max {max_diff:.2f}.",
                    "observed_value": float(diff),
                    "expected_parameters": {"paired_sensor": paired, "max_difference": max_diff},
                }

        # Unknown rule type
        return {
            "rule_id": rule_id,
            "applicable_sensor": col,
            "severity": severity,
            "status": "unknown",
            "explanation": f"Unsupported rule type '{rule_type}' for constraint '{rule_id}'.",
            "observed_value": float(val),
            "expected_parameters": c,
        }
