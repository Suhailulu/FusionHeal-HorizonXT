from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


class ReconstructionEngine:
    """Generates evidence-guided candidate replacement values for industrial sensor anomalies.

    Generates multiple candidate values when justified (linear interpolation, local median,
    and cross-sensor estimation) and preserves diagnostic assumptions and limitations without
    modifying source measurements in place.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def generate_candidates(
        self,
        df: pd.DataFrame,
        issue: dict[str, Any],
        evidence_list: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate all viable candidate replacement values for a detected issue."""
        pos = issue.get("row_position")
        col = issue.get("sensor_id")
        issue_id = issue.get("issue_id", "UNKNOWN_ISSUE")
        fault_type = issue.get("fault_type")
        original = issue.get("original_value")

        if pos is None or col not in df.columns:
            return []

        pos = int(pos)
        series = pd.to_numeric(df[col], errors="coerce").copy()
        n = len(series)
        candidates: list[dict[str, Any]] = []

        evidence_refs = [e.get("evidence_id") for e in (evidence_list or []) if e.get("evidence_id")]

        # Do not reconstruct structural or safety-critical issues automatically
        if fault_type in (
            "range_violation",
            "stuck",
            "drift",
            "timestamp_gap",
            "timestamp_duplicate",
            "timestamp_order",
            "timestamp_invalid",
            "cross_sensor_inconsistency",
        ):
            return []

        # 1. Method: Linear Interpolation (applicable to interior missing values and spikes)
        if fault_type in ("missing", "spike"):
            # Avoid extrapolation at endpoints
            if 0 < pos < n - 1:
                # Check preceding valid point and following valid point
                prev_indices = [i for i in range(pos - 1, -1, -1) if pd.notna(series.iloc[i])]
                next_indices = [i for i in range(pos + 1, n) if pd.notna(series.iloc[i])]

                if prev_indices and next_indices:
                    i_prev = prev_indices[0]
                    i_next = next_indices[0]
                    gap_size = i_next - i_prev

                    # Avoid interpolating across overly large gaps (> 5 samples)
                    max_gap = int(self.config.get("reconstruction", {}).get("max_interpolation_gap", 5))
                    if gap_size <= max_gap:
                        y0 = float(series.iloc[i_prev])
                        y1 = float(series.iloc[i_next])
                        # Linear interpolation: y = y0 + (y1 - y0) * (pos - i_prev) / (i_next - i_prev)
                        interpolated_val = y0 + (y1 - y0) * (pos - i_prev) / float(gap_size)

                        if np.isfinite(interpolated_val):
                            candidates.append({
                                "candidate_id": f"CAND-INTERP-{issue_id}",
                                "issue_id": issue_id,
                                "method": "linear_interpolation",
                                "original_value": original,
                                "candidate_value": float(round(interpolated_val, 4)),
                                "evidence_references": evidence_refs,
                                "assumptions": [
                                    "Physical process changes linearly across the unobserved interval.",
                                    "No unobserved step transitions or trips occurred during the gap.",
                                ],
                                "diagnostics": {
                                    "left_index": i_prev,
                                    "right_index": i_next,
                                    "gap_size_samples": gap_size,
                                    "left_value": y0,
                                    "right_value": y1,
                                },
                                "limitations": [
                                    "Unreliable during dynamic process load transitions or nonlinear dynamics.",
                                    "Cannot reconstruct multi-point systemic sensor dropouts.",
                                ],
                            })

        # 2. Method: Local Median (applicable to outliers, spikes, and missing values)
        if fault_type in ("outlier", "spike", "missing"):
            detection_cfg = self.config.get("detection", {})
            w_size = int(detection_cfg.get("rolling_window", 9))
            w_strategy = str(detection_cfg.get("window_strategy", "centered")).lower()

            if w_strategy == "past_only":
                left = max(0, pos - w_size)
                right = pos
            else:
                left = max(0, pos - w_size // 2)
                right = min(n, pos + w_size // 2 + 1)

            window_vals = series.iloc[left:right].to_numpy(dtype=float)
            # Remove current position and non-finite values
            valid_vals = [v for idx, v in enumerate(window_vals) if np.isfinite(v) and (left + idx) != pos]

            min_points = int(detection_cfg.get("min_history_points", 3))
            if len(valid_vals) >= min_points:
                med_val = float(np.median(valid_vals))
                if np.isfinite(med_val):
                    candidates.append({
                        "candidate_id": f"CAND-MEDIAN-{issue_id}",
                        "issue_id": issue_id,
                        "method": "local_median",
                        "original_value": original,
                        "candidate_value": float(round(med_val, 4)),
                        "evidence_references": evidence_refs,
                        "assumptions": [
                            "Local process neighborhood represents steady-state operating conditions.",
                            "Isolated anomaly does not reflect legitimate process setpoint movement.",
                        ],
                        "diagnostics": {
                            "valid_window_samples": len(valid_vals),
                            "window_strategy": w_strategy,
                            "window_range": [left, right],
                            "local_median": med_val,
                        },
                        "limitations": [
                            "Lags during persistent physical ramps or rapid thermal cycles.",
                        ],
                    })

        # 3. Method: Cross-Sensor Estimation (ONLY when a validated relationship is configured)
        cs_rules = (
            self.config.get("detection", {}).get("cross_sensor", {}).get("rules", [])
            or self.config.get("cross_sensor_rules", [])
        )
        for rule in cs_rules:
            sensor_a = rule.get("sensor_a")
            sensor_b = rule.get("sensor_b")
            paired = None
            if sensor_a == col and sensor_b in df.columns:
                paired = sensor_b
            elif sensor_b == col and sensor_a in df.columns:
                paired = sensor_a

            if paired is not None:
                paired_val = pd.to_numeric(df.iloc[pos][paired], errors="coerce")
                if np.isfinite(paired_val):
                    # Compute cross-sensor estimated value
                    rule_type = rule.get("type", "difference_bound")
                    offset = float(rule.get("nominal_offset", 0.0))
                    cross_val = float(paired_val + offset)

                    candidates.append({
                        "candidate_id": f"CAND-CROSS-{rule.get('rule_id', 'rel')}-{issue_id}",
                        "issue_id": issue_id,
                        "method": "cross_sensor_estimate",
                        "original_value": original,
                        "candidate_value": float(round(cross_val, 4)),
                        "evidence_references": evidence_refs,
                        "assumptions": [
                            f"Validated physical relationship with paired sensor '{paired}' is intact.",
                        ],
                        "diagnostics": {
                            "paired_sensor": paired,
                            "paired_value": float(paired_val),
                            "nominal_offset": offset,
                            "rule_type": rule_type,
                        },
                        "limitations": [
                            f"Vulnerable to common-mode failures or drift in paired sensor '{paired}'.",
                        ],
                    })

        return candidates


def generate_candidate(df: pd.DataFrame, issue: dict[str, Any]) -> dict[str, Any] | None:
    """Backwards-compatible interface returning the primary reconstruction candidate."""
    engine = ReconstructionEngine()
    candidates = engine.generate_candidates(df, issue)
    return candidates[0] if candidates else None
