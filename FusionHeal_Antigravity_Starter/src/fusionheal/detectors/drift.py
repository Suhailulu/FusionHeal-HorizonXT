from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .base import BaseDetector, format_issue


class DriftDetector(BaseDetector):
    """Detects systematic, sustained calibration drift using Cumulative Sum (CUSUM) control charts.

    Avoids false positives from isolated noise or small operating variations by requiring
    consistent accumulation past a slack parameter (k) and decision threshold (h).
    """

    name: str = "cusum_drift"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        detection_cfg = config.get("detection", {})
        drift_cfg = detection_cfg.get("drift", {})
        if isinstance(drift_cfg, bool) and not drift_cfg:
            return []
        if isinstance(drift_cfg, dict) and not drift_cfg.get("enabled", True):
            return []

        drift_threshold = float(drift_cfg.get("drift_threshold", 4.0)) if isinstance(drift_cfg, dict) else 4.0
        slack = float(drift_cfg.get("slack", 0.5)) if isinstance(drift_cfg, dict) else 0.5
        min_baseline_points = int(drift_cfg.get("min_baseline_points", 5)) if isinstance(drift_cfg, dict) else 5
        min_drift_samples = int(drift_cfg.get("min_drift_samples", 3)) if isinstance(drift_cfg, dict) else 3
        min_scale = float(drift_cfg.get("min_scale", 1e-4)) if isinstance(drift_cfg, dict) else 1e-4

        issues: list[dict[str, Any]] = []

        for col in sensor_columns:
            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            n = len(values)
            if n < min_baseline_points + min_drift_samples:
                continue

            # Establish initial baseline from initial finite observations
            baseline_slice = values[:min_baseline_points]
            valid_baseline = baseline_slice[np.isfinite(baseline_slice)]
            if len(valid_baseline) < 3:
                continue

            mu_0 = float(np.median(valid_baseline))
            mad_0 = float(np.median(np.abs(valid_baseline - mu_0)))
            sigma_0 = max(1.4826 * mad_0, min_scale)

            s_pos = 0.0
            s_neg = 0.0
            sustained_count = 0

            for pos in range(min_baseline_points, n):
                val = values[pos]
                if not np.isfinite(val):
                    continue

                # Standardized deviation
                z = (val - mu_0) / sigma_0

                # Two-sided tabular CUSUM recursion
                s_pos = max(0.0, s_pos + z - slack)
                s_neg = max(0.0, s_neg - z - slack)

                max_cusum = max(s_pos, s_neg)
                if max_cusum >= drift_threshold:
                    sustained_count += 1
                else:
                    sustained_count = max(0, sustained_count - 1)

                if sustained_count >= min_drift_samples:
                    direction = "positive (upward)" if s_pos > s_neg else "negative (downward)"
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-DRIFT-{col}-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id=col,
                            timestamp=str(row.get("timestamp_parsed", row.get("timestamp_original", ""))),
                            original_value=float(val),
                            fault_type="drift",
                            detector=self.name,
                            detection_score=float(max_cusum),
                            reason=(
                                f"Sustained {direction} drift detected: CUSUM statistic {max_cusum:.2f} "
                                f"exceeded control threshold {drift_threshold:.2f} (baseline mean: {mu_0:.2f})."
                            ),
                            metadata={
                                "cusum_statistic": float(max_cusum),
                                "control_threshold": float(drift_threshold),
                                "drift_direction": direction,
                                "baseline_reference_mean": float(mu_0),
                                "baseline_scale": float(sigma_0),
                                "sustained_samples": sustained_count,
                            },
                        )
                    )
        return issues
