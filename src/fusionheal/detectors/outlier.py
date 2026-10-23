from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .base import BaseDetector, format_issue


class RobustOutlierDetector(BaseDetector):
    """Detects statistical outliers using rolling Median Absolute Deviation (MAD).

    Distinguishes robust deviation score from calibrated probability and supports
    configurable windowing strategies (centered for batch analysis vs past-only
    to prevent future-data leakage in streaming/online pipelines).
    """

    name: str = "rolling_mad"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        detection_cfg = config.get("detection", {})
        rolling_window = int(detection_cfg.get("rolling_window", config.get("rolling_window", 9)))
        mad_threshold = float(detection_cfg.get("mad_threshold", config.get("mad_threshold", 5.0)))
        min_history_points = int(detection_cfg.get("min_history_points", config.get("min_history_points", 5)))
        window_strategy = str(detection_cfg.get("window_strategy", config.get("window_strategy", "centered"))).lower()
        min_scale = float(detection_cfg.get("min_scale", config.get("min_scale", 1e-4)))

        issues: list[dict[str, Any]] = []

        for col in sensor_columns:
            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            n = len(values)

            for pos, val in enumerate(values):
                if not np.isfinite(val):
                    continue

                if window_strategy == "past_only":
                    # Past-only causal window: uses only preceding points [pos - rolling_window, pos]
                    # to strictly avoid future leakage in streaming / online scenarios.
                    left = max(0, pos - rolling_window)
                    right = pos + 1
                else:
                    # Centered window: symmetrical around pos for retrospective offline batch analysis
                    left = max(0, pos - rolling_window // 2)
                    right = min(n, pos + rolling_window // 2 + 1)

                raw_window = values[left:right]
                valid_window = raw_window[np.isfinite(raw_window)]

                # Handle short datasets and insufficient valid history points safely
                if len(valid_window) < min_history_points:
                    continue

                median = float(np.median(valid_window))
                mad = float(np.median(np.abs(valid_window - median)))

                # Zero MAD handling: prevents divide-by-zero on flat/constant windows
                # Uses 1.4826 standard normal consistency factor, floored by min_scale & eps
                scale = max(1.4826 * mad, min_scale, np.finfo(float).eps * max(1.0, abs(median)))
                diff = abs(float(val) - median)

                # If the difference is below min_scale, reading is effectively on the median
                if diff < min_scale:
                    continue

                score = diff / scale

                if score >= mad_threshold:
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-OUTLIER-{col}-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id=col,
                            timestamp=str(row.get("timestamp_parsed", row.get("timestamp_original", ""))),
                            original_value=float(val),
                            fault_type="outlier",
                            detector=self.name,
                            detection_score=float(score),
                            reason=(
                                f"Robust deviation score {score:.2f} exceeded threshold {mad_threshold:.2f} "
                                f"(uncalibrated MAD distance, not a calibrated probability)."
                            ),
                            metadata={
                                "local_median": median,
                                "local_mad": mad,
                                "window_strategy": window_strategy,
                                "effective_scale": scale,
                                "score_interpretation": "Heuristic MAD-scale distance ratio, not a Bayesian/logistic probability.",
                            },
                        )
                    )
        return issues
