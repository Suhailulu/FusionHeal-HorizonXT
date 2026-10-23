from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .base import BaseDetector, format_issue


class SpikeDetector(BaseDetector):
    """Detects isolated abrupt transient impulses using first differences and local context.

    Distinguishes isolated spikes (jump followed by immediate reversion) from
    legitimate operating condition changes (step transitions that persist at a new level).
    """

    name: str = "first_difference_spike"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        detection_cfg = config.get("detection", {})
        spike_cfg = detection_cfg.get("spike", {})
        if isinstance(spike_cfg, bool) and not spike_cfg:
            return []
        if isinstance(spike_cfg, dict) and not spike_cfg.get("enabled", True):
            return []

        spike_threshold = float(spike_cfg.get("spike_threshold", 4.0)) if isinstance(spike_cfg, dict) else 4.0
        recovery_tolerance = float(spike_cfg.get("recovery_tolerance", 2.0)) if isinstance(spike_cfg, dict) else 2.0
        window_size = int(spike_cfg.get("window_size", 9)) if isinstance(spike_cfg, dict) else 9
        min_scale = float(spike_cfg.get("min_scale", 1e-4)) if isinstance(spike_cfg, dict) else 1e-4

        issues: list[dict[str, Any]] = []

        for col in sensor_columns:
            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            n = len(values)
            if n < 3:
                continue

            # Compute consecutive differences: diffs[i] = values[i] - values[i - 1]
            diffs = np.diff(values)

            # Robust scale of normal point-to-point fluctuations
            finite_diffs = np.abs(diffs[np.isfinite(diffs)])
            if len(finite_diffs) == 0:
                continue

            for pos in range(1, n - 1):
                val_prev = values[pos - 1]
                val_curr = values[pos]
                val_next = values[pos + 1]

                if not (np.isfinite(val_prev) and np.isfinite(val_curr) and np.isfinite(val_next)):
                    continue

                d1 = val_curr - val_prev  # Jump into current point
                d2 = val_next - val_curr  # Step from current point to next

                # To be an isolated spike, the value must jump and then immediately revert
                # (d1 and d2 must have opposite signs, so d1 * d2 < 0)
                if d1 * d2 >= 0:
                    # Same direction or flat continuation = step transition or ramp, NOT an isolated spike
                    continue

                # Local noise scale estimation within neighborhood
                w_left = max(0, pos - window_size // 2)
                w_right = min(len(diffs), pos + window_size // 2 + 1)
                local_diffs = np.abs(diffs[w_left:w_right])
                valid_local_diffs = local_diffs[np.isfinite(local_diffs)]
                if len(valid_local_diffs) < 3:
                    local_scale = max(float(np.median(finite_diffs)), min_scale)
                else:
                    local_scale = max(float(np.median(valid_local_diffs)), min_scale)

                jump_mag = min(abs(d1), abs(d2))
                reversion_gap = abs(val_next - val_prev)

                # Check if jump magnitude significantly exceeds local step fluctuations
                score = jump_mag / local_scale

                # Check recovery: next value must be close to previous baseline (not a step change)
                is_recovery = reversion_gap <= max(recovery_tolerance * local_scale, 0.5 * jump_mag)

                if score >= spike_threshold and is_recovery:
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-SPIKE-{col}-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id=col,
                            timestamp=str(row.get("timestamp_parsed", row.get("timestamp_original", ""))),
                            original_value=float(val_curr),
                            fault_type="spike",
                            detector=self.name,
                            detection_score=float(score),
                            reason=(
                                f"Isolated transient spike detected: jumped {d1:+.2f} then reverted {d2:+.2f} "
                                f"(spike score: {score:.2f} relative to local scale {local_scale:.3f})."
                            ),
                            metadata={
                                "jump_magnitude": float(jump_mag),
                                "local_scale": float(local_scale),
                                "preceding_difference": float(d1),
                                "following_difference": float(d2),
                                "baseline_recovery_gap": float(reversion_gap),
                            },
                        )
                    )
        return issues
