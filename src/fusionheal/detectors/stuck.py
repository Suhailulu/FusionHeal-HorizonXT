from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .base import BaseDetector, format_issue


class StuckSensorDetector(BaseDetector):
    """Detects stuck / frozen sensor readings (flatline failure).

    Accounts for sensor resolution, ADC quantization noise floors, and allows
    exempting sensors that legitimately operate at constant values.
    """

    name: str = "stuck_sensor"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        detection_cfg = config.get("detection", {})
        stuck_cfg = detection_cfg.get("stuck", {})
        if isinstance(stuck_cfg, bool) and not stuck_cfg:
            return []
        if isinstance(stuck_cfg, dict) and not stuck_cfg.get("enabled", True):
            return []

        min_stuck_samples = int(stuck_cfg.get("min_stuck_samples", 5)) if isinstance(stuck_cfg, dict) else 5
        tolerance = float(stuck_cfg.get("tolerance", 1e-4)) if isinstance(stuck_cfg, dict) else 1e-4
        exempt_sensors = set(stuck_cfg.get("exempt_sensors", [])) if isinstance(stuck_cfg, dict) else set()

        issues: list[dict[str, Any]] = []

        for col in sensor_columns:
            if col not in df.columns or col in exempt_sensors:
                continue

            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            n = len(values)
            if n < min_stuck_samples:
                continue

            # Track consecutive runs where |x[i] - x[i-1]| <= tolerance
            run_length = 1
            stuck_start = 0

            for pos in range(1, n):
                prev_val = values[pos - 1]
                curr_val = values[pos]

                if np.isfinite(prev_val) and np.isfinite(curr_val) and abs(curr_val - prev_val) <= tolerance:
                    run_length += 1
                else:
                    run_length = 1
                    stuck_start = pos

                # Once the run hits the threshold, flag current and preceding run samples
                if run_length >= min_stuck_samples:
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-STUCK-{col}-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id=col,
                            timestamp=str(row.get("timestamp_parsed", row.get("timestamp_original", ""))),
                            original_value=float(curr_val),
                            fault_type="stuck",
                            detector=self.name,
                            detection_score=None,
                            reason=(
                                f"Sensor reading remained constant at {curr_val:.4f} for {run_length} consecutive "
                                f"samples (threshold: {min_stuck_samples}, tolerance: {tolerance})."
                            ),
                            metadata={
                                "consecutive_samples": run_length,
                                "stuck_start_index": stuck_start,
                                "constant_value": float(curr_val),
                                "configured_tolerance": tolerance,
                            },
                        )
                    )
        return issues
