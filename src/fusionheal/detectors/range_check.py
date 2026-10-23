from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .base import BaseDetector, format_issue


class RangeViolationDetector(BaseDetector):
    """Detects values that breach sensor-specific minimum and maximum operational limits."""

    name: str = "range_check"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        sensor_limits = config.get("sensor_limits", {})
        issues: list[dict[str, Any]] = []

        for col in sensor_columns:
            if col not in df.columns:
                continue
            limit = sensor_limits.get(col, {})
            if not limit or ("min" not in limit and "max" not in limit):
                continue

            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            for pos, val in enumerate(values):
                if not np.isfinite(val):
                    continue

                violated_rule: str | None = None
                reason: str = ""
                min_val = limit.get("min")
                max_val = limit.get("max")

                if min_val is not None and val < min_val:
                    violated_rule = "min"
                    reason = (
                        f"Value {val} is below configured demo minimum limit of {min_val} "
                        f"(rule: min_limit). Note: configured limits are demonstration settings."
                    )
                elif max_val is not None and val > max_val:
                    violated_rule = "max"
                    reason = (
                        f"Value {val} is above configured demo maximum limit of {max_val} "
                        f"(rule: max_limit). Note: configured limits are demonstration settings."
                    )

                if violated_rule is not None:
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-RANGE-{col}-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id=col,
                            timestamp=str(row.get("timestamp_parsed", row.get("timestamp_original", ""))),
                            original_value=float(val),
                            fault_type="range_violation",
                            detector=self.name,
                            detection_score=None,
                            reason=reason,
                            metadata={
                                "violated_limit": violated_rule,
                                "configured_min": min_val,
                                "configured_max": max_val,
                                "demonstration_disclaimer": "Demo setting; not a certified safety threshold.",
                            },
                        )
                    )
        return issues
