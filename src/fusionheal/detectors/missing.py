from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .base import BaseDetector, format_issue


class MissingValueDetector(BaseDetector):
    """Detects null, NaN, and non-numeric sensor readings in industrial datasets."""

    name: str = "null_check"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for col in sensor_columns:
            if col not in df.columns:
                continue
            series = df[col]
            for pos, raw_val in enumerate(series):
                is_missing = False
                is_non_numeric = False
                numeric_val = None

                if pd.isna(raw_val):
                    is_missing = True
                else:
                    try:
                        numeric_val = float(raw_val)
                        if not np.isfinite(numeric_val):
                            is_missing = True
                    except (ValueError, TypeError):
                        is_missing = True
                        is_non_numeric = True

                if is_missing:
                    row = df.iloc[pos]
                    orig_val = None if (not is_non_numeric and (raw_val is None or pd.isna(raw_val))) else raw_val
                    reason = (
                        f"Non-numeric value '{raw_val}' could not be parsed."
                        if is_non_numeric
                        else "Value is missing or not numeric."
                    )
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-{col}-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id=col,
                            timestamp=str(row.get("timestamp_parsed", row.get("timestamp_original", ""))),
                            original_value=orig_val,
                            fault_type="missing",
                            detector=self.name,
                            detection_score=None,
                            reason=reason,
                            metadata={"raw_value": str(raw_val), "is_non_numeric": is_non_numeric},
                        )
                    )
        return issues
