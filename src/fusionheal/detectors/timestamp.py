from __future__ import annotations
from typing import Any
import pandas as pd
from .base import BaseDetector, format_issue


class TimestampIntegrityDetector(BaseDetector):
    """Detects irregular sampling intervals, temporal gaps, duplicates, and out-of-order timestamps.

    Reports data collection gaps without synthesizing or inventing missing timestamps or measurements.
    """

    name: str = "timestamp_integrity"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        detection_cfg = config.get("detection", {})
        ts_cfg = detection_cfg.get("timestamp", {})
        if isinstance(ts_cfg, bool) and not ts_cfg:
            return []
        if isinstance(ts_cfg, dict) and not ts_cfg.get("enabled", True):
            return []

        expected_interval = ts_cfg.get("expected_interval_seconds") if isinstance(ts_cfg, dict) else None
        gap_tolerance_multiplier = float(ts_cfg.get("gap_tolerance_multiplier", 1.5)) if isinstance(ts_cfg, dict) else 1.5
        check_duplicates = bool(ts_cfg.get("check_duplicates", True)) if isinstance(ts_cfg, dict) else True
        check_order = bool(ts_cfg.get("check_order", True)) if isinstance(ts_cfg, dict) else True

        issues: list[dict[str, Any]] = []

        ts_col = "timestamp_parsed" if "timestamp_parsed" in df.columns else config.get("timestamp_column", "timestamp")
        if ts_col not in df.columns:
            return []

        ts_series = pd.to_datetime(df[ts_col], errors="coerce")
        n = len(df)
        if n == 0:
            return []

        # 1. Check for unparseable / invalid timestamps
        for pos, val in enumerate(ts_series):
            if pd.isna(val):
                row = df.iloc[pos]
                orig_ts = row.get("timestamp_original", row.get(ts_col, ""))
                issues.append(
                    format_issue(
                        issue_id=f"ISS-TIMESTAMP-INVALID-{pos:08d}",
                        source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                        row_position=pos,
                        sensor_id="timestamp",
                        timestamp=str(orig_ts),
                        original_value=orig_ts,
                        fault_type="timestamp_invalid",
                        detector=self.name,
                        detection_score=None,
                        reason=f"Timestamp value '{orig_ts}' is invalid or could not be parsed.",
                        metadata={"raw_timestamp": str(orig_ts)},
                    )
                )

        # 2. Sequential interval, duplicate, and ordering checks
        prev_valid_ts = None
        prev_valid_pos = None

        # Auto-infer interval if not explicitly configured and enough timestamps exist
        if expected_interval is None:
            valid_diffs = ts_series.dropna().diff().dt.total_seconds()
            positive_diffs = valid_diffs[valid_diffs > 0]
            if len(positive_diffs) >= 3:
                expected_interval = float(positive_diffs.median())

        for pos in range(n):
            curr_ts = ts_series.iloc[pos]
            if pd.isna(curr_ts):
                continue

            if prev_valid_ts is not None:
                delta_sec = (curr_ts - prev_valid_ts).total_seconds()

                # Duplicate check
                if check_duplicates and delta_sec == 0:
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-TIMESTAMP-DUP-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id="timestamp",
                            timestamp=str(curr_ts),
                            original_value=str(curr_ts),
                            fault_type="timestamp_duplicate",
                            detector=self.name,
                            detection_score=None,
                            reason=f"Duplicate timestamp '{curr_ts}' identical to prior record at row {prev_valid_pos}.",
                            metadata={"prior_row_position": prev_valid_pos, "duplicate_timestamp": str(curr_ts)},
                        )
                    )

                # Chronological ordering check
                elif check_order and delta_sec < 0:
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-TIMESTAMP-ORDER-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id="timestamp",
                            timestamp=str(curr_ts),
                            original_value=str(curr_ts),
                            fault_type="timestamp_order",
                            detector=self.name,
                            detection_score=None,
                            reason=(
                                f"Timestamp '{curr_ts}' is out of order; occurs {-delta_sec:.1f}s earlier than "
                                f"prior timestamp '{prev_valid_ts}' at row {prev_valid_pos}."
                            ),
                            metadata={
                                "prior_timestamp": str(prev_valid_ts),
                                "negative_delta_seconds": float(delta_sec),
                                "prior_row_position": prev_valid_pos,
                            },
                        )
                    )

                # Sampling interval gap check
                elif expected_interval is not None and expected_interval > 0:
                    max_allowed_delta = expected_interval * gap_tolerance_multiplier
                    if delta_sec > max_allowed_delta:
                        row = df.iloc[pos]
                        issues.append(
                            format_issue(
                                issue_id=f"ISS-TIMESTAMP-GAP-{pos:08d}",
                                source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                                row_position=pos,
                                sensor_id="timestamp",
                                timestamp=str(curr_ts),
                                original_value=str(curr_ts),
                                fault_type="timestamp_gap",
                                detector=self.name,
                                detection_score=float(delta_sec / expected_interval),
                                reason=(
                                    f"Sampling gap of {delta_sec:.1f}s exceeded allowed threshold "
                                    f"({max_allowed_delta:.1f}s, based on expected interval of {expected_interval:.1f}s)."
                                ),
                                metadata={
                                    "expected_interval_seconds": float(expected_interval),
                                    "observed_gap_seconds": float(delta_sec),
                                    "prior_timestamp": str(prev_valid_ts),
                                    "prior_row_position": prev_valid_pos,
                                    "synthesized_data": False,
                                },
                            )
                        )

            prev_valid_ts = curr_ts
            prev_valid_pos = pos

        return issues
