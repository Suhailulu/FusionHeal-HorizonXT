from __future__ import annotations
import logging
from typing import Any
import numpy as np
import pandas as pd
from .base import BaseDetector, format_issue

logger = logging.getLogger(__name__)


class CrossSensorConsistencyDetector(BaseDetector):
    """Evaluates cross-sensor physical consistency only across explicitly validated engineering relationships.

    CRITICAL SAFETY PRINCIPLE:
    Does NOT infer or hallucinate arbitrary correlations between unverified sensors.
    If no validated physical relationship is configured, marks itself as UNAVAILABLE
    and yields no issues, avoiding spurious inter-sensor alarms.
    """

    name: str = "cross_sensor_consistency"

    def __init__(self) -> None:
        self.is_available: bool = False
        self.status_reason: str = "Uninitialized"

    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        detection_cfg = config.get("detection", {})
        cs_cfg = detection_cfg.get("cross_sensor", {})
        if isinstance(cs_cfg, bool) and not cs_cfg:
            self.is_available = False
            self.status_reason = "Detector explicitly disabled via configuration."
            return []

        rules = cs_cfg.get("rules", []) if isinstance(cs_cfg, dict) else config.get("cross_sensor_rules", [])

        # If no verified relationships are provided, do NOT fabricate correlation
        if not rules:
            self.is_available = False
            self.status_reason = (
                "UNAVAILABLE: No validated cross-sensor physical relationships configured. "
                "FusionHeal never assumes unverified correlations."
            )
            logger.info(self.status_reason)
            return []

        self.is_available = True
        self.status_reason = f"Active with {len(rules)} validated relationship rules."

        issues: list[dict[str, Any]] = []

        for rule in rules:
            rule_id = rule.get("rule_id", "unnamed_rule")
            sensor_a = rule.get("sensor_a")
            sensor_b = rule.get("sensor_b")
            rule_type = rule.get("type", "difference_bound")

            if not sensor_a or not sensor_b:
                continue
            if sensor_a not in df.columns or sensor_b not in df.columns:
                continue

            val_a = pd.to_numeric(df[sensor_a], errors="coerce").to_numpy(dtype=float)
            val_b = pd.to_numeric(df[sensor_b], errors="coerce").to_numpy(dtype=float)

            n = min(len(val_a), len(val_b))

            for pos in range(n):
                a = val_a[pos]
                b = val_b[pos]
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue

                violated = False
                detail = ""
                score = None

                if rule_type == "difference_bound":
                    max_diff = float(rule.get("max_difference", 10.0))
                    diff = abs(a - b)
                    if diff > max_diff:
                        violated = True
                        score = diff / max(max_diff, 1e-6)
                        detail = f"Difference |{sensor_a} ({a:.2f}) - {sensor_b} ({b:.2f})| = {diff:.2f} > max allowed {max_diff:.2f}."

                elif rule_type == "ratio_bound":
                    min_ratio = float(rule.get("min_ratio", 0.5))
                    max_ratio = float(rule.get("max_ratio", 2.0))
                    if abs(b) > 1e-6:
                        ratio = a / b
                        if ratio < min_ratio or ratio > max_ratio:
                            violated = True
                            score = ratio if ratio > max_ratio else (min_ratio / max(ratio, 1e-6))
                            detail = f"Ratio {sensor_a}/{sensor_b} = {ratio:.2f} outside valid envelope [{min_ratio}, {max_ratio}]."

                elif rule_type == "greater_than_or_equal":
                    # e.g., Inlet pressure must be >= Outlet pressure
                    margin = float(rule.get("margin", 0.0))
                    if a < b - margin:
                        violated = True
                        score = float(b - a)
                        detail = f"Expected {sensor_a} ({a:.2f}) >= {sensor_b} ({b:.2f}) - margin ({margin:.2f}), violated by {b - a:.2f}."

                if violated:
                    row = df.iloc[pos]
                    issues.append(
                        format_issue(
                            issue_id=f"ISS-CROSS-{rule_id}-{pos:08d}",
                            source_record_id=str(row.get("source_record_id", f"ROW-{pos:08d}")),
                            row_position=pos,
                            sensor_id=f"{sensor_a}:{sensor_b}",
                            timestamp=str(row.get("timestamp_parsed", row.get("timestamp_original", ""))),
                            original_value={"sensor_a": float(a), "sensor_b": float(b)},
                            fault_type="cross_sensor_inconsistency",
                            detector=self.name,
                            detection_score=score,
                            reason=f"Validated relationship '{rule_id}' breached: {detail}",
                            metadata={
                                "rule_id": rule_id,
                                "rule_type": rule_type,
                                "sensor_a": sensor_a,
                                "sensor_b": sensor_b,
                                "value_a": float(a),
                                "value_b": float(b),
                            },
                        )
                    )
        return issues
