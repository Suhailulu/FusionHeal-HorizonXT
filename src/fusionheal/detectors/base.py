from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseDetector(ABC):
    """Abstract base class for all FusionHeal fault detectors."""

    name: str = "base_detector"

    @abstractmethod
    def detect(self, df: pd.DataFrame, sensor_columns: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
        """Run detection on the provided dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe (must contain source_record_id and timestamp columns).
        sensor_columns : list[str]
            List of numeric sensor column names to inspect.
        config : dict[str, Any]
            Full or detector-specific configuration dictionary.

        Returns
        -------
        list[dict[str, Any]]
            List of detected issue records.
        """
        raise NotImplementedError


def format_issue(
    issue_id: str,
    source_record_id: str,
    row_position: int,
    sensor_id: str,
    timestamp: str,
    original_value: Any,
    fault_type: str,
    detector: str,
    detection_score: float | None,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to construct a standard, schema-compliant fault record."""
    return {
        "issue_id": issue_id,
        "source_record_id": str(source_record_id),
        "row_position": int(row_position) if row_position is not None else None,
        "sensor_id": str(sensor_id),
        "timestamp": str(timestamp),
        "original_value": original_value,
        "fault_type": str(fault_type),
        "detector": str(detector),
        "detection_score": float(detection_score) if detection_score is not None else None,
        "reason": str(reason),
        "metadata": metadata or {},
    }
