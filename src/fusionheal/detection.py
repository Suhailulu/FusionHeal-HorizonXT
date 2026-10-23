from __future__ import annotations
from typing import Any
import pandas as pd

from .detectors import (
    BaseDetector,
    MissingValueDetector,
    RangeViolationDetector,
    RobustOutlierDetector,
    SpikeDetector,
    StuckSensorDetector,
    DriftDetector,
    TimestampIntegrityDetector,
    CrossSensorConsistencyDetector,
)


class FaultDetectionEngine:
    """Orchestrates modular industrial time-series fault detectors.

    Applies detection rules in a defined, evidence-guided sequence to prevent
    duplicate issue reports for the same fault event on a given sensor reading.
    """

    def __init__(self, detectors: list[BaseDetector] | None = None) -> None:
        if detectors is not None:
            self.detectors = detectors
        else:
            self.detectors = [
                MissingValueDetector(),
                RangeViolationDetector(),
                SpikeDetector(),
                RobustOutlierDetector(),
                StuckSensorDetector(),
                DriftDetector(),
                TimestampIntegrityDetector(),
                CrossSensorConsistencyDetector(),
            ]

    def run(
        self,
        df: pd.DataFrame,
        sensor_columns: list[str],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run all configured detectors and return deduplicated issue records."""
        # Ensure source dataframe is never modified during detection
        df_copy = df.copy()

        # Track (sensor_id, row_position, fault_category) to avoid duplicates
        reported_events: set[tuple[str, int, str]] = set()
        all_issues: list[dict[str, Any]] = []

        # Run timestamp integrity check first
        ts_detector = next((d for d in self.detectors if isinstance(d, TimestampIntegrityDetector)), None)
        if ts_detector:
            ts_issues = ts_detector.detect(df_copy, sensor_columns, config)
            all_issues.extend(ts_issues)

        # Run missing value detection
        missing_detector = next((d for d in self.detectors if isinstance(d, MissingValueDetector)), None)
        missing_positions: set[tuple[str, int]] = set()
        if missing_detector:
            missing_issues = missing_detector.detect(df_copy, sensor_columns, config)
            for iss in missing_issues:
                pos = iss.get("row_position")
                sensor = iss.get("sensor_id")
                if pos is not None and sensor:
                    missing_positions.add((sensor, pos))
                all_issues.append(iss)

        # Run range violation check
        range_detector = next((d for d in self.detectors if isinstance(d, RangeViolationDetector)), None)
        range_positions: set[tuple[str, int]] = set()
        if range_detector:
            range_issues = range_detector.detect(df_copy, sensor_columns, config)
            for iss in range_issues:
                pos = iss.get("row_position")
                sensor = iss.get("sensor_id")
                if pos is not None and sensor:
                    if (sensor, pos) in missing_positions:
                        continue
                    range_positions.add((sensor, pos))
                all_issues.append(iss)

        # Run spike detector
        spike_detector = next((d for d in self.detectors if isinstance(d, SpikeDetector)), None)
        spike_positions: set[tuple[str, int]] = set()
        if spike_detector:
            spike_issues = spike_detector.detect(df_copy, sensor_columns, config)
            for iss in spike_issues:
                pos = iss.get("row_position")
                sensor = iss.get("sensor_id")
                if pos is not None and sensor:
                    if (sensor, pos) in missing_positions or (sensor, pos) in range_positions:
                        continue
                    spike_positions.add((sensor, pos))
                all_issues.append(iss)

        # Run robust outlier detector
        outlier_detector = next((d for d in self.detectors if isinstance(d, RobustOutlierDetector)), None)
        if outlier_detector:
            outlier_issues = outlier_detector.detect(df_copy, sensor_columns, config)
            for iss in outlier_issues:
                pos = iss.get("row_position")
                sensor = iss.get("sensor_id")
                if pos is not None and sensor:
                    # Avoid duplicate reporting if already flagged as missing, range violation, or specific spike
                    if (sensor, pos) in missing_positions or (sensor, pos) in range_positions or (sensor, pos) in spike_positions:
                        continue
                all_issues.append(iss)

        # Run stuck sensor detector
        stuck_detector = next((d for d in self.detectors if isinstance(d, StuckSensorDetector)), None)
        if stuck_detector:
            stuck_issues = stuck_detector.detect(df_copy, sensor_columns, config)
            for iss in stuck_issues:
                pos = iss.get("row_position")
                sensor = iss.get("sensor_id")
                if pos is not None and sensor and (sensor, pos) in missing_positions:
                    continue
                all_issues.append(iss)

        # Run drift detector
        drift_detector = next((d for d in self.detectors if isinstance(d, DriftDetector)), None)
        if drift_detector:
            drift_issues = drift_detector.detect(df_copy, sensor_columns, config)
            for iss in drift_issues:
                pos = iss.get("row_position")
                sensor = iss.get("sensor_id")
                if pos is not None and sensor and (sensor, pos) in missing_positions:
                    continue
                all_issues.append(iss)

        # Run cross-sensor consistency detector
        cross_detector = next((d for d in self.detectors if isinstance(d, CrossSensorConsistencyDetector)), None)
        if cross_detector:
            cross_issues = cross_detector.detect(df_copy, sensor_columns, config)
            all_issues.extend(cross_issues)

        # Final deduplication by unique issue_id
        seen_ids: set[str] = set()
        deduped_issues: list[dict[str, Any]] = []
        for issue in all_issues:
            iid = issue.get("issue_id", "")
            if iid not in seen_ids:
                seen_ids.add(iid)
                deduped_issues.append(issue)

        return deduped_issues


def detect_issues(
    df: pd.DataFrame,
    sensor_columns: list[str],
    sensor_limits: dict,
    rolling_window: int = 9,
    mad_threshold: float = 5.0,
    min_history_points: int = 5,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Return a list of detected issue records across configured detectors.

    Maintains backwards compatibility with starter interface while enabling
    the full multi-detector engine.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing sensor observations and record identifiers.
    sensor_columns : list[str]
        List of numeric sensor columns to analyze.
    sensor_limits : dict
        Mapping of sensor column names to dicts containing 'min' and/or 'max' demo limits.
    rolling_window : int, optional
        Window size for rolling statistics (default is 9).
    mad_threshold : float, optional
        Robust deviation threshold (default is 5.0).
    min_history_points : int, optional
        Minimum valid history points required before scoring (default is 5).
    config : dict, optional
        Full pipeline configuration dictionary.

    Returns
    -------
    list[dict[str, Any]]
        List of formatted fault records.
    """
    merged_config: dict[str, Any] = {}
    if config:
        merged_config.update(config)

    merged_config["sensor_limits"] = sensor_limits or merged_config.get("sensor_limits", {})

    detection_sub = merged_config.setdefault("detection", {})
    detection_sub.setdefault("rolling_window", rolling_window)
    detection_sub.setdefault("mad_threshold", mad_threshold)
    detection_sub.setdefault("min_history_points", min_history_points)

    for k, v in kwargs.items():
        detection_sub[k] = v

    engine = FaultDetectionEngine()
    return engine.run(df, sensor_columns, merged_config)
