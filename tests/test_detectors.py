from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from fusionheal.detection import FaultDetectionEngine, detect_issues
from fusionheal.detectors import (
    MissingValueDetector,
    RangeViolationDetector,
    RobustOutlierDetector,
    SpikeDetector,
    StuckSensorDetector,
    DriftDetector,
    TimestampIntegrityDetector,
    CrossSensorConsistencyDetector,
)


def create_base_df(n: int = 15) -> pd.DataFrame:
    """Helper to generate a clean baseline synthetic dataframe."""
    return pd.DataFrame({
        "source_record_id": [f"ROW-{i:08d}" for i in range(n)],
        "timestamp_original": [f"2026-10-23 10:{i:02d}:00" for i in range(n)],
        "timestamp_parsed": pd.date_range("2026-10-23 10:00:00", periods=n, freq="min"),
        "temperature": [50.0 + (i % 2) * 0.2 for i in range(n)],
        "pressure": [10.0 + (i % 3) * 0.1 for i in range(n)],
    })


# 1. Missing and non-numeric detector tests
def test_missing_and_non_numeric_values():
    df = create_base_df(10)
    df["temperature"] = df["temperature"].astype(object)
    df.loc[2, "temperature"] = None
    df.loc[5, "temperature"] = np.nan
    df.loc[7, "temperature"] = "INVALID_STR"

    detector = MissingValueDetector()
    issues = detector.detect(df, ["temperature"], {})

    assert len(issues) == 3
    positions = {i["row_position"] for i in issues}
    assert positions == {2, 5, 7}

    # Verify preservation of original input value and source record ID
    str_issue = next(i for i in issues if i["row_position"] == 7)
    assert str_issue["original_value"] == "INVALID_STR"
    assert str_issue["source_record_id"] == "ROW-00000007"
    assert str_issue["fault_type"] == "missing"
    assert str_issue["metadata"]["is_non_numeric"] is True


# 2. Constant-valued sensors and zero MAD
def test_constant_valued_sensors_and_zero_mad():
    df = create_base_df(12)
    # Perfectly flatline constant values
    df["temperature"] = [42.0] * 12

    detector = RobustOutlierDetector()
    issues = detector.detect(df, ["temperature"], {"detection": {"rolling_window": 7, "min_scale": 1e-4}})
    # No false alarms should be triggered when MAD is 0 and readings are on median
    assert len(issues) == 0

    # Inject one isolated anomaly into the flat series
    df.loc[6, "temperature"] = 99.0
    issues_with_outlier = detector.detect(df, ["temperature"], {"detection": {"rolling_window": 7, "min_scale": 1e-4}})
    assert len(issues_with_outlier) == 1
    assert issues_with_outlier[0]["row_position"] == 6
    assert issues_with_outlier[0]["fault_type"] == "outlier"


# 3. Short datasets and insufficient history
def test_short_datasets_and_insufficient_history():
    df = create_base_df(3)  # Only 3 points, min_history_points is 5
    df.loc[1, "temperature"] = 1000.0

    detector = RobustOutlierDetector()
    # Should safely skip without throwing error
    issues = detector.detect(df, ["temperature"], {"detection": {"min_history_points": 5}})
    assert len(issues) == 0


# 4. Spike detector vs legitimate process transitions
def test_spike_detector_and_legitimate_transitions():
    df = create_base_df(20)
    # Base is around 50.0
    # Inject isolated single-point spike at row 5: jumps to 120 and reverts to 50.2
    df.loc[5, "temperature"] = 120.0

    # Inject legitimate step change starting at row 12: jumps to 85.0 and stays at 85.0
    for r in range(12, 20):
        df.loc[r, "temperature"] = 85.0 + (r % 2) * 0.1

    detector = SpikeDetector()
    issues = detector.detect(df, ["temperature"], {"detection": {"spike": {"spike_threshold": 4.0}}})

    # The spike at row 5 must be detected
    spike_positions = [i["row_position"] for i in issues]
    assert 5 in spike_positions

    # The step transition starting at row 12 must NOT be flagged as a spike
    for r in range(12, 20):
        assert r not in spike_positions


# 5. Stuck-sensor detector
def test_stuck_sensor_detector():
    df = create_base_df(15)
    # Rows 4 to 10 are completely flatlined at 60.1234
    for r in range(4, 11):
        df.loc[r, "temperature"] = 60.1234

    detector = StuckSensorDetector()
    issues = detector.detect(
        df, ["temperature"],
        {"detection": {"stuck": {"min_stuck_samples": 5, "tolerance": 1e-4}}}
    )
    assert len(issues) >= 1
    assert all(i["fault_type"] == "stuck" for i in issues)
    assert issues[0]["metadata"]["consecutive_samples"] >= 5


def test_stuck_sensor_exemption():
    df = create_base_df(15)
    df["temperature"] = 25.0  # Constant
    detector = StuckSensorDetector()
    # Exempt temperature
    issues = detector.detect(
        df, ["temperature"],
        {"detection": {"stuck": {"min_stuck_samples": 5, "exempt_sensors": ["temperature"]}}}
    )
    assert len(issues) == 0


# 6. Drift detector (CUSUM)
def test_drift_detector_cusum():
    df = create_base_df(30)
    # Baseline for first 10 samples is ~50.0
    for i in range(10):
        df.loc[i, "temperature"] = 50.0 + (i % 2) * 0.05
    # Then gradual sustained upward drift
    for i in range(10, 30):
        df.loc[i, "temperature"] = 50.0 + (i - 9) * 0.5

    detector = DriftDetector()
    issues = detector.detect(
        df, ["temperature"],
        {"detection": {"drift": {"drift_threshold": 4.0, "slack": 0.5, "min_baseline_points": 8, "min_drift_samples": 3}}}
    )
    assert len(issues) > 0
    assert issues[0]["fault_type"] == "drift"
    assert "upward" in issues[0]["metadata"]["drift_direction"]


# 7. Range-violation detector with rule identification
def test_range_violation_rules():
    df = create_base_df(8)
    df.loc[2, "temperature"] = -10.0  # Below min 0
    df.loc[5, "temperature"] = 200.0  # Above max 150

    detector = RangeViolationDetector()
    config = {"sensor_limits": {"temperature": {"min": 0, "max": 150}}}
    issues = detector.detect(df, ["temperature"], config)

    assert len(issues) == 2
    min_issue = next(i for i in issues if i["row_position"] == 2)
    max_issue = next(i for i in issues if i["row_position"] == 5)

    assert min_issue["metadata"]["violated_limit"] == "min"
    assert "below configured demo minimum limit" in min_issue["reason"]
    assert max_issue["metadata"]["violated_limit"] == "max"
    assert "above configured demo maximum limit" in max_issue["reason"]


# 8. Timestamp integrity detector
def test_timestamp_irregularities():
    df = create_base_df(6)
    # Inject duplicate timestamp at row 2
    df.loc[2, "timestamp_parsed"] = df.loc[1, "timestamp_parsed"]
    # Inject out-of-order timestamp at row 4
    df.loc[4, "timestamp_parsed"] = pd.Timestamp("2026-10-23 09:00:00")

    detector = TimestampIntegrityDetector()
    issues = detector.detect(
        df, ["temperature"],
        {"detection": {"timestamp": {"check_duplicates": True, "check_order": True}}}
    )

    fault_types = [i["fault_type"] for i in issues]
    assert "timestamp_duplicate" in fault_types
    assert "timestamp_order" in fault_types


def test_timestamp_sampling_gap():
    df = create_base_df(5)
    # Normal interval is 60s; inject a 10-minute gap at row 3
    df.loc[3, "timestamp_parsed"] = pd.Timestamp("2026-10-23 10:15:00")
    df.loc[4, "timestamp_parsed"] = pd.Timestamp("2026-10-23 10:16:00")

    detector = TimestampIntegrityDetector()
    issues = detector.detect(
        df, ["temperature"],
        {"detection": {"timestamp": {"expected_interval_seconds": 60.0, "gap_tolerance_multiplier": 1.5}}}
    )
    gap_issues = [i for i in issues if i["fault_type"] == "timestamp_gap"]
    assert len(gap_issues) == 1
    assert gap_issues[0]["metadata"]["observed_gap_seconds"] > 60.0
    assert gap_issues[0]["metadata"]["synthesized_data"] is False


# 9. Cross-sensor consistency detector
def test_cross_sensor_unconfigured_is_unavailable():
    df = create_base_df(10)
    detector = CrossSensorConsistencyDetector()
    # No validated relationships configured
    issues = detector.detect(df, ["temperature", "pressure"], {"detection": {"cross_sensor": {"rules": []}}})
    assert len(issues) == 0
    assert detector.is_available is False
    assert "UNAVAILABLE" in detector.status_reason


def test_cross_sensor_validated_rule():
    df = create_base_df(10)
    # Configure physical rule: |temperature - pressure| <= 45.0
    # Baseline temperature is ~50, pressure is ~10 (diff is ~40).
    # At row 4, spike pressure to 80 (diff becomes |50 - 80| = 30, still within 45).
    # At row 6, spike temperature to 110 (diff becomes |110 - 10| = 100 > 45).
    df.loc[6, "temperature"] = 110.0

    detector = CrossSensorConsistencyDetector()
    config = {
        "detection": {
            "cross_sensor": {
                "rules": [
                    {
                        "rule_id": "temp_press_diff",
                        "sensor_a": "temperature",
                        "sensor_b": "pressure",
                        "type": "difference_bound",
                        "max_difference": 45.0,
                    }
                ]
            }
        }
    }
    issues = detector.detect(df, ["temperature", "pressure"], config)
    assert detector.is_available is True
    assert len(issues) == 1
    assert issues[0]["fault_type"] == "cross_sensor_inconsistency"
    assert issues[0]["row_position"] == 6


# 10. Multiple simultaneous sensor faults & data preservation
def test_multiple_simultaneous_sensor_faults_and_preservation():
    df = create_base_df(15)
    # Row 3 has faults across both temperature and pressure
    df.loc[3, "temperature"] = None  # Missing
    df.loc[3, "pressure"] = 999.0    # Range violation

    df_copy = df.copy(deep=True)

    config = {
        "sensor_limits": {"temperature": {"min": 0, "max": 100}, "pressure": {"min": 0, "max": 20}},
        "detection": {"rolling_window": 7, "min_history_points": 4},
    }

    issues = detect_issues(df, ["temperature", "pressure"], config["sensor_limits"], config=config)

    # Both faults must be identified
    row_3_issues = [i for i in issues if i.get("row_position") == 3]
    sensors = {i["sensor_id"] for i in row_3_issues}
    assert "temperature" in sensors
    assert "pressure" in sensors

    # Data preservation: input dataframe must not be altered
    pd.testing.assert_frame_equal(df, df_copy)


# 11. Repeatability test
def test_repeatable_results():
    df = create_base_df(15)
    df.loc[4, "temperature"] = 999.0
    df.loc[8, "temperature"] = None

    config = {"sensor_limits": {"temperature": {"min": 0, "max": 100}}}

    run_1 = detect_issues(df, ["temperature"], config["sensor_limits"], config=config)
    run_2 = detect_issues(df, ["temperature"], config["sensor_limits"], config=config)

    assert len(run_1) == len(run_2)
    assert [i["issue_id"] for i in run_1] == [i["issue_id"] for i in run_2]
    assert [i["fault_type"] for i in run_1] == [i["fault_type"] for i in run_2]


# 12. Streaming window strategy (no future leakage)
def test_streaming_window_strategy_no_future_leakage():
    df = create_base_df(20)
    # In past-only window, a massive future outlier at row 15 must NOT affect detection at row 10
    df.loc[15, "temperature"] = 500.0

    detector = RobustOutlierDetector()
    issues_past = detector.detect(
        df, ["temperature"],
        {"detection": {"rolling_window": 5, "min_history_points": 4, "window_strategy": "past_only"}}
    )
    # Outlier at row 15 should be detected
    assert any(i["row_position"] == 15 for i in issues_past)
    # But row 10 should not be flagged due to leakage
    assert not any(i["row_position"] == 10 for i in issues_past)
