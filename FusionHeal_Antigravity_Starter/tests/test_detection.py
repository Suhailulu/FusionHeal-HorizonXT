import pandas as pd
from fusionheal.detection import detect_issues

def test_missing_value_detected():
    df = pd.DataFrame({
        "source_record_id": [f"ROW-{i}" for i in range(9)],
        "timestamp_parsed": pd.date_range("2026-01-01", periods=9, freq="min"),
        "temperature": [10, 10, 11, 10, None, 10, 11, 10, 10],
    })
    issues = detect_issues(df, ["temperature"], {}, rolling_window=9, min_history_points=5)
    assert any(i["fault_type"] == "missing" for i in issues)

def test_range_violation_detected():
    df = pd.DataFrame({
        "source_record_id": [f"ROW-{i}" for i in range(7)],
        "timestamp_parsed": pd.date_range("2026-01-01", periods=7, freq="min"),
        "temperature": [10, 10, 11, 10, 999, 10, 11],
    })
    issues = detect_issues(df, ["temperature"], {"temperature": {"min": 0, "max": 100}},
                           rolling_window=7, min_history_points=5)
    assert any(i["fault_type"] == "range_violation" for i in issues)
