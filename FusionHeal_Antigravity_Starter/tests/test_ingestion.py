import pandas as pd
import pytest
from fusionheal.ingestion import DataIngestionError, load_dataset

def test_load_valid_dataset(tmp_path):
    path = tmp_path / "sensor.csv"
    pd.DataFrame({"timestamp": ["2026-10-23 10:00:00", "2026-10-23 10:01:00"],
                  "temperature": [65.0, 66.0]}).to_csv(path, index=False)
    result = load_dataset(path, "timestamp")
    assert len(result) == 2
    assert "source_record_id" in result
    assert "timestamp_original" in result
    assert "timestamp_parsed" in result

def test_missing_timestamp_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"temperature": [65.0]}).to_csv(path, index=False)
    with pytest.raises(DataIngestionError):
        load_dataset(path, "timestamp")

def test_missing_file():
    with pytest.raises(DataIngestionError):
        load_dataset("does_not_exist.csv", "timestamp")
