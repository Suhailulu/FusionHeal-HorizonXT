import json
import pandas as pd
from fusionheal.pipeline import run_pipeline

def test_pipeline_writes_outputs(tmp_path):
    input_path = tmp_path / "input.csv"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=12, freq="min").astype(str),
        "temperature": [20, 21, 20, 22, 21, None, 20, 21, 20, 22, 21, 20],
        "pressure": [5, 5.1, 5, 5.2, 5.1, 5, 5.1, 5, 5.2, 5.1, 5, 5.1],
    }).to_csv(input_path, index=False)
    config_path.write_text(
        "timestamp_column: timestamp\n"
        "sensor_limits:\n  temperature:\n    min: 0\n    max: 100\n"
        "detection:\n  rolling_window: 9\n  mad_threshold: 5.0\n  min_history_points: 5\n"
        "decision:\n  auto_accept_interpolation: true\n  auto_accept_outlier_replacement: false\n",
        encoding="utf-8"
    )
    summary = run_pipeline(input_path, config_path, output_dir)
    assert summary["input_rows"] == 12
    assert (output_dir / "processed_data.csv").exists()
    assert (output_dir / "audit.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
