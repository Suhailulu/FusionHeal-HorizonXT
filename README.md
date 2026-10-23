# FusionHeal — Antigravity Starter

An initial, runnable prototype for an evidence-guided industrial sensor-data quality pipeline.

## What this starter does
- Loads timestamped sensor data from CSV.
- Preserves source rows and original readings.
- Detects missing values, range violations, and robust rolling outliers.
- Generates conservative interpolation/rolling-median candidates.
- Applies basic configured range constraints.
- Produces explicit `ACCEPT`, `REVIEW`, `REJECT`, or `NO_ACTION` decisions.
- Writes a JSONL audit trail and a cleaned-output CSV.
- Includes a small Streamlit dashboard and synthetic demo dataset.

**Important:** This is a prototype, not a safety-certified industrial control system. Example limits are demo values. Do not connect it to live equipment or use its reconstructed values for control decisions without domain validation.

## 1. Open in Antigravity
1. Extract this ZIP into a folder named `FusionHeal`.
2. Open Antigravity and choose **Open Folder** (or the equivalent project-open action).
3. Select the extracted `FusionHeal` folder.
4. Open the integrated terminal in that folder.

## 2. Create a Python environment

Python 3.11 or 3.12 is recommended.

### Windows PowerShell
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```
Then run commands with `.\.venv\Scripts\python.exe`.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 3. Run tests
```bash
pytest -q
```

## 4. Run the pipeline
```bash
python -m fusionheal.cli --input data/synthetic/demo_sensor_data.csv --config config.yaml --output-dir outputs
```

Expected outputs:
- `outputs/processed_data.csv`
- `outputs/audit.jsonl`
- `outputs/summary.json`

## 5. Run the dashboard
```bash
streamlit run app.py
```
Upload a CSV with a timestamp column and numeric sensor columns, or use the included demo dataset.

## 6. Use Antigravity effectively
Ask the coding agent to work in small, testable tasks. Suggested first prompt:

> Inspect this FusionHeal repository. Do not rewrite everything. Run the tests, explain the architecture, identify the next smallest high-value implementation task, implement it with tests, and report exactly which files changed.

Then continue with:
1. Add detector tests for spike, stuck sensor, and drift.
2. Add timestamp-gap detection using the expected sampling interval.
3. Add evidence records that explain why a candidate was generated.
4. Add validation-based confidence calibration and clearly distinguish heuristic scores from probabilities.
5. Add benchmark scripts for injected faults with known ground truth.

## Design rules
- Preserve original data; never silently overwrite it.
- An anomaly is not proof that a sensor is faulty.
- Candidate generation is separate from decision-making.
- If evidence is weak, use `REVIEW` rather than inventing a value.
- Demo limits in `config.yaml` are placeholders and must be replaced with validated domain rules.
