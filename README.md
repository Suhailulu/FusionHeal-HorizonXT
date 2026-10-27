# 🛡️ FusionHeal — Industrial Data-Quality Gateway

**Evidence-Guided, Process-Constrained Self-Healing Telemetry Pipeline**  
*Built for the HorizonXT-PS8 Hackathon*

---

## 1. System Overview & Philosophy

Industrial facilities rely on continuous telemetric observations from **Multiple Manufacturing Assets (MMA)** such as boilers, compressors, pumps, and robotic cells. Unchecked sensor corruption (calibration drift, electrical spikes, frozen ADCs, packet drops) silently compromises downstream AI models and predictive maintenance systems.

### Core Design Philosophy
> **"Don't just clean industrial data. Make the data explain what happened to it."**

FusionHeal rejects opaque, binary overwriting. It operates as an **inline pre-analytics gate** that:
1. **Preserves raw telemetry immutably** before any candidate repair.
2. **Detects multi-stage data-quality anomalies** using physical bounds, CUSUM drift, and robust statistics.
3. **Decouples detection from repair**—an anomaly is not automatic proof that an instrument has failed.
4. **Enforces mandatory physical process constraints** that cannot be overridden by ML scores.
5. **Classifies every observation into a strict 3-state taxonomy**: `PRESERVE`, `REPAIR`, or `UNRESOLVED`.
6. **Maintains append-only, cryptographically chained audit trails** (SHA-256) aligned with SOC 2 Type II and EU AI Act governance.

---

## 2. Architecture & Processing Flow

```
Simulated Sensor Data (Boiler / Pump)
               │
               ▼
   Telemetry Generator / Injector
               │
               ▼
   FastAPI Ingestion Endpoint (POST /api/telemetry)
               │
               ▼
 Schema, Timestamp & Sequence Validation
               │
               ▼
   Raw Data Preservation (Table: raw_readings)
               │
               ▼
   Modular Data-Quality Detection Engine
   ├── Missing Data & Sequence Skip
   ├── Frozen / Stale Sensor Check
   ├── Sudden Spike (Rate-of-Change / MAD)
   ├── Gradual Drift (Page's CUSUM)
   ├── Communication Gap (Sampling Interval)
   └── Physical Plausibility Range Violation
               │
               ▼
   Reconstruction Model Interface
   └── [Placeholder Adapter] ──► (Future: SAITS / ImTransformer)
               │
               ▼
   Decision Engine
   ├── PRESERVE   (Observation valid; retained intact)
   ├── REPAIR     (Candidate verified against physical rules & policy)
   ├── REVIEW     (Flagged for operator review: borderline conf, staleness, soft warnings)
   └── UNRESOLVED (Model unavailable or candidate breaches constraints)
         │                     │
         ▼                     ▼
Accepted Telemetry       Audit Trail
(accepted_readings)     (audit_logs with SHA-256 hash chaining)
         │                     │
         └──────────┬──────────┘
                    ▼
           SQLite Database
                    ▼
            FastAPI REST API
```

---

## 3. Simulated Industrial Assets

| Asset Profile | Asset ID | Sensor ID | Nominal Unit | Operating Range | Rate-of-Change Max | Physical Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Industrial Boiler** | `boiler_01` | `temperature_01` | $^\circ\text{C}$ | $0.0 - 150.0^\circ\text{C}$ | $5.0^\circ\text{C/s}$ | High thermal inertia, gradual sinusoidal load variations + white noise |
| **Industrial Pump** | `pump_01` | `vibration_01` | $\text{m/s}^2$ | $0.0 - 25.0\,\text{m/s}^2$ | $8.0\,\text{m/s}^2/\text{s}$ | Rotational harmonics (10s & 3s cycles) + random mechanical jitter |

*Note: All simulated records are explicitly labeled with `"source_type": "simulated"`. The ingestion pipeline is decoupled from the simulator and ready to accept physical sensor adapters.*

---

## 4. Telemetry Data Contract

Every telemetric observation adheres to the following contract:

```json
{
  "reading_id": "READ-BOILER-8f4b2a1c",
  "asset_id": "boiler_01",
  "sensor_id": "temperature_01",
  "timestamp": "2026-10-27T08:00:00Z",
  "sequence_number": 142,
  "value": 72.4,
  "unit": "C",
  "source_type": "simulated"
}
```

* **Missing Values**: Represented as `"value": null` without confusing missing readings with `0.0`.
* **Idempotency**: Duplicate `reading_id` submissions are safely detected and preserved without re-processing.
* **Separation of Time**: Event timestamp (`timestamp`) is preserved separately from ingestion arrival time (`ingested_at`).

---

## 5. Decision Engine Status Taxonomy & Review Rules

The decision engine assigns exactly one primary decision status to each observation:

| Status | Definition & Operational Behavior | Candidate Stored? | Accepted Value |
| :--- | :--- | :---: | :--- |
| **`PRESERVE`** | Reading passes all physical plausibility, rate-of-change, and sequence rules. Telemetry is retained unmodified. | No | Original observed value |
| **`REPAIR`** | Fault detected; ML candidate is available; candidate passes independent physical constraints, high confidence threshold ($\ge 0.80$), and auto-accept policy. | Yes | Candidate reconstructed value |
| **`REVIEW`** | Fault detected or candidate exists, but automatic acceptance is withheld under explicit review rules. Requires operator sign-off. | Yes | `null` (withheld until human review) |
| **`UNRESOLVED`** | Fault detected, but ML model is unavailable, evidence is weak ($< 0.40$), or candidate breaches hard physical safety constraints. | No / Rejected | `null` (never fabricated) |

### Explicit Rules for Flagging `REVIEW`

1. **Rule R1: Manual Plant Policy Gate**
   * *Trigger*: Plant policy has `auto_accept_repairs = False` or sensor is designated for manual sign-off.
   * *Rationale*: Even when a mathematical candidate satisfies physical limits, plant regulations may require human engineering authorization before mutating the control/analytics stream.
2. **Rule R2: Borderline ML Confidence Gate**
   * *Trigger*: Candidate satisfies physical bounds, but model confidence score is borderline ($0.40 \le \text{confidence} < 0.80$).
   * *Rationale*: Prevents automated acceptance of tentative or noisy ML estimates while presenting the candidate to the engineer for contextual evaluation.
3. **Rule R3: Staleness / Frozen Sensor Ambiguity Gate**
   * *Trigger*: Anomaly detected is `FROZEN_SENSOR` ($\ge 5$ consecutive identical floats).
   * *Rationale*: A frozen sensor reading could represent an actual steady-state thermodynamic equilibrium or an electrical line freeze. Routing to `REVIEW` prevents falsely "repairing" legitimate steady-state processes.
4. **Rule R4: Soft Physical Constraint Warning (Rate-of-Change)**
   * *Trigger*: Candidate satisfies absolute range limits $[0, 150]^\circ\text{C}$ but approaches dynamic rate-of-change envelope ($> 70\%$ of maximum allowable $\Delta x/\Delta t$).
   * *Rationale*: Flags aggressive jumps that may destabilize downstream predictive maintenance models.


---

## 6. Fast-Track Setup & Execution

### Prerequisites
* Python 3.11 or Python 3.12
* PowerShell (Windows) or Bash (macOS/Linux)

### 1. Environment Setup
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest -v
```
*Executes all unit, integration, decision policy, and API contract tests (57 tests passing).*

### 3. Start the FastAPI Ingestion Service
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be live at:
👉 **`http://127.0.0.1:8000/docs`**

### 4. Run the Continuous Telemetry Simulator
In a separate terminal, launch the continuous sensor generator:
```bash
# Normal continuous telemetry (Boiler + Pump)
python -m app.simulator.runner --url http://127.0.0.1:8000/api/telemetry --interval 1.0

# Or inject a specific fault scenario after 5 seconds:
python -m app.simulator.runner --url http://127.0.0.1:8000/api/telemetry --fault spike --fault-after 5
```
Supported fault scenarios: `normal`, `spike`, `frozen`, `drift`, `missing_value`, `missing_message`, `invalid`.

---

## 7. REST API Endpoints & Examples

### Health Check
```http
GET /health
```
```json
{
  "status": "healthy",
  "service": "FusionHeal",
  "version": "0.1.0",
  "database": "connected",
  "environment": "development",
  "timestamp": "2026-10-27T08:00:00Z"
}
```

### Ingest Telemetry
```http
POST /api/telemetry
Content-Type: application/json

{
  "reading_id": "test-read-001",
  "asset_id": "boiler_01",
  "sensor_id": "temperature_01",
  "timestamp": "2026-10-27T08:00:00Z",
  "sequence_number": 1,
  "value": 72.5,
  "unit": "C",
  "source_type": "simulated"
}
```
**Response (201 Created):**
```json
{
  "reading_id": "test-read-001",
  "asset_id": "boiler_01",
  "sensor_id": "temperature_01",
  "timestamp": "2026-10-27T08:00:00Z",
  "original_value": 72.5,
  "final_value": 72.5,
  "status": "PRESERVE",
  "issues_detected": [],
  "decision_reason": "Observation conforms to physical plausibility bounds, continuous sequence, and nominal rate-of-change rules. Original telemetry preserved.",
  "audit_id": "AUD-20261027080000-8f4b2a1c"
}
```

### Operational Overview & Counts
```http
GET /api/overview
```
```json
{
  "total_readings_received": 150,
  "readings_preserved": 142,
  "repairs_accepted": 0,
  "unresolved_events": 8,
  "issues_detected": 8,
  "fault_distribution": {
    "SUDDEN_SPIKE": 3,
    "MISSING_DATA": 5
  }
}
```

### Cryptographic Audit Chain Verification
```http
GET /api/audit/verify
```
```json
{
  "status": "valid",
  "valid": true,
  "count": 150
}
```

---

## 8. Connecting the Separately Trained Transformer

Our teammate is developing the Spatial-Temporal Transformer model (e.g., **SAITS**, **ImTransformer**, or **Quantile BiLSTM**).

### Clean Integration Point: `ReconstructionModel`
All model inference is decoupled behind the abstract interface in [`app/pipeline/reconstruction.py`](file:///c:/Users/suhai/OneDrive/Desktop/FusionHeal/app/pipeline/reconstruction.py).

```python
from app.pipeline.reconstruction import ReconstructionModel, ModelPrediction

class SpatialTemporalTransformerAdapter(ReconstructionModel):
    def __init__(self, model_weights_path: str):
        # Load your PyTorch or ONNX Runtime model session here
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_weights_path)
        self.version = "1.0.0-saits"

    def predict(self, asset_id: str, sensor_id: str, sensor_window: list[dict]) -> ModelPrediction:
        # 1. Format input tensor from sensor_window: [Batch, Window_Size, Channels]
        # 2. Execute inference: point estimate + quantile prediction intervals
        predicted_val = 73.4
        q05 = 71.8
        q95 = 74.9
        confidence = 0.92

        return ModelPrediction(
            available=True,
            candidate_value=predicted_val,
            uncertainty_lower=q05,
            uncertainty_upper=q95,
            confidence_score=confidence,
            model_name="SAITSSpatialTemporalTransformer",
            model_version=self.version,
            reconstruction_method="dual_attention_imputation",
            notes="Quantile confidence score calculated from prediction interval spread."
        )
```

To activate the real model in production, instantiate `PipelineProcessor(settings, model=SpatialTemporalTransformerAdapter(...))` in [`app/ingestion/service.py`](file:///c:/Users/suhai/OneDrive/Desktop/FusionHeal/app/ingestion/service.py).
The Decision Engine will automatically validate the proposed candidate against physical bounds before accepting any repair.

---

## 9. Project Directory Structure

```
FusionHeal/
├── app/
│   ├── main.py                  # FastAPI application entrypoint & lifespan
│   ├── config.py                # Environment configuration & asset constraints
│   ├── api/                     # REST API routers
│   │   ├── health.py            # GET /health
│   │   ├── telemetry.py         # POST /api/telemetry
│   │   ├── readings.py          # GET /api/readings
│   │   ├── events.py            # GET /api/events & GET /api/events/{id}
│   │   ├── overview.py          # GET /api/overview
│   │   ├── audit.py             # GET /api/audit & GET /api/audit/verify
│   │   └── metrics.py           # GET /api/metrics
│   ├── ingestion/               # Telemetry ingestion & validation
│   │   ├── schemas.py           # Pydantic telemetry models
│   │   ├── validator.py         # Timestamp, sequence & range validation
│   │   └── service.py           # Ingestion coordinator
│   ├── pipeline/                # Pipeline intelligence
│   │   ├── detection.py         # Modular anomaly & fault detectors
│   │   ├── reconstruction.py    # Model interface & placeholder adapter
│   │   ├── decision_engine.py   # PRESERVE / REPAIR / UNRESOLVED policy
│   │   └── processing.py        # End-to-end pipeline orchestrator
│   ├── storage/                 # Persistence layer
│   │   ├── database.py          # SQLAlchemy connection & sessionmaker
│   │   ├── models.py            # SQLite schema (Raw, Events, Accepted, Audit)
│   │   └── repository.py        # Data access repository
│   ├── audit/                   # Governance & Compliance
│   │   └── logger.py            # Cryptographic SHA-256 hash chaining
│   ├── evaluation/              # Quantitative benchmarking
│   │   └── metrics.py           # TP, FP, FN, Precision, Recall, MAE, RMSE
│   └── simulator/               # Continuous telemetry simulator
│       ├── generator.py         # Synthetic Boiler & Pump generators
│       ├── fault_injector.py    # Controlled fault injection & ground truth
│       └── runner.py            # Continuous loop & CLI runner
├── tests/                       # Automated test suite (57 tests)
│   ├── test_fastapi_pipeline.py # End-to-end integration tests
│   ├── test_api.py              # Route & filter tests
│   ├── test_decisions.py        # Policy and constraint safety tests
│   ├── test_audit.py            # Tamper-evident hash chain verification
│   └── test_detection.py        # Anomaly detector tests
├── .env.example                 # Example configuration variables
├── requirements.txt             # Project dependencies
├── conftest.py                  # Centralized test database & fixtures
└── README.md                    # Documentation & user guide
```
