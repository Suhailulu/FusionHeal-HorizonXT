from __future__ import annotations
import json
from typing import Any, Dict, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.storage.database import get_db
from app.storage.repository import TelemetryRepository
from app.evaluation.metrics import run_reproducible_benchmark

router = APIRouter(prefix="/api", tags=["Evaluation Metrics"])


class MetricRunResponse(BaseModel):
    available: bool
    run_id: Optional[str] = None
    scenario_name: Optional[str] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    latency_ms: Optional[float] = None
    accepted_repairs: Optional[int] = None
    unresolved_events: Optional[int] = None
    unintentional_modifications: Optional[int] = None
    baseline_comparison: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    status_note: str
    confusion_matrix: Optional[Dict[str, int]] = None
    detection_by_fault_type: Optional[Dict[str, Any]] = None
    reconstruction_by_asset: Optional[Dict[str, Any]] = None
    reconstruction_by_fault_type: Optional[Dict[str, Any]] = None
    preservation_metrics: Optional[Dict[str, Any]] = None
    operational_metrics: Optional[Dict[str, Any]] = None
    decision_counts: Optional[Dict[str, Any]] = None


class RunBenchmarkRequest(BaseModel):
    seed: int = Field(42, description="Random seed for repeatable synthetic data generation")
    sample_count: int = Field(60, ge=10, le=500, description="Number of evaluation ticks across sensors")
    scenario_name: str = Field("reproducible_synthetic_benchmark", description="Scenario label")


@router.get("/metrics", response_model=MetricRunResponse, summary="Retrieve latest evaluation metrics")
def get_metrics(db: Session = Depends(get_db)):
    """Retrieve quantitative evaluation metrics from the most recent controlled benchmark run.

    CRITICAL RULE:
    If evaluation metrics have not yet been calculated, returns an explicit unavailable state
    with null values rather than inventing synthetic benchmark results.
    """
    repo = TelemetryRepository(db)
    latest = repo.get_latest_metrics()
    if not latest:
        return MetricRunResponse(
            available=False,
            status_note=(
                "No controlled evaluation run has been recorded yet. "
                "Execute the evaluation benchmark suite via POST /api/metrics/run to calculate precision, recall, MAE, and RMSE."
            ),
        )

    details = {}
    if latest.details_json:
        try:
            details = json.loads(latest.details_json)
        except Exception:
            details = {"raw": latest.details_json}

    baseline_comp = None
    if latest.baseline_comparison_json:
        try:
            baseline_comp = json.loads(latest.baseline_comparison_json)
        except Exception:
            baseline_comp = None
    elif isinstance(details, dict):
        baseline_comp = details.get("baseline_comparison")

    det_sec = details.get("detection", {}) if isinstance(details, dict) else {}
    rec_sec = details.get("reconstruction", {}) if isinstance(details, dict) else {}
    pres_sec = details.get("preservation", {}) if isinstance(details, dict) else {}
    perf_sec = details.get("performance", {}) if isinstance(details, dict) else {}
    dec_sec = details.get("decision_policy", {}) if isinstance(details, dict) else {}

    confusion_matrix = det_sec.get("confusion_matrix")
    detection_by_fault_type = det_sec.get("by_fault_type")
    reconstruction_by_asset = {
        "mae": rec_sec.get("mae_by_asset", {}),
        "rmse": rec_sec.get("rmse_by_asset", {}),
    } if ("mae_by_asset" in rec_sec or "rmse_by_asset" in rec_sec) else None
    reconstruction_by_fault_type = {
        "mae": rec_sec.get("mae_by_fault_type", {}),
        "rmse": rec_sec.get("rmse_by_fault_type", {}),
    } if ("mae_by_fault_type" in rec_sec or "rmse_by_fault_type" in rec_sec) else None

    return MetricRunResponse(
        available=True,
        run_id=latest.run_id,
        scenario_name=latest.scenario_name,
        precision=latest.precision,
        recall=latest.recall,
        f1_score=latest.f1_score,
        mae=latest.mae,
        rmse=latest.rmse,
        latency_ms=latest.latency_ms,
        accepted_repairs=latest.accepted_repairs,
        unresolved_events=latest.unresolved_events,
        unintentional_modifications=latest.unintentional_modifications,
        baseline_comparison=baseline_comp,
        details=details,
        created_at=latest.created_at,
        status_note="Metrics calculated against controlled fault-injected ground truth.",
        confusion_matrix=confusion_matrix,
        detection_by_fault_type=detection_by_fault_type,
        reconstruction_by_asset=reconstruction_by_asset,
        reconstruction_by_fault_type=reconstruction_by_fault_type,
        preservation_metrics=pres_sec if pres_sec else None,
        operational_metrics=perf_sec if perf_sec else None,
        decision_counts=dec_sec if dec_sec else None,
    )


@router.post("/metrics/run", response_model=MetricRunResponse, summary="Execute a reproducible benchmark evaluation run")
def run_benchmark_endpoint(
    request: Optional[RunBenchmarkRequest] = None,
    db: Session = Depends(get_db),
):
    """Execute a controlled benchmark run with ground-truth verification and calculate baseline comparisons.
    
    Evaluates:
    - Baseline 1: No-repair baseline
    - Baseline 2: Linear interpolation (strictly for missing values)
    - Baseline 3: FusionHeal pipeline
    """
    seed = request.seed if request else 42
    sample_count = request.sample_count if request else 60
    scenario_name = request.scenario_name if request else "reproducible_synthetic_benchmark"

    report = run_reproducible_benchmark(
        seed=seed,
        sample_count=sample_count,
        db=db,
        scenario_name=scenario_name,
    )

    rep_dict = report.to_dict()
    det_sec = rep_dict.get("detection", {})
    rec_sec = rep_dict.get("reconstruction", {})
    pres_sec = rep_dict.get("preservation", {})
    perf_sec = rep_dict.get("performance", {})
    dec_sec = rep_dict.get("decision_policy", {})

    reconstruction_by_asset = {
        "mae": report.mae_by_asset,
        "rmse": report.rmse_by_asset,
    }
    reconstruction_by_fault_type = {
        "mae": report.mae_by_fault_type,
        "rmse": report.rmse_by_fault_type,
    }

    return MetricRunResponse(
        available=True,
        run_id=report.run_id,
        scenario_name=report.scenario_name,
        precision=report.precision,
        recall=report.recall,
        f1_score=report.f1_score,
        mae=report.mae,
        rmse=report.rmse,
        latency_ms=report.latency_ms,
        accepted_repairs=report.accepted_repairs,
        unresolved_events=report.unresolved_events,
        unintentional_modifications=report.unintentional_modifications,
        baseline_comparison=report.baseline_comparison,
        details=report.details,
        created_at=datetime.now(),
        status_note=f"Reproducible evaluation executed with seed={seed}, sample_count={sample_count}.",
        confusion_matrix=report.confusion_matrix,
        detection_by_fault_type=report.detection_by_fault_type,
        reconstruction_by_asset=reconstruction_by_asset,
        reconstruction_by_fault_type=reconstruction_by_fault_type,
        preservation_metrics=pres_sec if pres_sec else None,
        operational_metrics=perf_sec if perf_sec else None,
        decision_counts=dec_sec if dec_sec else None,
    )
