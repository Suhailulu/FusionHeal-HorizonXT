from __future__ import annotations
import json
from typing import Any, Dict, Optional
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.storage.database import get_db
from app.storage.repository import TelemetryRepository

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
    details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    status_note: str


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
                "Execute the evaluation benchmark suite to calculate precision, recall, MAE, and RMSE."
            ),
        )

    details = {}
    if latest.details_json:
        try:
            details = json.loads(latest.details_json)
        except Exception:
            details = {"raw": latest.details_json}

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
        details=details,
        created_at=latest.created_at,
        status_note="Metrics calculated against controlled fault-injected ground truth.",
    )
