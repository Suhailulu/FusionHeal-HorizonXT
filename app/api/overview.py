from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.storage.database import get_db
from app.storage.repository import TelemetryRepository
from app.ingestion.schemas import OverviewResponse

router = APIRouter(prefix="/api", tags=["Overview"])


@router.get("/overview", response_model=OverviewResponse, summary="Operational overview and metrics summary")
def get_overview(db: Session = Depends(get_db)):
    """Aggregate counts across all processed telemetry, preserved observations, and detected faults.

    Strictly prevents double-counting by tallying accepted records once per primary decision.
    """
    repo = TelemetryRepository(db)
    counts = repo.get_overview_counts()
    return OverviewResponse(**counts)
