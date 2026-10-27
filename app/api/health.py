from __future__ import annotations
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.config import get_settings
from app.storage.database import get_db
from app.ingestion.schemas import HealthResponse

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)):
    """Check service health and database connectivity."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {str(exc)}"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        service=settings.app_name,
        version="0.1.0",
        database=db_status,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc),
    )
