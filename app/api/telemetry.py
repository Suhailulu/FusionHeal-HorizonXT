from __future__ import annotations
from typing import Any, List, Union
from fastapi import APIRouter, Depends, Body, status, HTTPException
from pydantic import ValidationError
from sqlalchemy.orm import Session
from app.storage.database import get_db
from app.ingestion.schemas import (
    TelemetryReadingInput,
    TelemetryBatchInput,
    TelemetryProcessingResult,
    TelemetryBatchResponse,
)
from app.ingestion.service import IngestionService

router = APIRouter(prefix="/api", tags=["Telemetry Ingestion"])
service = IngestionService()


@router.post(
    "/telemetry",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest simulated or physical telemetry readings",
)
def ingest_telemetry(
    payload: Any = Body(..., description="Single reading object, list of readings, or {'readings': [...]} batch"),
    db: Session = Depends(get_db),
) -> Any:
    """Continuous telemetry ingestion endpoint.

    Accepts single or batched sensor observations, validates schemas and timestamps,
    preserves raw telemetry immutably, performs quality detection, invokes the ML
    inference gate, applies policy decisions, and produces append-only audit logs.
    """
    try:
        if isinstance(payload, list):
            readings = [TelemetryReadingInput.model_validate(item) for item in payload]
            return service.ingest_batch(readings, db)
        elif isinstance(payload, dict):
            if "readings" in payload and isinstance(payload["readings"], list):
                batch = TelemetryBatchInput.model_validate(payload)
                return service.ingest_batch(batch.readings, db)
            else:
                reading = TelemetryReadingInput.model_validate(payload)
                return service.ingest_single(reading, db)
        else:
            raise ValueError("Payload must be a JSON object or array of telemetry readings.")
    except ValidationError as val_err:
        import json
        safe_errors = json.loads(json.dumps(val_err.errors(), default=str))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=safe_errors,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Telemetry ingestion failed: {str(exc)}",
        )

