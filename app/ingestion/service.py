from __future__ import annotations
from typing import List, Union
from sqlalchemy.orm import Session
from app.config import get_settings
from app.ingestion.schemas import TelemetryReadingInput, TelemetryProcessingResult, TelemetryBatchResponse
from app.pipeline.processing import PipelineProcessor

settings = get_settings()
default_processor = PipelineProcessor(settings)


class IngestionService:
    def __init__(self, processor: PipelineProcessor | None = None):
        self.processor = processor or default_processor

    def reset_state(self) -> None:
        self.processor.reset_state()

    def ingest_single(
        self,
        reading: TelemetryReadingInput,
        db: Session,
    ) -> TelemetryProcessingResult:
        return self.processor.process_reading(reading, db)

    def ingest_batch(
        self,
        readings: List[TelemetryReadingInput],
        db: Session,
    ) -> TelemetryBatchResponse:
        results = self.processor.process_batch(readings, db)
        return TelemetryBatchResponse(
            total_received=len(readings),
            processed=len(results),
            results=results,
        )
