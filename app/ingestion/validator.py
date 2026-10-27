from __future__ import annotations
from typing import Tuple, List, Optional
from datetime import datetime, timezone
import math
from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput


class ValidationOutcome:
    def __init__(self, is_valid: bool, status: str, errors: List[str]):
        self.is_valid = is_valid
        self.status = status  # VALID, MALFORMED, OUT_OF_ORDER, CORRUPTED
        self.errors = errors


class TelemetryValidator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_reading(
        self,
        reading: TelemetryReadingInput,
        last_timestamp: Optional[datetime] = None,
        last_sequence: Optional[int] = None,
    ) -> ValidationOutcome:
        errors: List[str] = []
        status = "VALID"

        # 1. Timestamp validation
        ts = reading.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if self.settings.enforce_wall_clock_sync:
            now = datetime.now(timezone.utc)
            skew = (ts - now).total_seconds()
            if skew > self.settings.allowed_clock_skew_seconds:
                errors.append(f"Timestamp is in the future by {skew:.1f}s (allowed: {self.settings.allowed_clock_skew_seconds}s).")
                status = "MALFORMED"

        # 2. Sequence order and timestamp ordering
        if last_timestamp is not None:
            if ts < last_timestamp:
                errors.append(f"Out-of-order reading detected: timestamp {ts.isoformat()} is earlier than previous {last_timestamp.isoformat()}.")
                status = "OUT_OF_ORDER"

        if last_sequence is not None:
            if reading.sequence_number < last_sequence:
                errors.append(f"Regressive sequence number: {reading.sequence_number} < last seen {last_sequence}.")
                status = "OUT_OF_ORDER"

        # 3. Numeric validity
        if reading.value is not None:
            if math.isnan(reading.value) or math.isinf(reading.value):
                errors.append(f"Non-finite numeric value: {reading.value}.")
                status = "CORRUPTED"

        # 4. Sensor limits check
        constraint = self.settings.get_constraint(reading.asset_id, reading.sensor_id)
        if constraint:
            if reading.unit.lower() != constraint.unit.lower():
                errors.append(f"Unit mismatch: got '{reading.unit}', expected '{constraint.unit}'.")
                status = "MALFORMED"

        is_valid = (len(errors) == 0)
        return ValidationOutcome(is_valid=is_valid, status=status, errors=errors)
