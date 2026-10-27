from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from app.ingestion.schemas import TelemetryReadingInput


class GroundTruthFaultRecord(BaseModel):
    timestamp: datetime
    asset_id: str
    sensor_id: str
    fault_type: str
    injected_value: Optional[float]
    clean_ground_truth_value: float
    reading_id: Optional[str] = None
    notes: str = ""


class FaultInjector:
    """Injects controlled, repeatable data-quality anomalies into telemetry streams.

    Maintains a separate ground-truth record for rigorous quantitative evaluation.
    CRITICAL RULE:
    The injector NEVER directly assigns pipeline decisions. The pipeline must detect
    and process faults independently.
    """

    def __init__(self):
        self.ground_truth_log: List[GroundTruthFaultRecord] = []
        self._drift_accumulator: Dict[str, float] = {}
        self._last_frozen_value: Dict[str, float] = {}

    def inject_fault(
        self,
        reading: TelemetryReadingInput,
        clean_value: float,
        scenario: str,
    ) -> Tuple[Optional[TelemetryReadingInput], Optional[GroundTruthFaultRecord]]:
        """Applies a controlled fault transformation to a clean reading.

        Returns:
            Tuple of (modified_reading_or_None, ground_truth_record_or_None).
            If the scenario is 'missing_message', modified_reading is None (dropped packet).
        """
        sensor_key = f"{reading.asset_id}:{reading.sensor_id}"
        scenario = scenario.lower()

        # 1. Normal Operation
        if scenario in ("normal", "clean"):
            return reading, None

        # 2. Missing Message (Simulates dropped network frame)
        if scenario == "missing_message":
            record = GroundTruthFaultRecord(
                timestamp=reading.timestamp,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="MISSING_DATA",
                injected_value=None,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                notes="Network frame dropped at field gateway level.",
            )
            self.ground_truth_log.append(record)
            return None, record  # dropped: do not send

        # 3. Explicit Missing Value (Payload arrives with null value)
        if scenario == "missing_value":
            reading.value = None
            record = GroundTruthFaultRecord(
                timestamp=reading.timestamp,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="MISSING_DATA",
                injected_value=None,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                notes="Explicit null telemetry measurement received.",
            )
            self.ground_truth_log.append(record)
            return reading, record

        # 4. Repeated Readings (Frozen / Stale Sensor)
        if scenario in ("repeated_readings", "frozen"):
            if sensor_key not in self._last_frozen_value:
                self._last_frozen_value[sensor_key] = clean_value
            frozen_val = self._last_frozen_value[sensor_key]
            reading.value = frozen_val
            record = GroundTruthFaultRecord(
                timestamp=reading.timestamp,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="FROZEN_SENSOR",
                injected_value=frozen_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                notes=f"Sensor value frozen at {frozen_val}.",
            )
            self.ground_truth_log.append(record)
            return reading, record

        # 5. Sudden Spike (Dirac impulse anomaly)
        if scenario in ("sudden_spike", "spike"):
            spike_magnitude = 45.0 if "boiler" in reading.asset_id else 18.0
            spiked_val = round(clean_value + spike_magnitude, 2)
            reading.value = spiked_val
            record = GroundTruthFaultRecord(
                timestamp=reading.timestamp,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="SUDDEN_SPIKE",
                injected_value=spiked_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                notes=f"Electrical transient spike of +{spike_magnitude}.",
            )
            self.ground_truth_log.append(record)
            return reading, record

        # 6. Gradual Drift (Cumulative calibration creep)
        if scenario in ("gradual_drift", "drift"):
            drift_rate = 0.8 if "boiler" in reading.asset_id else 0.4
            self._drift_accumulator[sensor_key] = self._drift_accumulator.get(sensor_key, 0.0) + drift_rate
            drifted_val = round(clean_value + self._drift_accumulator[sensor_key], 2)
            reading.value = drifted_val
            record = GroundTruthFaultRecord(
                timestamp=reading.timestamp,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="GRADUAL_DRIFT",
                injected_value=drifted_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                notes=f"Cumulative calibration creep: +{self._drift_accumulator[sensor_key]:.2f}.",
            )
            self.ground_truth_log.append(record)
            return reading, record

        # 7. Invalid or Corrupted Payload (Out of bounds or malformed)
        if scenario in ("invalid_corrupted", "invalid", "out_of_bounds"):
            invalid_val = 350.0 if "boiler" in reading.asset_id else 85.0
            reading.value = invalid_val
            record = GroundTruthFaultRecord(
                timestamp=reading.timestamp,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="INVALID_CORRUPTED",
                injected_value=invalid_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                notes=f"Gross physical limit breach: {invalid_val}.",
            )
            self.ground_truth_log.append(record)
            return reading, record

        # Fallback to normal if scenario not recognized
        return reading, None
