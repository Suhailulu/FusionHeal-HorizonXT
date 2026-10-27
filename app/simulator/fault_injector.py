from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
from app.ingestion.schemas import TelemetryReadingInput


class GroundTruthFaultRecord(BaseModel):
    timestamp: datetime
    event_timestamp: Optional[datetime] = None
    asset_id: str
    sensor_id: str
    fault_type: str
    injected_value: Optional[float] = None
    clean_ground_truth_value: Optional[float] = None
    reading_id: Optional[str] = None
    injection_parameters: Dict[str, Any] = Field(default_factory=dict)
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
        self._last_emitted_reading: Dict[str, TelemetryReadingInput] = {}

    def inject_fault(
        self,
        reading: TelemetryReadingInput,
        clean_value: float,
        scenario: str,
    ) -> Tuple[Optional[TelemetryReadingInput], Optional[GroundTruthFaultRecord]]:
        """Applies a controlled fault transformation to a clean reading.

        Returns:
            Tuple of (modified_reading_or_None, ground_truth_record_or_None).
            If scenario is 'missing_message', modified_reading is None (dropped packet).
        """
        sensor_key = f"{reading.asset_id}:{reading.sensor_id}"
        scenario = scenario.lower().strip()
        ts = reading.event_timestamp or reading.timestamp

        # 1. Normal Operation
        if scenario in ("normal", "clean"):
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, None

        # 2. Missing Message (Simulates dropped network frame)
        if scenario in ("missing_message", "dropped_packet"):
            record = GroundTruthFaultRecord(
                timestamp=ts,
                event_timestamp=ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="MISSING_DATA",
                injected_value=None,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "missing_message", "action": "packet_dropped"},
                notes="Network frame dropped at field gateway level.",
            )
            self.ground_truth_log.append(record)
            return None, record  # dropped: do not send to API

        # 3. Explicit Missing Value (Payload arrives with null value)
        if scenario in ("missing_value", "null_value"):
            reading.value = None
            record = GroundTruthFaultRecord(
                timestamp=ts,
                event_timestamp=ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="MISSING_DATA",
                injected_value=None,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "missing_value", "action": "set_null"},
                notes="Explicit null telemetry measurement received.",
            )
            self.ground_truth_log.append(record)
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, record

        # 4. Repeated Readings (Frozen / Stale Sensor)
        if scenario in ("repeated_readings", "frozen", "frozen_sensor"):
            if sensor_key not in self._last_frozen_value:
                self._last_frozen_value[sensor_key] = clean_value
            frozen_val = self._last_frozen_value[sensor_key]
            reading.value = frozen_val
            record = GroundTruthFaultRecord(
                timestamp=ts,
                event_timestamp=ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="FROZEN_SENSOR",
                injected_value=frozen_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "frozen", "frozen_value": frozen_val},
                notes=f"Sensor value frozen at {frozen_val}.",
            )
            self.ground_truth_log.append(record)
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, record

        # 5. Sudden Spike (Transient impulse anomaly)
        if scenario in ("sudden_spike", "spike"):
            spike_magnitude = 45.0 if "boiler" in reading.asset_id else 18.0
            spiked_val = round(clean_value + spike_magnitude, 2)
            reading.value = spiked_val
            record = GroundTruthFaultRecord(
                timestamp=ts,
                event_timestamp=ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="SUDDEN_SPIKE",
                injected_value=spiked_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "sudden_spike", "spike_magnitude": spike_magnitude},
                notes=f"Electrical transient spike of +{spike_magnitude}.",
            )
            self.ground_truth_log.append(record)
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, record

        # 6. Gradual Drift (Cumulative calibration creep)
        if scenario in ("gradual_drift", "drift"):
            drift_rate = 0.8 if "boiler" in reading.asset_id else 0.4
            self._drift_accumulator[sensor_key] = self._drift_accumulator.get(sensor_key, 0.0) + drift_rate
            drifted_val = round(clean_value + self._drift_accumulator[sensor_key], 2)
            reading.value = drifted_val
            record = GroundTruthFaultRecord(
                timestamp=ts,
                event_timestamp=ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="GRADUAL_DRIFT",
                injected_value=drifted_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "gradual_drift", "cumulative_drift": self._drift_accumulator[sensor_key]},
                notes=f"Cumulative calibration creep: +{self._drift_accumulator[sensor_key]:.2f}.",
            )
            self.ground_truth_log.append(record)
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, record

        # 7. Invalid or Corrupted Payload (Out of bounds or malformed unit)
        if scenario in ("invalid_corrupted", "invalid", "out_of_bounds", "corrupted"):
            invalid_val = 350.0 if "boiler" in reading.asset_id else 85.0
            reading.value = invalid_val
            record = GroundTruthFaultRecord(
                timestamp=ts,
                event_timestamp=ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="INVALID_CORRUPTED",
                injected_value=invalid_val,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "invalid_corrupted", "invalid_value": invalid_val},
                notes=f"Gross physical limit breach: {invalid_val}.",
            )
            self.ground_truth_log.append(record)
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, record

        # 8. Communication Gap (Injects sequence jump or time gap)
        if scenario in ("communication_gap", "gap"):
            reading.sequence_number += 5  # skip sequence
            new_ts = ts + timedelta(seconds=15)
            reading.event_timestamp = new_ts
            reading.timestamp = new_ts
            record = GroundTruthFaultRecord(
                timestamp=new_ts,
                event_timestamp=new_ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="COMMUNICATION_GAP",
                injected_value=clean_value,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "communication_gap", "time_jump_seconds": 15, "sequence_jump": 5},
                notes="Injected communications stall and skipped telemetry sequence.",
            )
            self.ground_truth_log.append(record)
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, record

        # 9. Duplicate Message
        if scenario in ("duplicate", "duplicate_message"):
            if sensor_key in self._last_emitted_reading:
                prev = self._last_emitted_reading[sensor_key]
                dup_reading = prev.model_copy(deep=True)
                record = GroundTruthFaultRecord(
                    timestamp=dup_reading.event_timestamp or dup_reading.timestamp,
                    event_timestamp=dup_reading.event_timestamp or dup_reading.timestamp,
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    fault_type="DUPLICATE_MESSAGE",
                    injected_value=dup_reading.value,
                    clean_ground_truth_value=clean_value,
                    reading_id=dup_reading.reading_id,
                    injection_parameters={"scenario": "duplicate", "duplicate_of": dup_reading.reading_id},
                    notes=f"Injected duplicate packet replay of reading {dup_reading.reading_id}.",
                )
                self.ground_truth_log.append(record)
                return dup_reading, record

        # 10. Out-of-Order Message
        if scenario in ("out_of_order", "late", "late_arriving"):
            delayed_ts = ts - timedelta(seconds=3)
            reading.event_timestamp = delayed_ts
            reading.timestamp = delayed_ts
            reading.sequence_number = max(0, reading.sequence_number - 2)
            record = GroundTruthFaultRecord(
                timestamp=delayed_ts,
                event_timestamp=delayed_ts,
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                fault_type="OUT_OF_ORDER",
                injected_value=clean_value,
                clean_ground_truth_value=clean_value,
                reading_id=reading.reading_id,
                injection_parameters={"scenario": "out_of_order", "offset_seconds": -3},
                notes="Injected out-of-order delayed sequence packet.",
            )
            self.ground_truth_log.append(record)
            self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
            return reading, record

        # Fallback to normal if scenario not recognized
        self._last_emitted_reading[sensor_key] = reading.model_copy(deep=True)
        return reading, None
