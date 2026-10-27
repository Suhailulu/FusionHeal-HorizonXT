from __future__ import annotations
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import numpy as np
from app.ingestion.schemas import TelemetryReadingInput


class SyntheticSensorGenerator:
    """Generates realistic synthetic industrial telemetry for software testing and demonstration.

    NOTICE:
    These signals are purely synthetic for pipeline validation and do NOT represent physical measurements
    from certified industrial equipment.
    """

    def __init__(self, random_seed: Optional[int] = 42):
        self.rng = np.random.default_rng(random_seed)
        self.sequence_counters: Dict[str, int] = {
            "boiler_01:temperature_01": 0,
            "pump_01:vibration_01": 0,
        }
        self.tick_counters: Dict[str, int] = {
            "boiler_01:temperature_01": 0,
            "pump_01:vibration_01": 0,
        }

    def generate_boiler_temperature(
        self,
        timestamp: Optional[datetime] = None,
        base_temp: float = 72.0,
        thermal_oscillation_period: float = 60.0,
    ) -> Tuple[TelemetryReadingInput, float]:
        """Asset A: Industrial Boiler (temperature in degrees Celsius).

        Generates realistic thermodynamic thermal inertia: slow sinusoidal variations
        plus small Gaussian white noise.
        """
        key = "boiler_01:temperature_01"
        self.sequence_counters[key] += 1
        seq = self.sequence_counters[key]
        tick = self.tick_counters[key]
        self.tick_counters[key] += 1

        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Physics: slow sinusoidal process change + Gaussian noise
        t_phase = 2 * math.pi * (tick % thermal_oscillation_period) / thermal_oscillation_period
        process_shift = 3.5 * math.sin(t_phase)
        noise = float(self.rng.normal(0.0, 0.25))
        true_clean_val = round(base_temp + process_shift + noise, 2)

        reading = TelemetryReadingInput(
            reading_id=f"READ-BOILER-{uuid.uuid4().hex[:8]}",
            asset_id="boiler_01",
            sensor_id="temperature_01",
            timestamp=ts,
            sequence_number=seq,
            value=true_clean_val,
            unit="C",
            source_type="simulated",
        )
        return reading, true_clean_val

    def generate_pump_vibration(
        self,
        timestamp: Optional[datetime] = None,
        base_vibration: float = 2.8,
    ) -> Tuple[TelemetryReadingInput, float]:
        """Asset B: Industrial Pump (vibration acceleration in m/s²).

        Generates rotational shaft dynamics: baseline mechanical oscillation
        plus high-frequency noise.
        """
        key = "pump_01:vibration_01"
        self.sequence_counters[key] += 1
        seq = self.sequence_counters[key]
        tick = self.tick_counters[key]
        self.tick_counters[key] += 1

        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Rotational mechanical harmonics + random vibration jitter
        harmonics = 0.4 * math.sin(2 * math.pi * tick / 10.0) + 0.15 * math.cos(2 * math.pi * tick / 3.0)
        jitter = float(self.rng.normal(0.0, 0.12))
        true_clean_val = round(max(0.05, base_vibration + harmonics + jitter), 3)

        reading = TelemetryReadingInput(
            reading_id=f"READ-PUMP-{uuid.uuid4().hex[:8]}",
            asset_id="pump_01",
            sensor_id="vibration_01",
            timestamp=ts,
            sequence_number=seq,
            value=true_clean_val,
            unit="m/s2",
            source_type="simulated",
        )
        return reading, true_clean_val
