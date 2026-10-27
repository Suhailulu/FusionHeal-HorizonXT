from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from collections import deque
import uuid
import math
import numpy as np

from app.config import Settings
from app.ingestion.schemas import TelemetryReadingInput


class QualityDetectionEvent:
    def __init__(
        self,
        event_id: str,
        asset_id: str,
        sensor_id: str,
        timestamp: datetime,
        issue_type: str,
        detection_rule: str,
        observed_evidence: Dict[str, Any],
        severity: str = "MEDIUM",
        end_timestamp: Optional[datetime] = None,
        reading_id: Optional[str] = None,
    ):
        self.event_id = event_id
        self.asset_id = asset_id
        self.sensor_id = sensor_id
        self.timestamp = timestamp
        self.end_timestamp = end_timestamp
        self.issue_type = issue_type
        self.detection_rule = detection_rule
        self.observed_evidence = observed_evidence
        self.severity = severity
        self.reading_id = reading_id
        self.detected_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "asset_id": self.asset_id,
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "issue_type": self.issue_type,
            "detection_rule": self.detection_rule,
            "observed_evidence": self.observed_evidence,
            "severity": self.severity,
            "reading_id": self.reading_id,
            "detected_at": self.detected_at.isoformat(),
        }


class SensorChannelState:
    """Maintains sequential state for a single sensor channel."""
    def __init__(self, asset_id: str, sensor_id: str, history_len: int = 50):
        self.asset_id = asset_id
        self.sensor_id = sensor_id
        self.history_len = history_len
        self.history: deque[tuple[datetime, Optional[float]]] = deque(maxlen=history_len)
        self.last_timestamp: Optional[datetime] = None
        self.last_sequence: Optional[int] = None
        self.last_valid_value: Optional[float] = None
        self.consecutive_identical_count: int = 1
        # CUSUM drift state
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.baseline_mean: Optional[float] = None


class QualityDetectionEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.channel_states: Dict[str, SensorChannelState] = {}

    def get_channel_state(self, asset_id: str, sensor_id: str) -> SensorChannelState:
        key = f"{asset_id}:{sensor_id}"
        if key not in self.channel_states:
            self.channel_states[key] = SensorChannelState(asset_id, sensor_id)
        return self.channel_states[key]

    def reset(self) -> None:
        """Reset all in-memory channel states."""
        self.channel_states.clear()

    def detect_issues(self, reading: TelemetryReadingInput) -> List[QualityDetectionEvent]:
        events: List[QualityDetectionEvent] = []
        state = self.get_channel_state(reading.asset_id, reading.sensor_id)
        constraint = self.settings.get_constraint(reading.asset_id, reading.sensor_id)

        current_val = reading.value
        current_ts = reading.timestamp
        if current_ts.tzinfo is None:
            current_ts = current_ts.replace(tzinfo=timezone.utc)

        # 1. Check Missing Data (Explicit null)
        if current_val is None:
            events.append(
                QualityDetectionEvent(
                    event_id=f"EVT-MISSING-{uuid.uuid4().hex[:8]}",
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    timestamp=current_ts,
                    issue_type="MISSING_DATA",
                    detection_rule="ExplicitNullRule",
                    observed_evidence={
                        "reason": "Explicit missing value (null) received in telemetry frame.",
                        "sequence_number": reading.sequence_number,
                    },
                    severity="HIGH",
                    reading_id=reading.reading_id,
                )
            )

        # 2. Check Sequence Skip / Communication Gap
        if state.last_timestamp is not None:
            dt = (current_ts - state.last_timestamp).total_seconds()
            nominal_interval = constraint.nominal_interval_seconds if constraint else self.settings.default_sampling_interval_seconds
            gap_threshold = nominal_interval * self.settings.communication_gap_multiplier

            # Sequence jump check
            if state.last_sequence is not None and reading.sequence_number > state.last_sequence + 1:
                skipped = reading.sequence_number - state.last_sequence - 1
                events.append(
                    QualityDetectionEvent(
                        event_id=f"EVT-SEQGAP-{uuid.uuid4().hex[:8]}",
                        asset_id=reading.asset_id,
                        sensor_id=reading.sensor_id,
                        timestamp=state.last_timestamp,
                        end_timestamp=current_ts,
                        issue_type="MISSING_DATA",
                        detection_rule="SequenceNumberSkipRule",
                        observed_evidence={
                            "last_sequence": state.last_sequence,
                            "current_sequence": reading.sequence_number,
                            "skipped_count": skipped,
                            "interval_seconds": dt,
                        },
                        severity="MEDIUM",
                        reading_id=reading.reading_id,
                    )
                )

            # Time gap check
            if dt > gap_threshold:
                events.append(
                    QualityDetectionEvent(
                        event_id=f"EVT-COMMGAP-{uuid.uuid4().hex[:8]}",
                        asset_id=reading.asset_id,
                        sensor_id=reading.sensor_id,
                        timestamp=state.last_timestamp,
                        end_timestamp=current_ts,
                        issue_type="COMMUNICATION_GAP",
                        detection_rule="SamplingIntervalExceededRule",
                        observed_evidence={
                            "elapsed_seconds": dt,
                            "nominal_interval": nominal_interval,
                            "threshold": gap_threshold,
                            "reason": f"Expected telemetry interval {nominal_interval}s exceeded ({dt:.2f}s).",
                        },
                        severity="HIGH",
                        reading_id=reading.reading_id,
                    )
                )

        # If current_val is valid numeric, evaluate value-dependent rules
        if current_val is not None:
            # 3. Check Physical Plausibility Constraints
            if constraint:
                if current_val < constraint.min_value or current_val > constraint.max_value:
                    events.append(
                        QualityDetectionEvent(
                            event_id=f"EVT-INVALID-{uuid.uuid4().hex[:8]}",
                            asset_id=reading.asset_id,
                            sensor_id=reading.sensor_id,
                            timestamp=current_ts,
                            issue_type="INVALID_CORRUPTED",
                            detection_rule="PhysicalPlausibilityRangeRule",
                            observed_evidence={
                                "observed_value": current_val,
                                "min_allowed": constraint.min_value,
                                "max_allowed": constraint.max_value,
                                "unit": constraint.unit,
                                "breach": "below_minimum" if current_val < constraint.min_value else "above_maximum",
                            },
                            severity="CRITICAL",
                            reading_id=reading.reading_id,
                        )
                    )

            # 4. Check Sudden Spike (Rate of Change & Statistical MAD)
            if state.last_valid_value is not None and state.last_timestamp is not None:
                dt_step = max(0.001, (current_ts - state.last_timestamp).total_seconds())
                delta = abs(current_val - state.last_valid_value)
                rate_of_change = delta / dt_step

                max_roc = constraint.max_rate_of_change if constraint else 10.0
                if rate_of_change > max_roc:
                    events.append(
                        QualityDetectionEvent(
                            event_id=f"EVT-SPIKE-{uuid.uuid4().hex[:8]}",
                            asset_id=reading.asset_id,
                            sensor_id=reading.sensor_id,
                            timestamp=current_ts,
                            issue_type="SUDDEN_SPIKE",
                            detection_rule="RateOfChangeSpikeRule",
                            observed_evidence={
                                "observed_value": current_val,
                                "previous_value": state.last_valid_value,
                                "rate_of_change": round(rate_of_change, 3),
                                "max_allowed_roc": max_roc,
                            },
                            severity="HIGH",
                            reading_id=reading.reading_id,
                        )
                    )

            # 5. Check Frozen / Stale Sensor
            if state.last_valid_value is not None:
                if math.isclose(current_val, state.last_valid_value, abs_tol=1e-6):
                    state.consecutive_identical_count += 1
                else:
                    state.consecutive_identical_count = 1

                if state.consecutive_identical_count >= self.settings.frozen_consecutive_count:
                    events.append(
                        QualityDetectionEvent(
                            event_id=f"EVT-FROZEN-{uuid.uuid4().hex[:8]}",
                            asset_id=reading.asset_id,
                            sensor_id=reading.sensor_id,
                            timestamp=current_ts,
                            issue_type="FROZEN_SENSOR",
                            detection_rule="ConsecutiveIdenticalValueRule",
                            observed_evidence={
                                "repeated_value": current_val,
                                "consecutive_count": state.consecutive_identical_count,
                                "threshold": self.settings.frozen_consecutive_count,
                                "warning": "Sensor repeatedly emitting identical float value; potential ADC freeze or stalled line.",
                            },
                            severity="MEDIUM",
                            reading_id=reading.reading_id,
                        )
                    )

            # 6. Check Gradual Drift (Page's CUSUM with moving baseline)
            valid_hist = [v for _, v in state.history if v is not None]
            if len(valid_hist) >= 10:
                if state.baseline_mean is None:
                    state.baseline_mean = float(np.mean(valid_hist[:10]))

                ref_mean = state.baseline_mean
                k_slack = self.settings.drift_cusum_slack
                h_thresh = self.settings.drift_cusum_threshold

                # CUSUM accumulation
                state.cusum_pos = max(0.0, state.cusum_pos + (current_val - ref_mean - k_slack))
                state.cusum_neg = max(0.0, state.cusum_neg - (current_val - ref_mean + k_slack))

                if state.cusum_pos > h_thresh or state.cusum_neg > h_thresh:
                    drift_dir = "positive_drift" if state.cusum_pos > h_thresh else "negative_drift"
                    score = max(state.cusum_pos, state.cusum_neg)
                    events.append(
                        QualityDetectionEvent(
                            event_id=f"EVT-DRIFT-{uuid.uuid4().hex[:8]}",
                            asset_id=reading.asset_id,
                            sensor_id=reading.sensor_id,
                            timestamp=current_ts,
                            issue_type="GRADUAL_DRIFT",
                            detection_rule="CUSUMCumulativeDriftRule",
                            observed_evidence={
                                "observed_value": current_val,
                                "reference_baseline": round(ref_mean, 3),
                                "cusum_score": round(score, 3),
                                "threshold": h_thresh,
                                "direction": drift_dir,
                            },
                            severity="MEDIUM",
                            reading_id=reading.reading_id,
                        )
                    )
            else:
                if state.baseline_mean is None:
                    state.baseline_mean = current_val

            # Update last valid value
            state.last_valid_value = current_val

        # Update channel state
        state.last_timestamp = current_ts
        state.last_sequence = reading.sequence_number
        state.history.append((current_ts, current_val))

        return events
