from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from collections import deque
import uuid
import math
import numpy as np

from app.config import Settings, SensorConstraint
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
        event_timestamp: Optional[datetime] = None,
    ):
        self.event_id = event_id
        self.asset_id = asset_id
        self.sensor_id = sensor_id
        ts = event_timestamp or timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        self.event_timestamp = ts
        self.timestamp = ts
        if end_timestamp is not None:
            if end_timestamp.tzinfo is None:
                end_timestamp = end_timestamp.replace(tzinfo=timezone.utc)
            else:
                end_timestamp = end_timestamp.astimezone(timezone.utc)
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
            "event_timestamp": self.event_timestamp.isoformat(),
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

    # -------------------------------------------------------------------------
    # Individual Detection Functions (A through F)
    # -------------------------------------------------------------------------

    def detect_missing_value(
        self, reading: TelemetryReadingInput, current_ts: datetime
    ) -> Optional[QualityDetectionEvent]:
        """A1. Detect explicit null measurement."""
        if reading.value is None:
            return QualityDetectionEvent(
                event_id=f"EVT-MISSING-{uuid.uuid4().hex[:8]}",
                asset_id=reading.asset_id,
                sensor_id=reading.sensor_id,
                timestamp=current_ts,
                event_timestamp=current_ts,
                issue_type="MISSING_DATA",
                detection_rule="ExplicitNullRule",
                observed_evidence={
                    "reason": "Explicit missing value (null) received in telemetry frame.",
                    "sequence_number": reading.sequence_number,
                },
                severity="HIGH",
                reading_id=reading.reading_id,
            )
        return None

    def detect_sequence_gap(
        self,
        reading: TelemetryReadingInput,
        state: SensorChannelState,
        current_ts: datetime,
    ) -> Optional[QualityDetectionEvent]:
        """A2. Detect missing messages via sequence number jump."""
        if state.last_timestamp is not None and state.last_sequence is not None:
            if reading.sequence_number > state.last_sequence + 1:
                skipped = reading.sequence_number - state.last_sequence - 1
                dt = (current_ts - state.last_timestamp).total_seconds()
                return QualityDetectionEvent(
                    event_id=f"EVT-SEQGAP-{uuid.uuid4().hex[:8]}",
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    timestamp=state.last_timestamp,
                    event_timestamp=state.last_timestamp,
                    end_timestamp=current_ts,
                    issue_type="MISSING_DATA",
                    detection_rule="SequenceNumberSkipRule",
                    observed_evidence={
                        "last_sequence": state.last_sequence,
                        "current_sequence": reading.sequence_number,
                        "skipped_count": skipped,
                        "interval_seconds": dt,
                        "reason": f"Telemetric sequence jump detected: {skipped} intermediate frame(s) missing.",
                    },
                    severity="MEDIUM",
                    reading_id=reading.reading_id,
                )
        return None

    def detect_communication_gap(
        self,
        reading: TelemetryReadingInput,
        state: SensorChannelState,
        current_ts: datetime,
        constraint: Optional[SensorConstraint] = None,
    ) -> Optional[QualityDetectionEvent]:
        """E. Detect missing telemetry according to sensor schedule + lateness tolerance."""
        if state.last_timestamp is not None:
            dt = (current_ts - state.last_timestamp).total_seconds()
            nominal = constraint.nominal_interval_seconds if constraint else self.settings.default_sampling_interval_seconds
            gap_threshold = nominal * self.settings.communication_gap_multiplier

            if dt > gap_threshold:
                return QualityDetectionEvent(
                    event_id=f"EVT-COMMGAP-{uuid.uuid4().hex[:8]}",
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    timestamp=state.last_timestamp,
                    event_timestamp=state.last_timestamp,
                    end_timestamp=current_ts,
                    issue_type="COMMUNICATION_GAP",
                    detection_rule="SamplingIntervalExceededRule",
                    observed_evidence={
                        "elapsed_seconds": round(dt, 3),
                        "nominal_interval": nominal,
                        "threshold": round(gap_threshold, 3),
                        "reason": f"Expected telemetry interval {nominal}s exceeded ({dt:.2f}s). Potential network outage.",
                    },
                    severity="HIGH",
                    reading_id=reading.reading_id,
                )
        return None

    def detect_physical_range_violation(
        self,
        reading: TelemetryReadingInput,
        current_val: float,
        current_ts: datetime,
        constraint: Optional[SensorConstraint] = None,
    ) -> Optional[QualityDetectionEvent]:
        """F. Detect gross physical constraint breaches / corrupted signal."""
        if constraint:
            if current_val < constraint.min_value or current_val > constraint.max_value:
                return QualityDetectionEvent(
                    event_id=f"EVT-INVALID-{uuid.uuid4().hex[:8]}",
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    timestamp=current_ts,
                    event_timestamp=current_ts,
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
        return None

    def detect_sudden_spike(
        self,
        reading: TelemetryReadingInput,
        state: SensorChannelState,
        current_val: float,
        current_ts: datetime,
        constraint: Optional[SensorConstraint] = None,
    ) -> Optional[QualityDetectionEvent]:
        """C. Detect sudden rate-of-change spikes (treated as suspicious, not automatically invalid)."""
        if state.last_valid_value is not None and state.last_timestamp is not None:
            dt_step = max(0.001, (current_ts - state.last_timestamp).total_seconds())
            delta = abs(current_val - state.last_valid_value)
            rate_of_change = delta / dt_step

            max_roc = constraint.max_rate_of_change if constraint else 10.0
            if rate_of_change > max_roc:
                return QualityDetectionEvent(
                    event_id=f"EVT-SPIKE-{uuid.uuid4().hex[:8]}",
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    timestamp=current_ts,
                    event_timestamp=current_ts,
                    issue_type="SUDDEN_SPIKE",
                    detection_rule="RateOfChangeSpikeRule",
                    observed_evidence={
                        "observed_value": current_val,
                        "previous_value": state.last_valid_value,
                        "rate_of_change": round(rate_of_change, 3),
                        "max_allowed_roc": max_roc,
                        "warning": "Sudden rate-of-change surge. Suspicious transient event.",
                    },
                    severity="HIGH",
                    reading_id=reading.reading_id,
                )
        return None

    def detect_frozen_sensor(
        self,
        reading: TelemetryReadingInput,
        state: SensorChannelState,
        current_val: float,
        current_ts: datetime,
    ) -> Optional[QualityDetectionEvent]:
        """B. Detect repeated readings / insufficient variation across configured window."""
        if state.last_valid_value is not None:
            if math.isclose(current_val, state.last_valid_value, abs_tol=1e-6):
                state.consecutive_identical_count += 1
            else:
                state.consecutive_identical_count = 1

            if state.consecutive_identical_count >= self.settings.frozen_consecutive_count:
                return QualityDetectionEvent(
                    event_id=f"EVT-FROZEN-{uuid.uuid4().hex[:8]}",
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    timestamp=current_ts,
                    event_timestamp=current_ts,
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
        return None

    def detect_gradual_drift(
        self,
        reading: TelemetryReadingInput,
        state: SensorChannelState,
        current_val: float,
        current_ts: datetime,
    ) -> Optional[QualityDetectionEvent]:
        """D. Detect cumulative drift using statistical Page's CUSUM rule."""
        valid_hist = [v for _, v in state.history if v is not None]
        if len(valid_hist) >= 10:
            if state.baseline_mean is None:
                state.baseline_mean = float(np.mean(valid_hist[:10]))

            ref_mean = state.baseline_mean
            k_slack = self.settings.drift_cusum_slack
            h_thresh = self.settings.drift_cusum_threshold

            state.cusum_pos = max(0.0, state.cusum_pos + (current_val - ref_mean - k_slack))
            state.cusum_neg = max(0.0, state.cusum_neg - (current_val - ref_mean + k_slack))

            if state.cusum_pos > h_thresh or state.cusum_neg > h_thresh:
                drift_dir = "positive_drift" if state.cusum_pos > h_thresh else "negative_drift"
                score = max(state.cusum_pos, state.cusum_neg)
                return QualityDetectionEvent(
                    event_id=f"EVT-DRIFT-{uuid.uuid4().hex[:8]}",
                    asset_id=reading.asset_id,
                    sensor_id=reading.sensor_id,
                    timestamp=current_ts,
                    event_timestamp=current_ts,
                    issue_type="GRADUAL_DRIFT",
                    detection_rule="CUSUMCumulativeDriftRule",
                    observed_evidence={
                        "observed_value": current_val,
                        "reference_baseline": round(ref_mean, 3),
                        "cusum_score": round(score, 3),
                        "threshold": h_thresh,
                        "direction": drift_dir,
                        "note": "Statistical cumulative sum drift detected relative to baseline.",
                    },
                    severity="MEDIUM",
                    reading_id=reading.reading_id,
                )
        else:
            if state.baseline_mean is None:
                state.baseline_mean = current_val
        return None

    def detect_issues(self, reading: TelemetryReadingInput) -> List[QualityDetectionEvent]:
        """Orchestrate all individual detection checks across the telemetry observation."""
        events: List[QualityDetectionEvent] = []
        state = self.get_channel_state(reading.asset_id, reading.sensor_id)
        constraint = self.settings.get_constraint(reading.asset_id, reading.sensor_id)

        current_val = reading.value
        current_ts = reading.event_timestamp or reading.timestamp
        if current_ts.tzinfo is None:
            current_ts = current_ts.replace(tzinfo=timezone.utc)
        else:
            current_ts = current_ts.astimezone(timezone.utc)

        # 1. Missing Data (explicit null)
        evt_missing = self.detect_missing_value(reading, current_ts)
        if evt_missing:
            events.append(evt_missing)

        # 2. Sequence Skip (missing frames)
        evt_seq = self.detect_sequence_gap(reading, state, current_ts)
        if evt_seq:
            events.append(evt_seq)

        # 3. Communication Gap (sampling schedule exceeded)
        evt_gap = self.detect_communication_gap(reading, state, current_ts, constraint)
        if evt_gap:
            events.append(evt_gap)

        # Value-dependent checks for non-null readings
        if current_val is not None:
            # 4. Physical Plausibility Limits
            evt_range = self.detect_physical_range_violation(reading, current_val, current_ts, constraint)
            if evt_range:
                events.append(evt_range)

            # 5. Sudden Spike
            evt_spike = self.detect_sudden_spike(reading, state, current_val, current_ts, constraint)
            if evt_spike:
                events.append(evt_spike)

            # 6. Frozen / Stale Sensor
            evt_frozen = self.detect_frozen_sensor(reading, state, current_val, current_ts)
            if evt_frozen:
                events.append(evt_frozen)

            # 7. Gradual Drift
            evt_drift = self.detect_gradual_drift(reading, state, current_val, current_ts)
            if evt_drift:
                events.append(evt_drift)

            state.last_valid_value = current_val

        # Update channel state
        state.last_timestamp = current_ts
        state.last_sequence = reading.sequence_number
        state.history.append((current_ts, current_val))

        return events
