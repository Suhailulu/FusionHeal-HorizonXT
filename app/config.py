from __future__ import annotations
import os
import json
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Load from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _parse_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _parse_list(val: Optional[str]) -> List[str]:
    if not val:
        return []
    return [item.strip() for item in val.split(",") if item.strip()]


class SensorConstraint(BaseModel):
    sensor_id: str
    asset_id: str
    unit: str
    min_value: float
    max_value: float
    max_rate_of_change: float  # max absolute delta per second
    nominal_interval_seconds: float = 1.0
    allowed_lateness_seconds: float = 5.0


class Settings(BaseModel):
    app_name: str = "FusionHeal"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./fusionheal.db"
    host: str = "127.0.0.1"
    port: int = 8000

    # Timing & detection tolerances
    default_sampling_interval_seconds: float = 1.0
    communication_gap_multiplier: float = 2.5
    allowed_lateness_seconds: float = 5.0
    frozen_consecutive_count: int = 5
    spike_mad_threshold: float = 5.0
    drift_cusum_slack: float = 0.5
    drift_cusum_threshold: float = 5.0
    allowed_clock_skew_seconds: float = 10.0
    enforce_wall_clock_sync: bool = False


    # Policy and Governance
    policy_version: str = "1.0.0-horizonxt"
    rule_config_version: str = "2026.1"
    auto_accept_repairs: bool = True
    min_auto_accept_confidence: float = 0.80  # candidates with confidence >= 0.80 can be auto-accepted
    min_review_confidence: float = 0.40       # candidates with 0.40 <= confidence < 0.80 routed to REVIEW
    manual_review_sensors: List[str] = Field(default_factory=list)  # specific sensors requiring mandatory human review

    # External ML Model Configuration (Weights stored outside Git)
    transformer_model_path: str = "models/saits_spatial_temporal.onnx"
    transformer_device: str = "cpu"

    # Cloud & Supabase Integration (Optional credentials, never committed)
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None

    # Configured Asset/Sensor constraints (Boiler & Pump as required)
    sensor_constraints: Dict[str, SensorConstraint] = Field(default_factory=lambda: {
        "boiler_01:temperature_01": SensorConstraint(
            asset_id="boiler_01",
            sensor_id="temperature_01",
            unit="C",
            min_value=0.0,
            max_value=150.0,
            max_rate_of_change=5.0,
            nominal_interval_seconds=1.0,
        ),
        "pump_01:vibration_01": SensorConstraint(
            asset_id="pump_01",
            sensor_id="vibration_01",
            unit="m/s2",
            min_value=0.0,
            max_value=25.0,
            max_rate_of_change=8.0,
            nominal_interval_seconds=1.0,
        ),
    })

    def get_constraint(self, asset_id: str, sensor_id: str) -> SensorConstraint | None:
        key = f"{asset_id}:{sensor_id}"
        return self.sensor_constraints.get(key)

    def compute_config_hash(self) -> str:
        """Deterministic hash of active configuration for audit and governance."""
        data = {
            "policy_version": self.policy_version,
            "rule_config_version": self.rule_config_version,
            "sampling_interval": self.default_sampling_interval_seconds,
            "gap_multiplier": self.communication_gap_multiplier,
            "frozen_count": self.frozen_consecutive_count,
            "spike_mad": self.spike_mad_threshold,
            "drift_cusum": self.drift_cusum_threshold,
            "auto_accept_repairs": self.auto_accept_repairs,
            "min_auto_accept_confidence": self.min_auto_accept_confidence,
            "min_review_confidence": self.min_review_confidence,
            "manual_review_sensors": self.manual_review_sensors,
            "constraints": {k: v.model_dump() for k, v in self.sensor_constraints.items()},
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "FusionHeal"),
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./fusionheal.db"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        default_sampling_interval_seconds=float(os.getenv("DEFAULT_SAMPLING_INTERVAL_SECONDS", "1.0")),
        communication_gap_multiplier=float(os.getenv("COMMUNICATION_GAP_MULTIPLIER", "2.5")),
        allowed_clock_skew_seconds=float(os.getenv("ALLOWED_CLOCK_SKEW_SECONDS", "10.0")),
        enforce_wall_clock_sync=_parse_bool(os.getenv("ENFORCE_WALL_CLOCK_SYNC", "false")),
        frozen_consecutive_count=int(os.getenv("FROZEN_CONSECUTIVE_COUNT", "5")),
        spike_mad_threshold=float(os.getenv("SPIKE_MAD_THRESHOLD", "5.0")),
        drift_cusum_slack=float(os.getenv("DRIFT_CUSUM_SLACK", "0.5")),
        drift_cusum_threshold=float(os.getenv("DRIFT_CUSUM_THRESHOLD", "5.0")),
        policy_version=os.getenv("POLICY_VERSION", "1.0.0-horizonxt"),
        rule_config_version=os.getenv("RULE_CONFIG_VERSION", "2026.1"),
        auto_accept_repairs=_parse_bool(os.getenv("AUTO_ACCEPT_REPAIRS", "true")),
        min_auto_accept_confidence=float(os.getenv("MIN_AUTO_ACCEPT_CONFIDENCE", "0.80")),
        min_review_confidence=float(os.getenv("MIN_REVIEW_CONFIDENCE", "0.40")),
        manual_review_sensors=_parse_list(os.getenv("MANUAL_REVIEW_SENSORS", "")),
        transformer_model_path=os.getenv("TRANSFORMER_MODEL_PATH", "models/saits_spatial_temporal.onnx"),
        transformer_device=os.getenv("TRANSFORMER_DEVICE", "cpu"),
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY"),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    )
