from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from app.ingestion.schemas import TelemetryReadingInput


class BaseTelemetryAdapter(ABC):
    """Abstract interface for all hardware and software telemetry ingress adapters.

    Every adapter—whether reading from a software simulator, an ESP32 microcontroller over HTTP,
    an industrial MQTT broker, or an OPC UA server—must normalize raw source data
    into the canonical TelemetryReadingInput schema before handing it to the pipeline.
    """

    @abstractmethod
    def normalize(self, raw_data: Any) -> TelemetryReadingInput:
        """Converts raw source-specific payload into canonical TelemetryReadingInput."""
        raise NotImplementedError


class SimulatorAdapter(BaseTelemetryAdapter):
    """Adapter for the synthetic simulator telemetry generator."""

    def normalize(self, raw_data: Dict[str, Any]) -> TelemetryReadingInput:
        data = dict(raw_data)
        if "source_type" not in data:
            data["source_type"] = "simulated"
        return TelemetryReadingInput.model_validate(data)


class HttpSensorAdapter(BaseTelemetryAdapter):
    """Adapter for physical or gateway HTTP ingest (e.g. ESP32, Raspberry Pi, HTTP edge sensor)."""

    def normalize(self, raw_data: Dict[str, Any]) -> TelemetryReadingInput:
        data = dict(raw_data)
        if "source_type" not in data:
            data["source_type"] = "physical"
        return TelemetryReadingInput.model_validate(data)


class MqttSensorAdapter(BaseTelemetryAdapter):
    """Adapter for industrial MQTT message payloads.

    Example MQTT Topic: 'factory/asset/boiler_01/sensor/temperature_01'
    """

    def normalize(self, raw_data: Dict[str, Any]) -> TelemetryReadingInput:
        # Normalize MQTT message dictionary with topic or payload keys
        asset_id = raw_data.get("asset_id") or raw_data.get("asset") or "unknown_asset"
        sensor_id = raw_data.get("sensor_id") or raw_data.get("sensor") or "unknown_sensor"
        val = raw_data.get("value") or raw_data.get("val")

        return TelemetryReadingInput(
            reading_id=str(raw_data.get("reading_id") or raw_data.get("msg_id")),
            asset_id=str(asset_id),
            sensor_id=str(sensor_id),
            event_timestamp=raw_data.get("event_timestamp") or raw_data.get("timestamp") or datetime.now(timezone.utc),
            sequence_number=int(raw_data.get("sequence_number") or raw_data.get("seq", 0)),
            value=float(val) if val is not None else None,
            unit=str(raw_data.get("unit", "unknown")),
            source_type=str(raw_data.get("source_type", "physical")),
        )


class OpcUaSensorAdapter(BaseTelemetryAdapter):
    """Adapter for industrial OPC UA Node attribute readouts."""

    def normalize(self, raw_data: Dict[str, Any]) -> TelemetryReadingInput:
        val = raw_data.get("Value") or raw_data.get("value")
        return TelemetryReadingInput(
            reading_id=str(raw_data.get("reading_id") or raw_data.get("NodeId")),
            asset_id=str(raw_data.get("asset_id", "opcua_asset")),
            sensor_id=str(raw_data.get("sensor_id") or raw_data.get("DisplayName", "opcua_sensor")),
            event_timestamp=raw_data.get("SourceTimestamp") or raw_data.get("event_timestamp") or datetime.now(timezone.utc),
            sequence_number=int(raw_data.get("sequence_number", 0)),
            value=float(val) if val is not None else None,
            unit=str(raw_data.get("unit", "C")),
            source_type=str(raw_data.get("source_type", "physical")),
        )
