from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, Text, Index, ForeignKey
)
from app.storage.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RawReading(Base):
    __tablename__ = "raw_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reading_id = Column(String(64), unique=True, index=True, nullable=False)
    asset_id = Column(String(64), index=True, nullable=False)
    sensor_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    ingested_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    sequence_number = Column(Integer, nullable=False)
    value = Column(Float, nullable=True)  # explicit None indicates missing measurement
    is_missing = Column(Boolean, default=False, nullable=False)
    unit = Column(String(16), nullable=False)
    source_type = Column(String(32), default="simulated", nullable=False)
    raw_payload = Column(Text, nullable=False)
    validation_status = Column(String(32), default="VALID", nullable=False)
    validation_error = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_raw_asset_sensor_time", "asset_id", "sensor_id", "timestamp"),
    )


class QualityEvent(Base):
    __tablename__ = "quality_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)
    reading_id = Column(String(64), nullable=True, index=True)
    asset_id = Column(String(64), index=True, nullable=False)
    sensor_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    end_timestamp = Column(DateTime(timezone=True), nullable=True)
    issue_type = Column(String(48), index=True, nullable=False)
    detection_rule = Column(String(64), nullable=False)
    observed_evidence = Column(Text, nullable=False)  # JSON formatted details
    severity = Column(String(16), default="MEDIUM", nullable=False)
    decision_status = Column(String(16), default="UNRESOLVED", index=True, nullable=False)
    detected_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        Index("idx_evt_asset_sensor_time", "asset_id", "sensor_id", "timestamp"),
    )


class AcceptedReading(Base):
    __tablename__ = "accepted_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reading_id = Column(String(64), index=True, nullable=False)
    asset_id = Column(String(64), index=True, nullable=False)
    sensor_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    original_value = Column(Float, nullable=True)
    accepted_value = Column(Float, nullable=True)
    status = Column(String(16), index=True, nullable=False)  # PRESERVE, REPAIR, UNRESOLVED
    repair_method = Column(String(64), nullable=True)
    processed_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        Index("idx_accepted_asset_sensor_time", "asset_id", "sensor_id", "timestamp"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(String(64), unique=True, index=True, nullable=False)
    reading_id = Column(String(64), nullable=True, index=True)
    event_id = Column(String(64), nullable=True, index=True)
    asset_id = Column(String(64), index=True, nullable=False)
    sensor_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    original_value = Column(Float, nullable=True)
    candidate_value = Column(Float, nullable=True)
    accepted_value = Column(Float, nullable=True)
    decision_status = Column(String(16), index=True, nullable=False)
    validation_result = Column(String(32), nullable=False)
    decision_reason = Column(Text, nullable=False)
    model_version = Column(String(32), nullable=True)
    rule_config_version = Column(String(32), nullable=False)
    config_hash = Column(String(64), nullable=False)
    record_hash = Column(String(64), nullable=False)
    prev_record_hash = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class EvaluationMetricRun(Base):
    __tablename__ = "evaluation_metric_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    scenario_name = Column(String(64), nullable=False)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    accepted_repairs = Column(Integer, default=0, nullable=False)
    unresolved_events = Column(Integer, default=0, nullable=False)
    unintentional_modifications = Column(Integer, default=0, nullable=False)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
