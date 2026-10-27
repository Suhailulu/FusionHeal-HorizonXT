from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.storage.models import RawReading, QualityEvent, AcceptedReading, AuditLog, EvaluationMetricRun


class TelemetryRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_raw_reading(self, reading: RawReading) -> RawReading:
        self.db.add(reading)
        self.db.commit()
        self.db.refresh(reading)
        return reading

    def get_raw_reading_by_id(self, reading_id: str) -> Optional[RawReading]:
        return self.db.query(RawReading).filter(RawReading.reading_id == reading_id).first()

    def get_readings(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[RawReading]:
        query = self.db.query(RawReading)
        if asset_id:
            query = query.filter(RawReading.asset_id == asset_id)
        if sensor_id:
            query = query.filter(RawReading.sensor_id == sensor_id)
        if start_time:
            query = query.filter(RawReading.timestamp >= start_time)
        if end_time:
            query = query.filter(RawReading.timestamp <= end_time)
        return query.order_by(RawReading.timestamp.asc()).offset(offset).limit(limit).all()

    def save_quality_event(self, event: QualityEvent) -> QualityEvent:
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def get_events(
        self,
        issue_type: Optional[str] = None,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        decision_status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QualityEvent]:
        query = self.db.query(QualityEvent)
        if issue_type:
            query = query.filter(QualityEvent.issue_type == issue_type)
        if asset_id:
            query = query.filter(QualityEvent.asset_id == asset_id)
        if sensor_id:
            query = query.filter(QualityEvent.sensor_id == sensor_id)
        if decision_status:
            query = query.filter(QualityEvent.decision_status == decision_status)
        return query.order_by(QualityEvent.timestamp.desc()).offset(offset).limit(limit).all()

    def get_event_by_id(self, event_id: str) -> Optional[QualityEvent]:
        return self.db.query(QualityEvent).filter(QualityEvent.event_id == event_id).first()

    def save_accepted_reading(self, reading: AcceptedReading) -> AcceptedReading:
        self.db.add(reading)
        self.db.commit()
        self.db.refresh(reading)
        return reading

    def get_accepted_readings(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AcceptedReading]:
        query = self.db.query(AcceptedReading)
        if asset_id:
            query = query.filter(AcceptedReading.asset_id == asset_id)
        if sensor_id:
            query = query.filter(AcceptedReading.sensor_id == sensor_id)
        return query.order_by(AcceptedReading.timestamp.asc()).offset(offset).limit(limit).all()

    def get_latest_audit_hash(self) -> Optional[str]:
        latest = self.db.query(AuditLog.record_hash).order_by(AuditLog.id.desc()).first()
        return latest[0] if latest else None

    def save_audit_log(self, audit: AuditLog) -> AuditLog:
        self.db.add(audit)
        self.db.commit()
        self.db.refresh(audit)
        return audit

    def get_audit_logs(
        self,
        asset_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        decision_status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        query = self.db.query(AuditLog)
        if asset_id:
            query = query.filter(AuditLog.asset_id == asset_id)
        if sensor_id:
            query = query.filter(AuditLog.sensor_id == sensor_id)
        if decision_status:
            query = query.filter(AuditLog.decision_status == decision_status)
        return query.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()

    def get_overview_counts(self) -> Dict[str, Any]:
        """Compute exact aggregate counts across telemetry and decision records.
        Strictly prevents double counting.
        """
        total_readings = self.db.query(func.count(RawReading.id)).scalar() or 0
        total_issues = self.db.query(func.count(QualityEvent.id)).scalar() or 0

        # AcceptedReading carries exactly one primary decision per reading
        preserved = (
            self.db.query(func.count(AcceptedReading.id))
            .filter(AcceptedReading.status == "PRESERVE")
            .scalar() or 0
        )
        repaired = (
            self.db.query(func.count(AcceptedReading.id))
            .filter(AcceptedReading.status == "REPAIR")
            .scalar() or 0
        )
        unresolved = (
            self.db.query(func.count(AcceptedReading.id))
            .filter(AcceptedReading.status == "UNRESOLVED")
            .scalar() or 0
        )

        # Issue type distribution
        issue_counts = (
            self.db.query(QualityEvent.issue_type, func.count(QualityEvent.id))
            .group_by(QualityEvent.issue_type)
            .all()
        )
        fault_distribution = {row[0]: row[1] for row in issue_counts}

        return {
            "total_readings_received": total_readings,
            "readings_preserved": preserved,
            "repairs_accepted": repaired,
            "unresolved_events": unresolved,
            "issues_detected": total_issues,
            "fault_distribution": fault_distribution,
        }

    def save_metric_run(self, metric: EvaluationMetricRun) -> EvaluationMetricRun:
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        return metric

    def get_latest_metrics(self) -> Optional[EvaluationMetricRun]:
        return self.db.query(EvaluationMetricRun).order_by(EvaluationMetricRun.id.desc()).first()
