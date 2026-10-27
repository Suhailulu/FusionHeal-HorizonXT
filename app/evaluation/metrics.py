from __future__ import annotations
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

from app.storage.models import EvaluationMetricRun
from app.storage.repository import TelemetryRepository
from app.simulator.fault_injector import GroundTruthFaultRecord


class BenchmarkReport:
    def __init__(
        self,
        run_id: str,
        scenario_name: str,
        total_eval_points: int,
        tp: int,
        fp: int,
        fn: int,
        precision: Optional[float],
        recall: Optional[float],
        f1_score: Optional[float],
        mae: Optional[float],
        rmse: Optional[float],
        latency_ms: float,
        accepted_repairs: int,
        unresolved_events: int,
        unintentional_modifications: int,
        details: Dict[str, Any],
    ):
        self.run_id = run_id
        self.scenario_name = scenario_name
        self.total_eval_points = total_eval_points
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.mae = mae
        self.rmse = rmse
        self.latency_ms = latency_ms
        self.accepted_repairs = accepted_repairs
        self.unresolved_events = unresolved_events
        self.unintentional_modifications = unintentional_modifications
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "total_eval_points": self.total_eval_points,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mae": self.mae,
            "rmse": self.rmse,
            "latency_ms": round(self.latency_ms, 2),
            "accepted_repairs": self.accepted_repairs,
            "unresolved_events": self.unresolved_events,
            "unintentional_modifications": self.unintentional_modifications,
            "details": self.details,
        }


class EvaluationSuite:
    """Rigorous evaluation suite comparing pipeline decisions and candidate repairs against ground truth."""

    @staticmethod
    def evaluate(
        scenario_name: str,
        ground_truth_records: List[GroundTruthFaultRecord],
        processed_results: List[Dict[str, Any]],
        start_time_perf: float,
        end_time_perf: float,
        repo: Optional[TelemetryRepository] = None,
    ) -> BenchmarkReport:
        run_id = f"EVAL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        total_points = len(processed_results)
        latency_ms = (end_time_perf - start_time_perf) * 1000.0

        # Build lookup of injected faults by (asset_id, sensor_id, reading_id)
        fault_by_id = {f.reading_id: f for f in ground_truth_records if f.reading_id}

        tp = 0
        fp = 0
        fn = 0
        unintentional_modifications = 0
        accepted_repairs = 0
        unresolved_events = 0

        abs_errors: List[float] = []
        sq_errors: List[float] = []

        for res in processed_results:
            rid = res.get("reading_id")
            status = res.get("status")
            issues = res.get("issues_detected", [])
            final_val = res.get("final_value")
            orig_val = res.get("original_value")

            if status == "REPAIR":
                accepted_repairs += 1
            elif status == "UNRESOLVED":
                unresolved_events += 1

            was_injected = rid in fault_by_id
            fault_record = fault_by_id.get(rid)

            has_detection = len(issues) > 0 or status in ("REPAIR", "UNRESOLVED")

            if was_injected:
                if has_detection:
                    tp += 1
                else:
                    fn += 1

                # If repair was accepted, compute reconstruction accuracy against clean ground truth
                if status == "REPAIR" and final_val is not None and fault_record:
                    clean_true = fault_record.clean_ground_truth_value
                    diff = abs(final_val - clean_true)
                    abs_errors.append(diff)
                    sq_errors.append(diff ** 2)
            else:
                # Was a clean observation
                if has_detection:
                    fp += 1

                # Check if clean data was unintentionally modified
                if status == "REPAIR" and orig_val is not None and final_val is not None:
                    if not math.isclose(orig_val, final_val, abs_tol=1e-5):
                        unintentional_modifications += 1

        # Calculate Precision, Recall, F1
        denom_prec = tp + fp
        precision = round(tp / denom_prec, 4) if denom_prec > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)

        denom_rec = tp + fn
        recall = round(tp / denom_rec, 4) if denom_rec > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)

        if precision + recall > 0:
            f1 = round(2 * (precision * recall) / (precision + recall), 4)
        else:
            f1 = 0.0 if denom_rec > 0 else None

        # MAE & RMSE on repaired ground-truth points
        mae = round(float(np.mean(abs_errors)), 4) if abs_errors else None
        rmse = round(float(np.sqrt(np.mean(sq_errors))), 4) if sq_errors else None

        report = BenchmarkReport(
            run_id=run_id,
            scenario_name=scenario_name,
            total_eval_points=total_points,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mae=mae,
            rmse=rmse,
            latency_ms=latency_ms,
            accepted_repairs=accepted_repairs,
            unresolved_events=unresolved_events,
            unintentional_modifications=unintentional_modifications,
            details={
                "injected_faults_count": len(ground_truth_records),
                "reconstruction_eval_samples": len(abs_errors),
                "average_latency_per_record_ms": round(latency_ms / max(1, total_points), 3),
            },
        )

        # Persist to database if repository provided
        if repo is not None:
            db_metric = EvaluationMetricRun(
                run_id=run_id,
                scenario_name=scenario_name,
                precision=precision,
                recall=recall,
                f1_score=f1,
                mae=mae,
                rmse=rmse,
                latency_ms=latency_ms,
                accepted_repairs=accepted_repairs,
                unresolved_events=unresolved_events,
                unintentional_modifications=unintentional_modifications,
                details_json=str(report.details),
            )
            repo.save_metric_run(db_metric)

        return report
