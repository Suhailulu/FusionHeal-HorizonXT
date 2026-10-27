from __future__ import annotations
import math
import time
import uuid
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np

from app.storage.models import EvaluationMetricRun
from app.storage.repository import TelemetryRepository
from app.simulator.fault_injector import GroundTruthFaultRecord, FaultInjector
from app.simulator.generator import SyntheticSensorGenerator
from app.pipeline.reconstruction import ReconstructionModel, ModelPrediction, PlaceholderReconstructionAdapter


class DeterministicBenchmarkModel(ReconstructionModel):
    """Deterministic, transparent reconstruction model for reproducible evaluation.

    CRITICAL ARCHITECTURE & DATA-INTEGRITY RULES:
    1. ZERO access to hidden ground-truth labels. Operates exclusively on historical
       sensor observation windows provided by the pipeline.
    2. Exercises the 4-status decision engine with ordinary and ambiguous cases:
       - Standard missing value: computes moving-average candidate with high confidence (0.92) -> REPAIR
       - Ambiguous reading / soft warning: assigns borderline confidence (0.65) -> REVIEW
       - Extreme spike anomaly: proposes candidate violating physical bounds -> REJECTED -> UNRESOLVED
    """

    def __init__(self, model_name: str = "DeterministicBenchmarkModel", version: str = "1.0.0-bench"):
        self.model_name = model_name
        self.version = version

    @property
    def model_version(self) -> str:
        return self.version

    def predict(
        self,
        asset_id: str,
        sensor_id: str,
        sensor_window: List[Dict[str, Any]],
        target_timestamp: Optional[datetime] = None,
    ) -> ModelPrediction:
        valid_vals = [pt["value"] for pt in sensor_window if pt.get("value") is not None]
        base_estimate = float(np.mean(valid_vals[-5:])) if valid_vals else (72.0 if "boiler" in asset_id else 2.8)

        # 1. Candidate Rejection / Constraint Violation condition -> evaluates UNRESOLVED
        # Periodically propose an out-of-bounds candidate (210 °C for boiler, 35 m/s² for pump)
        # to test that gateway hard constraints reject invalid repairs and mark UNRESOLVED.
        tick_id = int(target_timestamp.timestamp()) if target_timestamp else len(sensor_window)
        if tick_id % 7 == 0:
            out_of_bounds_val = 210.0 if "boiler" in asset_id else 35.0
            return ModelPrediction(
                available=True,
                candidate_value=out_of_bounds_val,
                uncertainty_lower=out_of_bounds_val - 5.0,
                uncertainty_upper=out_of_bounds_val + 5.0,
                confidence_score=0.95,
                model_name=self.model_name,
                model_version=self.version,
                reconstruction_method="unconstrained_extrapolation",
                notes="Candidate intentionally exceeds physical plausibility bounds to test gateway constraint rejection.",
            )

        # 2. Ambiguous condition: high historical fluctuation in window
        # Returns candidate with borderline confidence (0.65) to test REVIEW status
        if len(valid_vals) >= 3 and abs(valid_vals[-1] - base_estimate) > 3.0:
            return ModelPrediction(
                available=True,
                candidate_value=round(base_estimate, 2),
                uncertainty_lower=round(base_estimate - 2.5, 2),
                uncertainty_upper=round(base_estimate + 2.5, 2),
                confidence_score=0.65,  # Borderline confidence (0.40 <= c < 0.80) -> routes to REVIEW
                model_name=self.model_name,
                model_version=self.version,
                reconstruction_method="moving_average_uncertain",
                notes="Borderline reconstruction confidence; operator evaluation required.",
            )

        # 3. Standard condition: high confidence valid candidate -> REPAIR if physical constraints pass
        return ModelPrediction(
            available=True,
            candidate_value=round(base_estimate, 2),
            uncertainty_lower=round(base_estimate - 0.5, 2),
            uncertainty_upper=round(base_estimate + 0.5, 2),
            confidence_score=0.92,  # High confidence (c >= 0.80)
            model_name=self.model_name,
            model_version=self.version,
            reconstruction_method="historical_exponential_smoothing",
            notes="High confidence spatial-temporal estimate within physical bounds.",
        )


class BenchmarkReport:
    def __init__(
        self,
        run_id: str,
        scenario_name: str,
        total_eval_points: int,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        precision: Optional[float],
        recall: Optional[float],
        f1_score: Optional[float],
        mae: Optional[float],
        rmse: Optional[float],
        mae_by_fault_type: Dict[str, float],
        mae_by_gap_length: Dict[str, float],
        valid_preserved_count: int,
        valid_preserved_pct: float,
        unintentional_modifications: int,
        false_positive_rate: float,
        avg_ingestion_delay_ms: float,
        avg_processing_latency_ms: float,
        avg_e2e_latency_ms: float,
        throughput_readings_per_sec: float,
        failed_processing_count: int,
        accepted_repairs: int,
        unresolved_events: int,
        repair_acceptance_rate: float,
        details: Dict[str, Any],
        model_version: str = "unavailable",
        dataset_version: str = "synthetic-1.0",
        baseline_comparison: Optional[Dict[str, Any]] = None,
        confusion_matrix: Optional[Dict[str, int]] = None,
        detection_by_fault_type: Optional[Dict[str, Any]] = None,
        mae_by_asset: Optional[Dict[str, float]] = None,
        rmse_by_asset: Optional[Dict[str, float]] = None,
        rmse_by_fault_type: Optional[Dict[str, float]] = None,
        candidates_generated_count: int = 0,
        rejected_candidates_count: int = 0,
        review_events_count: int = 0,
        preserved_readings_count: int = 0,
        unnecessary_modification_rate: float = 0.0,
        review_rate: float = 0.0,
        unresolved_rate: float = 0.0,
        min_processing_latency_ms: float = 0.0,
        max_processing_latency_ms: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.run_id = run_id
        self.scenario_name = scenario_name
        self.total_eval_points = total_eval_points
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.mae = mae
        self.rmse = rmse
        self.mae_by_fault_type = mae_by_fault_type
        self.rmse_by_fault_type = rmse_by_fault_type or {}
        self.mae_by_gap_length = mae_by_gap_length
        self.mae_by_asset = mae_by_asset or {}
        self.rmse_by_asset = rmse_by_asset or {}
        self.valid_preserved_count = valid_preserved_count
        self.valid_preserved_pct = valid_preserved_pct
        self.unintentional_modifications = unintentional_modifications
        self.unnecessary_modification_rate = unnecessary_modification_rate
        self.false_positive_rate = false_positive_rate
        self.avg_ingestion_delay_ms = avg_ingestion_delay_ms
        self.avg_processing_latency_ms = avg_processing_latency_ms
        self.min_processing_latency_ms = min_processing_latency_ms
        self.max_processing_latency_ms = max_processing_latency_ms
        self.avg_e2e_latency_ms = avg_e2e_latency_ms
        self.throughput_readings_per_sec = throughput_readings_per_sec
        self.failed_processing_count = failed_processing_count
        self.candidates_generated_count = candidates_generated_count
        self.accepted_repairs = accepted_repairs
        self.rejected_candidates_count = rejected_candidates_count
        self.review_events_count = review_events_count
        self.unresolved_events = unresolved_events
        self.preserved_readings_count = preserved_readings_count
        self.repair_acceptance_rate = repair_acceptance_rate
        self.review_rate = review_rate
        self.unresolved_rate = unresolved_rate
        self.details = details
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.seed = seed
        self.baseline_comparison = baseline_comparison or {}
        self.confusion_matrix = confusion_matrix or {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        self.detection_by_fault_type = detection_by_fault_type or {}

    @property
    def latency_ms(self) -> float:
        return self.avg_processing_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "total_eval_points": self.total_eval_points,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "seed": self.seed,
            # Detection Metrics
            "detection": {
                "true_positives": self.tp,
                "false_positives": self.fp,
                "false_negatives": self.fn,
                "true_negatives": self.tn,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "confusion_matrix": self.confusion_matrix,
                "by_fault_type": self.detection_by_fault_type,
            },
            # Reconstruction Metrics
            "reconstruction": {
                "mae": self.mae,
                "rmse": self.rmse,
                "mae_by_fault_type": self.mae_by_fault_type,
                "rmse_by_fault_type": self.rmse_by_fault_type,
                "mae_by_asset": self.mae_by_asset,
                "rmse_by_asset": self.rmse_by_asset,
                "mae_by_gap_length": self.mae_by_gap_length,
                "candidates_generated": self.candidates_generated_count,
                "accepted_repairs": self.accepted_repairs,
                "rejected_candidates": self.rejected_candidates_count,
                "events_requiring_review": self.review_events_count,
                "unresolved_events": self.unresolved_events,
            },
            # Preservation Metrics
            "preservation": {
                "known_valid_tested_count": self.details.get("clean_eval_samples", self.valid_preserved_count),
                "known_valid_preserved_count": self.valid_preserved_count,
                "known_valid_preserved_percentage": self.valid_preserved_pct,
                "unintentional_modifications": self.unintentional_modifications,
                "unnecessary_modification_rate": self.unnecessary_modification_rate,
                "false_positive_rate": self.false_positive_rate,
            },
            # Pipeline Performance
            "performance": {
                "avg_ingestion_delay_ms": self.avg_ingestion_delay_ms,
                "avg_processing_latency_ms": self.avg_processing_latency_ms,
                "min_processing_latency_ms": self.min_processing_latency_ms,
                "max_processing_latency_ms": self.max_processing_latency_ms,
                "avg_end_to_end_latency_ms": self.avg_e2e_latency_ms,
                "throughput_readings_per_sec": self.throughput_readings_per_sec,
                "failed_processing_count": self.failed_processing_count,
            },
            # Decision Policy
            "decision_policy": {
                "preserved_readings": self.preserved_readings_count,
                "accepted_repairs": self.accepted_repairs,
                "review_events": self.review_events_count,
                "unresolved_events": self.unresolved_events,
                "repair_acceptance_rate": self.repair_acceptance_rate,
                "review_rate": self.review_rate,
                "unresolved_rate": self.unresolved_rate,
            },
            # Baseline Comparison
            "baseline_comparison": self.baseline_comparison,
            "details": self.details,
        }


class EvaluationSuite:
    """Rigorous evaluation suite evaluating detection, reconstruction, preservation, pipeline performance, and decision policy."""

    @staticmethod
    def evaluate(
        scenario_name: str,
        ground_truth_records: List[GroundTruthFaultRecord],
        processed_results: List[Dict[str, Any]],
        start_time_perf: float,
        end_time_perf: float,
        repo: Optional[TelemetryRepository] = None,
        model_version: str = "unavailable",
        dataset_version: str = "synthetic-1.0",
        seed: Optional[int] = None,
    ) -> BenchmarkReport:
        run_id = f"EVAL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        total_points = len(processed_results)
        total_elapsed_sec = max(0.0001, end_time_perf - start_time_perf)
        throughput = round(total_points / total_elapsed_sec, 2)

        # Lookup of injected faults by reading_id
        fault_by_id = {f.reading_id: f for f in ground_truth_records if f.reading_id}

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        unintentional_modifications = 0
        valid_preserved_count = 0
        total_clean_points = 0

        # Decision & Candidate tracking
        candidates_generated = 0
        accepted_repairs = 0
        rejected_candidates = 0
        review_events = 0
        unresolved_events = 0
        preserved_readings = 0
        failed_processing = 0

        # Error tracking
        abs_errors: List[float] = []
        sq_errors: List[float] = []
        errors_by_type: Dict[str, List[float]] = {}
        sq_errors_by_type: Dict[str, List[float]] = {}
        errors_by_gap: Dict[str, List[float]] = {}
        errors_by_asset: Dict[str, List[float]] = {"boiler_01": [], "pump_01": []}
        sq_errors_by_asset: Dict[str, List[float]] = {"boiler_01": [], "pump_01": []}

        # Detection by fault type tracking
        target_fault_types = [
            "MISSING_DATA",
            "COMMUNICATION_GAP",
            "FROZEN_SENSOR",
            "SUDDEN_SPIKE",
            "GRADUAL_DRIFT",
            "INVALID_CORRUPTED",
        ]
        by_fault_type_counts: Dict[str, Dict[str, int]] = {
            ft: {"tp": 0, "fn": 0, "fp": 0, "tested_count": 0} for ft in target_fault_types
        }

        ingestion_delays_ms: List[float] = []
        processing_latencies_ms: List[float] = []
        e2e_latencies_ms: List[float] = []

        for res in processed_results:
            rid = res.get("reading_id")
            asset_id = res.get("asset_id", "boiler_01")
            status = res.get("decision_status") or res.get("status")
            issues = res.get("issues_detected") or res.get("detected_issues") or []
            final_val = res.get("final_value")
            orig_val = res.get("original_value")
            cand_val = res.get("candidate_value")

            # Ingestion & processing timing if timestamps available
            ev_ts = res.get("event_timestamp") or res.get("timestamp")
            rec_ts = res.get("received_at")
            proc_ts = res.get("processed_at")
            if isinstance(ev_ts, str):
                try:
                    ev_ts = datetime.fromisoformat(ev_ts.replace("Z", "+00:00"))
                except Exception:
                    ev_ts = None
            if isinstance(rec_ts, str):
                try:
                    rec_ts = datetime.fromisoformat(rec_ts.replace("Z", "+00:00"))
                except Exception:
                    rec_ts = None
            if isinstance(proc_ts, str):
                try:
                    proc_ts = datetime.fromisoformat(proc_ts.replace("Z", "+00:00"))
                except Exception:
                    proc_ts = None

            if ev_ts and rec_ts:
                ingestion_delays_ms.append(max(0.0, (rec_ts - ev_ts).total_seconds() * 1000.0))
            if rec_ts and proc_ts:
                processing_latencies_ms.append(max(0.0, (proc_ts - rec_ts).total_seconds() * 1000.0))
            if ev_ts and proc_ts:
                e2e_latencies_ms.append(max(0.0, (proc_ts - ev_ts).total_seconds() * 1000.0))

            # Candidate & decision status tracking
            if cand_val is not None:
                candidates_generated += 1
                if status != "REPAIR":
                    rejected_candidates += 1

            if status == "PRESERVE":
                preserved_readings += 1
            elif status == "REPAIR":
                accepted_repairs += 1
            elif status == "REVIEW":
                review_events += 1
            elif status == "UNRESOLVED":
                unresolved_events += 1
            elif status == "FAILED":
                failed_processing += 1

            was_injected = rid in fault_by_id
            fault_record = fault_by_id.get(rid)
            has_detection = len(issues) > 0 or status in ("REPAIR", "UNRESOLVED", "REVIEW")

            if was_injected:
                ftype = fault_record.fault_type
                if ftype in by_fault_type_counts:
                    by_fault_type_counts[ftype]["tested_count"] += 1

                if has_detection:
                    tp += 1
                    if ftype in by_fault_type_counts:
                        by_fault_type_counts[ftype]["tp"] += 1
                else:
                    fn += 1
                    if ftype in by_fault_type_counts:
                        by_fault_type_counts[ftype]["fn"] += 1

                # If repair was accepted, compute reconstruction accuracy against clean ground truth
                if status == "REPAIR" and final_val is not None and fault_record:
                    clean_true = fault_record.clean_ground_truth_value
                    if clean_true is not None:
                        diff = abs(final_val - clean_true)
                        abs_errors.append(diff)
                        sq_errors.append(diff ** 2)

                        errors_by_type.setdefault(ftype, []).append(diff)
                        sq_errors_by_type.setdefault(ftype, []).append(diff ** 2)

                        gap_len = fault_record.injection_parameters.get("time_jump_seconds", 1)
                        errors_by_gap.setdefault(str(gap_len), []).append(diff)

                        if asset_id in errors_by_asset:
                            errors_by_asset[asset_id].append(diff)
                            sq_errors_by_asset[asset_id].append(diff ** 2)
            else:
                total_clean_points += 1
                if has_detection:
                    fp += 1
                    for iss in issues:
                        if iss in by_fault_type_counts:
                            by_fault_type_counts[iss]["fp"] += 1
                else:
                    tn += 1

                # Clean readings preservation evaluation
                if orig_val is not None and final_val is not None:
                    if math.isclose(orig_val, final_val, abs_tol=1e-5):
                        valid_preserved_count += 1
                    else:
                        unintentional_modifications += 1
                elif status == "PRESERVE":
                    valid_preserved_count += 1

        # Calculate Precision, Recall, F1
        denom_prec = tp + fp
        precision = round(tp / denom_prec, 4) if denom_prec > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)

        denom_rec = tp + fn
        recall = round(tp / denom_rec, 4) if denom_rec > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)

        if precision + recall > 0:
            f1 = round(2 * (precision * recall) / (precision + recall), 4)
        else:
            f1 = 0.0 if denom_rec > 0 else None

        confusion_matrix = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

        # Detection by fault type
        detection_by_fault_type: Dict[str, Dict[str, Any]] = {}
        for ft, cnts in by_fault_type_counts.items():
            ft_tp = cnts["tp"]
            ft_fn = cnts["fn"]
            ft_fp = cnts["fp"]
            ft_prec = round(ft_tp / (ft_tp + ft_fp), 4) if (ft_tp + ft_fp) > 0 else (1.0 if ft_tp == 0 and ft_fp == 0 else 0.0)
            ft_rec = round(ft_tp / (ft_tp + ft_fn), 4) if (ft_tp + ft_fn) > 0 else (1.0 if ft_tp == 0 and ft_fn == 0 else 0.0)
            ft_f1 = round(2 * ft_prec * ft_rec / (ft_prec + ft_rec), 4) if (ft_prec + ft_rec) > 0 else 0.0
            detection_by_fault_type[ft] = {
                "tp": ft_tp,
                "fn": ft_fn,
                "fp": ft_fp,
                "tested_count": cnts["tested_count"],
                "precision": ft_prec,
                "recall": ft_rec,
                "f1_score": ft_f1,
            }

        # MAE & RMSE
        mae = round(float(np.mean(abs_errors)), 4) if abs_errors else None
        rmse = round(float(np.sqrt(np.mean(sq_errors))), 4) if sq_errors else None

        mae_by_fault_type = {
            k: round(float(np.mean(v)), 4) for k, v in errors_by_type.items() if v
        }
        rmse_by_fault_type = {
            k: round(float(np.sqrt(np.mean(v))), 4) for k, v in sq_errors_by_type.items() if v
        }
        mae_by_gap_length = {
            k: round(float(np.mean(v)), 4) for k, v in errors_by_gap.items() if v
        }

        mae_by_asset = {
            asset: round(float(np.mean(diffs)), 4) if diffs else None
            for asset, diffs in errors_by_asset.items()
        }
        rmse_by_asset = {
            asset: round(float(np.sqrt(np.mean(sq_diffs))), 4) if sq_diffs else None
            for asset, sq_diffs in sq_errors_by_asset.items()
        }

        # Preservation metrics
        valid_preserved_pct = (
            round((valid_preserved_count / total_clean_points) * 100.0, 2)
            if total_clean_points > 0
            else 100.0
        )
        unnecessary_modification_rate = (
            round(unintentional_modifications / total_clean_points, 4)
            if total_clean_points > 0
            else 0.0
        )
        false_positive_rate = (
            round(fp / total_clean_points, 4) if total_clean_points > 0 else 0.0
        )

        # Performance latencies
        avg_ingestion_delay = (
            round(float(np.mean(ingestion_delays_ms)), 2)
            if ingestion_delays_ms
            else 0.0
        )
        avg_processing_latency = (
            round(float(np.mean(processing_latencies_ms)), 2)
            if processing_latencies_ms
            else round((total_elapsed_sec * 1000.0) / max(1, total_points), 2)
        )
        min_processing_latency = (
            round(float(np.min(processing_latencies_ms)), 2)
            if processing_latencies_ms
            else avg_processing_latency
        )
        max_processing_latency = (
            round(float(np.max(processing_latencies_ms)), 2)
            if processing_latencies_ms
            else avg_processing_latency
        )
        avg_e2e_latency = (
            round(float(np.mean(e2e_latencies_ms)), 2)
            if e2e_latencies_ms
            else avg_processing_latency
        )

        # Decision policy rates
        total_decisions_on_anomalies = accepted_repairs + unresolved_events + review_events
        repair_acceptance_rate = (
            round(accepted_repairs / total_decisions_on_anomalies, 4)
            if total_decisions_on_anomalies > 0
            else 0.0
        )
        review_rate = (
            round(review_events / max(1, total_points), 4)
        )
        unresolved_rate = (
            round(unresolved_events / max(1, total_points), 4)
        )

        # ---------------------------------------------------------------------
        # 3-Way Baseline Comparison Calculation
        # ---------------------------------------------------------------------
        reading_eval_list = []
        for res in processed_results:
            rid = res.get("reading_id")
            if not rid:
                continue
            orig_v = res.get("original_value")
            final_v = res.get("final_value")
            gt = fault_by_id.get(rid)
            clean_v = gt.clean_ground_truth_value if gt else orig_v
            is_fault = (gt is not None)
            fault_type = gt.fault_type if gt else None
            reading_eval_list.append({
                "reading_id": rid,
                "orig_v": orig_v,
                "final_v": final_v,
                "clean_v": clean_v,
                "is_fault": is_fault,
                "fault_type": fault_type,
            })

        # Baseline 1: No-repair (retains uncorrected raw observations)
        b1_diffs = []
        b1_missing = 0
        for item in reading_eval_list:
            if item["clean_v"] is not None:
                if item["orig_v"] is not None:
                    b1_diffs.append(abs(item["orig_v"] - item["clean_v"]))
                else:
                    b1_missing += 1
        mae_b1 = float(np.mean(b1_diffs)) if b1_diffs else None
        rmse_b1 = float(np.sqrt(np.mean([d**2 for d in b1_diffs]))) if b1_diffs else None

        # Baseline 2: Linear interpolation (strictly for missing values)
        b2_diffs = []
        b2_interpolated = 0
        b2_missing = 0
        n_pts = len(reading_eval_list)
        for i, item in enumerate(reading_eval_list):
            val = item["orig_v"]
            if val is None:
                # Search previous valid point within 10 ticks
                prev_v = None
                prev_idx = None
                for j in range(i - 1, max(-1, i - 11), -1):
                    if reading_eval_list[j]["orig_v"] is not None:
                        prev_v = reading_eval_list[j]["orig_v"]
                        prev_idx = j
                        break
                # Search next valid point within 10 ticks
                next_v = None
                next_idx = None
                for k in range(i + 1, min(n_pts, i + 11)):
                    if reading_eval_list[k]["orig_v"] is not None:
                        next_v = reading_eval_list[k]["orig_v"]
                        next_idx = k
                        break
                if prev_v is not None and next_v is not None and prev_idx is not None and next_idx is not None:
                    fraction = (i - prev_idx) / (next_idx - prev_idx)
                    val = prev_v + fraction * (next_v - prev_v)
                    b2_interpolated += 1
                else:
                    b2_missing += 1

            if item["clean_v"] is not None and val is not None:
                b2_diffs.append(abs(val - item["clean_v"]))

        mae_b2 = float(np.mean(b2_diffs)) if b2_diffs else None
        rmse_b2 = float(np.sqrt(np.mean([d**2 for d in b2_diffs]))) if b2_diffs else None

        # Baseline 3: FusionHeal pipeline
        b3_diffs = []
        for item in reading_eval_list:
            if item["clean_v"] is not None and item["final_v"] is not None:
                b3_diffs.append(abs(item["final_v"] - item["clean_v"]))
        mae_b3 = float(np.mean(b3_diffs)) if b3_diffs else (mae if mae is not None else 0.0)
        rmse_b3 = float(np.sqrt(np.mean([d**2 for d in b3_diffs]))) if b3_diffs else (rmse if rmse is not None else 0.0)

        # Decision distribution count
        dec_counts = {
            "PRESERVE": preserved_readings,
            "REPAIR": accepted_repairs,
            "REVIEW": review_events,
            "UNRESOLVED": unresolved_events,
        }

        baseline_comparison = {
            "baseline_1_no_repair": {
                "name": "Baseline 1: No-Repair Baseline",
                "mae": round(mae_b1, 4) if mae_b1 is not None else None,
                "rmse": round(rmse_b1, 4) if rmse_b1 is not None else None,
                "unrepaired_missing_count": b1_missing,
                "description": "Retains raw uncorrected observations without repair.",
            },
            "baseline_2_linear_interpolation": {
                "name": "Baseline 2: Linear Interpolation",
                "mae": round(mae_b2, 4) if mae_b2 is not None else None,
                "rmse": round(rmse_b2, 4) if rmse_b2 is not None else None,
                "interpolated_count": b2_interpolated,
                "unrepaired_missing_count": b2_missing,
                "description": "Applies linear interpolation strictly to missing values with surrounding neighbors. Spikes, drift, and corrupted signals remain untouched.",
            },
            "baseline_3_fusionheal": {
                "name": "Baseline 3: FusionHeal Pipeline",
                "mae": round(mae_b3, 4) if mae_b3 is not None else None,
                "rmse": round(rmse_b3, 4) if rmse_b3 is not None else None,
                "detection_precision": precision,
                "detection_recall": recall,
                "detection_f1": f1,
                "valid_preserved_percentage": valid_preserved_pct,
                "decision_distribution": dec_counts,
                "description": "Multi-detector anomaly identification, physical-rule verification, and evidence-governed auditable repairs.",
            },
        }

        report = BenchmarkReport(
            run_id=run_id,
            scenario_name=scenario_name,
            total_eval_points=total_points,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mae=mae,
            rmse=rmse,
            mae_by_fault_type=mae_by_fault_type,
            mae_by_gap_length=mae_by_gap_length,
            valid_preserved_count=valid_preserved_count,
            valid_preserved_pct=valid_preserved_pct,
            unintentional_modifications=unintentional_modifications,
            false_positive_rate=false_positive_rate,
            avg_ingestion_delay_ms=avg_ingestion_delay,
            avg_processing_latency_ms=avg_processing_latency,
            avg_e2e_latency_ms=avg_e2e_latency,
            throughput_readings_per_sec=throughput,
            failed_processing_count=failed_processing,
            accepted_repairs=accepted_repairs,
            unresolved_events=unresolved_events,
            repair_acceptance_rate=repair_acceptance_rate,
            details={
                "injected_faults_count": len(ground_truth_records),
                "reconstruction_eval_samples": len(abs_errors),
                "clean_eval_samples": total_clean_points,
                "total_tested_readings": total_points,
            },
            model_version=model_version,
            dataset_version=dataset_version,
            baseline_comparison=baseline_comparison,
            confusion_matrix=confusion_matrix,
            detection_by_fault_type=detection_by_fault_type,
            mae_by_asset=mae_by_asset,
            rmse_by_asset=rmse_by_asset,
            rmse_by_fault_type=rmse_by_fault_type,
            candidates_generated_count=candidates_generated,
            rejected_candidates_count=rejected_candidates,
            review_events_count=review_events,
            preserved_readings_count=preserved_readings,
            unnecessary_modification_rate=unnecessary_modification_rate,
            review_rate=review_rate,
            unresolved_rate=unresolved_rate,
            min_processing_latency_ms=min_processing_latency,
            max_processing_latency_ms=max_processing_latency,
            seed=seed,
        )

        # Persist to database if repository provided
        if repo is not None:
            db_metric = EvaluationMetricRun(
                run_id=run_id,
                scenario_name=scenario_name,
                model_version=model_version,
                dataset_version=dataset_version,
                precision=precision,
                recall=recall,
                f1_score=f1,
                mae=mae,
                rmse=rmse,
                latency_ms=avg_processing_latency,
                accepted_repairs=accepted_repairs,
                unresolved_events=unresolved_events,
                unintentional_modifications=unintentional_modifications,
                details_json=json.dumps(report.to_dict()),
                baseline_comparison_json=json.dumps(baseline_comparison),
            )
            repo.save_metric_run(db_metric)

        return report


def run_reproducible_benchmark(
    seed: int = 42,
    sample_count: int = 60,
    db: Optional[Any] = None,
    scenario_name: str = "reproducible_synthetic_benchmark",
    model: Optional[ReconstructionModel] = None,
    use_connected_model: bool = True,
) -> BenchmarkReport:
    """Execute a fully controlled, reproducible benchmark run across Boiler and Pump telemetry.

    Generates all 6 fault types + clean telemetry:
    - Missing values
    - Communication gaps
    - Frozen sensors
    - Sudden spikes
    - Gradual drift
    - Corrupted signals (physical bound breaches)

    Evaluates both ordinary and ambiguous cases.
    CRITICAL RULE: The pipeline is given ZERO access to hidden ground-truth labels during execution.
    """
    from app.config import get_settings
    from app.pipeline.processing import PipelineProcessor
    from app.storage.database import SessionLocal

    settings = get_settings()
    gen = SyntheticSensorGenerator(random_seed=seed)
    injector = FaultInjector()

    should_close_db = False
    session = db
    if session is None:
        session = SessionLocal()
        should_close_db = True

    repo = TelemetryRepository(session)

    # If no model provided, default to DeterministicBenchmarkModel if use_connected_model is True
    active_model = model
    if active_model is None:
        if use_connected_model:
            active_model = DeterministicBenchmarkModel()
        else:
            active_model = PlaceholderReconstructionAdapter()

    processor = PipelineProcessor(settings, model=active_model)
    processor.reset_state()

    # Fault rotation across sequence to evaluate all 6 fault modes, clean points, and ambiguous cases
    fault_schedule = [
        "clean", "clean", "clean", "clean",
        "missing_value", "clean",
        "frozen", "frozen", "frozen", "clean",
        "sudden_spike", "clean",
        "gradual_drift", "gradual_drift", "clean",
        "invalid_corrupted", "clean",
        "communication_gap", "clean",
    ]

    base_time = datetime(2026, 10, 27, 8, 0, 0, tzinfo=timezone.utc)
    readings_to_process = []

    for idx in range(sample_count):
        cur_ts = base_time + timedelta(seconds=idx)
        scen = fault_schedule[idx % len(fault_schedule)]

        # 1. Industrial Boiler (Temperature in C)
        b_rd, b_clean = gen.generate_boiler_temperature(timestamp=cur_ts)
        mod_b, _ = injector.inject_fault(b_rd, b_clean, scen)
        if mod_b is not None:
            readings_to_process.append(mod_b)

        # 2. Industrial Pump (Vibration in m/s2)
        p_rd, p_clean = gen.generate_pump_vibration(timestamp=cur_ts)
        mod_p, _ = injector.inject_fault(p_rd, p_clean, scen)
        if mod_p is not None:
            readings_to_process.append(mod_p)

    t0 = time.time()
    processed_results: List[Dict[str, Any]] = []
    for rd in readings_to_process:
        res = processor.process_reading(rd, session)
        processed_results.append(res.model_dump())
    t1 = time.time()

    report = EvaluationSuite.evaluate(
        scenario_name=scenario_name,
        ground_truth_records=injector.ground_truth_log,
        processed_results=processed_results,
        start_time_perf=t0,
        end_time_perf=t1,
        repo=repo,
        model_version=getattr(processor.model, "model_version", getattr(processor.model, "version", "unavailable")),
        dataset_version=f"synthetic-seed-{seed}",
        seed=seed,
    )

    if should_close_db:
        session.close()

    return report

