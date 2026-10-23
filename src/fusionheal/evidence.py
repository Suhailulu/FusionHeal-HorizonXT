from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


class EvidenceEngine:
    """Collects, validates, and aggregates multi-source evidence for proposed sensor corrections.

    Preserves supporting and contradicting evidence separately, handles missing or
    insufficient history gracefully, and enforces that cross-sensor relationships
    are evaluated only when explicitly validated by domain engineering configuration.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def collect_evidence(
        self,
        df: pd.DataFrame,
        issue: dict[str, Any],
        candidate: dict[str, Any] | None = None,
        flagged_positions: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Collect all applicable evidence records for an issue and candidate."""
        flagged = flagged_positions or set()
        evidence_list: list[dict[str, Any]] = []

        col = issue.get("sensor_id")
        pos = issue.get("row_position")
        issue_id = issue.get("issue_id", "UNKNOWN_ISSUE")

        if col not in df.columns or pos is None:
            return evidence_list

        pos = int(pos)
        candidate_val = candidate.get("candidate_value") if candidate else None

        # 1. Temporal Evidence
        temp_ev = self._evaluate_temporal_evidence(df, col, pos, issue_id, candidate_val, flagged)
        evidence_list.extend(temp_ev)

        # 2. Historical Evidence
        hist_ev = self._evaluate_historical_evidence(df, col, pos, issue_id, candidate_val)
        evidence_list.extend(hist_ev)

        # 3. Cross-Sensor Evidence
        cross_ev = self._evaluate_cross_sensor_evidence(df, col, pos, issue_id, candidate_val)
        evidence_list.extend(cross_ev)

        # 4. Process-Rule Evidence
        rule_ev = self._evaluate_process_rule_evidence(col, issue_id, candidate_val)
        evidence_list.extend(rule_ev)

        return evidence_list

    def _evaluate_temporal_evidence(
        self,
        df: pd.DataFrame,
        col: str,
        pos: int,
        issue_id: str,
        candidate_val: float | None,
        flagged: set[int],
    ) -> list[dict[str, Any]]:
        """Evaluate temporal continuity and neighboring observation quality."""
        records: list[dict[str, Any]] = []
        series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        n = len(series)

        detection_cfg = self.config.get("detection", {})
        window_strategy = str(detection_cfg.get("window_strategy", "centered")).lower()
        window_size = int(detection_cfg.get("rolling_window", 9))

        # Determine valid window bounds based on online vs batch mode
        if window_strategy == "past_only":
            left = max(0, pos - window_size)
            right = pos
            mode_desc = "past-only causal mode (no future leakage)"
        else:
            left = max(0, pos - window_size // 2)
            right = min(n, pos + window_size // 2 + 1)
            mode_desc = "centered retrospective mode"

        # Filter out invalid, NaN, or already flagged neighboring measurements
        valid_neighbor_vals: list[float] = []
        for idx in range(left, right):
            if idx == pos:
                continue
            if idx in flagged:
                continue
            val = series[idx]
            if np.isfinite(val):
                valid_neighbor_vals.append(float(val))

        # Check neighbor availability
        if len(valid_neighbor_vals) < 2:
            records.append({
                "evidence_id": f"EVID-TEMP-UNAVAIL-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "temporal",
                "direction": "neutral",
                "explanation": f"Insufficient valid temporal neighbors ({len(valid_neighbor_vals)} found) in {mode_desc}.",
                "source_sensor": col,
                "source_record_id": str(df.iloc[pos].get("source_record_id", f"ROW-{pos:08d}")),
                "strength": 0.2,
                "metadata": {"valid_neighbors_count": len(valid_neighbor_vals), "window_strategy": window_strategy},
            })
            return records

        med = float(np.median(valid_neighbor_vals))
        std = float(np.std(valid_neighbor_vals))

        if candidate_val is not None:
            delta = abs(candidate_val - med)
            scale = max(std, 1e-4)
            dev_ratio = delta / scale

            if dev_ratio <= 2.0:
                direction = "supporting"
                strength = max(0.5, 1.0 - (dev_ratio / 3.0))
                expl = (
                    f"Candidate value {candidate_val:.2f} is consistent with neighboring {mode_desc} "
                    f"median {med:.2f} (deviation ratio: {dev_ratio:.2f})."
                )
            else:
                direction = "contradicting"
                strength = min(1.0, dev_ratio / 3.0)
                expl = (
                    f"Candidate value {candidate_val:.2f} deviates significantly from neighboring {mode_desc} "
                    f"median {med:.2f} (deviation ratio: {dev_ratio:.2f} > 2.0)."
                )

            records.append({
                "evidence_id": f"EVID-TEMP-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "temporal",
                "direction": direction,
                "explanation": expl,
                "source_sensor": col,
                "source_record_id": str(df.iloc[pos].get("source_record_id", f"ROW-{pos:08d}")),
                "strength": float(round(strength, 3)),
                "metadata": {
                    "local_median": med,
                    "local_std": std,
                    "deviation_ratio": float(round(dev_ratio, 3)),
                    "window_strategy": window_strategy,
                },
            })
        else:
            records.append({
                "evidence_id": f"EVID-TEMP-CTX-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "temporal",
                "direction": "neutral",
                "explanation": f"Temporal context evaluated ({len(valid_neighbor_vals)} valid points, median {med:.2f}).",
                "source_sensor": col,
                "source_record_id": str(df.iloc[pos].get("source_record_id", f"ROW-{pos:08d}")),
                "strength": 0.5,
                "metadata": {"local_median": med, "window_strategy": window_strategy},
            })

        return records

    def _evaluate_historical_evidence(
        self,
        df: pd.DataFrame,
        col: str,
        pos: int,
        issue_id: str,
        candidate_val: float | None,
    ) -> list[dict[str, Any]]:
        """Compare candidate against verified historical sensor distribution."""
        records: list[dict[str, Any]] = []
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        min_history_points = int(self.config.get("detection", {}).get("min_history_points", 5))

        if len(series) < min_history_points:
            records.append({
                "evidence_id": f"EVID-HIST-INSUFF-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "historical",
                "direction": "neutral",
                "explanation": f"Insufficient historical data ({len(series)} samples < required {min_history_points}).",
                "source_sensor": col,
                "source_record_id": str(df.iloc[pos].get("source_record_id", f"ROW-{pos:08d}")),
                "strength": 0.1,
                "metadata": {"sample_count": len(series), "min_required": min_history_points},
            })
            return records

        q01 = float(np.percentile(series, 1))
        q99 = float(np.percentile(series, 99))
        hist_med = float(np.median(series))

        if candidate_val is not None:
            if q01 <= candidate_val <= q99:
                direction = "supporting"
                strength = 0.8
                expl = (
                    f"Candidate value {candidate_val:.2f} falls within normal historical range "
                    f"[{q01:.2f}, {q99:.2f}] (median {hist_med:.2f})."
                )
            else:
                direction = "contradicting"
                strength = 0.85
                expl = (
                    f"Candidate value {candidate_val:.2f} falls outside historical 1st-99th percentile envelope "
                    f"[{q01:.2f}, {q99:.2f}]."
                )

            records.append({
                "evidence_id": f"EVID-HIST-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "historical",
                "direction": direction,
                "explanation": expl,
                "source_sensor": col,
                "source_record_id": str(df.iloc[pos].get("source_record_id", f"ROW-{pos:08d}")),
                "strength": strength,
                "metadata": {"hist_q01": q01, "hist_q99": q99, "hist_median": hist_med},
            })

        return records

    def _evaluate_cross_sensor_evidence(
        self,
        df: pd.DataFrame,
        col: str,
        pos: int,
        issue_id: str,
        candidate_val: float | None,
    ) -> list[dict[str, Any]]:
        """Evaluate cross-sensor consistency using ONLY validated configured relationships."""
        records: list[dict[str, Any]] = []

        detection_cfg = self.config.get("detection", {})
        cs_cfg = detection_cfg.get("cross_sensor", {})
        rules = cs_cfg.get("rules", []) if isinstance(cs_cfg, dict) else self.config.get("cross_sensor_rules", [])

        # Filter rules relevant to this sensor
        applicable_rules = [
            r for r in rules
            if r.get("sensor_a") == col or r.get("sensor_b") == col
        ]

        # Critical Safety Rule: If no validated relationship exists, mark UNAVAILABLE
        if not applicable_rules:
            records.append({
                "evidence_id": f"EVID-CROSS-UNAVAIL-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "cross_sensor",
                "direction": "neutral",
                "explanation": (
                    "UNAVAILABLE: No validated cross-sensor physical relationships configured for this sensor. "
                    "Unverified correlations are intentionally not evaluated."
                ),
                "source_sensor": col,
                "source_record_id": str(df.iloc[pos].get("source_record_id", f"ROW-{pos:08d}")),
                "strength": 0.0,
                "metadata": {"status": "unavailable", "rules_configured": 0},
            })
            return records

        for rule in applicable_rules:
            rule_id = rule.get("rule_id", "cross_sensor_rule")
            paired_sensor = rule.get("sensor_b") if rule.get("sensor_a") == col else rule.get("sensor_a")

            if paired_sensor not in df.columns:
                continue

            paired_val = pd.to_numeric(df.iloc[pos][paired_sensor], errors="coerce")
            if not np.isfinite(paired_val):
                records.append({
                    "evidence_id": f"EVID-CROSS-{rule_id}-{issue_id}",
                    "issue_id": issue_id,
                    "evidence_type": "cross_sensor",
                    "direction": "neutral",
                    "explanation": f"Paired sensor '{paired_sensor}' has invalid/missing reading at row {pos}.",
                    "source_sensor": f"{col}:{paired_sensor}",
                    "strength": 0.1,
                    "metadata": {"paired_sensor": paired_sensor, "status": "paired_missing"},
                })
                continue

            if candidate_val is not None:
                max_diff = float(rule.get("max_difference", 10.0))
                observed_diff = abs(candidate_val - paired_val)

                if observed_diff <= max_diff:
                    direction = "supporting"
                    strength = 0.85
                    expl = (
                        f"Candidate value {candidate_val:.2f} satisfies validated cross-sensor rule '{rule_id}' "
                        f"with paired sensor {paired_sensor} ({paired_val:.2f}): difference {observed_diff:.2f} <= {max_diff:.2f}."
                    )
                else:
                    direction = "contradicting"
                    strength = 0.9
                    expl = (
                        f"Candidate value {candidate_val:.2f} breaches validated cross-sensor rule '{rule_id}' "
                        f"with paired sensor {paired_sensor} ({paired_val:.2f}): difference {observed_diff:.2f} > {max_diff:.2f}."
                    )

                records.append({
                    "evidence_id": f"EVID-CROSS-{rule_id}-{issue_id}",
                    "issue_id": issue_id,
                    "evidence_type": "cross_sensor",
                    "direction": direction,
                    "explanation": expl,
                    "source_sensor": f"{col}:{paired_sensor}",
                    "source_record_id": str(df.iloc[pos].get("source_record_id", f"ROW-{pos:08d}")),
                    "strength": strength,
                    "metadata": {
                        "rule_id": rule_id,
                        "paired_sensor": paired_sensor,
                        "paired_value": float(paired_val),
                        "observed_difference": float(round(observed_diff, 3)),
                    },
                })

        return records

    def _evaluate_process_rule_evidence(
        self,
        col: str,
        issue_id: str,
        candidate_val: float | None,
    ) -> list[dict[str, Any]]:
        """Evaluate soft process-rule evidence against configured demonstration parameters."""
        records: list[dict[str, Any]] = []
        limits = self.config.get("sensor_limits", {}).get(col, {})
        if not limits or candidate_val is None:
            return records

        min_val = limits.get("min")
        max_val = limits.get("max")

        is_valid = True
        if min_val is not None and candidate_val < min_val:
            is_valid = False
        if max_val is not None and candidate_val > max_val:
            is_valid = False

        if is_valid:
            records.append({
                "evidence_id": f"EVID-RULE-BOUNDS-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "process_rule",
                "direction": "supporting",
                "explanation": (
                    f"Candidate value {candidate_val:.2f} falls within configured process range limits "
                    f"[{min_val}, {max_val}]."
                ),
                "source_sensor": col,
                "strength": 0.75,
                "metadata": {"configured_min": min_val, "configured_max": max_val, "is_hard_rule": False},
            })
        else:
            records.append({
                "evidence_id": f"EVID-RULE-BOUNDS-{issue_id}",
                "issue_id": issue_id,
                "evidence_type": "process_rule",
                "direction": "contradicting",
                "explanation": (
                    f"Candidate value {candidate_val:.2f} falls outside configured process range limits "
                    f"[{min_val}, {max_val}]."
                ),
                "source_sensor": col,
                "strength": 0.95,
                "metadata": {"configured_min": min_val, "configured_max": max_val, "is_hard_rule": False},
            })

        return records

    @staticmethod
    def aggregate_evidence(evidence_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate evidence records, partitioning into supporting and contradicting categories."""
        supporting = [e for e in evidence_list if e.get("direction") == "supporting"]
        contradicting = [e for e in evidence_list if e.get("direction") == "contradicting"]
        neutral = [e for e in evidence_list if e.get("direction") == "neutral"]

        return {
            "total_evidence_records": len(evidence_list),
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
            "neutral_count": len(neutral),
            "supporting": supporting,
            "contradicting": contradicting,
            "neutral": neutral,
            "has_contradictions": len(contradicting) > 0,
            "is_sufficient": (len(supporting) + len(contradicting)) >= 2,
        }
