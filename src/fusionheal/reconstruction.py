from __future__ import annotations
import numpy as np
import pandas as pd

def generate_candidate(df: pd.DataFrame, issue: dict) -> dict | None:
    """Generate a conservative candidate; never modifies the source dataframe."""
    pos = issue.get("row_position")
    col = issue.get("sensor_id")
    if pos is None or col not in df.columns:
        return None
    pos = int(pos)
    series = pd.to_numeric(df[col], errors="coerce").copy()
    original = issue.get("original_value")
    method = None
    candidate = None

    if issue["fault_type"] == "missing":
        # Time-aware interpolation can be added later; row-order interpolation is
        # used only for this starter demo.
        series.iloc[pos] = np.nan
        interpolated = series.interpolate(limit_area="inside")
        if pos < len(interpolated) and pd.notna(interpolated.iloc[pos]):
            candidate = float(interpolated.iloc[pos])
            method = "linear_interpolation"
    elif issue["fault_type"] == "outlier":
        median = issue.get("metadata", {}).get("local_median", issue.get("local_median"))
        if median is not None and np.isfinite(median):
            candidate = float(median)
            method = "local_median"
    elif issue["fault_type"] == "spike":
        # Conservative local interpolation for isolated single-point spike
        prev_val = series.iloc[pos - 1] if pos > 0 else np.nan
        next_val = series.iloc[pos + 1] if pos < len(series) - 1 else np.nan
        if pd.notna(prev_val) and pd.notna(next_val):
            candidate = float((prev_val + next_val) / 2.0)
            method = "local_interpolation"
    elif issue["fault_type"] in (
        "range_violation",
        "stuck",
        "drift",
        "timestamp_gap",
        "timestamp_duplicate",
        "timestamp_order",
        "timestamp_invalid",
        "cross_sensor_inconsistency",
    ):
        # Do not automatically reconstruct structural or domain-level safety issues
        return None

    if candidate is None or not np.isfinite(candidate):
        return None
    return {
        "candidate_id": f"CAND-{issue['issue_id']}",
        "issue_id": issue["issue_id"],
        "method": method,
        "original_value": original,
        "candidate_value": candidate,
        "assumptions": ["Local temporal context is representative."],
    }
