from __future__ import annotations
import sys
from pathlib import Path

# Ensure src and root are in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
src_dir = root_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import json
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone

from app.config import get_settings
from app.storage.database import SessionLocal, init_db
from app.storage.models import RawReading, QualityEvent, AcceptedReading, AuditLog, EvaluationMetricRun
from app.storage.repository import TelemetryRepository
from app.audit.logger import AuditService
from app.ingestion.schemas import ReviewDecisionInput
from app.api.events import _process_review_decision
from app.evaluation.metrics import run_reproducible_benchmark
from app.simulator.generator import SyntheticSensorGenerator
from app.simulator.fault_injector import FaultInjector
from app.pipeline.processing import PipelineProcessor

# Try to import legacy offline pipeline runner
try:
    from fusionheal.pipeline import run_pipeline
    HAS_LEGACY_PIPELINE = True
except ImportError:
    HAS_LEGACY_PIPELINE = False

# Ensure DB tables exist
init_db()
settings = get_settings()

st.set_page_config(
    page_title="FusionHeal — Industrial Data-Quality Gateway",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ FusionHeal — Industrial Data-Quality Gateway")
st.caption("Inline Industrial Sensor Data-Quality Intelligence, Multi-Detector Diagnostics, and Evidence-Governed Auditable Repair")

# Quick helper functions to query backend database
def get_db():
    return SessionLocal()


def load_overview_data():
    with get_db() as db:
        repo = TelemetryRepository(db)
        return repo.get_overview_counts()


def load_recent_readings(limit=150):
    with get_db() as db:
        repo = TelemetryRepository(db)
        readings = repo.get_accepted_readings(limit=limit)
        data = []
        for r in readings:
            data.append({
                "timestamp": r.event_timestamp or r.timestamp,
                "asset_id": r.asset_id,
                "sensor_id": r.sensor_id,
                "reading_id": r.reading_id,
                "original_value": r.original_value,
                "candidate_value": r.candidate_value,
                "final_value": r.final_value,
                "unit": r.unit,
                "decision_status": r.decision_status or r.status,
                "quality_status": r.quality_status,
                "human_review_required": r.human_review_required,
                "validation_status": r.validation_status,
                "reason": r.reason,
            })
        return pd.DataFrame(data)


def load_events(limit=200):
    with get_db() as db:
        repo = TelemetryRepository(db)
        events = repo.get_events(limit=limit)
        data = []
        for e in events:
            ev_json = {}
            try:
                ev_json = json.loads(e.observed_evidence) if e.observed_evidence else {}
            except Exception:
                ev_json = {"raw": e.observed_evidence}

            data.append({
                "event_id": e.event_id,
                "reading_id": e.reading_id,
                "asset_id": e.asset_id,
                "sensor_id": e.sensor_id,
                "timestamp": e.event_timestamp or e.timestamp,
                "issue_type": e.issue_type,
                "quality_status": e.quality_status,
                "decision_status": e.decision_status,
                "severity": e.severity,
                "original_value": e.original_value,
                "candidate_value": e.candidate_value,
                "final_value": e.final_value,
                "human_review_required": e.human_review_required,
                "detection_rule": e.detection_rule,
                "reason": e.reason,
                "observed_evidence": ev_json,
            })
        return pd.DataFrame(data)


def load_pending_reviews(limit=100):
    with get_db() as db:
        repo = TelemetryRepository(db)
        events = repo.get_pending_reviews(limit=limit)
        data = []
        for e in events:
            data.append({
                "event_id": e.event_id,
                "reading_id": e.reading_id,
                "asset_id": e.asset_id,
                "sensor_id": e.sensor_id,
                "timestamp": e.event_timestamp or e.timestamp,
                "issue_type": e.issue_type,
                "severity": e.severity,
                "original_value": e.original_value,
                "candidate_value": e.candidate_value,
                "final_value": e.final_value,
                "reason": e.reason,
                "detection_rule": e.detection_rule,
            })
        return pd.DataFrame(data)


def load_audit_logs(limit=100):
    with get_db() as db:
        repo = TelemetryRepository(db)
        logs = repo.get_audit_logs(limit=limit)
        data = []
        for l in logs:
            data.append({
                "audit_id": l.audit_id,
                "timestamp": l.created_at or l.timestamp,
                "action": l.action,
                "actor": l.actor,
                "asset_id": l.asset_id,
                "sensor_id": l.sensor_id,
                "reading_id": l.reading_id,
                "event_id": l.event_id,
                "prev_status": l.prev_status,
                "new_status": l.new_status or l.decision_status,
                "original_value": l.original_value,
                "accepted_value": l.accepted_value,
                "validation_result": l.validation_result,
                "decision_reason": l.decision_reason,
                "record_hash": l.record_hash,
                "prev_record_hash": l.prev_record_hash,
            })
        return pd.DataFrame(data)


def load_latest_metrics():
    with get_db() as db:
        repo = TelemetryRepository(db)
        metric = repo.get_latest_metrics()
        if not metric:
            return None
        details = {}
        if metric.details_json:
            try:
                details = json.loads(metric.details_json)
            except Exception:
                details = {}
        baseline_comp = {}
        if metric.baseline_comparison_json:
            try:
                baseline_comp = json.loads(metric.baseline_comparison_json)
            except Exception:
                baseline_comp = {}
        elif "baseline_comparison" in details:
            baseline_comp = details["baseline_comparison"]

        return {
            "run_id": metric.run_id,
            "scenario_name": metric.scenario_name,
            "precision": metric.precision,
            "recall": metric.recall,
            "f1_score": metric.f1_score,
            "mae": metric.mae,
            "rmse": metric.rmse,
            "latency_ms": metric.latency_ms,
            "accepted_repairs": metric.accepted_repairs,
            "unresolved_events": metric.unresolved_events,
            "unintentional_modifications": metric.unintentional_modifications,
            "details": details,
            "baseline_comparison": baseline_comp,
            "created_at": metric.created_at,
        }


# =============================================================================
# Sidebar Controls & Live Telemetry Ingestion Simulator
# =============================================================================
with st.sidebar:
    st.header("⚙️ Gateway Controls")
    st.info(f"**Policy Version**: `{settings.policy_version}`\n\n**Rule Config**: `{settings.rule_config_version}`")

    if st.button("🔄 Refresh Live Dashboard", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.subheader("📡 Live Telemetry Generator")
    st.caption("Inject simulated telemetry with controlled faults into the live database.")

    selected_scenario = st.selectbox(
        "Fault Injection Scenario",
        [
            "clean",
            "missing_value",
            "communication_gap",
            "frozen",
            "sudden_spike",
            "gradual_drift",
            "invalid_corrupted",
        ],
        index=0,
    )
    ticks_to_generate = st.slider("Readings to Generate", min_value=2, max_value=20, value=6, step=2)

    if st.button("⚡ Inject Telemetry Ticks", type="primary", use_container_width=True):
        try:
            with st.spinner("Generating and processing telemetry ticks..."):
                gen = SyntheticSensorGenerator()
                inj = FaultInjector()
                proc = PipelineProcessor(settings)
                with get_db() as db:
                    now = datetime.now(timezone.utc)
                    for i in range(ticks_to_generate // 2):
                        ts = now.replace(second=(now.second + i) % 60)
                        # Boiler
                        b_rd, b_clean = gen.generate_boiler_temperature(timestamp=ts)
                        mod_b, _ = inj.inject_fault(b_rd, b_clean, selected_scenario)
                        if mod_b is not None:
                            proc.process_reading(mod_b, db)
                        # Pump
                        p_rd, p_clean = gen.generate_pump_vibration(timestamp=ts)
                        mod_p, _ = inj.inject_fault(p_rd, p_clean, selected_scenario)
                        if mod_p is not None:
                            proc.process_reading(mod_p, db)

                st.success(f"Injected {ticks_to_generate} readings with scenario '{selected_scenario}'!")
                st.rerun()
        except Exception as ex:
            st.error(f"Injection failed: {ex}")

    st.markdown("---")
    if st.button("🧪 Run Reproducible Benchmark", use_container_width=True):
        with st.spinner("Running full 6-fault benchmark evaluation..."):
            with get_db() as db:
                rep = run_reproducible_benchmark(seed=42, sample_count=60, db=db)
            st.success(f"Benchmark completed! F1={rep.f1_score}, MAE={rep.mae}")
            st.rerun()


# =============================================================================
# Main Dashboard Overview Cards
# =============================================================================
overview = load_overview_data()

st.markdown("### 📊 Operational Quality Overview")
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

kpi1.metric("Readings Received", overview.get("total_readings_received", 0))
kpi2.metric("Detected Issues", overview.get("issues_detected", 0))
kpi3.metric("Repairs Accepted", overview.get("repairs_accepted", 0), help="Officially REPAIR status with verified replacement")
kpi4.metric("Pending Review", overview.get("readings_under_review", 0), help="Officially REVIEW status requiring operator decision")
kpi5.metric("Preserved", overview.get("readings_preserved", 0), help="Officially PRESERVE status retaining raw telemetry")
kpi6.metric("Unresolved", overview.get("unresolved_events", 0), help="Officially UNRESOLVED status with preserved raw values")

# Fault distribution breakdown if issues exist
faults = overview.get("fault_distribution", {})
if faults:
    with st.expander("🔍 Anomaly Fault Distribution", expanded=False):
        fault_df = pd.DataFrame(list(faults.items()), columns=["Anomaly Issue Type", "Count"]).sort_values("Count", ascending=False)
        st.dataframe(fault_df, use_container_width=True, hide_index=True)


# =============================================================================
# Dashboard Tabs
# =============================================================================
tab_monitoring, tab_investigation, tab_review, tab_audit, tab_metrics, tab_batch = st.tabs([
    "📈 Live Telemetry Monitoring",
    "🔍 Event Investigation",
    "⚖️ Review Queue & Decision Workbench",
    "📜 Auditable History & Hash Chain",
    "🧪 Evaluation Metrics & Baselines",
    "📁 Batch CSV Processing (Offline)",
])


# -----------------------------------------------------------------------------
# TAB 1: Live Telemetry Monitoring
# -----------------------------------------------------------------------------
with tab_monitoring:
    st.subheader("Live Process Telemetry Streams")
    readings_df = load_recent_readings(limit=150)

    if readings_df.empty:
        st.info("No telemetry readings recorded yet in the gateway database. Click **'⚡ Inject Telemetry Ticks'** in the sidebar to simulate live data.")
    else:
        col_boiler, col_pump = st.columns(2)

        # Asset A: Industrial Boiler Temperature (0-150 C)
        with col_boiler:
            st.markdown("#### 🔥 Asset A: Industrial Boiler (`boiler_01:temperature_01`)")
            st.caption("Normal process range: 0.0°C – 150.0°C (Nominal baseline ~72°C)")
            boiler_df = readings_df[readings_df["sensor_id"] == "temperature_01"].copy()
            if not boiler_df.empty:
                boiler_df = boiler_df.sort_values("timestamp")
                chart_data = boiler_df[["timestamp", "original_value", "final_value"]].set_index("timestamp")
                st.line_chart(chart_data)
                
                # Show recent repairs / issues
                repairs_b = boiler_df[boiler_df["decision_status"] == "REPAIR"]
                if not repairs_b.empty:
                    st.caption(f"🔧 **{len(repairs_b)} repairs accepted** in current view window.")
            else:
                st.caption("No readings received yet for Boiler.")

        # Asset B: Industrial Pump Vibration (0-25 m/s2)
        with col_pump:
            st.markdown("#### ⚙️ Asset B: Industrial Pump (`pump_01:vibration_01`)")
            st.caption("Normal process range: 0.0 – 25.0 m/s² (Nominal baseline ~2.8 m/s²)")
            pump_df = readings_df[readings_df["sensor_id"] == "vibration_01"].copy()
            if not pump_df.empty:
                pump_df = pump_df.sort_values("timestamp")
                chart_data_p = pump_df[["timestamp", "original_value", "final_value"]].set_index("timestamp")
                st.line_chart(chart_data_p)

                repairs_p = pump_df[pump_df["decision_status"] == "REPAIR"]
                if not repairs_p.empty:
                    st.caption(f"🔧 **{len(repairs_p)} repairs accepted** in current view window.")
            else:
                st.caption("No readings received yet for Pump.")

        st.markdown("---")
        st.markdown("#### Recent Processed Telemetry Records")
        st.dataframe(readings_df.head(50), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: Event Investigation Details
# -----------------------------------------------------------------------------
with tab_investigation:
    st.subheader("Data-Quality Anomalies & Physical Rule Violations")
    events_df = load_events(limit=200)

    if events_df.empty:
        st.success("✅ Clean operation: No data-quality anomalies or physical violations detected yet.")
    else:
        # Filters
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            status_filter = st.multiselect(
                "Filter Decision Status",
                options=["PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"],
                default=["PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"],
            )
        with c_f2:
            issue_filter = st.multiselect(
                "Filter Issue Type",
                options=list(events_df["issue_type"].unique()),
                default=list(events_df["issue_type"].unique()),
            )
        with c_f3:
            asset_filter = st.multiselect(
                "Filter Asset",
                options=list(events_df["asset_id"].unique()),
                default=list(events_df["asset_id"].unique()),
            )

        filtered_events = events_df[
            events_df["decision_status"].isin(status_filter)
            & events_df["issue_type"].isin(issue_filter)
            & events_df["asset_id"].isin(asset_filter)
        ]

        st.caption(f"Showing {len(filtered_events)} of {len(events_df)} detected events.")
        st.dataframe(
            filtered_events[
                [
                    "event_id",
                    "timestamp",
                    "asset_id",
                    "sensor_id",
                    "issue_type",
                    "quality_status",
                    "decision_status",
                    "severity",
                    "original_value",
                    "candidate_value",
                    "final_value",
                    "human_review_required",
                    "reason",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("#### 🔬 Detailed Event Evidence Inspector")
        event_options = list(filtered_events["event_id"].values)
        if event_options:
            selected_eid = st.selectbox("Select Event ID to Inspect Evidence", event_options)
            chosen_row = filtered_events[filtered_events["event_id"] == selected_eid].iloc[0]

            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**Asset / Sensor**: `{chosen_row['asset_id']}` / `{chosen_row['sensor_id']}`")
            d1.markdown(f"**Issue Type**: `{chosen_row['issue_type']}`")
            d1.markdown(f"**Rule**: `{chosen_row['detection_rule']}`")

            d2.markdown(f"**Original Value**: `{chosen_row['original_value']}`")
            d2.markdown(f"**Candidate Value**: `{chosen_row['candidate_value']}`")
            d2.markdown(f"**Accepted Final Value**: `{chosen_row['final_value']}`")

            d3.markdown(f"**Quality Status**: `{chosen_row['quality_status']}`")
            d3.markdown(f"**Decision Status**: `{chosen_row['decision_status']}`")
            d3.markdown(f"**Human Review**: `{'REQUIRED' if chosen_row['human_review_required'] else 'NO'}`")

            st.markdown(f"**Technical Diagnostic Rationale**: {chosen_row['reason']}")
            st.json(chosen_row["observed_evidence"])


# -----------------------------------------------------------------------------
# TAB 3: Operator Review Queue & Decision Workbench
# -----------------------------------------------------------------------------
with tab_review:
    st.subheader("⚖️ Operator Review Queue & Decision Workbench")
    st.markdown(
        "Industrial operators can assess events requiring human evaluation, apply domain knowledge, "
        "and record auditable transitions between **PRESERVE**, **REPAIR**, **REVIEW**, and **UNRESOLVED**."
    )

    pending_df = load_pending_reviews(limit=100)

    if pending_df.empty:
        st.success("🎉 No pending events in the review queue. All telemetry decisions are fully resolved.")
    else:
        st.warning(f"⚠️ **{len(pending_df)} event(s) awaiting human operator review.**")
        st.dataframe(pending_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### ✍️ Record Operator Review Outcome")

        with st.form("operator_review_form"):
            selected_review_eid = st.selectbox(
                "Select Event to Review",
                options=list(pending_df["event_id"].values),
            )

            # Look up chosen event details
            chosen_evt = pending_df[pending_df["event_id"] == selected_review_eid].iloc[0]
            st.caption(
                f"Asset: **{chosen_evt['asset_id']}** | Sensor: **{chosen_evt['sensor_id']}** | "
                f"Issue: **{chosen_evt['issue_type']}** | Original: **{chosen_evt['original_value']}** | "
                f"Candidate: **{chosen_evt['candidate_value']}**"
            )

            c_rev1, c_rev2, c_rev3 = st.columns(3)
            with c_rev1:
                new_status = st.selectbox(
                    "New Official Decision Status",
                    ["PRESERVE", "REPAIR", "REVIEW", "UNRESOLVED"],
                    index=1,
                    help="PRESERVE: Keep original. REPAIR: Replace with approved value. UNRESOLVED: Flag as unable to correct. REVIEW: Keep in review.",
                )
            with c_rev2:
                default_cand = float(chosen_evt["candidate_value"]) if pd.notnull(chosen_evt["candidate_value"]) else 72.0
                approved_value = st.number_input(
                    "Approved Replacement Value (for REPAIR)",
                    value=default_cand,
                    help="Mandatory and validated against physical sensor bounds if status is REPAIR.",
                )
            with c_rev3:
                operator_actor = st.text_input("Operator Identifier", value="operator_01")

            review_reason = st.text_area(
                "Technical Justification / Rationale",
                value="Operator verified machine state and process logs. Value confirmed valid.",
                help="Explain the operational rationale for this decision transition.",
            )

            submit_btn = st.form_submit_button("💾 Commit Review Decision & Write Audit Log", type="primary")

            if submit_btn:
                try:
                    with get_db() as db:
                        review_input = ReviewDecisionInput(
                            new_status=new_status,
                            actor=operator_actor,
                            reason=review_reason,
                            accepted_value=approved_value if new_status == "REPAIR" else None,
                        )
                        result = _process_review_decision(selected_review_eid, review_input, db)
                    st.success(f"✅ Successfully transitioned event '{selected_review_eid}' to **{new_status}**! Final value: {result.final_value}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Review decision submission failed: {ex}")


# -----------------------------------------------------------------------------
# TAB 4: Auditable History & Hash Chain
# -----------------------------------------------------------------------------
with tab_audit:
    st.subheader("📜 Cryptographic Audit History")
    st.caption("Every pipeline decision, review transition, and parameter override is permanently recorded in a SHA-256 tamper-evident hash chain.")

    col_btn_v, col_lbl_v = st.columns([1, 3])
    with col_btn_v:
        verify_clicked = st.button("🔒 Verify Cryptographic Chain Integrity", use_container_width=True)

    if verify_clicked:
        with get_db() as db:
            repo = TelemetryRepository(db)
            audit_svc = AuditService(repo, settings)
            ver = audit_svc.verify_integrity()
        if ver.get("valid"):
            st.success(f"✅ Cryptographic verification PASSED: All {ver.get('count', 0)} audit records strictly chained without tampering!")
        else:
            st.error(f"❌ Cryptographic verification FAILED: Chain break detected at audit record '{ver.get('failed_audit_id')}'. Expected {ver.get('expected_prev')}, found {ver.get('actual_prev')}.")

    audit_df = load_audit_logs(limit=100)
    if audit_df.empty:
        st.info("No audit logs recorded yet in the database.")
    else:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# TAB 5: Evaluation Metrics & 3-Way Baseline Comparison
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("🧪 Quantitative Benchmark Evaluation")
    st.markdown("Rigorous quantitative verification of anomaly detection, data preservation, and fault repair against controlled ground truth.")

    latest_metric = load_latest_metrics()

    if not latest_metric:
        st.warning("⚠️ No controlled evaluation run recorded yet. Click the button below to execute a reproducible benchmark run.")
        if st.button("▶ Execute Initial Benchmark Run", type="primary"):
            with st.spinner("Executing benchmark across Boiler and Pump telemetry..."):
                with get_db() as db:
                    run_reproducible_benchmark(seed=42, sample_count=60, db=db)
                st.success("Benchmark completed!")
                st.rerun()
    else:
        details = latest_metric.get("details", {})
        det_sec = details.get("detection", {})
        rec_sec = details.get("reconstruction", {})
        pres_sec = details.get("preservation", {})
        perf_sec = details.get("performance", {})
        dec_sec = details.get("decision_policy", {})

        # Display Summary KPI cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precision", f"{latest_metric['precision']:.3f}" if latest_metric['precision'] is not None else "N/A")
        m2.metric("Recall", f"{latest_metric['recall']:.3f}" if latest_metric['recall'] is not None else "N/A")
        m3.metric("F1 Score", f"{latest_metric['f1_score']:.3f}" if latest_metric['f1_score'] is not None else "N/A")
        m4.metric("MAE", f"{latest_metric['mae']:.3f}" if latest_metric['mae'] is not None else "N/A", help="Mean Absolute Error of accepted repairs vs clean ground truth")
        m5.metric("RMSE", f"{latest_metric['rmse']:.3f}" if latest_metric['rmse'] is not None else "N/A")

        st.markdown("---")
        # 1. Confusion Matrix & Detection Breakdown
        st.markdown("#### 🎯 Anomaly Detection & Confusion Matrix")
        cm = det_sec.get("confusion_matrix", {})
        tp = cm.get("tp", 0)
        fp = cm.get("fp", 0)
        fn = cm.get("fn", 0)
        tn = cm.get("tn", 0)

        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        cm_col1.metric("True Positives (TP)", tp, help="Correctly flagged anomalies")
        cm_col2.metric("False Positives (FP)", fp, help="Clean telemetry incorrectly flagged")
        cm_col3.metric("False Negatives (FN)", fn, help="Injected faults missed")
        cm_col4.metric("True Negatives (TN)", tn, help="Clean telemetry correctly preserved")

        # Per Fault Type Breakdown Table
        st.markdown("##### 🔍 Results by Injected Fault Type")
        by_ft = det_sec.get("by_fault_type", {})
        mae_ft = rec_sec.get("mae_by_fault_type", {})
        rmse_ft = rec_sec.get("rmse_by_fault_type", {})

        ft_rows = []
        for ft_name, ft_stats in by_ft.items():
            ft_rows.append({
                "Fault Mode": ft_name,
                "Tested Readings": ft_stats.get("tested_count", 0),
                "TP": ft_stats.get("tp", 0),
                "FN": ft_stats.get("fn", 0),
                "FP": ft_stats.get("fp", 0),
                "Precision": f"{ft_stats.get('precision', 0.0):.3f}",
                "Recall": f"{ft_stats.get('recall', 0.0):.3f}",
                "F1 Score": f"{ft_stats.get('f1_score', 0.0):.3f}",
                "Reconstruction MAE": f"{mae_ft.get(ft_name, 'N/A')}",
                "Reconstruction RMSE": f"{rmse_ft.get(ft_name, 'N/A')}",
            })
        if ft_rows:
            st.dataframe(pd.DataFrame(ft_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No per-fault breakdown data available in this run.")

        st.markdown("---")
        # 2. Decision Policy & Preservation Quality
        st.markdown("#### 🛡️ Decision Engine Outcomes & Preservation Metrics")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("PRESERVE Count", dec_sec.get("preserved_readings", 0), help="Clean readings untouched")
        d2.metric("REPAIR Count", dec_sec.get("accepted_repairs", latest_metric['accepted_repairs']), help="Accepted validated reconstructions")
        d3.metric("REVIEW Count", dec_sec.get("review_events", 0), help="Ambiguous cases routed to operator queue")
        d4.metric("UNRESOLVED Count", dec_sec.get("unresolved_events", latest_metric['unresolved_events']), help="Failed bounds or no valid repair")

        pr1, pr2, pr3, pr4 = st.columns(4)
        pres_pct = pres_sec.get("known_valid_preserved_percentage", 100.0)
        unnec_pct = pres_sec.get("unnecessary_modification_rate", 0.0) * 100.0
        rep_acc_pct = dec_sec.get("repair_acceptance_rate", 0.0) * 100.0
        rev_pct = dec_sec.get("review_rate", 0.0) * 100.0
        pr1.metric("Preservation Accuracy", f"{pres_pct:.1f}%")
        pr2.metric("Unnecessary Modification Rate", f"{unnec_pct:.2f}%")
        pr3.metric("Repair Acceptance Rate", f"{rep_acc_pct:.1f}%")
        pr4.metric("Review Routing Rate", f"{rev_pct:.1f}%")

        st.markdown("---")
        # 3. Reconstruction Accuracy by Industrial Asset
        st.markdown("#### 🏭 Reconstruction Quality by Asset")
        mae_asset = rec_sec.get("mae_by_asset", {})
        rmse_asset = rec_sec.get("rmse_by_asset", {})

        as_col1, as_col2 = st.columns(2)
        with as_col1:
            st.markdown("**Industrial Boiler (`boiler_01` — Temperature in °C)**")
            b_mae = mae_asset.get("boiler_01", "N/A")
            b_rmse = rmse_asset.get("boiler_01", "N/A")
            st.metric("Boiler MAE", f"{b_mae:.3f} °C" if isinstance(b_mae, (int, float)) else str(b_mae))
            st.metric("Boiler RMSE", f"{b_rmse:.3f} °C" if isinstance(b_rmse, (int, float)) else str(b_rmse))

        with as_col2:
            st.markdown("**Industrial Pump (`pump_01` — Vibration in m/s²)**")
            p_mae = mae_asset.get("pump_01", "N/A")
            p_rmse = rmse_asset.get("pump_01", "N/A")
            st.metric("Pump MAE", f"{p_mae:.3f} m/s²" if isinstance(p_mae, (int, float)) else str(p_mae))
            st.metric("Pump RMSE", f"{p_rmse:.3f} m/s²" if isinstance(p_rmse, (int, float)) else str(p_rmse))

        # Operational Latency
        st.markdown("##### ⏱️ Operational Latency & Throughput")
        lat1, lat2, lat3, lat4 = st.columns(4)
        lat1.metric("Avg Latency", f"{perf_sec.get('avg_processing_latency_ms', latest_metric['latency_ms']):.2f} ms")
        lat2.metric("Min Latency", f"{perf_sec.get('min_processing_latency_ms', 0.0):.2f} ms")
        lat3.metric("Max Latency", f"{perf_sec.get('max_processing_latency_ms', 0.0):.2f} ms")
        lat4.metric("Throughput", f"{perf_sec.get('throughput_readings_per_sec', 0.0):.1f} readings/s")

        st.markdown("---")
        st.markdown("### 🏆 3-Way Baseline Comparison")
        st.markdown(
            "Comparison of the complete industrial data-quality pipeline against standard industrial baselines:"
        )

        b_comp = latest_metric.get("baseline_comparison", {})
        if b_comp:
            b1 = b_comp.get("baseline_1_no_repair", {})
            b2 = b_comp.get("baseline_2_linear_interpolation", {})
            b3 = b_comp.get("baseline_3_fusionheal", {})

            comp_table = [
                {
                    "Method": b1.get("name", "Baseline 1: No-Repair Baseline"),
                    "MAE": b1.get("mae", "N/A"),
                    "RMSE": b1.get("rmse", "N/A"),
                    "Missing Unaddressed": b1.get("unrepaired_missing_count", "N/A"),
                    "Description": b1.get("description", "Raw uncorrected telemetry"),
                },
                {
                    "Method": b2.get("name", "Baseline 2: Linear Interpolation"),
                    "MAE": b2.get("mae", "N/A"),
                    "RMSE": b2.get("rmse", "N/A"),
                    "Missing Unaddressed": b2.get("unrepaired_missing_count", "N/A"),
                    "Description": b2.get("description", "Strictly interpolates missing values; ignores spikes & drift"),
                },
                {
                    "Method": b3.get("name", "Baseline 3: FusionHeal Gateway"),
                    "MAE": b3.get("mae", "N/A"),
                    "RMSE": b3.get("rmse", "N/A"),
                    "Missing Unaddressed": 0,
                    "Description": b3.get("description", "Multi-detector physical validation & evidence-governed repair"),
                },
            ]
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)
        else:
            st.info("Baseline comparison metrics not available in this run. Re-run benchmark to calculate.")

        st.markdown("---")
        st.markdown("#### 🔁 Run Custom Reproducible Benchmark")
        with st.form("custom_benchmark_form"):
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                bench_seed = st.number_input("Random Seed", value=42, min_value=1, max_value=9999)
            with c_s2:
                bench_samples = st.slider("Sample Count", min_value=20, max_value=200, value=60, step=10)

            run_bench_btn = st.form_submit_button("▶ Run Benchmark Experiment")
            if run_bench_btn:
                with st.spinner(f"Running benchmark with seed={bench_seed}, samples={bench_samples}..."):
                    with get_db() as db:
                        rep = run_reproducible_benchmark(seed=bench_seed, sample_count=bench_samples, db=db)
                    st.success(f"Benchmark finished! Run ID: {rep.run_id}")
                    st.rerun()



# -----------------------------------------------------------------------------
# TAB 6: Batch CSV Processing (Offline Mode)
# -----------------------------------------------------------------------------
with tab_batch:
    st.subheader("📁 Batch CSV Processing (Offline Analysis)")
    st.caption("Upload static industrial CSV files for historical batch quality cleansing and reconstruction.")

    demo_path = Path("data/synthetic/demo_sensor_data.csv")
    config_path = Path("config.yaml")

    if HAS_LEGACY_PIPELINE:
        c_up1, c_up2 = st.columns([2, 1])
        with c_up1:
            uploaded_csv = st.file_uploader("Upload Sensor CSV", type=["csv"], key="batch_csv_uploader")
        with c_up2:
            st.markdown("**Bundled Demo Dataset:**")
            if demo_path.exists():
                if st.button("▶ Run Bundled Demo CSV", use_container_width=True):
                    try:
                        with st.spinner("Processing demo dataset..."):
                            summary_res = run_pipeline(demo_path, config_path, "outputs")
                            st.session_state["batch_summary"] = summary_res
                            st.session_state["batch_data"] = pd.read_csv("outputs/processed_data.csv")
                            st.success("Batch pipeline completed!")
                    except Exception as ex:
                        st.error(f"Execution failed: {ex}")

        if uploaded_csv is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                in_p = Path(temp_dir) / uploaded_csv.name
                in_p.write_bytes(uploaded_csv.getvalue())
                try:
                    with st.spinner("Processing uploaded dataset..."):
                        summary_res = run_pipeline(in_p, config_path, Path(temp_dir) / "outputs")
                        st.session_state["batch_summary"] = summary_res
                        st.session_state["batch_data"] = pd.read_csv(Path(temp_dir) / "outputs" / "processed_data.csv")
                        st.success("Uploaded dataset processed!")
                except Exception as ex:
                    st.error(f"Processing failed: {ex}")

        if "batch_summary" in st.session_state and "batch_data" in st.session_state:
            st.markdown("#### Batch Processing Results")
            st.json(st.session_state["batch_summary"])
            st.dataframe(st.session_state["batch_data"], use_container_width=True)
            st.download_button(
                "📥 Download Cleaned CSV",
                st.session_state["batch_data"].to_csv(index=False),
                file_name="fusionheal_cleaned.csv",
                mime="text/csv",
            )
    else:
        st.info("Legacy batch pipeline module not available in current environment.")

