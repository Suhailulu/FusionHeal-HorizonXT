from __future__ import annotations
from pathlib import Path
import json
import tempfile
import pandas as pd
import streamlit as st
from fusionheal.pipeline import run_pipeline

st.set_page_config(page_title="FusionHeal", layout="wide", page_icon="🛡️")
st.title("🛡️ FusionHeal")
st.caption("Evidence-guided and process-constrained industrial sensor data-quality intelligence")
st.warning("⚠️ Prototype only. Configured demo constraints are illustrative and not certified industrial safety limits.")

demo_path = Path("data/synthetic/demo_sensor_data.csv")
config_path = Path("config.yaml")

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Reset / Clear View", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Quick Actions:**")
    if demo_path.exists():
        if st.button("▶ Run Demo Pipeline", type="primary", use_container_width=True):
            try:
                with st.spinner("Executing FusionHeal Intelligence Pipeline..."):
                    summary = run_pipeline(demo_path, config_path, "outputs")
                    st.session_state["summary"] = summary
                    st.session_state["result_data"] = pd.read_csv("outputs/processed_data.csv")
                    st.session_state["audit_text"] = Path("outputs/audit.jsonl").read_text(encoding="utf-8")
                    st.success("Pipeline execution complete!")
                    st.rerun()
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

    outputs_summary_path = Path("outputs/summary.json")
    if outputs_summary_path.exists() and "summary" not in st.session_state:
        if st.button("📂 Load Previous Run from outputs/", use_container_width=True):
            try:
                st.session_state["summary"] = json.loads(outputs_summary_path.read_text(encoding="utf-8"))
                st.session_state["result_data"] = pd.read_csv("outputs/processed_data.csv")
                st.session_state["audit_text"] = Path("outputs/audit.jsonl").read_text(encoding="utf-8")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load outputs: {exc}")

uploaded = st.file_uploader("Upload a sensor CSV", type=["csv"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / uploaded.name
        input_path.write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Processing uploaded dataset..."):
                summary = run_pipeline(input_path, config_path, Path(temp_dir) / "outputs")
                st.session_state["summary"] = summary
                st.session_state["result_path"] = str(Path(temp_dir) / "outputs" / "processed_data.csv")
                st.session_state["audit_path"] = str(Path(temp_dir) / "outputs" / "audit.jsonl")
                st.session_state["result_data"] = pd.read_csv(Path(temp_dir) / "outputs" / "processed_data.csv")
                st.session_state["audit_text"] = (Path(temp_dir) / "outputs" / "audit.jsonl").read_text(encoding="utf-8")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

if "summary" in st.session_state:
    summary = st.session_state["summary"]
    st.markdown("### 📊 Pipeline Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Input Rows", summary.get("input_rows", 0))
    c2.metric("Issues Detected", summary.get("issues_detected", 0))
    c3.metric("Accepted", summary.get("accepted", 0))
    c4.metric("Needs Review", summary.get("review", 0))
    c5.metric("Rejected", summary.get("rejected", 0))

    tab1, tab2, tab3 = st.tabs(["Processed Data", "Run Summary", "Audit Records"])

    with tab1:
        st.subheader("Cleaned & Reconstructed Dataset")
        st.dataframe(st.session_state["result_data"], use_container_width=True)
        c_down1, c_down2 = st.columns(2)
        with c_down1:
            st.download_button(
                "📥 Download processed CSV",
                st.session_state["result_data"].to_csv(index=False),
                file_name="processed_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_down2:
            st.download_button(
                "📥 Download audit JSONL",
                st.session_state["audit_text"],
                file_name="audit.jsonl",
                mime="application/x-ndjson",
                use_container_width=True,
            )

    with tab2:
        st.subheader("Pipeline Execution Metadata")
        st.json(summary)

    with tab3:
        st.subheader("Evidence & Audit Trail")
        audit_lines = st.session_state["audit_text"].strip().split("\n")
        st.caption(f"Showing {len(audit_lines)} audit records")
        parsed_records = []
        for line in audit_lines:
            try:
                parsed_records.append(json.loads(line))
            except Exception:
                pass
        if parsed_records:
            st.json(parsed_records[:10])
            if len(parsed_records) > 10:
                st.info(f"Showing first 10 of {len(parsed_records)} records. Download the complete JSONL above.")
else:
    st.info("Upload a CSV file above or click **'▶ Run Demo Pipeline'** in the sidebar to run with the bundled dataset.")
    if demo_path.exists():
        st.subheader("Bundled Demo Dataset Preview (`data/synthetic/demo_sensor_data.csv`)")
        st.dataframe(pd.read_csv(demo_path).head(20), use_container_width=True)
