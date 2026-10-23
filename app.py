from pathlib import Path
import json
import tempfile
import pandas as pd
import streamlit as st
from fusionheal.pipeline import run_pipeline

st.set_page_config(page_title="FusionHeal", layout="wide")
st.title("FusionHeal")
st.caption("Evidence-guided industrial sensor data quality prototype")
st.warning("Prototype only. Demo constraints are not approved industrial safety limits.")

uploaded = st.file_uploader("Upload a sensor CSV", type=["csv"])
demo_path = Path("data/synthetic/demo_sensor_data.csv")
config_path = Path("config.yaml")

if uploaded is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / uploaded.name
        input_path.write_bytes(uploaded.getvalue())
        try:
            summary = run_pipeline(input_path, config_path, Path(temp_dir) / "outputs")
            st.session_state["summary"] = summary
            st.session_state["result_path"] = str(Path(temp_dir) / "outputs" / "processed_data.csv")
            st.session_state["audit_path"] = str(Path(temp_dir) / "outputs" / "audit.jsonl")
            # Temporary files are removed after this block; show data before cleanup.
            result = pd.read_csv(Path(temp_dir) / "outputs" / "processed_data.csv")
            audit_text = (Path(temp_dir) / "outputs" / "audit.jsonl").read_text(encoding="utf-8")
            st.session_state["result_data"] = result
            st.session_state["audit_text"] = audit_text
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

if "summary" in st.session_state:
    summary = st.session_state["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", summary["input_rows"])
    c2.metric("Issues", summary["issues_detected"])
    c3.metric("Accepted", summary["accepted"])
    c4.metric("Needs review", summary["review"])
    st.subheader("Run summary")
    st.json(summary)
    st.subheader("Processed data")
    st.dataframe(st.session_state["result_data"], use_container_width=True)
    st.download_button("Download processed CSV", st.session_state["result_data"].to_csv(index=False),
                       file_name="processed_data.csv", mime="text/csv")
    st.download_button("Download audit JSONL", st.session_state["audit_text"],
                       file_name="audit.jsonl", mime="application/x-ndjson")
else:
    st.info("Upload a CSV to run the pipeline. The bundled demo file is available at data/synthetic/demo_sensor_data.csv.")
    if demo_path.exists():
        st.subheader("Bundled demo dataset preview")
        st.dataframe(pd.read_csv(demo_path).head(20), use_container_width=True)
