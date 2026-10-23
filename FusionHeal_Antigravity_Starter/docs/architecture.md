# Architecture

1. **Ingestion** reads CSV and creates stable row IDs.
2. **Detection** flags missing values, configured range violations, and robust rolling outliers.
3. **Reconstruction** proposes interpolation for interior missing values or local median for outliers.
4. **Decision** checks candidate limits and applies explicit acceptance policy.
5. **Audit** records issue, candidate, decision, reason, and run timestamp as JSONL.
6. **Reporting** emits processed CSV and JSON summary; Streamlit presents them.

## Prototype limitations
- Rolling statistics are centered and suitable only for offline demonstration.
- No cross-sensor inference is enabled yet.
- Heuristic confidence values are not calibrated probabilities.
- Timestamp gaps, stuck sensors, drift, unit conversion, and multi-sensor fault linking are planned follow-ups.
- Example sensor limits are illustrative only.
