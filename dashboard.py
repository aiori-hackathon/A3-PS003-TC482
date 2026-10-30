from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="HiLo-Fed Research Dashboard", layout="wide")
st.title("HiLo-Fed | Hierarchical Federated Traffic Forecasting")
st.caption("Research prototype using the provided METR-LA / PEMS-BAY datasets. Displayed values are loaded from experiment output files.")

choices = {"METR-LA": "", "PEMS-BAY": "_pems_bay"}
dataset_label = st.sidebar.selectbox("Dataset results", list(choices))
suffix = choices[dataset_label]
results = ROOT / "results"
metrics_path = results / f"forecast_metrics{suffix}.csv"
costs_path = results / f"communication_costs{suffix}.csv"
round_path = results / f"round_logs{suffix}.csv"
meta_path = results / f"run_metadata{suffix}.json"
if not metrics_path.exists() or not costs_path.exists():
    st.warning("No results found yet. Run `python -m hilofed.pipeline --dataset both --fast` first.")
    st.stop()
metrics = pd.read_csv(metrics_path)
costs = pd.read_csv(costs_path)
rounds = pd.read_csv(round_path) if round_path.exists() else pd.DataFrame()
metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

c1,c2,c3,c4 = st.columns(4)
c1.metric("Dataset", dataset_label)
c2.metric("Sensors", metadata.get("sensor_count", "—"))
c3.metric("Training rounds", metadata.get("rounds", "—"))
c4.metric("Clients / edges", f"{metadata.get('client_count','—')} / {metadata.get('edge_count','—')}")
if metadata:
    st.info(f"Run mode: {'fast subset' if metadata.get('fast_mode') else 'full dataset'} · Rows: {metadata.get('rows_used','—')} · Seed: {metadata.get('seed','—')} · Runtime: {metadata.get('runtime_seconds',0):.1f}s")

methods = sorted(metrics.method.unique())
selected = st.multiselect("Methods", methods, default=[x for x in ["HiLo-Fed", "FedAvg", "FedAvg + INT8", "Centralized", "Last-value"] if x in methods])
horizon = st.select_slider("Forecast horizon", options=[15,30,60], value=60)
metric_name = st.radio("Forecast error", ["rmse_mph", "mae_mph"], horizontal=True, format_func=lambda x: "RMSE (mph)" if x=="rmse_mph" else "MAE (mph)")
view = metrics[(metrics.horizon_minutes == horizon) & metrics.method.isin(selected)].sort_values(metric_name)
st.subheader(f"Forecast accuracy · {horizon}-minute horizon")
st.dataframe(view[["method","horizon_minutes",metric_name,"n_valid"]], use_container_width=True, hide_index=True)
chart = metrics[metrics.method.isin(selected)].pivot(index="horizon_minutes",columns="method",values=metric_name)
st.line_chart(chart)

st.subheader("Measured communication cost")
finite_costs = costs[pd.to_numeric(costs.total_bytes,errors="coerce").notna()].copy()
if not finite_costs.empty:
    st.bar_chart(finite_costs.set_index("method")[["total_bytes"]])
    st.dataframe(finite_costs[["method","client_upload_bytes","client_downlink_bytes","edge_upload_bytes","edge_downlink_bytes","total_bytes","upload_reduction_pct"]], use_container_width=True, hide_index=True)
else:
    st.write("No federated communication measurements are available yet.")
st.caption("Centralized and non-trained reference baselines have no comparable federated-update communication total. Communication accounting includes encoded model payloads and a 16-byte message header; it is not a packet-capture measurement.")

if not rounds.empty:
    st.subheader("Round-by-round behavior")
    round_methods = st.multiselect("Round-log methods", sorted(rounds.method.unique()), default=["HiLo-Fed"] if "HiLo-Fed" in rounds.method.unique() else [])
    rview = rounds[rounds.method.isin(round_methods)].sort_values(["method","round"])
    st.dataframe(rview, use_container_width=True, hide_index=True)
    if not rview.empty:
        st.line_chart(rview.pivot(index="round",columns="method",values="validation_mae_60m_mph"))

st.subheader("Run metadata and caveats")
with st.expander("Inspect run metadata"):
    st.json(metadata)
st.caption("This is a reconstructed research prototype, not recovered original source code. Validate full-dataset and multi-seed results before making research claims.")
