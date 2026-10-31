from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from hilofed.core import load_dataset
from hilofed.ui_service import run_experiment

st.set_page_config(page_title="HiLo-Fed | Spatial Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080d16; --panel:#101925; --line:#223247; --text:#e7eef8; --muted:#8fa2b8; --cyan:#49d7e8; --amber:#ffbd69; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background:radial-gradient(ellipse at 30% -10%, #15263a 0%, #080d16 52%); color:var(--text); }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.2rem; max-width:1600px; }
[data-testid="stSidebar"] { background:#0b121d; border-right:1px solid #1e2c3e; }
[data-testid="stMetric"] { background:#101925; border:1px solid #223247; border-radius:14px; padding:14px 16px; }
[data-testid="stMetricLabel"] { color:#91a5bb; }
[data-testid="stMetricValue"] { color:#eaf4ff; }
h1,h2,h3 { letter-spacing:-.04em; }
.eyebrow { color:#49d7e8; font:500 11px 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; }
.hero { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:8px 0 20px; }
.brand { font-size:30px; font-weight:800; letter-spacing:-.06em; }
.brand span { color:#49d7e8; }
.subtitle { color:#91a5bb; font-size:13px; margin-top:3px; }
.status { border:1px solid #25535d; color:#7fe8d7; background:#0d282c; padding:7px 11px; border-radius:99px; font:11px 'DM Mono',monospace; white-space:nowrap; }
.panel-title { font-size:16px; font-weight:750; margin-bottom:3px; }
.panel-sub { color:#91a5bb; font-size:12px; margin-bottom:12px; }
.small-mono { font:11px 'DM Mono',monospace; color:#91a5bb; }
div[data-testid="stTabs"] button { font-weight:650; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = ROOT / "data"
@st.cache_data(show_spinner=False)
def load_bundle(dataset):
    bundle = load_dataset(DATA_DIR, dataset, fast=False)
    locations_path = DATA_DIR / ("metr_la/graph_sensor_locations.csv" if dataset == "metr_la" else "pems_bay/graph_sensor_locations_bay.csv")
    loc = pd.read_csv(locations_path)
    if dataset == "metr_la":
        loc["sensor_id"] = loc["sensor_id"].astype(str)
        loc = loc[["sensor_id", "latitude", "longitude"]]
    else:
        # PEMS-BAY location file is a headerless edge-like table; first row is data.
        loc = pd.read_csv(locations_path, header=None, names=["sensor_id", "latitude", "longitude"])
        loc = loc.drop_duplicates("sensor_id")
        loc["sensor_id"] = loc["sensor_id"].astype(str)
    loc["latitude"] = pd.to_numeric(loc["latitude"], errors="coerce")
    loc["longitude"] = pd.to_numeric(loc["longitude"], errors="coerce")
    loc = loc.dropna(subset=["latitude", "longitude"])
    return bundle, loc

st.sidebar.markdown('<div class="eyebrow">HI LO / FEDERATED INTELLIGENCE</div>', unsafe_allow_html=True)
st.sidebar.title("Workspace")
dataset_label = st.sidebar.selectbox("Traffic dataset", ["METR-LA", "PEMS-BAY"])
dataset = "metr_la" if dataset_label == "METR-LA" else "pems_bay"
view = st.sidebar.radio("Workspace", ["Spatial overview", "Forecast explorer", "Federated topology", "Communication lab", "Experiments & reports"])
st.sidebar.divider()
st.sidebar.caption("Research controls")
metric = st.sidebar.selectbox("Map layer", ["Sensor network", "Mean traffic speed", "Traffic variability"])
st.sidebar.caption("Map points represent sensor locations from the supplied metadata. Network links are shown only where source graph metadata is available.")

with st.sidebar.expander("Run a real experiment", expanded=False):
    st.caption("Executes the HiLo-Fed training pipeline on the selected original dataset. Results are written to results/ and then read by this dashboard.")
    run_rounds = st.slider("Federated rounds", min_value=1, max_value=10, value=3)
    run_rows = st.select_slider("Fast-run rows", options=[300, 600, 900, 1200, 1800, 2400], value=1200)
    run_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=17, step=1)
    run_partition = st.selectbox("Client partition", ["sensor", "iid"], index=0)
    run_clicked = st.button("Run HiLo-Fed pipeline", type="primary", use_container_width=True)
if run_clicked:
    try:
        with st.spinner(f"Training federated models on {dataset_label}. This can take several minutes..."):
            run_result = run_experiment(
                ROOT, dataset, rounds=int(run_rounds), max_rows=int(run_rows),
                seed=int(run_seed), partition=run_partition,
            )
        st.session_state["last_run_summary"] = run_result["summary"]
        st.session_state["last_run_stdout"] = run_result["stdout"]
        st.session_state["last_run_dataset"] = dataset_label
        st.success(f"Pipeline completed for {dataset_label}. Dashboard artifacts refreshed.")
        st.rerun()
    except Exception as exc:
        st.error(f"Experiment failed: {exc}")
        st.stop()

if "last_run_summary" in st.session_state:
    with st.expander(f"Last pipeline run · {st.session_state.get('last_run_dataset', 'dataset')}", expanded=False):
        st.json(st.session_state["last_run_summary"])
        st.code(st.session_state.get("last_run_stdout", ""), language="text")

bundle, locations = load_bundle(dataset)
sensor_ids = bundle.sensor_ids
if not locations.empty:
    locations["sensor_id"] = locations["sensor_id"].astype(str)
    locations = locations[locations.sensor_id.isin(sensor_ids)].copy()
    # Reindex values to the sensor IDs actually available in the traffic matrix.
    locations = locations.drop_duplicates("sensor_id")
    loc_lookup = locations.set_index("sensor_id")
    common_ids = [sid for sid in sensor_ids if sid in loc_lookup.index]
    locations = loc_lookup.loc[common_ids].reset_index()
else:
    common_ids = []
values = bundle.raw
overall_mean = float(np.nanmean(np.where(bundle.mask, values, np.nan)))
overall_std = float(np.nanstd(np.where(bundle.mask, values, np.nan)))
st.markdown(f"""
<div class="hero">
 <div><div class="eyebrow">SPATIAL INTELLIGENCE PLATFORM · PHASE 3 · INTEGRATED EXPERIMENTS</div>
 <div class="brand">HiLo<span>·</span>Fed</div>
 <div class="subtitle">Hierarchical learning for traffic forecasting and communication-aware intelligence</div></div>
 <div class="status">● DATASET LOADED · {dataset_label}</div>
</div>
""", unsafe_allow_html=True)

m1,m2,m3,m4 = st.columns(4)
m1.metric("Traffic sensors", f"{len(sensor_ids):,}")
m2.metric("Time records", f"{len(bundle.raw):,}")
m3.metric("Mean observed value", f"{overall_mean:.2f}")
m4.metric("Graph edges supplied", f"{len(bundle.distances):,}")

def sensor_frame():
    if not locations.empty:
        frame = locations.copy()
        # Metrics are aligned by sensor ID using the bundle's sensor order.
        pos = {sid:i for i,sid in enumerate(sensor_ids)}
        frame["sensor_index"] = frame.sensor_id.map(pos)
        frame["mean_value"] = frame.sensor_index.map(lambda i: float(np.mean(bundle.raw[:, int(i)][bundle.mask[:, int(i)]])) if i is not None and bundle.mask[:, int(i)].any() else np.nan)
        frame["variability"] = frame.sensor_index.map(lambda i: float(np.std(bundle.raw[:, int(i)][bundle.mask[:, int(i)]])) if i is not None and bundle.mask[:, int(i)].any() else np.nan)
        return frame.dropna(subset=["latitude","longitude"])
    return pd.DataFrame(columns=["sensor_id","latitude","longitude","mean_value","variability"])

sf = sensor_frame()
if view == "Spatial overview":
    left, right = st.columns([2.25, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">Sensor network / geographic view</div><div class="panel-sub">Explore the supplied sensor coordinates. Select a sensor in the right panel for detailed analysis.</div>', unsafe_allow_html=True)
        if len(sf):
            vals = sf["mean_value"] if metric == "Mean traffic speed" else sf["variability"] if metric == "Traffic variability" else pd.Series(np.ones(len(sf)), index=sf.index)
            lo, hi = float(vals.min()), float(vals.max())
            norm = (vals - lo) / max(hi-lo, 1e-9)
            sf["radius"] = 45 + norm * 120
            sf["color"] = norm.map(lambda x: [40, int(130+100*x), int(190+45*x), 220])
            import pydeck as pdk
            layer = pdk.Layer("ScatterplotLayer", data=sf, get_position="[longitude, latitude]", get_radius="radius", get_fill_color="color", get_line_color=[220,245,255,180], line_width_min_pixels=1, pickable=True, auto_highlight=True)
            view_state = pdk.ViewState(latitude=float(sf.latitude.mean()), longitude=float(sf.longitude.mean()), zoom=9 if dataset=="metr_la" else 10, pitch=20)
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text":"Sensor: {sensor_id}\\nMean: {mean_value}\\nVariability: {variability}"}, map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json")
            st.pydeck_chart(deck, use_container_width=True, height=540)
            st.caption("Node size and color encode the selected layer. Map tiles require internet access; sensor data and coordinates are read locally.")
        else:
            st.warning("Usable geographic coordinates were not found for this dataset. Review the location metadata before interpreting a map.")
    with right:
        st.markdown('<div class="panel-title">Sensor inspector</div><div class="panel-sub">Choose a sensor to inspect historical observations.</div>', unsafe_allow_html=True)
        selected = st.selectbox("Sensor ID", sensor_ids, index=0)
        si = sensor_ids.index(selected)
        observed = bundle.raw[:, si]
        valid = bundle.mask[:, si] & np.isfinite(observed)
        st.metric("Sensor mean", f"{np.mean(observed[valid]):.2f}" if valid.any() else "N/A")
        st.metric("Sensor variability", f"{np.std(observed[valid]):.2f}" if valid.any() else "N/A")
        st.markdown("**Recent observations**")
        hist = pd.DataFrame({"record":np.arange(max(0,len(observed)-288),len(observed)), "observed":observed[-288:]})
        hist.loc[~bundle.mask[-288:, si], "observed"] = np.nan
        st.line_chart(hist.set_index("record"), height=220)
        st.caption("Forecasts appear after an experiment has generated compatible prediction outputs.")
elif view == "Forecast explorer":
    st.markdown('<div class="panel-title">Forecast explorer</div><div class="panel-sub">Inspect measured outputs from saved runs. Use “Run HiLo-Fed pipeline” in the sidebar to generate fresh results.</div>', unsafe_allow_html=True)
    metrics_path = ROOT / "results" / ("forecast_metrics_pems_bay.csv" if dataset=="pems_bay" else "forecast_metrics.csv")
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        st.dataframe(df, use_container_width=True, hide_index=True)
        num = df.select_dtypes(include="number").columns.tolist()
        if num:
            st.markdown("**Metric trends**")
            st.line_chart(df.set_index(num[0])[num[1:3]] if len(num)>2 else df[num])
    else:
        st.info("No compatible forecast metric file exists yet. Run the Phase 1 pipeline to generate one.")
elif view == "Federated topology":
    st.markdown('<div class="panel-title">Federated topology</div><div class="panel-sub">Logical topology view. Run logs below are loaded from pipeline outputs; the diagram remains a conceptual topology, not a live network connection.</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.markdown("### Client nodes"); a.write("Local data partition and model update")
    b.markdown("### Edge aggregators"); b.write("Sample-weighted regional aggregation")
    c.markdown("### Cloud model"); c.write("Global model aggregation and distribution")
    st.graphviz_chart('digraph G {rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#142638", color="#49d7e8", fontcolor="#e7eef8"]; edge [color="#52728b"]; Cloud [label="Cloud Aggregator"]; EdgeA [label="Edge Aggregator A"]; EdgeB [label="Edge Aggregator B"]; C1 [label="Client 1"]; C2 [label="Client 2"]; C3 [label="Client 3"]; C4 [label="Client 4"]; Cloud -> EdgeA; Cloud -> EdgeB; EdgeA -> C1; EdgeA -> C2; EdgeB -> C3; EdgeB -> C4;}')
    st.info("This view describes the intended logical topology. Live round status is shown only when backed by a current run log.")
    log_path = ROOT / "results" / ("round_logs_pems_bay.csv" if dataset=="pems_bay" else "round_logs.csv")
    if log_path.exists():
        logs = pd.read_csv(log_path)
        st.dataframe(logs.tail(30), use_container_width=True, hide_index=True)
elif view == "Communication lab":
    st.markdown('<div class="panel-title">Communication lab</div><div class="panel-sub">Payload measurements from actual pipeline experiment artifacts.</div>', unsafe_allow_html=True)
    p = ROOT / "results" / ("communication_costs_pems_bay.csv" if dataset=="pems_bay" else "communication_costs.csv")
    if p.exists():
        df = pd.read_csv(p)
        st.dataframe(df, use_container_width=True, hide_index=True)
        numeric = df.select_dtypes(include="number")
        if not numeric.empty: st.bar_chart(numeric)
    else:
        st.info("No communication report found. Run an experiment first.")
    st.caption("Interpret compression ratios only alongside the codec, payload accounting method, and experiment configuration.")
else:
    st.markdown('<div class="panel-title">Experiments & reports</div><div class="panel-sub">Artifacts currently available in this project package.</div>', unsafe_allow_html=True)
    for p in [ROOT/"reports"/"PHASE1_END_TO_END_VALIDATION.md", ROOT/"reports"/"PHASE1_VERIFICATION_BASELINE.md", ROOT/"docs"/"COMPETITOR_REFERENCE_NOTES.md"]:
        if p.exists():
            with st.expander(p.name):
                st.markdown(p.read_text(encoding="utf-8"))
    st.markdown("**Result files**")
    files = sorted((ROOT/"results").glob("*"))
    st.write("\n".join(f"- `{f.name}`" for f in files if f.is_file()))

st.divider()
st.markdown('<span class="small-mono">HILO-FED · PHASE 3 INTEGRATION · Live pipeline execution writes measured artifacts · No synthetic results presented as measured results</span>', unsafe_allow_html=True)
