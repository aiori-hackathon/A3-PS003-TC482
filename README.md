# HiLo-Fed — Phase 1: Research Core

This Phase 1 package focuses on the HiLo-Fed backend and research core. It preserves the supplied METR-LA and PEMS-BAY datasets and their associated graph/distance metadata. Competitor projects are references only; their datasets do not replace HiLo-Fed's primary data.

## Included

- Dataset loader with chronological train/validation/test splits and train-only normalization.
- Sensor-aligned, distance-weighted adjacency from supplied metadata and neighbour-history features.
- Multi-horizon traffic forecasting pipeline and evaluation at 15, 30 and 60 minutes.
- Local client training, flat FedAvg and client → edge → cloud hierarchical aggregation.
- Sample-weighted aggregation.
- Binary update serialization codecs: float32, INT8, top-k, and top-k + INT8; decoded payloads feed aggregation.
- Volatility-based upload thresholds and maximum-staleness upload rule.
- Frozen persistence baseline with a low-rank trainable correction mode.
- Reference baselines, per-round/client logs, communication accounting, convergence tables, plots and multi-seed helper scripts where included in source.
- Automated tests.

## Important scope note

This is the Phase 1 research core, not the final map-first frontend. The forecaster currently uses local history and graph-neighbour history features; it is not a deep message-passing GNN. Communication accounting measures serialized application payloads and a HiLo-Fed header, not packet captures, physical network latency or energy. The low-rank correction is a lightweight adapter-style method, not a claim of a transformer/graph-network LoRA implementation. See `reports/PHASE1_VERIFICATION_BASELINE.md` for the previously recorded verification context; rerun tests locally to confirm your environment.

## Setup (Windows PowerShell)

From this extracted folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Run a fast experiment on both original datasets

```powershell
.\.venv\Scripts\python.exe -m hilofed.pipeline --dataset both --fast --max-rows 1200 --rounds 3
```

## Run a full-file, sampled-window experiment

This loads all rows from each dataset but uses a temporal stride to control memory and runtime. It is not an exhaustive every-window benchmark.

```powershell
.\.venv\Scripts\python.exe -m hilofed.pipeline --dataset both --rounds 3 --window-stride 50 --output-dir results
```

## Phase 1 completion criteria

1. Dataset loader reads the supplied datasets and metadata without substitution.
2. Federated clients perform local updates and aggregation executes across rounds.
3. Hierarchical aggregation, update codecs, adaptive upload decisions and staleness enforcement are exercised by tests/experiments.
4. The experiment runner writes metric and communication tables.
5. Test output is reported from the actual local run; no result is assumed from packaging alone.

The frontend/map interface is intentionally reserved for Phase 2.


## Phase 2 — Map-first frontend

Install dependencies, then launch the interactive dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

The dashboard includes a sensor map using supplied location metadata, sensor inspection, forecast artifacts, federated topology, communication reports, and experiment documentation. Map tiles require internet access. The topology screen is explicitly labelled as a logical topology, not a live network visualization.


## Phase 3 — Integrated dashboard and experiment execution

The dashboard can now launch the actual research pipeline for the selected dataset. Open the sidebar expander **Run a real experiment**, choose rounds/rows/seed/partition, and click **Run HiLo-Fed pipeline**. Real pipeline artifacts are written into `results/` and consumed by the forecast, topology-log, and communication views. Start with 300–1200 rows and 1–3 rounds for interactive runs; these are smoke experiments, not final benchmark evidence. See `docs/PHASE3_INTEGRATION.md`.
