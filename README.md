# HiLo-Fed — Hierarchical Federated Traffic Forecasting

**Priority:** HiLo-Fed and the datasets provided by the user. The project uses the included METR-LA and PEMS-BAY CSVs; competitor projects were inspected for evaluation ideas, not substituted for the project's datasets or identity.

This ZIP contains a runnable reconstructed research prototype. It is **not recovered original Antigravity source code**: the originally supplied `hilofed.zip` contained datasets but no original Python implementation. The current implementation is a lightweight, testable baseline intended to make the key mechanisms explicit and measurable.

## Implemented mechanisms (v0.3.0)

- User-provided **METR-LA and PEMS-BAY** data loading, chronological 70/10/20 split, train-only normalization, zero/non-finite target masking, sensor-ID order preservation for METR-LA, and sensor-aligned distance-graph construction. Distance metadata is filtered to sensor IDs present in each dataset; PEMS-BAY distances are parsed from the headerless file. A normalized, distance-weighted adjacency matrix produces graph-neighbour historical features that are consumed by the forecaster.
- Local client training using SGD on a shared multi-horizon linear forecaster; actual model parameters change during local training.
- Flat FedAvg and hierarchical **client → edge → cloud** aggregation with sample-weighted update averaging.
- Frozen persistence backbone with trainable rank-1 `A @ B` adapter correction (LoRA-style; the base and bias are frozen) and a full-parameter ablation.
- Actual binary serialization and receiver-side decoding for float32, INT8, top-k, and top-k + INT8 updates. Sparse indices and values are transmitted in the encoded representation, decoded updates affect aggregation, and byte counts are measured from the actual serialized payload including the 16-byte HiLo-Fed message header.
- Traffic-volatility-based adaptive upload threshold, fixed-threshold ablation, no-trigger ablation, and forced uploads after a maximum staleness of two silent rounds.
- Cloud aggregation cadence (every two edge rounds, with a final-round flush), per-client/per-round logs, convergence-tolerance summaries, MAE/RMSE at 15/30/60 minutes in original speed units, centralized and persistence reference baselines, and communication-versus-accuracy plots.
- Optional IID random-sample partition for comparison with sensor-partitioned non-IID clients.
- Optional multi-seed runner for variability reporting.
- Streamlit dashboard backed by result files rather than fabricated display values.

## Evaluation scope and measurement definitions

1. The forecasting model is a graph-informed spatial-temporal linear multi-horizon forecaster. It consumes each sensor’s own history plus a distance-weighted neighbour-history aggregate. It is not a deep message-passing GNN; that distinction should be kept clear in research claims.
2. The frozen base for LoRA-style training is a persistence forecast. The adapter learns a low-rank correction. This is parameter-efficient adapter training, but claims of equivalence to a particular transformer/graph-model LoRA design would be inaccurate.
3. Communication bytes measure encoded model payloads and a 16-byte per-message header. They do not include IP/TCP/TLS framing, retries, wall-clock network latency, or energy use. The results are analytical payload accounting, not packet captures.
4. The centralized baseline pools training samples in one process and has no comparable federated-update byte total; its communication cost is intentionally blank, not zero.
5. A short fast-mode run is a smoke/iteration result, not publication-quality evidence. Use both datasets, multiple seeds, and full-data runs before making general claims.
6. The initial ZIP did not contain original source or original tests. QA tests in this package test this reconstructed implementation.

## Windows PowerShell setup

```powershell
cd path\to\HiLo-Fed-Research-Prototype
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Activation is not required. These direct commands also avoid PowerShell execution-policy issues with `Activate.ps1`.

## Run a quick experiment

```powershell
.\.venv\Scripts\python.exe -m hilofed.pipeline --dataset metr_la --fast --max-rows 1200 --rounds 3
```

Run both user-provided datasets:

```powershell
.\.venv\Scripts\python.exe -m hilofed.pipeline --dataset both --fast --max-rows 1200 --rounds 3
```

Run on all rows (more expensive; defaults to a stride of 5 between forecast windows to keep memory use manageable; use `--window-stride 1` if you specifically need every overlapping window):

```powershell
.\.venv\Scripts\python.exe -m hilofed.pipeline --dataset metr_la --rounds 5
```

Optional IID partition:

```powershell
.\.venv\Scripts\python.exe -m hilofed.pipeline --dataset metr_la --fast --partition iid
```

## Test and dashboard

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

## Multi-seed experiment

```powershell
.\.venv\Scripts\python.exe scripts\run_seed_sweep.py --dataset metr_la --fast --max-rows 1200 --rounds 3 --seeds 17,23,41
```

## Outputs

The pipeline writes CSVs and plots to `results/` and metadata to `results/run_metadata*.json`. Important files include:

- `forecast_metrics.csv` — MAE/RMSE at each horizon
- `communication_costs.csv` — measured encoded payloads and communication accounting
- `round_logs.csv` — per-round uploads, forced uploads, validation metrics and cloud aggregation cadence
- `client_round_metrics.csv` — per-client drift, volatility, thresholds and upload decisions
- `convergence.csv` — tolerance-based convergence summary
- `plots/` — accuracy, communication and trade-off charts

## Dependencies

Python 3.10+, NumPy, pandas, Matplotlib, pytest, and Streamlit. PyTorch is not required.
