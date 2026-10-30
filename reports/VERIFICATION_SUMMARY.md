# HiLo-Fed Packaging Verification Summary

## Project state inspected

- The packaged project is a **reconstructed HiLo-Fed research prototype**, not recovered original Antigravity source code. The originally supplied `hilofed.zip` contained the user's datasets and metadata files, but no original Python implementation.
- The project includes the user's METR-LA and PEMS-BAY datasets and metadata. All eight dataset/metadata files checked against `hilofed.zip` matched byte-for-byte.
- Competitor archives were used as references for experiment tracking, IID/non-IID evaluation, client-level reporting, convergence summaries and multi-seed evaluation ideas. Their datasets and projects did not replace HiLo-Fed's data or project identity.

## Latest actual verification runs

Environment: Python 3.13.5, Linux x86_64.

### Automated tests

Command: `python -m pytest -v`

- Latest result after graph-feature, volatility, and serialized-payload changes: **18 passed** (latest rerun recorded in `reports/pytest_output.txt`).
- Failures: 0; errors: 0.

### Full-dataset three-round experiments

Command: `PYTHONPATH=src python -m hilofed.pipeline --dataset both --rounds 3 --window-stride 50 --output-dir results`

- Result: **SUCCESS** for both supplied datasets.
- METR-LA: all 20,640 rows loaded, 207 sensors, 3 rounds, 30 forecasting metric rows and 10 methods; runtime **4.10 seconds**.
- PEMS-BAY: all 52,116 rows loaded, 325 sensors, 3 rounds, 30 forecasting metric rows and 10 methods; runtime **7.18 seconds**.
- Window stride 50 samples every 50th eligible temporal window to keep compute manageable; this is a full-file-data experiment but not an every-window exhaustive run.

### Three-seed fast-mode evaluation

- METR-LA: seeds 17, 23 and 41 completed successfully; runtimes 4.6, 4.1 and 4.5 seconds.
- PEMS-BAY: seeds 17, 23 and 41 completed successfully; runtimes 5.3, 6.6 and 6.2 seconds.
- Per-seed metrics and mean/standard-deviation summaries are saved as `results/seed_sweep_metrics_*.csv` and `results/seed_sweep_summary_*.csv`.
- These seed sweeps use 1,200 chronological rows and three rounds, so they test repeatability on a fast subset rather than providing full-data uncertainty estimates.

### Additional validation

- All 8 supplied dataset/metadata files matched the original `hilofed.zip` byte-for-byte.
- Update codecs serialize the actual update payload, including the 16-byte HiLo-Fed header, and the receiver-side decoded values are used in aggregation.
- Graph metadata is converted to a sensor-aligned normalized distance-weighted adjacency matrix and used to derive neighbour-history features.
- Report generation succeeded and updated metrics/plots are included in the package.

## Included feature scope

The code includes local client training, flat FedAvg, client → edge → cloud aggregation, sample-weighted aggregation, float32/INT8/top-k/top-k+INT8 update codecs, volatility-based upload thresholds, staleness-triggered uploads, a frozen persistence backbone with a low-rank trainable correction, forecasting metrics at 15/30/60 minutes, baselines, communication accounting, per-round/client logs, convergence summaries, IID partition option, multi-seed runner, plots and a results-backed Streamlit dashboard.

## Scope and measurement definitions

- The forecaster is a graph-informed spatial-temporal linear multi-horizon model: it consumes own-sensor history and distance-weighted neighbour-history features. It is not a deep message-passing GNN.
- The low-rank adapter is applied to a frozen persistence backbone; it should not be represented as equivalent to a LoRA implementation on a transformer or graph neural network.
- Communication accounting measures encoded payload arrays and a 16-byte message header, not real network packet captures, transport overhead, latency or energy.
- The fast multi-seed runs are repeatability checks on a 1,200-row subset. Full-file experiments use a temporal stride of 50. HiLo-Fed does not consistently beat every baseline on forecasting error; results show an accuracy–communication trade-off rather than universal superiority.
- These tests validate this reconstructed prototype, not missing original source code.


### Dashboard verification

- `dashboard.py` syntax check is run during packaging. Streamlit is declared in `requirements.txt`, but the validation runtime did not have Streamlit installed, so a live dashboard launch could not be verified in this environment. Run the documented setup command to install dependencies before launching it.
