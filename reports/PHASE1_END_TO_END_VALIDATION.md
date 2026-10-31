# HiLo-Fed Phase 1 — End-to-End Federated Learning Validation

## Actual commands run in the packaging environment

Environment: Python runtime available in the execution container; project source loaded with `PYTHONPATH=src`.

1. `python -m pytest -q`
   - Result: **18 passed in 8.38 seconds** (rerun after strengthening the end-to-end hierarchy assertions).
2. `PYTHONPATH=src python -m hilofed.pipeline --dataset both --fast --max-rows 1200 --rounds 3 --output-dir /tmp/hilofed_phase1_validation`
   - Result: **SUCCESS** for METR-LA and PEMS-BAY.
   - METR-LA: 4.79 seconds, 30 forecast-metric rows, 10 methods.
   - PEMS-BAY: 6.30 seconds, 30 forecast-metric rows, 10 methods.
3. `PYTHONPATH=src python -m hilofed.pipeline --dataset both --rounds 3 --window-stride 50 --output-dir /tmp/hilofed_phase1_fullfile_validation`
   - Result: **SUCCESS** for METR-LA and PEMS-BAY.
   - METR-LA: 3.49 seconds, 30 forecast-metric rows, 10 methods.
   - PEMS-BAY: 6.55 seconds, 30 forecast-metric rows, 10 methods.

The full-file runs read all rows from the supplied CSV files, but sample eligible temporal windows at stride 50. They are not exhaustive every-window benchmarks. The fast runs use 1,200 rows and three rounds.

## What was exercised

- Local model training changes model parameters.
- Flat FedAvg and hierarchical client → edge → cloud methods execute through multiple rounds.
- Client update payloads are serialized, decoded, and decoded values are used by aggregation.
- INT8 and top-k codecs are included in the experiment methods.
- Adaptive upload control and maximum-staleness behavior are recorded in per-client logs.
- Edge-to-cloud synchronization is recorded in round logs; the final round flushes pending edge state.
- Sample-weighted aggregation, dataset loading, chronological splits, graph-neighbour input construction, and metrics are covered by tests.
- Experiment outputs include forecasting metrics at 15/30/60 minutes, per-client and per-round logs, communication-cost tables, convergence summaries, dataset audits, sensor indexes, and plots.

## Important interpretation

This validates that the reconstructed prototype executes its implemented FL path and produces measured software-level experiment outputs. It does **not** prove that HiLo-Fed beats every baseline or that it is publication-ready. Communication bytes represent serialized application payloads plus the defined 16-byte header; they are not physical network packet captures. The current forecaster uses own-sensor history and distance-weighted neighbour-history features; it is not a deep message-passing GNN. The adapter mode is a lightweight low-rank correction to a frozen persistence baseline, not LoRA applied to a transformer/GNN backbone.

## Generated results

Fresh full-file sampled-window outputs from the run above are included under `results/`, including separate METR-LA and PEMS-BAY CSVs and plots. Run metadata records the exact configuration and limitations.
