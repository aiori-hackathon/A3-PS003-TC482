# HiLo-Fed Phase 3 — Research Fix Validation Report

**Status: implementation and local validation complete for requested items 2, 5, and 6.** The results below are from the actual pipeline run described here, not hand-entered simulation output. Browser-level Streamlit interaction was not automated; the dashboard controls are wired to the real quick-run pipeline and source/tests were checked locally.

## Reproduction configuration

- Dataset: supplied METR-LA data (`data/metr_la/METR-LA.csv`)
- Run mode: fast chronological subset, 300 rows
- Clients / edge clusters / rounds: **30 / 4 / 30**
- Seed: 17; `tau=0.006`; LoRA rank 1; cloud aggregation interval `K=2`; `w_edge=1.0`; `w_cloud=1.0`
- Runtime reported by pipeline: approximately 8.63 seconds
- Communication accounting: encoded update payloads plus the existing 16-byte message header. The byte counter/serializer was not changed.

## Automated test result

Command:

```bash
PYTHONPATH=src python -m pytest -q
```

Result: **23 passed, 0 failed**.

## Item 2 — Drift-trigger and forced uploads

Drift-triggered and staleness-forced uploads are now classified separately and mutually exclusively. `tau` is the fixed base threshold; the adaptive threshold decreases with the measured volatility value. The adaptive and fixed-policy comparisons use paired initialization/local-training seeds to make the policy comparison more controlled.

| Method | Client uploads across 30 rounds | Drift-triggered uploads | Staleness-forced uploads | Total communication bytes |
|---|---:|---:|---:|---:|
| HiLo-Fed (adaptive) | 502 | 321 | 181 | 183,280 |
| HiLo-Fed (fixed threshold) | 418 | 182 | 236 | 179,920 |

The adaptive and fixed policies are measurably different in this run: **84 more client uploads** and **3,360 more total bytes** for the adaptive policy. Adaptive triggering does not win on communication volume for this particular run; no claim of lower communication cost is made. Round-level and client-level logs contain separate drift and forced counters.

## Item 5 — Research outputs

Implemented and emitted by the pipeline:

- Weighted communication cost with independent `w_edge` and `w_cloud` weights.
- Accuracy-per-MB metric.
- Pareto plot: 60-minute RMSE against weighted communication MB.
- MB-to-target report with exact first round reaching the target, cumulative weighted MB, and RMSE at the first hit. The per-round target uses **1.05 × FedAvg validation 60-minute RMSE**, so both the baseline and round-wise comparisons use the validation split. The final method-level target in `communication_costs.csv` is separately based on 1.05 × FedAvg test 60-minute RMSE.
- Upload heatmap by time of day, using the supplied dataset timestamps in UTC. Each round is associated with the midpoint timestamp of its chronological training segment; this is a round-segment summary, not a packet-level event timestamp.
- Flat (no hierarchy) ablation.

Generated plots include `pareto_rmse_weighted_mb.png`, `mb_to_target_rmse.png`, `upload_heatmap_by_time_of_day.png`, and `upload_heatmap_by_round.png`, alongside accuracy/communication plots. The run produced 33 forecast rows across 11 methods, with RMSE and MAE at 15, 30, and 60 minutes.

### Selected measured results

| Method | 60-min RMSE (mph) | 60-min MAE (mph) | Total bytes | Weighted MB | Total-byte reduction vs FedAvg |
|---|---:|---:|---:|---:|---:|
| HiLo-Fed | 4.7259 | 2.7426 | 183,280 | 0.18328 | 91.63% |
| HiLo-Fed (fixed threshold) | 4.7257 | 2.7424 | 179,920 | 0.17992 | 91.78% |
| HiLo-Fed (flat ablation) | 4.7289 | 2.7456 | 180,000 | 0.18000 | 91.78% |
| FedAvg | 6.0094 | 4.3639 | 2,188,800 | 2.18880 | 0.00% |

These values are for the stated 300-row experiment only and should not be generalized to the full dataset without a full-data run.

## Item 6 — Dashboard controls

The Streamlit quick-run panel now includes controls for:

- Upload threshold `tau`
- LoRA rank
- Cloud sync interval `K`
- Edge communication weight `w_edge`
- Cloud communication weight `w_cloud`
- Dataset, rounds, row subset, seed, and client partition

These values are passed into the real pipeline command and are recorded in run metadata. The quick-run remains intentionally small (up to 10 rounds in the dashboard); the 30-round experiment above was run directly through the pipeline. Browser automation was not available, so visual/browser interaction is not claimed as tested.

## Output files

- `results/forecast_metrics.csv`
- `results/communication_costs.csv`
- `results/round_logs.csv`
- `results/client_round_metrics.csv`
- `results/mb_to_target_rmse.csv`
- `results/run_metadata.json`
- `results/plots/`

## Remaining scope caveat

This is a reconstructed research prototype and the run uses a 300-row fast subset. The source report distinguishes measured outputs from implementation limitations; this report does not imply a full-dataset benchmark or browser-level UI test.
