# HiLo-Fed Initial Research Prototype Report

> This report summarizes the latest recorded experiments on the supplied datasets. It reports measured outputs and does not claim HiLo-Fed currently outperforms all baselines.

## Experiment configuration

### METR-LA

- Rows loaded: **34272** (full supplied dataset; temporal window stride: 50)
- Sensors: **207**
- Train / validation / test rows: **23990 / 3427 / 6855**
- Clients / edges: **6 / 3**
- Rounds: **3**; seed: **17**; runtime: **3.46s**
- LoRA-style adapter parameters: **36 / 300** (12.0%)
- Parsed distance graph edges: **11753**; graph-neighbour history features enabled: **True**

## 60-minute forecasting results — METR-LA

| Method | RMSE (mph) | MAE (mph) |
|---|---:|---:|
| HiLo-Fed | 10.538 | 5.521 |
| HiLo-Fed (no LoRA) | 9.989 | 5.423 |
| HiLo-Fed (fixed threshold) | 10.540 | 5.522 |
| HiLo-Fed (no trigger) | 10.535 | 5.519 |
| FedAvg | 9.799 | 5.640 |
| FedAvg + INT8 | 9.795 | 5.629 |
| FedAvg + top-k | 9.784 | 5.532 |
| Centralized | 9.449 | 5.555 |
| Last-value | 10.206 | 6.455 |
| Historical-average | 11.905 | 7.495 |

## Communication accounting — METR-LA

| Method | Client uplink bytes | Total measured bytes | Total bytes saved vs FedAvg |
|---|---:|---:|---:|
| HiLo-Fed | 240 | 5040 | 88.5% |
| HiLo-Fed (no LoRA) | 1020 | 37500 | 14.3% |
| HiLo-Fed (fixed threshold) | 240 | 5040 | 88.5% |
| HiLo-Fed (no trigger) | 2880 | 7680 | 82.5% |
| FedAvg | 21888 | 43776 | 0.0% |
| FedAvg + INT8 | 5760 | 27648 | 36.8% |
| FedAvg + top-k | 4608 | 26496 | 39.5% |

### HiLo-Fed communication control observations

- Client uploads across rounds: **6**
- Forced uploads due to staleness cap: **6**
- Cloud aggregation rounds (including final flush): **2**

### PEMS-BAY

- Rows loaded: **52116** (full supplied dataset; temporal window stride: 50)
- Sensors: **325**
- Train / validation / test rows: **36481 / 5211 / 10424**
- Clients / edges: **6 / 3**
- Rounds: **3**; seed: **17**; runtime: **6.55s**
- LoRA-style adapter parameters: **36 / 300** (12.0%)
- Parsed distance graph edges: **8358**; graph-neighbour history features enabled: **True**

## 60-minute forecasting results — PEMS-BAY

| Method | RMSE (mph) | MAE (mph) |
|---|---:|---:|
| HiLo-Fed | 6.940 | 3.036 |
| HiLo-Fed (no LoRA) | 6.416 | 3.142 |
| HiLo-Fed (fixed threshold) | 6.943 | 3.037 |
| HiLo-Fed (no trigger) | 6.841 | 3.003 |
| FedAvg | 6.345 | 3.220 |
| FedAvg + INT8 | 6.344 | 3.214 |
| FedAvg + top-k | 6.395 | 3.139 |
| Centralized | 6.187 | 3.194 |
| Last-value | 7.361 | 4.164 |
| Historical-average | 9.303 | 5.391 |

## Communication accounting — PEMS-BAY

| Method | Client uplink bytes | Total measured bytes | Total bytes saved vs FedAvg |
|---|---:|---:|---:|
| HiLo-Fed | 240 | 5040 | 88.5% |
| HiLo-Fed (no LoRA) | 1020 | 37500 | 14.3% |
| HiLo-Fed (fixed threshold) | 240 | 5040 | 88.5% |
| HiLo-Fed (no trigger) | 2880 | 7680 | 82.5% |
| FedAvg | 21888 | 43776 | 0.0% |
| FedAvg + INT8 | 5760 | 27648 | 36.8% |
| FedAvg + top-k | 4608 | 26496 | 39.5% |

### HiLo-Fed communication control observations

- Client uploads across rounds: **6**
- Forced uploads due to staleness cap: **6**
- Cloud aggregation rounds (including final flush): **2**

## Interpretation and measurement scope

- The automated test suite validates the reconstructed implementation, not any missing original Antigravity source code.
- In these results, HiLo-Fed does not consistently achieve the lowest 60-minute forecasting error across both datasets. The system exposes an accuracy–communication trade-off; the results do not establish universal forecasting superiority.
- The primary tables use the full supplied files, 3 rounds, 6 clients and 3 edges, with temporal window stride 50. The separate multi-seed repeatability sweeps use 1,200-row fast subsets.
- Communication totals are measured from serialized update payloads and a 16-byte HiLo-Fed message header. They do not include network-stack overhead, energy, or measured wall-clock network latency.
- The forecaster consumes own-sensor history and distance-weighted graph-neighbour history. It is a graph-informed spatial-temporal linear model, not a deep message-passing GNN.
- Distance metadata is filtered to IDs present in the selected dataset, aligned to sensor order, normalized into a weighted adjacency matrix, and used to derive graph-neighbour input features.
- Centralized and non-trained reference baselines have blank communication totals because their transfer costs are not comparable to federated model-update traffic.
- The full-file experiments process every row in the supplied CSV files but sample every 50th eligible temporal window. They are multi-round full-file experiments, not exhaustive every-window runs.
- Multi-seed summary files are included for both datasets; their fast subsets provide repeatability evidence, not full-data confidence intervals.
