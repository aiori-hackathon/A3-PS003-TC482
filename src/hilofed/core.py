from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

DATA_FILES = {
    "metr_la": ("metr_la", "METR-LA.csv"),
    "pems_bay": ("pems_bay", "PEMS-BAY.csv"),
}

@dataclass
class DatasetBundle:
    name: str
    raw: np.ndarray
    normalized: np.ndarray
    mask: np.ndarray
    sensor_ids: list[str]
    train: slice
    val: slice
    test: slice
    mean: np.ndarray
    std: np.ndarray
    distances: pd.DataFrame
    adjacency: np.ndarray
    timestamps: np.ndarray


def load_dataset(data_dir: str | Path, dataset: str = "metr_la", fast: bool = False,
                 max_rows: int = 2000) -> DatasetBundle:
    """Load user-provided METR-LA or PEMS-BAY data with train-only normalization."""
    dataset = dataset.lower().replace("-", "_")
    if dataset not in DATA_FILES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose metr_la or pems_bay")
    folder, filename = DATA_FILES[dataset]
    d = Path(data_dir) / folder
    csv_path = d / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required user-provided dataset not found: {csv_path}")
    # The supplied CSV stores timestamps in its first, unnamed index column.
    frame = pd.read_csv(csv_path, index_col=0)
    timestamps = frame.index.astype(str).to_numpy()
    values = frame.apply(pd.to_numeric, errors="coerce")
    # METR-LA's graph ID file defines the canonical sensor ordering.
    ids_path = d / "graph_sensor_ids.txt"
    if dataset == "metr_la" and ids_path.exists():
        graph_ids = [x.strip() for x in ids_path.read_text().replace("\n", ",").split(",") if x.strip()]
        missing = [sid for sid in graph_ids if sid not in values.columns]
        if missing:
            raise ValueError(f"{len(missing)} graph sensors absent from METR-LA; first={missing[:5]}")
        values = values.loc[:, graph_ids]
    else:
        values.columns = values.columns.astype(str)
    if fast and len(values) > max_rows:
        values = values.iloc[:max_rows].copy()
        timestamps = timestamps[:max_rows]
    raw = values.to_numpy(dtype=np.float64)
    valid = np.isfinite(raw) & (raw != 0)
    n = len(raw)
    n_train = int(n * .70)
    n_val = int(n * .10)
    if n_train <= 12 or n - n_train - n_val <= 12:
        raise ValueError("Dataset slice is too short for chronological train/validation/test windows")
    train, val, test = slice(0, n_train), slice(n_train, n_train+n_val), slice(n_train+n_val, n)
    train_raw, train_valid = raw[train], valid[train]
    mean = np.zeros(raw.shape[1], dtype=np.float64)
    std = np.ones(raw.shape[1], dtype=np.float64)
    for j in range(raw.shape[1]):
        col = train_raw[:, j][train_valid[:, j]]
        if len(col):
            mean[j] = col.mean()
            std[j] = col.std()
    std = np.where(std > 1e-8, std, 1.0)
    filled = raw.copy()
    for j in range(filled.shape[1]):
        bad = ~np.isfinite(filled[:, j]) | (filled[:, j] == 0)
        filled[bad, j] = mean[j]
    normalized = (filled - mean) / std
    dist_path = d / ("distances_la_2012.csv" if dataset == "metr_la" else "distances_bay_2017.csv")
    distances = pd.DataFrame(columns=["from", "to", "cost"])
    if dist_path.exists():
        if dataset == "pems_bay":
            # The supplied PEMS-BAY distance file has no header; its first row is data.
            distances = pd.read_csv(dist_path, header=None, names=["from", "to", "cost"])
        else:
            distances = pd.read_csv(dist_path)
        distances.columns = [str(c).strip().lower() for c in distances.columns]
        # Normalize common schema variants without fabricating missing graph edges.
        if set(["from", "to", "cost"]).issubset(distances.columns):
            distances["from"] = distances["from"].astype(str)
            distances["to"] = distances["to"].astype(str)
            distances["cost"] = pd.to_numeric(distances["cost"], errors="coerce")
            ids = set(values.columns.astype(str))
            distances = distances[distances["from"].isin(ids) & distances["to"].isin(ids)].dropna(subset=["cost"])
    # Build a sensor-aligned, distance-weighted graph from the user's metadata.
    # The graph is used by make_windows to create spatial-neighbour features.
    sensor_ids = values.columns.astype(str).tolist()
    index = {sid: i for i, sid in enumerate(sensor_ids)}
    adjacency = np.zeros((len(sensor_ids), len(sensor_ids)), dtype=np.float32)
    if len(distances):
        valid_dist = distances[(distances["from"] != distances["to"]) & (distances["cost"] >= 0)]
        positive = valid_dist.loc[valid_dist["cost"] > 0, "cost"].to_numpy(dtype=float)
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1.0)
        for row in valid_dist.itertuples(index=False):
            i, j = index.get(str(row[0])), index.get(str(row[1]))
            if i is None or j is None:
                continue
            weight = float(np.exp(-float(row[2]) / scale))
            if weight >= 0.01:
                adjacency[i, j] = max(adjacency[i, j], weight)
                adjacency[j, i] = max(adjacency[j, i], weight)
    # Ensure every sensor has a safe fallback; diagonal is excluded from neighbour aggregation.
    row_sums = adjacency.sum(axis=1, keepdims=True)
    isolated = row_sums[:, 0] <= 0
    adjacency[isolated, isolated] = 1.0
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / np.maximum(row_sums, 1e-12)
    return DatasetBundle(dataset, raw, normalized, valid, sensor_ids,
                         train, val, test, mean, std, distances.reset_index(drop=True), adjacency, timestamps)


def make_windows(bundle: DatasetBundle, split: slice, lookback: int = 12, horizon: int = 12, stride: int = 1):
    """Return one sample per sensor and target time; samples stay inside each split."""
    start = max(split.start + lookback, lookback)
    stop = split.stop - horizon + 1
    if stop <= start:
        raise ValueError("Split too short for requested lookback/horizon")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    times = range(start, stop, stride)
    n_sensors = bundle.normalized.shape[1]
    xs, ys, masks, sensor_ids, target_times = [], [], [], [], []
    for t in times:
        own_history = bundle.normalized[t-lookback:t, :].T
        # Graph diffusion feature: weighted historical signal from connected sensors.
        neighbour_history = (bundle.normalized[t-lookback:t, :] @ bundle.adjacency.T).T
        xs.append(np.concatenate([own_history, neighbour_history], axis=1))
        ys.append(bundle.normalized[t:t+horizon, :].T)
        masks.append(bundle.mask[t:t+horizon, :].T)
        sensor_ids.append(np.arange(n_sensors, dtype=np.int32))
        target_times.append(np.full(n_sensors, t, dtype=np.int32))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(masks).astype(bool), np.concatenate(sensor_ids), np.concatenate(target_times))


def masked_metrics(y_true, y_pred, mask) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    err = np.asarray(y_pred)[valid] - np.asarray(y_true)[valid]
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))), "n": int(valid.sum())}


def split_sensor_clients(sensor_ids: list[str], n_clients: int = 6) -> dict[int, list[int]]:
    if not sensor_ids:
        return {}
    n_clients = max(1, min(int(n_clients), len(sensor_ids)))
    return {client: list(range(client, len(sensor_ids), n_clients)) for client in range(n_clients)}


def lora_parameter_counts(in_dim: int = 12, out_dim: int = 12, rank: int = 2) -> tuple[int, int, float]:
    full = in_dim * out_dim + out_dim
    adapter = rank * (in_dim + out_dim)
    return full, adapter, adapter / full


def tensor_bytes(n_values: int, dtype: str = "float32", include_scale: bool = False) -> int:
    sizes = {"float32": 4, "int8": 1, "float64": 8, "int16": 2}
    if dtype not in sizes:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return int(n_values * sizes[dtype] + (4 if include_scale else 0))


def weighted_average(arrays: list[np.ndarray], sample_counts: list[int]) -> np.ndarray:
    if len(arrays) != len(sample_counts) or not arrays:
        raise ValueError("arrays and sample_counts must be non-empty and equal length")
    weights = np.asarray(sample_counts, dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample counts must sum to a positive value")
    if any(np.shape(a) != np.shape(arrays[0]) for a in arrays):
        raise ValueError("All arrays must have identical shapes")
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, weight in zip(arrays, weights):
        out += np.asarray(arr, dtype=np.float64) * weight
    return (out / weights.sum()).astype(np.float32)


def adaptive_threshold(volatility: float, base: float = 0.08) -> float:
    """Higher traffic volatility lowers the relative-update trigger threshold."""
    return float(base / (1.0 + max(0.0, float(volatility))))


def should_upload(drift: float, threshold: float, silent_rounds: int, staleness_cap: int) -> bool:
    return float(drift) >= float(threshold) or int(silent_rounds) >= int(staleness_cap)


def cloud_should_aggregate(edge_round: int, every_k: int = 2) -> bool:
    if every_k < 1:
        raise ValueError("every_k must be >= 1")
    return edge_round % every_k == 0


def weighted_cost(client_bytes: float, edge_cloud_bytes: float, client_weight: float = 1.0,
                  cloud_weight: float = 1.0) -> float:
    return client_bytes * client_weight + edge_cloud_bytes * cloud_weight
