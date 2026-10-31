from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .core import load_dataset, make_windows, masked_metrics, adaptive_threshold, should_upload, cloud_should_aggregate, split_sensor_clients, lora_parameter_counts

ROOT = Path(__file__).resolve().parents[2]
LOOKBACK = 12
FEATURE_DIM = LOOKBACK * 2  # own-sensor history + graph-neighbour history
HORIZON = 12
RANK = 1
HEADER_BYTES = 16
CLIENT_COUNT = 6
EDGE_COUNT = 3
TAU = 0.006
W_EDGE = 1.0
W_CLOUD = 1.0


def _base_matrix(lookback=FEATURE_DIM, horizon=HORIZON):
    # Persistence forecast uses the last own-sensor value; graph features follow it.
    w = np.zeros((lookback, horizon), dtype=np.float32)
    own_last_index = lookback // 2 - 1 if lookback % 2 == 0 else lookback - 1
    w[own_last_index, :] = 1.0
    return w


def _init_state(mode: str, seed: int, lookback=FEATURE_DIM, horizon=HORIZON, rank=RANK):
    rng = np.random.default_rng(seed)
    if mode == "lora":
        return {"A": rng.normal(0, 0.02, (lookback, rank)).astype(np.float32),
                "B": rng.normal(0, 0.02, (rank, horizon)).astype(np.float32)}
    return {"W": _base_matrix(lookback, horizon), "b": np.zeros(horizon, dtype=np.float32)}


def _state_keys(state):
    return sorted(state.keys())


def _flatten_state(state):
    return np.concatenate([np.asarray(state[k], dtype=np.float32).ravel() for k in _state_keys(state)])


def _unflatten_like(vector, template):
    out, offset = {}, 0
    for key in _state_keys(template):
        shape = template[key].shape
        size = int(np.prod(shape))
        out[key] = np.asarray(vector[offset:offset+size], dtype=np.float32).reshape(shape).copy()
        offset += size
    if offset != len(vector):
        raise ValueError("Vector length does not match model state")
    return out


def _state_add(state, delta):
    return {k: (np.asarray(state[k], dtype=np.float32) + np.asarray(delta[k], dtype=np.float32)).astype(np.float32)
            for k in state}


def _state_delta(new, old):
    return {k: (np.asarray(new[k], dtype=np.float32) - np.asarray(old[k], dtype=np.float32)).astype(np.float32)
            for k in old}


def _predict(X, state, mode, base_w=None):
    if mode == "lora":
        if base_w is None:
            base_w = _base_matrix(X.shape[1], state["B"].shape[1])
        return X @ base_w + (X @ state["A"]) @ state["B"] + state.get("b", 0.0)
    return X @ state["W"] + state["b"]


def _local_train(X, Y, M, initial_state, mode="full", seed=0, epochs=1, batch_size=512, lr=0.025):
    """Actual local SGD on a shared linear multi-horizon forecaster; returns trained state."""
    if len(X) == 0:
        return {k: v.copy() for k, v in initial_state.items()}
    state = {k: v.copy() for k, v in initial_state.items()}
    rng = np.random.default_rng(seed)
    base_w = _base_matrix(X.shape[1], Y.shape[1])
    n = len(X)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start+batch_size]
            xb, yb, mb = X[idx], Y[idx], M[idx].astype(np.float32)
            if mb.sum() == 0:
                continue
            pred = _predict(xb, state, mode, base_w)
            err = (pred - yb) * mb
            scale = 2.0 / max(float(mb.sum()), 1.0)
            if mode == "lora":
                xa = xb @ state["A"]
                gB = (xa.T @ err) * scale
                gA = (xb.T @ (err @ state["B"].T)) * scale
                # Base forecast and bias remain frozen; only low-rank A/B adapters train.
                norm = np.sqrt(float(np.sum(gA*gA) + np.sum(gB*gB)))
                factor = min(1.0, 5.0 / max(norm, 1e-12))
                state["A"] -= (lr * factor * gA).astype(np.float32)
                state["B"] -= (lr * factor * gB).astype(np.float32)
            else:
                gW = (xb.T @ err) * scale
                gb = err.sum(axis=0) * scale
                norm = np.sqrt(float(np.sum(gW*gW) + np.sum(gb*gb)))
                factor = min(1.0, 5.0 / max(norm, 1e-12))
                state["W"] -= (lr * factor * gW).astype(np.float32)
                state["b"] -= (lr * factor * gb).astype(np.float32)
    return state


def _encode_update(delta_vector, codec="float32", topk_fraction=0.10):
    """Serialize a real update, decode that payload, and return decoded values/bytes/count.

    The 16-byte header is physically included in the serialized payload. Payload sizes
    therefore come from len(payload), not a separate estimate of tensor sizes.
    """
    import struct
    x = np.asarray(delta_vector, dtype=np.float32).ravel()
    n = len(x)
    codec_ids = {"float32": 1, "int8": 2, "topk": 3, "topk_int8": 4}
    if codec not in codec_ids:
        raise ValueError(f"Unknown codec: {codec}")
    # 4-byte magic + 1-byte codec + 3 reserved bytes + 4-byte value count + 4-byte auxiliary count.
    def header(aux=0):
        return struct.pack("<4sB3xII", b"HFL1", codec_ids[codec], n, int(aux))
    if codec == "float32":
        body = x.astype("<f4", copy=False).tobytes()
        payload = header(0) + body
        decoded = np.frombuffer(payload[HEADER_BYTES:], dtype="<f4").astype(np.float32, copy=True)
        return decoded, len(payload), int(n)
    if codec == "int8":
        scale = float(np.max(np.abs(x)) / 127.0) if n and np.max(np.abs(x)) > 0 else 1.0
        q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
        body = struct.pack("<f", scale) + q.tobytes()
        payload = header(1) + body
        decoded_scale = struct.unpack("<f", payload[HEADER_BYTES:HEADER_BYTES+4])[0]
        decoded_q = np.frombuffer(payload[HEADER_BYTES+4:], dtype=np.int8)
        return decoded_q.astype(np.float32) * decoded_scale, len(payload), int(n)
    k = max(1, min(n, int(np.ceil(n * topk_fraction)))) if n else 0
    idx = np.argpartition(np.abs(x), -k)[-k:] if k else np.array([], dtype=np.int64)
    vals = x[idx]
    if codec == "topk":
        wire_idx = idx.astype("<u4")
        wire_vals = vals.astype("<f4")
        payload = header(k) + wire_idx.tobytes() + wire_vals.tobytes()
        offset = HEADER_BYTES
        decoded_idx = np.frombuffer(payload[offset:offset+4*k], dtype="<u4").astype(np.int64)
        decoded_vals = np.frombuffer(payload[offset+4*k:], dtype="<f4").astype(np.float32)
        decoded = np.zeros(n, dtype=np.float32); decoded[decoded_idx] = decoded_vals
    else:
        scale = float(np.max(np.abs(vals)) / 127.0) if len(vals) and np.max(np.abs(vals)) > 0 else 1.0
        q = np.clip(np.rint(vals / scale), -127, 127).astype(np.int8)
        payload = header(k) + struct.pack("<f", scale) + idx.astype("<u4").tobytes() + q.tobytes()
        offset = HEADER_BYTES
        decoded_scale = struct.unpack("<f", payload[offset:offset+4])[0]; offset += 4
        decoded_idx = np.frombuffer(payload[offset:offset+4*k], dtype="<u4").astype(np.int64); offset += 4*k
        decoded_q = np.frombuffer(payload[offset:], dtype=np.int8)
        decoded = np.zeros(n, dtype=np.float32); decoded[decoded_idx] = decoded_q.astype(np.float32) * decoded_scale
    return decoded, len(payload), int(k)


def _relative_drift(delta_vector, reference_vector):
    return float(np.linalg.norm(delta_vector) / max(float(np.linalg.norm(reference_vector)), 1e-8))


def _volatility(X):
    """Mean one-step change across own-sensor and graph-neighbour histories."""
    if len(X) == 0 or X.shape[1] < 4:
        return 0.0
    half = X.shape[1] // 2
    own_change = np.abs(X[:, half - 1] - X[:, half - 2])
    neighbour_change = np.abs(X[:, -1] - X[:, -2])
    return float(0.5 * (np.mean(own_change) + np.mean(neighbour_change)))


def _client_data(X, Y, M, S, n_clients=6, partition="sensor", seed=17):
    n_sensors = int(S.max()) + 1 if len(S) else 0
    rng = np.random.default_rng(seed)
    if partition == "sensor":
        assignment = split_sensor_clients([str(i) for i in range(n_sensors)], n_clients)
        return {c: np.isin(S, sensors) for c, sensors in assignment.items()}
    if partition == "iid":
        order = rng.permutation(len(X))
        bins = np.array_split(order, n_clients)
        masks = {}
        for c, ids in enumerate(bins):
            mask = np.zeros(len(X), dtype=bool); mask[ids] = True; masks[c] = mask
        return masks
    raise ValueError("partition must be sensor or iid")


def _evaluate(X, Y, M, S, state, mode, bundle, horizons=(2, 5, 11)):
    pred = _predict(X, state, mode, _base_matrix(X.shape[1], Y.shape[1]))
    rows = []
    for h in horizons:
        true_mph = Y[:, h] * bundle.std[S] + bundle.mean[S]
        pred_mph = pred[:, h] * bundle.std[S] + bundle.mean[S]
        met = masked_metrics(true_mph, pred_mph, M[:, h])
        rows.append((h, met["rmse"], met["mae"], met["n"]))
    return pred, rows


def _metric_rows(method, dataset, X, Y, M, S, state, mode, bundle):
    _, vals = _evaluate(X, Y, M, S, state, mode, bundle)
    minutes = {2: 15, 5: 30, 11: 60}
    return [{"dataset": dataset, "method": method, "horizon_minutes": minutes[h],
             "rmse_mph": rmse, "mae_mph": mae, "n_valid": n} for h, rmse, mae, n in vals]


def _payload_bytes_for_state(state):
    return int(_flatten_state(state).nbytes + HEADER_BYTES)


def _run_flat(name, mode, codec, data, val, rounds, seed, batch_size, local_epochs, partition="sensor", n_clients=None, rank=None):
    Xtr, Ytr, Mtr, Str = data
    Xv, Yv, Mv, Sv = val
    client_masks = _client_data(Xtr, Ytr, Mtr, Str, n_clients=int(n_clients or CLIENT_COUNT), partition=partition, seed=seed)
    global_state = _init_state(mode, seed, rank=int(rank or RANK))
    client_up = client_down = 0
    round_logs, client_logs = [], []
    val_history = []
    n_params = len(_flatten_state(global_state))
    for r in range(1, rounds + 1):
        client_down += len(client_masks) * (n_params * 4 + HEADER_BYTES)
        deltas, weights, raw_full_bytes = [], [], 0
        for cid, cmask in client_masks.items():
            init = {k: v.copy() for k, v in global_state.items()}
            local = _local_train(Xtr[cmask], Ytr[cmask], Mtr[cmask], init, mode,
                                 seed + 1000*r + cid, epochs=local_epochs, batch_size=batch_size)
            delta = _flatten_state(_state_delta(local, global_state))
            decoded, nbytes, sent = _encode_update(delta, codec)
            client_up += nbytes; raw_full_bytes += len(delta)*4 + HEADER_BYTES
            deltas.append(_unflatten_like(decoded, global_state))
            weights.append(max(1, int(Mtr[cmask].sum())))
            client_logs.append({"method": name, "round": r, "client_id": cid, "uploaded": True,
                                "forced_upload": False, "drift": _relative_drift(delta, _flatten_state(global_state)),
                                "threshold": 0.0, "volatility": _volatility(Xtr[cmask]),
                                "payload_bytes": nbytes, "values_sent": sent, "valid_target_count": weights[-1]})
        # Aggregate decoded updates, not merely byte estimates.
        avg_delta = _flatten_state(weighted_state_average(deltas, weights))
        global_state = _state_add(global_state, _unflatten_like(avg_delta, global_state))
        _, vals = _evaluate(Xv, Yv, Mv, Sv, global_state, mode, _BUNDLE_CURRENT)
        val_rmse = vals[-1][1]
        val_mae = vals[-1][2]
        val_history.append(val_mae)
        round_logs.append({"method": name, "round": r, "client_upload_bytes": int(sum(x["payload_bytes"] for x in client_logs if x["method"]==name and x["round"]==r)),
                           "client_downlink_bytes": len(client_masks)*(n_params*4+HEADER_BYTES), "edge_upload_bytes": 0,
                           "edge_downlink_bytes": 0, "total_bytes": int(sum(x["payload_bytes"] for x in client_logs if x["method"]==name and x["round"]==r) + len(client_masks)*(n_params*4+HEADER_BYTES)),
                           "client_uploads": len(client_masks), "drift_triggered_uploads": 0, "forced_uploads": 0, "mean_drift": float(np.mean([x["drift"] for x in client_logs if x["method"]==name and x["round"]==r])),
                           "mean_threshold": 0.0, "cloud_aggregated": True, "validation_mae_60m_mph": val_mae, "validation_rmse_60m_mph": val_rmse})
    return global_state, round_logs, client_logs, {"client_upload_bytes": client_up, "client_downlink_bytes": client_down,
        "edge_upload_bytes": 0, "edge_downlink_bytes": 0, "total_bytes": client_up+client_down,
        "counterfactual_full_upload_bytes": rounds*len(client_masks)*(n_params*4+HEADER_BYTES), "rounds": rounds}


def weighted_state_average(states, weights):
    total = float(sum(weights))
    if not states or total <= 0:
        raise ValueError("Cannot aggregate empty or zero-weight model updates")
    return {k: (sum(np.asarray(s[k], dtype=np.float64)*w for s,w in zip(states,weights))/total).astype(np.float32)
            for k in states[0]}


# The bundle is set at run time so round evaluation can use its inverse normalization.
_BUNDLE_CURRENT = None


def _run_hierarchical(name, mode, codec, trigger, data, val, rounds, seed, batch_size, local_epochs, partition="sensor", n_edges=None, cloud_every=2, n_clients=None, rank=None, tau=None):
    Xtr, Ytr, Mtr, Str = data
    Xv, Yv, Mv, Sv = val
    n_edges = int(n_edges or EDGE_COUNT)
    client_masks = _client_data(Xtr, Ytr, Mtr, Str, n_clients=int(n_clients or CLIENT_COUNT), partition=partition, seed=seed)
    client_ids = list(client_masks)
    edge_for_client = {cid: cid % n_edges for cid in client_ids}
    cloud = _init_state(mode, seed, rank=int(rank or RANK))
    edge_states = {e: {k:v.copy() for k,v in cloud.items()} for e in range(n_edges)}
    silent = {cid: 0 for cid in client_ids}
    volatility_segments = {cid: [_volatility(seg) for seg in np.array_split(Xtr[client_masks[cid]], rounds)] for cid in client_ids}
    client_up = client_down = edge_up = edge_down = 0
    round_logs, client_logs = [], []
    n_params = len(_flatten_state(cloud))
    val_history=[]
    for r in range(1, rounds+1):
        round_client_up = round_edge_up = round_forced = round_drift_uploads = round_uploads = 0
        drifts, thresholds = [], []
        for e in range(n_edges):
            member_ids = [cid for cid in client_ids if edge_for_client[cid] == e]
            edge_state = edge_states[e]
            uploaded_deltas, weights = [], []
            for cid in member_ids:
                cmask = client_masks[cid]
                local = _local_train(Xtr[cmask], Ytr[cmask], Mtr[cmask], edge_state, mode,
                                     seed + 1000*r + cid, epochs=local_epochs, batch_size=batch_size)
                delta = _flatten_state(_state_delta(local, edge_state))
                drift = _relative_drift(delta, _flatten_state(edge_state))
                volatility_now = volatility_segments[cid][r-1]
                tau_value = float(tau if tau is not None else TAU)
                # tau is the interpretable base relative-update threshold. The adaptive
                # policy lowers it as traffic volatility rises; fixed uses tau unchanged.
                threshold = 0.0 if trigger == "none" else (tau_value if trigger == "fixed" else adaptive_threshold(volatility_now, base=tau_value))
                drift_condition = bool(trigger != "none" and drift >= threshold)
                forced_condition = bool(trigger != "none" and silent[cid] >= 2 and not drift_condition)
                upload = bool(trigger == "none" or drift_condition or forced_condition)
                # Trigger categories are mutually exclusive: a staleness upload is
                # counted as forced only when the drift criterion did not fire.
                forced = bool(upload and forced_condition)
                drift_triggered = bool(upload and trigger in {"adaptive", "fixed"} and drift_condition)
                drifts.append(drift); thresholds.append(threshold)
                if upload:
                    decoded, nbytes, sent = _encode_update(delta, codec)
                    uploaded_deltas.append(_unflatten_like(decoded, edge_state))
                    weights.append(max(1, int(Mtr[cmask].sum())))
                    client_up += nbytes; round_client_up += nbytes; round_uploads += 1
                    if forced: round_forced += 1
                    if drift_triggered: round_drift_uploads += 1
                    silent[cid] = 0
                else:
                    nbytes, sent = 0, 0
                    silent[cid] += 1
                client_logs.append({"method": name, "round": r, "client_id": cid, "edge_id": e,
                                    "uploaded": bool(upload), "forced_upload": forced, "drift_triggered_upload": drift_triggered,
                                    "drift": drift, "threshold": threshold, "volatility": volatility_now,
                                    "payload_bytes": nbytes, "values_sent": sent,
                                    "valid_target_count": int(Mtr[cmask].sum())})
            if uploaded_deltas:
                edge_states[e] = _state_add(edge_state, weighted_state_average(uploaded_deltas, weights))
        # Client downlink: current edge model parameters, each client every round.
        down_this = len(client_ids) * (n_params*4 + HEADER_BYTES)
        client_down += down_this
        # Flush pending edge updates at the final round so the reported model includes them.
        edge_sync = cloud_should_aggregate(r, cloud_every) or r == rounds
        if edge_sync:
            edge_deltas, edge_weights = [], []
            for e in range(n_edges):
                delta = _state_delta(edge_states[e], cloud)
                edge_deltas.append(delta)
                edge_weights.append(max(1, sum(int(Mtr[client_masks[cid]].sum()) for cid in client_ids if edge_for_client[cid] == e)))
                nbytes = n_params*4 + HEADER_BYTES
                edge_up += nbytes; round_edge_up += nbytes
            cloud = _state_add(cloud, weighted_state_average(edge_deltas, edge_weights))
            # Broadcast synchronized cloud state to each edge.
            sync_bytes = n_edges * (n_params*4 + HEADER_BYTES)
            edge_down += sync_bytes
            for e in range(n_edges):
                edge_states[e] = {k:v.copy() for k,v in cloud.items()}
        _, vals = _evaluate(Xv, Yv, Mv, Sv, cloud, mode, _BUNDLE_CURRENT)
        val_rmse = vals[-1][1]
        val_mae = vals[-1][2]
        val_history.append(val_mae)
        total = round_client_up + down_this + round_edge_up + (n_edges*(n_params*4+HEADER_BYTES) if edge_sync else 0)
        round_logs.append({"method": name, "round": r, "client_upload_bytes": round_client_up,
                           "client_downlink_bytes": down_this, "edge_upload_bytes": round_edge_up,
                           "edge_downlink_bytes": n_edges*(n_params*4+HEADER_BYTES) if edge_sync else 0,
                           "total_bytes": total, "client_uploads": round_uploads, "drift_triggered_uploads": round_drift_uploads, "forced_uploads": round_forced,
                           "mean_drift": float(np.mean(drifts)) if drifts else 0.0,
                           "mean_threshold": float(np.mean(thresholds)) if thresholds else 0.0,
                           "cloud_aggregated": edge_sync, "validation_mae_60m_mph": val_mae, "validation_rmse_60m_mph": val_rmse})
    total_up = sum(x["client_upload_bytes"] for x in round_logs)
    total_down = sum(x["client_downlink_bytes"] for x in round_logs)
    total_edge_up = sum(x["edge_upload_bytes"] for x in round_logs)
    total_edge_down = sum(x["edge_downlink_bytes"] for x in round_logs)
    return cloud, round_logs, client_logs, {"client_upload_bytes": total_up, "client_downlink_bytes": total_down,
        "edge_upload_bytes": total_edge_up, "edge_downlink_bytes": total_edge_down,
        "total_bytes": total_up+total_down+total_edge_up+total_edge_down,
        "counterfactual_full_upload_bytes": rounds*len(client_ids)*(n_params*4+HEADER_BYTES), "rounds": rounds}


def _run_centralized(mode, data, val, rounds, seed, batch_size, local_epochs):
    Xtr,Ytr,Mtr,_ = data
    state = _init_state(mode, seed)
    state = _local_train(Xtr,Ytr,Mtr,state,mode,seed=seed+90000,epochs=rounds*local_epochs,batch_size=batch_size)
    return state


def _baseline_rows(dataset, bundle, Xte, Yte, Mte, Ste):
    rows=[]
    for name, pred in [("Last-value", np.repeat(Xte[:, -1, None], Yte.shape[1], axis=1)),
                       ("Historical-average", np.zeros_like(Yte))]:
        for h, minutes in [(2,15),(5,30),(11,60)]:
            truth = Yte[:,h]*bundle.std[Ste]+bundle.mean[Ste]
            guess = pred[:,h]*bundle.std[Ste]+bundle.mean[Ste]
            met=masked_metrics(truth,guess,Mte[:,h])
            rows.append({"dataset":dataset,"method":name,"horizon_minutes":minutes,"rmse_mph":met["rmse"],"mae_mph":met["mae"],"n_valid":met["n"]})
    return rows


def _write_plots(metrics, costs, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    metric_methods = metrics[metrics.method.isin(["HiLo-Fed", "Centralized", "FedAvg", "FedAvg + INT8", "FedAvg + top-k", "Last-value", "Historical-average"])]
    for fname, col, title in [("rmse_by_horizon.png","rmse_mph","RMSE by forecast horizon"),("mae_by_horizon.png","mae_mph","MAE by forecast horizon")]:
        fig,ax=plt.subplots(figsize=(10,5))
        for method,g in metric_methods.groupby("method"):
            g=g.sort_values("horizon_minutes"); ax.plot(g.horizon_minutes,g[col],marker="o",label=method)
        ax.set(title=title,xlabel="Forecast horizon (minutes)",ylabel="Speed error (mph)"); ax.grid(True,alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(plot_dir/fname,dpi=140); plt.close(fig)
    finite=costs[np.isfinite(pd.to_numeric(costs.total_bytes,errors="coerce"))].sort_values("total_bytes")
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(finite.method,finite.total_bytes); ax.set(title="Measured communication payload bytes",ylabel="Bytes (uplink + downlink)"); ax.set_ylim(bottom=0); ax.tick_params(axis="x",rotation=60); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(plot_dir/"communication_bytes.png",dpi=140); plt.close(fig)
    merged=metrics[metrics.horizon_minutes==60].merge(finite[["method","total_bytes"]],on="method",how="inner")
    fig,ax=plt.subplots(figsize=(9,5)); ax.scatter(merged.total_bytes,merged.rmse_mph)
    for _,r in merged.iterrows(): ax.annotate(r.method,(r.total_bytes,r.rmse_mph),fontsize=7)
    ax.set(title="Communication–accuracy trade-off (60-minute RMSE)",xlabel="Measured communication bytes",ylabel="RMSE (mph)"); ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(plot_dir/"communication_accuracy_tradeoff.png",dpi=140); plt.close(fig)
    # Pareto plot uses weighted communication MB and 60-minute RMSE from measured output tables.
    if "weighted_cost_mb" in costs.columns:
        pm = metrics[metrics.horizon_minutes.eq(60)][["method","rmse_mph"]].merge(costs[["method","weighted_cost_mb"]],on="method",how="inner").dropna()
        fig,ax=plt.subplots(figsize=(9,5)); ax.scatter(pm.weighted_cost_mb,pm.rmse_mph)
        for _,r in pm.iterrows(): ax.annotate(r.method,(r.weighted_cost_mb,r.rmse_mph),fontsize=7)
        ax.set(title="Pareto view: 60-minute RMSE vs weighted communication",xlabel="Weighted communication (MB)",ylabel="RMSE (mph)"); ax.grid(True,alpha=.3)
        fig.tight_layout(); fig.savefig(plot_dir/"pareto_rmse_weighted_mb.png",dpi=140); plt.close(fig)
        target = costs.loc[costs.method.eq("FedAvg"),"rmse_mph"]
        if not target.empty and pd.notna(target.iloc[0]):
            threshold=float(target.iloc[0])*1.05
            fig,ax=plt.subplots(figsize=(10,5)); ax.bar(pm.method,pm.weighted_cost_mb); ax.axhline(0,linewidth=.5)
            for i,(_,r) in enumerate(pm.iterrows()): ax.text(i,r.weighted_cost_mb,f"RMSE {r.rmse_mph:.2f}",rotation=90,va="bottom",fontsize=7)
            ax.set(title=f"Weighted MB by method (target RMSE ≤ {threshold:.3f} mph)",ylabel="Weighted MB",xlabel="Method"); ax.set_ylim(bottom=0); ax.tick_params(axis="x",rotation=60); ax.grid(axis="y",alpha=.3)
            fig.tight_layout(); fig.savefig(plot_dir/"mb_to_target_rmse.png",dpi=140); plt.close(fig)


def run_pipeline(dataset="metr_la", fast=True, max_rows=1200, rounds=30, seed=17, data_dir=None, output_dir=None,
                 batch_size=512, local_epochs=1, partition="sensor", window_stride=None, n_clients=30, n_edges=4, tau=0.006, rank=1, k=2, w_edge=1.0, w_cloud=1.0):
    global _BUNDLE_CURRENT, CLIENT_COUNT, EDGE_COUNT, TAU, RANK, W_EDGE, W_CLOUD
    CLIENT_COUNT, EDGE_COUNT, TAU, RANK, W_EDGE, W_CLOUD = int(n_clients), int(n_edges), float(tau), int(rank), float(w_edge), float(w_cloud)
    t0=time.perf_counter()
    root=ROOT
    data_dir=Path(data_dir) if data_dir else root/"data"
    output_dir=Path(output_dir) if output_dir else root
    results_dir=output_dir/"results"; reports_dir=output_dir/"reports"
    results_dir.mkdir(parents=True,exist_ok=True); reports_dir.mkdir(parents=True,exist_ok=True)
    bundle=load_dataset(data_dir,dataset,fast=fast,max_rows=max_rows)
    _BUNDLE_CURRENT=bundle
    window_stride = int(window_stride if window_stride is not None else (1 if fast else 5))
    Xtr,Ytr,Mtr,Str,_=make_windows(bundle,bundle.train,LOOKBACK,HORIZON,window_stride)
    Xv,Yv,Mv,Sv,_=make_windows(bundle,bundle.val,LOOKBACK,HORIZON,window_stride)
    Xte,Yte,Mte,Ste,_=make_windows(bundle,bundle.test,LOOKBACK,HORIZON,window_stride)
    data=(Xtr,Ytr,Mtr,Str); val=(Xv,Yv,Mv,Sv)
    metrics_rows=[]; cost_rows=[]; round_rows=[]; client_rows=[]
    suffix="" if dataset=="metr_la" else f"_{dataset}"
    methods=[
        ("HiLo-Fed","lora","topk_int8","adaptive","hierarchical"),
        ("HiLo-Fed (no LoRA)","full","topk_int8","adaptive","hierarchical"),
        ("HiLo-Fed (fixed threshold)","lora","topk_int8","fixed","hierarchical"),
        ("HiLo-Fed (no trigger)","lora","float32","none","hierarchical"),
        ("HiLo-Fed (flat ablation)","lora","topk_int8","adaptive","flat"),
        ("FedAvg","full","float32","none","flat"),
        ("FedAvg + INT8","full","int8","none","flat"),
        ("FedAvg + top-k","full","topk","none","flat"),
    ]
    states={}
    for mi,(name,mode,codec,trigger,topology) in enumerate(methods):
        # Pair trigger-policy comparisons with the same initialization and local-SGD seeds.
        method_seed = seed if name in {"HiLo-Fed", "HiLo-Fed (fixed threshold)", "HiLo-Fed (no trigger)"} else seed + mi * 101
        if topology=="flat":
            state,rlogs,clogs,cost=_run_flat(name,mode,codec,data,val,rounds,method_seed,batch_size,local_epochs,partition,n_clients=n_clients,rank=rank)
        else:
            state,rlogs,clogs,cost=_run_hierarchical(name,mode,codec,trigger,data,val,rounds,method_seed,batch_size,local_epochs,partition,n_edges=n_edges,cloud_every=k,n_clients=n_clients,rank=rank,tau=tau)
        states[name]=(state,mode)
        metrics_rows.extend(_metric_rows(name,dataset,Xte,Yte,Mte,Ste,state,mode,bundle))
        round_rows.extend(rlogs); client_rows.extend(clogs)
        nparams=len(_flatten_state(_init_state(mode,method_seed)))
        counterfactual=max(1,int(cost["counterfactual_full_upload_bytes"]))
        cost_rows.append({"dataset":dataset,"method":name,**cost,"trainable_parameters":nparams,
                          "upload_reduction_pct":np.nan, "total_reduction_pct":np.nan, "weighted_cost_bytes": W_EDGE*(cost["client_upload_bytes"]+cost["client_downlink_bytes"]) + W_CLOUD*(cost["edge_upload_bytes"]+cost["edge_downlink_bytes"]), "weighted_cost_mb": (W_EDGE*(cost["client_upload_bytes"]+cost["client_downlink_bytes"]) + W_CLOUD*(cost["edge_upload_bytes"]+cost["edge_downlink_bytes"]))/1e6,
                          "communication_scope":"client and edge/cloud model payloads; 16-byte per-message header included"})
    central=_run_centralized("full",data,val,rounds,seed+80000,batch_size,local_epochs)
    metrics_rows.extend(_metric_rows("Centralized",dataset,Xte,Yte,Mte,Ste,central,"full",bundle))
    cost_rows.append({"dataset":dataset,"method":"Centralized","client_upload_bytes":np.nan,"client_downlink_bytes":np.nan,
                      "edge_upload_bytes":np.nan,"edge_downlink_bytes":np.nan,"total_bytes":np.nan,
                      "counterfactual_full_upload_bytes":np.nan,"rounds":rounds,"trainable_parameters":len(_flatten_state(central)),
                      "upload_reduction_pct":np.nan,"total_reduction_pct":np.nan,"weighted_cost_bytes":np.nan,"weighted_cost_mb":np.nan,"communication_scope":"Not comparable: centralized pooled-data baseline; data-transfer cost excluded"})
    metrics_rows.extend(_baseline_rows(dataset,bundle,Xte,Yte,Mte,Ste))
    for name in ["Last-value","Historical-average"]:
        cost_rows.append({"dataset":dataset,"method":name,"client_upload_bytes":np.nan,"client_downlink_bytes":np.nan,
                          "edge_upload_bytes":np.nan,"edge_downlink_bytes":np.nan,"total_bytes":np.nan,
                          "counterfactual_full_upload_bytes":np.nan,"rounds":0,"trainable_parameters":0,
                          "upload_reduction_pct":np.nan,"total_reduction_pct":np.nan,"weighted_cost_bytes":np.nan,"weighted_cost_mb":np.nan,"communication_scope":"Non-trained reference baseline; communication not applicable"})
    costs=pd.DataFrame(cost_rows)
    fedavg = costs.loc[costs.method.eq("FedAvg")]
    if not fedavg.empty:
        fedavg_row = fedavg.iloc[0]
        valid_upload = pd.to_numeric(costs.client_upload_bytes, errors="coerce")
        valid_total = pd.to_numeric(costs.total_bytes, errors="coerce")
        costs["upload_reduction_pct"] = np.where(valid_upload.notna() & (float(fedavg_row.client_upload_bytes) > 0), 100*(1-valid_upload/float(fedavg_row.client_upload_bytes)), np.nan)
        costs["total_reduction_pct"] = np.where(valid_total.notna() & (float(fedavg_row.total_bytes) > 0), 100*(1-valid_total/float(fedavg_row.total_bytes)), np.nan)
    metrics=pd.DataFrame(metrics_rows); round_df=pd.DataFrame(round_rows); client_df=pd.DataFrame(client_rows)
    # Associate each training round with the midpoint timestamp of its chronological
    # training-data segment. This enables a time-of-day upload summary without claiming
    # packet-level timestamps that the simulator does not generate.
    train_timestamps = pd.to_datetime(bundle.timestamps[bundle.train], errors="coerce", utc=True)
    if len(train_timestamps) and train_timestamps.notna().any():
        timestamp_chunks = np.array_split(np.arange(len(train_timestamps)), max(1, int(rounds)))
        round_hour = {}
        round_timestamp = {}
        for ri, idxs in enumerate(timestamp_chunks, start=1):
            if len(idxs):
                ts = train_timestamps.iloc[int(idxs[len(idxs)//2])] if hasattr(train_timestamps, "iloc") else train_timestamps[int(idxs[len(idxs)//2])]
                if pd.notna(ts):
                    round_hour[ri] = int(ts.hour)
                    round_timestamp[ri] = ts.isoformat()
        for frame in (round_df, client_df):
            if not frame.empty and "round" in frame:
                frame["hour_of_day_utc"] = frame["round"].map(round_hour)
                frame["round_segment_timestamp_utc"] = frame["round"].map(round_timestamp)
    rmse60 = metrics[metrics.horizon_minutes.eq(60)][["method","rmse_mph"]]
    costs = costs.merge(rmse60, on="method", how="left")
    costs["accuracy_per_mb"] = 1.0 / (costs["rmse_mph"] * costs["weighted_cost_mb"].replace(0, np.nan))
    fedavg_rmse = metrics.loc[(metrics.method == "FedAvg") & (metrics.horizon_minutes == 60), "rmse_mph"]
    target = float(fedavg_rmse.iloc[0]) * 1.05 if not fedavg_rmse.empty else np.nan
    costs["target_rmse_mph"] = target
    costs["target_rmse_reached"] = costs["rmse_mph"] <= target if np.isfinite(target) else False
    # First-round-to-target uses a validation baseline and validation RMSE for every
    # method (same split); the test-set target remains separately reported in costs.
    if not round_df.empty and "validation_rmse_60m_mph" in round_df.columns:
        fedavg_validation = round_df.loc[round_df.method.eq("FedAvg"), "validation_rmse_60m_mph"]
        validation_target = float(fedavg_validation.iloc[0]) * 1.05 if not fedavg_validation.empty else np.nan
        reach_rows=[]
        for method, grp in round_df.groupby("method"):
            grp=grp.sort_values("round").copy()
            grp["cumulative_weighted_mb"] = (grp["client_upload_bytes"].cumsum()*W_EDGE + (grp["edge_upload_bytes"]+grp["edge_downlink_bytes"]).cumsum()*W_CLOUD + grp["client_downlink_bytes"].cumsum()*W_EDGE)/1e6
            hit=grp[grp["validation_rmse_60m_mph"] <= validation_target] if np.isfinite(validation_target) else grp.iloc[0:0]
            reach_rows.append({"method":method,"target_rmse_mph":validation_target,"target_basis":"1.05 x FedAvg validation 60-minute RMSE",
                               "reached":not hit.empty,
                               "first_round":int(hit.iloc[0]["round"]) if not hit.empty else np.nan,
                               "weighted_mb_to_target":float(hit.iloc[0]["cumulative_weighted_mb"]) if not hit.empty else np.nan,
                               "validation_rmse_at_first_hit":float(hit.iloc[0]["validation_rmse_60m_mph"]) if not hit.empty else np.nan})
        pd.DataFrame(reach_rows).to_csv(results_dir/f"mb_to_target_rmse{suffix}.csv",index=False)
    # Save each dataset's own outputs; the shared project may later combine METR-LA and PEMS-BAY runs.
    metrics.to_csv(results_dir/f"forecast_metrics{suffix}.csv",index=False)
    costs.to_csv(results_dir/f"communication_costs{suffix}.csv",index=False)
    round_df.to_csv(results_dir/f"round_logs{suffix}.csv",index=False)
    client_df.to_csv(results_dir/f"client_round_metrics{suffix}.csv",index=False)
    pd.DataFrame({"sensor_index":range(len(bundle.sensor_ids)),"sensor_id":bundle.sensor_ids}).to_csv(results_dir/f"sensor_index{suffix}.csv",index=False)
    distance_ids=set(bundle.distances["from"].astype(str)) | set(bundle.distances["to"].astype(str)) if len(bundle.distances) else set()
    audit={"dataset":dataset,"sensor_count":len(bundle.sensor_ids),"distance_edges_usable":int(len(bundle.distances)),
           "distance_graph_sensor_coverage":len(distance_ids),"distance_graph_compatible":bool(len(bundle.distances)>0),
           "note":"Distance edges are filtered to sensor IDs present in the selected traffic dataset; unmatched edges are excluded rather than remapped by guess."}
    pd.DataFrame([audit]).to_csv(results_dir/f"dataset_audit{suffix}.csv",index=False)
    plot_dir=results_dir/"plots" if dataset=="metr_la" else results_dir/f"plots_{dataset}"
    _write_plots(metrics,costs,plot_dir)
    # Upload heatmap by time of day, based on each round's associated chronological
    # training-segment midpoint timestamp (UTC); it is not a packet-level event timestamp.
    if not client_df.empty and {"round","client_id","uploaded"}.issubset(client_df.columns):
        if "hour_of_day_utc" in client_df and client_df.hour_of_day_utc.notna().any():
            heat=client_df.pivot_table(index="client_id",columns="hour_of_day_utc",values="uploaded",aggfunc="mean",fill_value=0)
            fig,ax=plt.subplots(figsize=(12,6)); ax.imshow(heat.to_numpy(),aspect="auto",interpolation="nearest",origin="lower",vmin=0,vmax=1)
            ax.set(title="Upload heatmap by time of day (UTC; round-segment timestamp)",xlabel="Hour of day (UTC)",ylabel="Client ID")
            ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels([f"{int(h):02d}:00" for h in heat.columns],fontsize=8)
            fig.tight_layout(); fig.savefig(plot_dir/"upload_heatmap_by_time_of_day.png",dpi=140); plt.close(fig)
        heat=client_df.pivot_table(index="client_id",columns="round",values="uploaded",aggfunc="mean",fill_value=0)
        fig,ax=plt.subplots(figsize=(12,6)); ax.imshow(heat.to_numpy(),aspect="auto",interpolation="nearest",origin="lower",vmin=0,vmax=1)
        ax.set(title="Client upload heatmap by training round",xlabel="Training round",ylabel="Client ID")
        ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(list(heat.columns),fontsize=7)
        fig.tight_layout(); fig.savefig(plot_dir/"upload_heatmap_by_round.png",dpi=140); plt.close(fig)
    # Convergence summary uses measured validation MAE for each trained method.
    convergence=[]
    for method,g in round_df.groupby("method"):
        vals=g.sort_values("round").validation_mae_60m_mph.to_numpy(dtype=float)
        stable_round=None; patience=2; tol=1e-4
        for i in range(patience,len(vals)):
            if all(abs(vals[j]-vals[j-1])<tol for j in range(i-patience+1,i+1)):
                stable_round=int(g.sort_values("round").iloc[i]["round"]); break
        convergence.append({"method":method,"rounds_observed":len(vals),"validation_mae_60m_mph_last":float(vals[-1]) if len(vals) else np.nan,
                            "converged_by_tolerance":stable_round is not None,"convergence_round":stable_round,"absolute_improvement_tolerance":tol,"patience_rounds":patience})
    pd.DataFrame(convergence).to_csv(results_dir/f"convergence{suffix}.csv",index=False)
    full_params, adapter_params, adapter_ratio = lora_parameter_counts(FEATURE_DIM, HORIZON, RANK)
    metadata={"project":"HiLo-Fed research prototype","version":"0.3.0","dataset":dataset,"seed":seed,"partition":partition,
              "fast_mode":fast,"max_rows_requested":max_rows if fast else None,"rows_used":len(bundle.raw),"sensor_count":len(bundle.sensor_ids),
              "train_rows":bundle.train.stop-bundle.train.start,"validation_rows":bundle.val.stop-bundle.val.start,"test_rows":bundle.test.stop-bundle.test.start,
              "training_samples":len(Xtr),"validation_samples":len(Xv),"test_samples":len(Xte),"window_stride":window_stride,"rounds":rounds,"local_epochs":local_epochs,
              "client_count":int(n_clients),"edge_count":int(n_edges),"cloud_aggregation_every_rounds":int(k), "tau":float(tau), "lora_rank":int(rank), "w_edge":float(w_edge), "w_cloud":float(w_cloud),"lookback_steps":LOOKBACK,"input_features_per_sensor":FEATURE_DIM,"graph_neighbour_features_used":True,"forecast_horizon_steps":HORIZON,
              "full_model_trainable_parameters":full_params,"lora_adapter_trainable_parameters":adapter_params,"lora_trainable_parameter_ratio":adapter_ratio,
              "target_minutes":[15,30,60],"runtime_seconds":time.perf_counter()-t0,"python":sys.version,"platform":platform.platform(),
              "source_dataset_files": [str(bundle.name),"provided METR-LA.csv / PEMS-BAY.csv; no synthetic traffic substituted"],
              "usable_distance_graph_edges": int(len(bundle.distances)),
              "distance_graph_note": "Distance edges are aligned to sensor IDs and converted to normalized distance-weighted adjacency. Each input contains own-sensor history and graph-neighbour aggregate history; graph metadata contributes to model inputs.",
              "communication_accounting":"Measured encoded update arrays and model payloads, plus 16-byte message headers; network protocol overhead outside this model is not included.",
              "research_caveats":["This is a reconstructed prototype, not recovered original Antigravity source.",
                 "Graph-aware linear multi-horizon forecaster is a lightweight spatial-temporal baseline, not a deep graph neural network.",
                 "Results are from the requested run configuration and must not be generalized beyond the evaluated split.",
                 "LoRA is implemented as a trainable low-rank correction to a frozen persistence backbone."]}
    (results_dir/f"run_metadata{suffix}.json").write_text(json.dumps(metadata,indent=2))
    return metadata,metrics,costs


def main():
    p=argparse.ArgumentParser(description="Run HiLo-Fed federated traffic forecasting experiments")
    p.add_argument("--dataset",choices=["metr_la","pems_bay","both"],default="metr_la")
    p.add_argument("--fast",action="store_true",help="Use a contiguous chronological subset for quick experiments")
    p.add_argument("--max-rows",type=int,default=1200,help="Rows to use in fast mode")
    p.add_argument("--rounds",type=int,default=None,help="Training rounds; defaults to 3 in --fast mode and 30 in full mode")
    p.add_argument("--clients",type=int,default=30)
    p.add_argument("--edges",type=int,default=4)
    p.add_argument("--tau",type=float,default=0.006)
    p.add_argument("--rank",type=int,default=1)
    p.add_argument("--k",type=int,default=2)
    p.add_argument("--w-edge",type=float,default=1.0)
    p.add_argument("--w-cloud",type=float,default=1.0)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--data-dir",default=None)
    p.add_argument("--output-dir",default=None)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--local-epochs",type=int,default=1)
    p.add_argument("--window-stride",type=int,default=None,help="Temporal stride between training windows; defaults to 1 in fast mode and 5 for full data to control memory use")
    p.add_argument("--partition",choices=["sensor","iid"],default="sensor",help="Sensor-held-out/non-IID or randomly mixed IID client partition")
    args=p.parse_args()
    datasets=["metr_la","pems_bay"] if args.dataset=="both" else [args.dataset]
    all_summaries=[]
    rounds = int(args.rounds if args.rounds is not None else (3 if args.fast else 30))
    for ds in datasets:
        meta,metrics,costs=run_pipeline(ds,args.fast,args.max_rows,rounds,args.seed,args.data_dir,args.output_dir,args.batch_size,args.local_epochs,args.partition,args.window_stride,args.clients,args.edges,args.tau,args.rank,args.k,args.w_edge,args.w_cloud)
        all_summaries.append({"dataset":ds,"runtime_seconds":round(meta["runtime_seconds"],2),"metrics_rows":len(metrics),"methods":len(costs)})
    print(json.dumps({"status":"SUCCESS","datasets":all_summaries,"output_dir":str(Path(args.output_dir) if args.output_dir else ROOT/"results")},indent=2))

if __name__=="__main__":
    main()
