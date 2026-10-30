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


def _run_flat(name, mode, codec, data, val, rounds, seed, batch_size, local_epochs, partition="sensor"):
    Xtr, Ytr, Mtr, Str = data
    Xv, Yv, Mv, Sv = val
    client_masks = _client_data(Xtr, Ytr, Mtr, Str, n_clients=6, partition=partition, seed=seed)
    global_state = _init_state(mode, seed)
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
        val_mae = vals[-1][2]
        val_history.append(val_mae)
        round_logs.append({"method": name, "round": r, "client_upload_bytes": int(sum(x["payload_bytes"] for x in client_logs if x["method"]==name and x["round"]==r)),
                           "client_downlink_bytes": len(client_masks)*(n_params*4+HEADER_BYTES), "edge_upload_bytes": 0,
                           "edge_downlink_bytes": 0, "total_bytes": int(sum(x["payload_bytes"] for x in client_logs if x["method"]==name and x["round"]==r) + len(client_masks)*(n_params*4+HEADER_BYTES)),
                           "client_uploads": len(client_masks), "forced_uploads": 0, "mean_drift": float(np.mean([x["drift"] for x in client_logs if x["method"]==name and x["round"]==r])),
                           "mean_threshold": 0.0, "cloud_aggregated": True, "validation_mae_60m_mph": val_mae})
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


def _run_hierarchical(name, mode, codec, trigger, data, val, rounds, seed, batch_size, local_epochs, partition="sensor", n_edges=3, cloud_every=2):
    Xtr, Ytr, Mtr, Str = data
    Xv, Yv, Mv, Sv = val
    client_masks = _client_data(Xtr, Ytr, Mtr, Str, n_clients=6, partition=partition, seed=seed)
    client_ids = list(client_masks)
    edge_for_client = {cid: cid % n_edges for cid in client_ids}
    cloud = _init_state(mode, seed)
    edge_states = {e: {k:v.copy() for k,v in cloud.items()} for e in range(n_edges)}
    silent = {cid: 0 for cid in client_ids}
    volatility_segments = {cid: [_volatility(seg) for seg in np.array_split(Xtr[client_masks[cid]], rounds)] for cid in client_ids}
    client_up = client_down = edge_up = edge_down = 0
    round_logs, client_logs = [], []
    n_params = len(_flatten_state(cloud))
    val_history=[]
    for r in range(1, rounds+1):
        round_client_up = round_edge_up = round_forced = round_uploads = 0
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
                threshold = 0.0 if trigger == "none" else (0.25 if trigger == "fixed" else adaptive_threshold(volatility_now, base=0.25))
                forced = silent[cid] >= 2
                upload = True if trigger == "none" else should_upload(drift, threshold, silent[cid], staleness_cap=2)
                drifts.append(drift); thresholds.append(threshold)
                if upload:
                    decoded, nbytes, sent = _encode_update(delta, codec)
                    uploaded_deltas.append(_unflatten_like(decoded, edge_state))
                    weights.append(max(1, int(Mtr[cmask].sum())))
                    client_up += nbytes; round_client_up += nbytes; round_uploads += 1
                    if forced: round_forced += 1
                    silent[cid] = 0
                else:
                    nbytes, sent = 0, 0
                    silent[cid] += 1
                client_logs.append({"method": name, "round": r, "client_id": cid, "edge_id": e,
                                    "uploaded": bool(upload), "forced_upload": bool(upload and forced),
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
        val_mae = vals[-1][2]
        val_history.append(val_mae)
        total = round_client_up + down_this + round_edge_up + (n_edges*(n_params*4+HEADER_BYTES) if edge_sync else 0)
        round_logs.append({"method": name, "round": r, "client_upload_bytes": round_client_up,
                           "client_downlink_bytes": down_this, "edge_upload_bytes": round_edge_up,
                           "edge_downlink_bytes": n_edges*(n_params*4+HEADER_BYTES) if edge_sync else 0,
                           "total_bytes": total, "client_uploads": round_uploads, "forced_uploads": round_forced,
                           "mean_drift": float(np.mean(drifts)) if drifts else 0.0,
                           "mean_threshold": float(np.mean(thresholds)) if thresholds else 0.0,
                           "cloud_aggregated": edge_sync, "validation_mae_60m_mph": val_mae})
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
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(finite.method,finite.total_bytes); ax.set(title="Measured communication payload bytes",ylabel="Bytes (uplink + downlink)"); ax.tick_params(axis="x",rotation=60); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(plot_dir/"communication_bytes.png",dpi=140); plt.close(fig)
    merged=metrics[metrics.horizon_minutes==60].merge(finite[["method","total_bytes"]],on="method",how="inner")
    fig,ax=plt.subplots(figsize=(9,5)); ax.scatter(merged.total_bytes,merged.rmse_mph)
    for _,r in merged.iterrows(): ax.annotate(r.method,(r.total_bytes,r.rmse_mph),fontsize=7)
    ax.set(title="Communication–accuracy trade-off (60-minute RMSE)",xlabel="Measured communication bytes",ylabel="RMSE (mph)"); ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(plot_dir/"communication_accuracy_tradeoff.png",dpi=140); plt.close(fig)


def run_pipeline(dataset="metr_la", fast=True, max_rows=1200, rounds=3, seed=17, data_dir=None, output_dir=None,
                 batch_size=512, local_epochs=1, partition="sensor", window_stride=None):
    global _BUNDLE_CURRENT
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
    methods=[
        ("HiLo-Fed","lora","topk_int8","adaptive","hierarchical"),
        ("HiLo-Fed (no LoRA)","full","topk_int8","adaptive","hierarchical"),
        ("HiLo-Fed (fixed threshold)","lora","topk_int8","fixed","hierarchical"),
        ("HiLo-Fed (no trigger)","lora","float32","none","hierarchical"),
        ("FedAvg","full","float32","none","flat"),
        ("FedAvg + INT8","full","int8","none","flat"),
        ("FedAvg + top-k","full","topk","none","flat"),
    ]
    states={}
    for mi,(name,mode,codec,trigger,topology) in enumerate(methods):
        method_seed=seed+mi*101
        if topology=="flat":
            state,rlogs,clogs,cost=_run_flat(name,mode,codec,data,val,rounds,method_seed,batch_size,local_epochs,partition)
        else:
            state,rlogs,clogs,cost=_run_hierarchical(name,mode,codec,trigger,data,val,rounds,method_seed,batch_size,local_epochs,partition)
        states[name]=(state,mode)
        metrics_rows.extend(_metric_rows(name,dataset,Xte,Yte,Mte,Ste,state,mode,bundle))
        round_rows.extend(rlogs); client_rows.extend(clogs)
        nparams=len(_flatten_state(_init_state(mode,method_seed)))
        counterfactual=max(1,int(cost["counterfactual_full_upload_bytes"]))
        cost_rows.append({"dataset":dataset,"method":name,**cost,"trainable_parameters":nparams,
                          "upload_reduction_pct":100*(1-cost["client_upload_bytes"]/counterfactual),
                          "communication_scope":"client and edge/cloud model payloads; 16-byte per-message header included"})
    central=_run_centralized("full",data,val,rounds,seed+80000,batch_size,local_epochs)
    metrics_rows.extend(_metric_rows("Centralized",dataset,Xte,Yte,Mte,Ste,central,"full",bundle))
    cost_rows.append({"dataset":dataset,"method":"Centralized","client_upload_bytes":np.nan,"client_downlink_bytes":np.nan,
                      "edge_upload_bytes":np.nan,"edge_downlink_bytes":np.nan,"total_bytes":np.nan,
                      "counterfactual_full_upload_bytes":np.nan,"rounds":rounds,"trainable_parameters":len(_flatten_state(central)),
                      "upload_reduction_pct":np.nan,"communication_scope":"Not comparable: centralized pooled-data baseline; data-transfer cost excluded"})
    metrics_rows.extend(_baseline_rows(dataset,bundle,Xte,Yte,Mte,Ste))
    for name in ["Last-value","Historical-average"]:
        cost_rows.append({"dataset":dataset,"method":name,"client_upload_bytes":np.nan,"client_downlink_bytes":np.nan,
                          "edge_upload_bytes":np.nan,"edge_downlink_bytes":np.nan,"total_bytes":np.nan,
                          "counterfactual_full_upload_bytes":np.nan,"rounds":0,"trainable_parameters":0,
                          "upload_reduction_pct":np.nan,"communication_scope":"Non-trained reference baseline; communication not applicable"})
    metrics=pd.DataFrame(metrics_rows); costs=pd.DataFrame(cost_rows); round_df=pd.DataFrame(round_rows); client_df=pd.DataFrame(client_rows)
    # Save each dataset's own outputs; the shared project may later combine METR-LA and PEMS-BAY runs.
    suffix="" if dataset=="metr_la" else f"_{dataset}"
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
    _write_plots(metrics,costs,results_dir/"plots" if dataset=="metr_la" else results_dir/f"plots_{dataset}")
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
              "client_count":6,"edge_count":3,"cloud_aggregation_every_rounds":2,"lookback_steps":LOOKBACK,"input_features_per_sensor":FEATURE_DIM,"graph_neighbour_features_used":True,"forecast_horizon_steps":HORIZON,
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
    p.add_argument("--rounds",type=int,default=3)
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
    for ds in datasets:
        meta,metrics,costs=run_pipeline(ds,args.fast,args.max_rows,args.rounds,args.seed,args.data_dir,args.output_dir,args.batch_size,args.local_epochs,args.partition,args.window_stride)
        all_summaries.append({"dataset":ds,"runtime_seconds":round(meta["runtime_seconds"],2),"metrics_rows":len(metrics),"methods":len(costs)})
    print(json.dumps({"status":"SUCCESS","datasets":all_summaries,"output_dir":str(Path(args.output_dir) if args.output_dir else ROOT/"results")},indent=2))

if __name__=="__main__":
    main()
