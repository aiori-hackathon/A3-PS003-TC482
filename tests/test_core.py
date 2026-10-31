from pathlib import Path
import numpy as np
import pytest
from hilofed.core import (load_dataset, make_windows, masked_metrics, adaptive_threshold,
                          should_upload, cloud_should_aggregate, weighted_average, split_sensor_clients, lora_parameter_counts)
from hilofed.pipeline import _encode_update, _init_state, _flatten_state, _local_train, _predict, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

@pytest.mark.parametrize("dataset,expected_min_sensors", [("metr_la", 200), ("pems_bay", 300)])
def test_user_provided_datasets_load(dataset, expected_min_sensors):
    bundle = load_dataset(DATA, dataset, fast=True, max_rows=120)
    assert bundle.raw.shape[1] >= expected_min_sensors
    assert len(bundle.timestamps) == 120
    assert bundle.train.stop < bundle.test.stop
    assert np.isfinite(bundle.normalized).all()
    if dataset == "pems_bay":
        assert len(bundle.distances) > 0
        sensor_set = set(bundle.sensor_ids)
        assert set(bundle.distances["from"]).issubset(sensor_set)
        assert set(bundle.distances["to"]).issubset(sensor_set)


def test_metr_la_graph_sensor_order_is_preserved():
    bundle = load_dataset(DATA, "metr_la", fast=True, max_rows=120)
    ids = [x.strip() for x in (DATA / "metr_la" / "graph_sensor_ids.txt").read_text().replace("\n", ",").split(",") if x.strip()]
    assert bundle.sensor_ids == ids
    assert bundle.adjacency.shape == (len(ids), len(ids))
    assert np.allclose(bundle.adjacency.sum(axis=1), 1.0)


def test_windows_are_finite_and_target_mask_is_boolean():
    bundle = load_dataset(DATA, "metr_la", fast=True, max_rows=180)
    X,Y,M,S,T = make_windows(bundle, bundle.train)
    assert X.shape[1] == 24 and Y.shape[1] == 12
    assert M.shape == Y.shape and M.dtype == bool
    assert len(S) == len(X) == len(T)
    assert np.isfinite(X).all() and np.isfinite(Y).all()



def test_graph_neighbour_features_are_present_and_finite():
    bundle=load_dataset(DATA,"metr_la",fast=True,max_rows=180)
    X,Y,M,S,T=make_windows(bundle,bundle.train)
    # First 12 features are local history; second 12 are adjacency-weighted neighbour history.
    assert X.shape[1] == 24
    assert np.isfinite(X[:,12:]).all()
    assert bundle.adjacency.shape == (len(bundle.sensor_ids), len(bundle.sensor_ids))
    assert np.allclose(bundle.adjacency.sum(axis=1), 1.0)

def test_masked_metrics_ignores_invalid_target():
    got = masked_metrics(np.array([1., 1000., 3.]), np.array([1., -1000., 2.]), np.array([1,0,1], dtype=bool))
    assert got["n"] == 2 and np.isclose(got["rmse"], np.sqrt(.5))


def test_adaptive_threshold_lowers_when_volatility_rises():
    assert adaptive_threshold(2.0) < adaptive_threshold(0.1)


def test_staleness_cap_forces_upload():
    assert should_upload(0.0, 10.0, silent_rounds=2, staleness_cap=2)


def test_cloud_aggregation_cadence():
    assert [cloud_should_aggregate(i,2) for i in (1,2,3,4)] == [False,True,False,True]


def test_weighted_average_is_sample_weighted():
    assert np.allclose(weighted_average([np.array([1.]),np.array([5.])],[1,3]), [4.])


def test_sensor_partition_assigns_every_sensor_once():
    groups=split_sensor_clients([str(i) for i in range(207)],6)
    all_ids=[x for group in groups.values() for x in group]
    assert sorted(all_ids)==list(range(207)) and len(set(all_ids))==207

@pytest.mark.parametrize("codec", ["float32", "int8", "topk", "topk_int8"])
def test_update_codecs_roundtrip_shape_and_payload(codec):
    delta=np.linspace(-1,1,144,dtype=np.float32)
    decoded,nbytes,sent=_encode_update(delta,codec)
    assert decoded.shape==delta.shape and np.isfinite(decoded).all()
    assert nbytes >= 16
    assert nbytes>16 and 0<sent<=len(delta)
    if codec=="int8": assert nbytes < len(delta)*4+16
    if codec=="topk_int8": assert sent < len(delta)


def test_lora_adapter_uses_fewer_trainable_parameters():
    full, adapter, ratio = lora_parameter_counts(12, 12, 1)
    assert adapter == 24 and adapter < full and ratio < 0.20


def test_local_training_changes_actual_model_parameters():
    bundle=load_dataset(DATA,"metr_la",fast=True,max_rows=180)
    X,Y,M,S,T=make_windows(bundle,bundle.train)
    init=_init_state("full",17)
    trained=_local_train(X[:1000],Y[:1000],M[:1000],init,"full",seed=3,epochs=1,batch_size=128)
    assert np.linalg.norm(_flatten_state(trained)-_flatten_state(init))>0


def test_pipeline_smoke_produces_real_result_tables(tmp_path):
    meta, metrics, costs = run_pipeline(dataset="metr_la", fast=True, max_rows=360, rounds=3,
                                        seed=7, data_dir=DATA, output_dir=tmp_path, batch_size=256)
    assert meta["rows_used"] == 360
    assert {"HiLo-Fed","FedAvg","FedAvg + INT8","FedAvg + top-k","Centralized","Last-value"}.issubset(set(metrics.method))
    assert (tmp_path/"results"/"round_logs.csv").exists()
    assert (tmp_path/"results"/"client_round_metrics.csv").exists()
    hilo=costs[costs.method=="HiLo-Fed"].iloc[0]
    assert hilo.total_bytes > 0 and hilo.client_upload_bytes > 0
    logs = __import__("pandas").read_csv(tmp_path/"results"/"round_logs.csv")
    hilo_logs = logs[logs.method == "HiLo-Fed"]
    assert hilo_logs.client_uploads.max() > 0
    assert hilo_logs.mean_threshold.nunique() > 1
    # Verify the hierarchical route is actually exercised, not just listed in the UI.
    client_logs = __import__("pandas").read_csv(tmp_path/"results"/"client_round_metrics.csv")
    hilo_clients = client_logs[client_logs.method == "HiLo-Fed"]
    assert {"edge_id", "uploaded", "forced_upload", "payload_bytes"}.issubset(hilo_clients.columns)
    assert hilo_clients.edge_id.nunique() >= 2
    assert hilo_clients.uploaded.any()
    assert hilo_clients.loc[hilo_clients.uploaded, "payload_bytes"].gt(16).all()
    assert hilo_clients.forced_upload.any()
    assert hilo_logs.cloud_aggregated.any()
    assert hilo_logs.edge_upload_bytes.sum() > 0
    assert (tmp_path/"results"/"convergence.csv").exists()
    assert (tmp_path/"results"/"plots"/"communication_accuracy_tradeoff.png").exists()
