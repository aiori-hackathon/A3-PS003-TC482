from pathlib import Path
import pytest

from hilofed.ui_service import run_experiment


def test_run_experiment_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="dataset"):
        run_experiment(tmp_path, "unknown")


def test_run_experiment_rejects_invalid_rounds(tmp_path):
    with pytest.raises(ValueError, match="rounds"):
        run_experiment(tmp_path, "metr_la", rounds=0)


def test_run_experiment_requires_dataset_directory(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError, match="Dataset directory"):
        run_experiment(tmp_path, "metr_la")


def test_run_experiment_rejects_invalid_partition(tmp_path):
    with pytest.raises(ValueError, match="partition"):
        run_experiment(tmp_path, "metr_la", partition="random")


def test_run_experiment_passes_weight_controls_to_pipeline(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    from types import SimpleNamespace
    (tmp_path / "data" / "metr_la").mkdir(parents=True)
    captured = {}
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "SUCCESS"}), stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_experiment(tmp_path, "metr_la", rounds=2, max_rows=300,
                            tau=0.01, rank=3, k=4, w_edge=1.7, w_cloud=2.3)
    assert result["returncode"] == 0
    cmd = captured["cmd"]
    assert cmd[cmd.index("--tau") + 1] == "0.01"
    assert cmd[cmd.index("--rank") + 1] == "3"
    assert cmd[cmd.index("--k") + 1] == "4"
    assert cmd[cmd.index("--w-edge") + 1] == "1.7"
    assert cmd[cmd.index("--w-cloud") + 1] == "2.3"
