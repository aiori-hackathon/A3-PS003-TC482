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
