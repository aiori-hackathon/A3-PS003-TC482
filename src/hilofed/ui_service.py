"""Dashboard-to-research-pipeline integration helpers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_experiment(
    project_root: str | Path,
    dataset: str,
    *,
    rounds: int = 3,
    max_rows: int = 1200,
    seed: int = 17,
    partition: str = "sensor",
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run the real HiLo-Fed pipeline and return its captured process result.

    Outputs are written to <project_root>/results, the same directory consumed
    by the dashboard. No metrics are fabricated by this adapter.
    """
    root = Path(project_root).resolve()
    if dataset not in {"metr_la", "pems_bay"}:
        raise ValueError("dataset must be 'metr_la' or 'pems_bay'")
    if not 1 <= int(rounds) <= 50:
        raise ValueError("rounds must be between 1 and 50")
    if not 300 <= int(max_rows) <= 100_000:
        raise ValueError("max_rows must be between 300 and 100000")
    if partition not in {"sensor", "iid"}:
        raise ValueError("partition must be 'sensor' or 'iid'")
    if not (root / "data" / dataset).is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root / 'data' / dataset}")
    (root / "results").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable, "-m", "hilofed.pipeline",
        "--dataset", dataset,
        "--fast", "--max-rows", str(int(max_rows)),
        "--rounds", str(int(rounds)), "--seed", str(int(seed)),
        "--partition", partition,
        "--data-dir", str(root / "data"),
        "--output-dir", str(root),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=root, env=env, text=True, capture_output=True,
            timeout=int(timeout_seconds), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Experiment exceeded {timeout_seconds} seconds. Reduce rows or rounds and retry."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"HiLo-Fed pipeline failed (exit {proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout[-12000:]}\nSTDERR:\n{proc.stderr[-12000:]}"
        )
    try:
        summary = json.loads(proc.stdout)
    except (ValueError, json.JSONDecodeError):
        summary = {"status": "COMPLETED", "stdout": proc.stdout[-4000:]}
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "summary": summary,
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
    }
