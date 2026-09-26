"""Portable path resolution must work on Linux, notebooks, and local machines."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_config_honors_explicit_dataset_and_run_roots(tmp_path: Path) -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "dataset"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(implementation_root),
            "AMAZON_ML_DATASET_DIR": str(data_root),
            "AMAZON_ML_RUNS_DIR": str(runs_root),
            "AMAZON_ML_RUN_ID": "portable-run",
        }
    )
    command = (
        "import json; from src import config; "
        "print(json.dumps({'data': config.DATASET_DIR, 'checkpoints': config.CHECKPOINT_DIR, "
        "'output': config.OUTPUT_DIR}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    paths = json.loads(completed.stdout.strip())

    assert Path(paths["data"]) == data_root.resolve()
    assert Path(paths["checkpoints"]) == runs_root.resolve() / "portable-run" / "checkpoints"
    assert Path(paths["output"]) == runs_root.resolve() / "portable-run" / "output"
