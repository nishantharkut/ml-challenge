"""Deterministic hashing helpers for provenance-safe pipeline artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Portable roots for one immutable pipeline run namespace."""

    run_id: str
    data_root: Path
    implementation_root: Path
    runs_root: Path

    def __post_init__(self) -> None:
        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError(
                "run_id must be 1-80 characters using letters, digits, '.', '_' or '-'"
            )
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser().resolve())
        object.__setattr__(
            self, "implementation_root", Path(self.implementation_root).expanduser().resolve()
        )
        object.__setattr__(self, "runs_root", Path(self.runs_root).expanduser().resolve())


    @property
    def run_root(self) -> Path:
        return self.runs_root / self.run_id

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_root / "checkpoints"

    @property
    def output_dir(self) -> Path:
        return self.run_root / "output"

    @classmethod
    def from_environment(
        cls,
        implementation_root: os.PathLike[str] | str | None = None,
        run_id: str | None = None,
        data_root: os.PathLike[str] | str | None = None,
        runs_root: os.PathLike[str] | str | None = None,
    ) -> "RunContext":
        implementation = Path(
            implementation_root or Path(__file__).resolve().parents[1]
        ).resolve()
        selected_data_root = data_root or os.environ.get("AMAZON_ML_DATASET_DIR")
        if selected_data_root is None:
            candidates = (
                implementation.parent
                / "dataset"
                / "6ab10eb3b23ba_student_resource"
                / "student_resource"
                / "dataset",
                implementation.parent / "dataset",
                Path("/kaggle/input/amazon-ml-challenge-2026/dataset"),
                Path("/content/dataset"),
            )
            selected_data_root = next(
                (candidate for candidate in candidates if candidate.is_dir()), candidates[0]
            )
        selected_runs_root = (
            runs_root
            or os.environ.get("AMAZON_ML_RUNS_DIR")
            or implementation / "runs"
        )
        selected_run_id = run_id or os.environ.get("AMAZON_ML_RUN_ID") or "default"
        return cls(
            run_id=selected_run_id,
            data_root=Path(selected_data_root),
            implementation_root=implementation,
            runs_root=Path(selected_runs_root),
        )


def canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically for hashing and manifests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 digest of the canonical UTF-8 JSON representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fingerprint_file(path: os.PathLike[str] | str) -> dict[str, int | str]:
    """Hash an entire file and return its byte size and SHA-256 digest."""
    artifact_path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with artifact_path.open("rb") as artifact:
        while chunk := artifact.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def fingerprint_artifacts(
    artifacts: Mapping[str, os.PathLike[str] | str] | None,
) -> dict[str, dict[str, int | str]]:
    """Fingerprint named artifacts, recording stable names and resolved paths."""
    if artifacts is None:
        return {}
    if not isinstance(artifacts, Mapping):
        raise TypeError("artifacts must be a mapping of logical name to file path")

    fingerprints: dict[str, dict[str, int | str]] = {}
    for name, path in sorted(artifacts.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not name:
            raise ValueError("artifact names must be non-empty strings")
        resolved_path = Path(path).resolve()
        fingerprint = fingerprint_file(resolved_path)
        fingerprints[name] = {"path": str(resolved_path), **fingerprint}
    return fingerprints
