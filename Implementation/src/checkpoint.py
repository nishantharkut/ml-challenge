"""Checkpoint persistence and provenance validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import polars as pl

from .config import CHECKPOINT_DIR
from .run_context import (
    MANIFEST_SCHEMA_VERSION,
    canonical_json,
    canonical_json_sha256,
    fingerprint_artifacts,
    fingerprint_file,
)

def compute_file_hash(filepath, block_size=65536):
    """Return a full-file SHA-256 digest, or ``None`` when absent."""
    if not os.path.exists(filepath):
        return None
    return fingerprint_file(filepath)["sha256"]

def get_manifest_path(stage_name):
    """Get file path for a stage manifest."""
    return os.path.join(CHECKPOINT_DIR, f"{stage_name}_manifest.json")


def _resolve_manifest_path(manifest_path: os.PathLike[str] | str) -> Path:
    """Accept an explicit manifest path or the legacy stage-name shorthand."""
    if isinstance(manifest_path, os.PathLike):
        return Path(manifest_path)
    candidate = Path(manifest_path)
    if candidate.suffix.lower() == ".json" or candidate.parent != Path("."):
        return candidate
    return Path(get_manifest_path(manifest_path))


def _contains_expected(actual: Any, expected: Any) -> bool:
    """Return whether ``actual`` recursively contains every expected field."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _contains_expected(actual[key], value)
            for key, value in expected.items()
        )
    return actual == expected


def _fingerprint_matches(path: os.PathLike[str] | str, recorded: Mapping[str, Any]) -> bool:
    try:
        current = fingerprint_file(path)
    except (OSError, TypeError, ValueError):
        return False
    return (
        current.get("size") == recorded.get("size")
        and current.get("sha256") == recorded.get("sha256")
    )


def is_checkpoint_valid(manifest_path, expected=None, artifacts=None):
    """Validate completion, provenance fields, manifest integrity, and artifacts.

    ``manifest_path`` may be an explicit path or a legacy stage name. Legacy
    manifests remain readable only when no provenance expectation is requested;
    they cannot validate artifact identity because they contain no fingerprints.
    """
    resolved_manifest = _resolve_manifest_path(manifest_path)
    if not resolved_manifest.is_file():
        return False
    try:
        with resolved_manifest.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict) or manifest.get("completed") is not True:
            return False

        schema_version = manifest.get("schema_version")
        if schema_version is None:
            return False
        if schema_version != MANIFEST_SCHEMA_VERSION:
            return False

        recorded_content_hash = manifest.get("content_hash")
        if not isinstance(recorded_content_hash, str):
            return False
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("content_hash", None)
        if canonical_json_sha256(unsigned_manifest) != recorded_content_hash:
            return False

        if expected is not None:
            if not isinstance(expected, Mapping) or not _contains_expected(manifest, expected):
                return False

        recorded_artifacts = manifest.get("artifacts", {})
        if not isinstance(recorded_artifacts, Mapping):
            return False
        if artifacts is not None:
            if not isinstance(artifacts, Mapping):
                return False
            if set(artifacts) != set(recorded_artifacts):
                return False
            for name, path in artifacts.items():
                recorded = recorded_artifacts.get(name)
                if not isinstance(recorded, Mapping) or not _fingerprint_matches(path, recorded):
                    return False
        else:
            for recorded in recorded_artifacts.values():
                if not isinstance(recorded, Mapping) or not isinstance(recorded.get("path"), str):
                    return False
                if not _fingerprint_matches(recorded["path"], recorded):
                    return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def save_manifest(manifest_path, metadata, artifacts=None):
    """Atomically write a signed, schema-versioned manifest.

    Existing stage-name call sites remain supported. The supplied metadata is
    copied and is never mutated.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    resolved_manifest = _resolve_manifest_path(manifest_path)
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.update(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed": True,
            "artifacts": fingerprint_artifacts(artifacts),
        }
    )
    payload.pop("content_hash", None)
    payload["content_hash"] = canonical_json_sha256(payload)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved_manifest.name}.",
        suffix=".tmp",
        dir=resolved_manifest.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, resolved_manifest)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise
    return resolved_manifest

def save_parquet_atomic(df, filepath):
    """Save Polars DataFrame atomically to Parquet."""
    destination = Path(filepath)
    if not isinstance(df, pl.DataFrame):
        raise ValueError("df must be a Polars DataFrame")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        df.write_parquet(temporary_path, compression="zstd")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

def load_parquet(filepath):
    """Load Parquet file using Polars."""
    return pl.read_parquet(filepath)
