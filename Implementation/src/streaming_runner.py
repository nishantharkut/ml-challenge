"""Streaming inference runner helpers and contract validators.

Implements shard-streamed country execution where S2 and S3 target indices
are built and freed independently, and candidate merge/scoring runs per S1 shard.
"""
import sqlite3
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import polars as pl

from .streaming_contracts import _COMPACT_METADATA_FIELDS

_REQUIRED_IDENTIFIERS = ("ordinal", "s1_id", "target_id")


class SqliteWinnerResolver:
    """Resolve target query exclusivity globally across all shards on disk."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.con = sqlite3.connect(self.db_path)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS winners (
                target_id TEXT PRIMARY KEY,
                ordinal INTEGER,
                s1_id TEXT,
                prob REAL
            ) WITHOUT ROWID;
        """)
        self.con.commit()

    def add_pairs(self, pairs: Iterable[tuple[int, str, str, float]]) -> None:
        """Insert candidate pairs; updates only if prob is strictly higher or tied with smaller s1_id."""
        pairs_list = list(pairs)
        if not pairs_list:
            return
        self.con.executemany("""
            INSERT INTO winners(ordinal, s1_id, target_id, prob) VALUES (?, ?, ?, ?)
            ON CONFLICT(target_id) DO UPDATE SET
                ordinal = excluded.ordinal,
                s1_id = excluded.s1_id,
                prob = excluded.prob
            WHERE excluded.prob > winners.prob OR (excluded.prob = winners.prob AND excluded.s1_id < winners.s1_id);
        """, pairs_list)
        self.con.commit()

    def get_matches_by_s1(self) -> dict[str, list[str]]:
        """Return deterministic mapping {s1_id: [sorted target_ids]} in source ordinal order."""
        cur = self.con.execute("SELECT s1_id, target_id FROM winners ORDER BY ordinal, target_id")
        matches: dict[str, list[str]] = defaultdict(list)
        for s1_id, target_id in cur:
            matches[s1_id].append(target_id)
        return dict(matches)

    def close(self) -> None:
        self.con.close()



def parse_source1_records_by_country(
    source1_path: str | Path,
    country: str,
) -> list[dict[str, Any]]:
    """Parse Source-1 rows for a specific country, preserving global file ordinal."""
    records: list[dict[str, Any]] = []
    path = Path(source1_path)
    target_country = (country or "").strip().upper()

    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if not header or header[0] not in {"entity_id", "source1_entity_id"}:
            raise ValueError(f"{path.name} must have entity_id as its first column")

        for ordinal, line in enumerate(handle):
            fields = line.rstrip("\r\n").split("\t")
            if not fields or not fields[0]:
                continue
            row_country = fields[3].strip().upper() if len(fields) > 3 else ""
            if row_country == target_country:
                records.append({
                    "ordinal": ordinal,
                    "entity_id": fields[0],
                    "business_name": fields[1] if len(fields) > 1 else "",
                    "business_address": fields[2] if len(fields) > 2 else "",
                    "country": row_country,
                })

    return records


def validate_feature_table_columns(
    df: pl.DataFrame,
    expected_feature_names: Sequence[str],
) -> bool:
    """Validate that a feature DataFrame contains required identifiers, features, and metadata."""
    cols = set(df.columns)

    for identifier in _REQUIRED_IDENTIFIERS:
        if identifier not in cols:
            raise ValueError(f"missing required identifier: {identifier}")

    for feat in expected_feature_names:
        if feat not in cols:
            raise ValueError(f"missing feature column: {feat}")

    for meta_col in _COMPACT_METADATA_FIELDS:
        if meta_col not in cols:
            raise ValueError(f"missing compact metadata column: {meta_col}")

    return True


def write_shard_manifest(
    manifest_path: str | Path,
    manifest_data: Mapping[str, Any],
) -> None:
    """Atomically write a JSON manifest for a completed shard."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest_data), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())

    temp_path.replace(path)


def is_shard_manifest_valid(
    manifest_path: str | Path,
    expected_fingerprints: Mapping[str, Any],
) -> bool:
    """Check if a shard manifest exists, completed, and matches expected fingerprints."""
    path = Path(manifest_path)
    if not path.is_file():
        return False

    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return False

    if manifest.get("status") != "completed":
        return False

    for key, expected_val in expected_fingerprints.items():
        if manifest.get(key) != expected_val:
            return False

    return True
