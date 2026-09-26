"""Tests for shard-streamed inference runner helpers."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import polars as pl

from src.streaming_runner import (
    SqliteWinnerResolver,
    parse_source1_records_by_country,
    validate_feature_table_columns,
    write_shard_manifest,
    is_shard_manifest_valid,
)


def test_source1_parsing_retains_global_file_ordinal(tmp_path: Path) -> None:
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "s1_0\tCompany Zero\t100 Main\tINDIA\n"
        "s1_1\tCompany One\t200 Main\tFRANCE\n"
        "s1_2\tCompany Two\t300 Main\tINDIA\n"
        "s1_3\tCompany Three\t400 Main\tUS\n"
        "s1_4\tCompany Four\t500 Main\tINDIA\n"
    )
    tsv_path = tmp_path / "test_source1.tsv"
    tsv_path.write_text(tsv_content, encoding="utf-8")

    india_records = parse_source1_records_by_country(tsv_path, country="INDIA")
    assert len(india_records) == 3
    # Ordinals must reflect the original line positions in test_source1.tsv (0, 2, 4)
    assert [r["ordinal"] for r in india_records] == [0, 2, 4]
    assert [r["entity_id"] for r in india_records] == ["s1_0", "s1_2", "s1_4"]
    assert india_records[0]["business_name"] == "Company Zero"
    assert india_records[0]["business_address"] == "100 Main"
    assert india_records[0]["country"] == "INDIA"


def test_feature_table_column_validation_rejects_missing_or_extra_columns() -> None:
    expected_features = ["feat_name_jw", "feat_addr_set"]
    valid_cols = {
        "ordinal": [0],
        "s1_id": ["s1_0"],
        "target_id": ["S2_10"],
        "feat_name_jw": [0.95],
        "feat_addr_set": [0.80],
        "source": ["S2"],
        "channel_count": [2],
        "best_rank": [1],
        "exact_name": [1.0],
        "exact_addr": [0.0],
        "rare_token": [0.0],
        "address_anchor": [0.0],
        "tfidf_name_score": [0.85],
        "tfidf_addr_score": [0.0],
        "number_overlap": [0.0],
    }
    df_valid = pl.DataFrame(valid_cols)
    assert validate_feature_table_columns(df_valid, expected_features) is True

    # Missing a required feature column
    df_missing = df_valid.drop("feat_name_jw")
    with pytest.raises(ValueError, match="missing feature column"):
        validate_feature_table_columns(df_missing, expected_features)

    # Missing a required identity column
    df_no_ord = df_valid.drop("ordinal")
    with pytest.raises(ValueError, match="missing required identifier"):
        validate_feature_table_columns(df_no_ord, expected_features)


def test_shard_manifest_validates_fingerprints_and_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "shard_000_manifest.json"
    manifest_data = {
        "shard_id": 0,
        "country": "FRANCE",
        "row_count": 500,
        "source_file_hash": "hash_s1_v1",
        "target_file_hash": "hash_s2_v1",
        "model_hash": "hash_model_v1",
        "feature_names": ["f1", "f2"],
        "config_hash": "cfg_v1",
        "status": "completed",
    }
    write_shard_manifest(manifest_path, manifest_data)

    expected = {
        "source_file_hash": "hash_s1_v1",
        "target_file_hash": "hash_s2_v1",
        "model_hash": "hash_model_v1",
        "feature_names": ["f1", "f2"],
        "config_hash": "cfg_v1",
    }
    assert is_shard_manifest_valid(manifest_path, expected) is True

    # Mismatched model hash
    bad_model = dict(expected, model_hash="hash_model_v2")
    assert is_shard_manifest_valid(manifest_path, bad_model) is False

    # Mismatched source hash
    bad_source = dict(expected, source_file_hash="hash_s1_changed")
    assert is_shard_manifest_valid(manifest_path, bad_source) is False

    # Incomplete manifest (corrupted or missing keys)
    manifest_path.write_text(json.dumps({"incomplete": True}), encoding="utf-8")
    assert is_shard_manifest_valid(manifest_path, expected) is False


def test_sqlite_winner_resolver_matches_better_exclusive_winner(tmp_path: Path) -> None:
    db_path = tmp_path / "winners.db"
    resolver = SqliteWinnerResolver(db_path)

    # Initial pair
    resolver.add_pairs([(1, "s1_b", "T1", 0.80)])
    # Lower prob does not replace
    resolver.add_pairs([(0, "s1_a", "T1", 0.70)])
    assert resolver.get_matches_by_s1() == {"s1_b": ["T1"]}

    # Higher prob replaces
    resolver.add_pairs([(2, "s1_c", "T1", 0.90)])
    assert resolver.get_matches_by_s1() == {"s1_c": ["T1"]}

    # Tie broken by smaller s1_id (s1_a < s1_c)
    resolver.add_pairs([(0, "s1_a", "T1", 0.90)])
    assert resolver.get_matches_by_s1() == {"s1_a": ["T1"]}

    # Multiple targets
    resolver.add_pairs([(0, "s1_a", "T2", 0.85), (1, "s1_b", "T3", 0.95)])
    matches = resolver.get_matches_by_s1()
    assert matches["s1_a"] == ["T1", "T2"]
    assert matches["s1_b"] == ["T3"]

    resolver.close()

