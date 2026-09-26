"""Focused contracts for compact, shard-local streamed inference data."""
from __future__ import annotations

import pytest

from src.blocking import SourceSeparatedBlocker
from src.streaming_contracts import (
    balanced_merge_compact_candidates,
    better_exclusive_winner,
    compact_candidate_metadata,
    reconstruct_candidate_metadata,
    validate_shard_rows,
)


def _meta(source: str, *, score: float, channels: int = 1) -> dict:
    """Minimal blocker metadata with a controllable TF-IDF contribution."""
    return {
        "source": source,
        "channels": {f"channel_{index}" for index in range(channels)},
        "channel_ranks": {"channel_0": 1},
        "exact_name": 0,
        "exact_addr": 0,
        "rare_token": 0,
        "address_anchor": 0,
        "tfidf_name_score": score,
        "tfidf_addr_score": 0.0,
        "best_rank": 1,
        "number_overlap": 0.0,
    }


def test_compact_metadata_reconstructs_exact_score_relevant_fields() -> None:
    compact = compact_candidate_metadata(_meta("S2", score=0.72, channels=3))

    assert compact == {
        "source": "S2",
        "channel_count": 3,
        "exact_name": 0,
        "exact_addr": 0,
        "rare_token": 0,
        "address_anchor": 0,
        "tfidf_name_score": 0.72,
        "tfidf_addr_score": 0.0,
        "best_rank": 1,
        "number_overlap": 0.0,
    }
    rebuilt = reconstruct_candidate_metadata(compact)
    assert rebuilt["source"] == "S2"
    assert len(rebuilt["channels"]) == 3
    assert rebuilt["channel_ranks"] == {}
    for field in (
        "exact_name",
        "exact_addr",
        "rare_token",
        "address_anchor",
        "tfidf_name_score",
        "tfidf_addr_score",
        "best_rank",
        "number_overlap",
    ):
        assert rebuilt[field] == compact[field]


def test_compact_metadata_rejects_an_unreachable_zero_channel_candidate() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        compact_candidate_metadata(_meta("S2", score=0.5, channels=0))


def test_balanced_merge_is_deterministic_and_preserves_source_quotas() -> None:
    source2 = {
        "S2_z": compact_candidate_metadata(_meta("S2", score=0.90)),
        "S2_a": compact_candidate_metadata(_meta("S2", score=0.80)),
    }
    source3 = {
        "S3_b": compact_candidate_metadata(_meta("S3", score=0.20)),
        "S3_a": compact_candidate_metadata(_meta("S3", score=0.20)),
    }

    merged = balanced_merge_compact_candidates(source2, source3, total_k=2)

    # One item per source (ceil/floor quotas) even though S2 scores higher.
    assert set(item["source"] for item in merged.values()) == {"S2", "S3"}
    # The source-3 tie is resolved by target ID, exactly as the blocker ranks it.
    assert list(merged) == ["S2_z", "S3_a"]


def test_balanced_merge_rejects_cross_source_duplicate_candidate_ids() -> None:
    source2 = {"target": compact_candidate_metadata(_meta("S2", score=0.9))}
    source3 = {"target": compact_candidate_metadata(_meta("S3", score=0.8))}

    with pytest.raises(ValueError, match="both source inputs"):
        balanced_merge_compact_candidates(source2, source3, total_k=2)


def test_compact_balanced_merge_matches_production_blocker_semantics() -> None:
    native_s2 = {
        "S2_high": _meta("S2", score=0.91, channels=2),
        "S2_low": _meta("S2", score=0.11),
    }
    native_s3 = {
        "S3_high": _meta("S3", score=0.81),
        "S3_low": _meta("S3", score=0.05, channels=3),
    }
    expected = SourceSeparatedBlocker(total_k=3)._balanced_merge(native_s2, native_s3)

    actual = balanced_merge_compact_candidates(
        {candidate_id: compact_candidate_metadata(meta) for candidate_id, meta in native_s2.items()},
        {candidate_id: compact_candidate_metadata(meta) for candidate_id, meta in native_s3.items()},
        total_k=3,
    )

    assert list(actual) == list(expected)


def test_validate_shard_rows_accepts_sorted_deduplicated_subset() -> None:
    rows = [
        {
            "ordinal": 7,
            "s1_id": "S1_b",
            "candidate_ids": ["S2_a", "S3_z"],
            "prediction_ids": ["S3_z"],
        },
        {
            "ordinal": 8,
            "s1_id": "S1_c",
            "candidate_ids": [],
            "prediction_ids": [],
        },
    ]

    assert validate_shard_rows(rows, expected_rows=[(7, "S1_b"), (8, "S1_c")]) is None


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"ordinal": 0, "s1_id": "S1_a", "candidate_ids": [], "prediction_ids": []},
                {"ordinal": 0, "s1_id": "S1_b", "candidate_ids": [], "prediction_ids": []},
            ],
            "duplicate ordinal",
        ),
        (
            [
                {"ordinal": 0, "s1_id": "S1_a", "candidate_ids": [], "prediction_ids": []},
                {"ordinal": 1, "s1_id": "S1_a", "candidate_ids": [], "prediction_ids": []},
            ],
            "duplicate S1 ID",
        ),
        (
            [
                {"ordinal": 0, "s1_id": "S1_a", "candidate_ids": [], "prediction_ids": []},
                {"ordinal": 2, "s1_id": "S1_c", "candidate_ids": [], "prediction_ids": []},
            ],
            "missing ordinal",
        ),
        (
            [
                {
                    "ordinal": 0,
                    "s1_id": "S1_a",
                    "candidate_ids": ["S3_z", "S2_a"],
                    "prediction_ids": [],
                },
            ],
            "sorted",
        ),
        (
            [
                {
                    "ordinal": 0,
                    "s1_id": "S1_a",
                    "candidate_ids": ["S2_a", "S2_a"],
                    "prediction_ids": [],
                },
            ],
            "duplicate candidate ID",
        ),
        (
            [
                {
                    "ordinal": 0,
                    "s1_id": "S1_a",
                    "candidate_ids": ["S2_a"],
                    "prediction_ids": ["S3_z"],
                },
            ],
            "not present in candidates",
        ),
    ],
)
def test_validate_shard_rows_rejects_malformed_or_incomplete_rows(rows, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_shard_rows(rows)


def test_validate_shard_rows_rejects_noncontiguous_expected_coverage() -> None:
    rows = [
        {"ordinal": 0, "s1_id": "S1_a", "candidate_ids": [], "prediction_ids": []},
        {"ordinal": 2, "s1_id": "S1_c", "candidate_ids": [], "prediction_ids": []},
    ]

    with pytest.raises(ValueError, match="missing ordinal"):
        validate_shard_rows(rows, expected_rows=[(0, "S1_a"), (2, "S1_c")])


def test_better_exclusive_winner_uses_probability_then_s1_id() -> None:
    assert better_exclusive_winner("S1_b", 0.91, "S1_a", 0.90)
    assert better_exclusive_winner("S1_a", 0.90, "S1_b", 0.90)
    assert not better_exclusive_winner("S1_b", 0.90, "S1_a", 0.90)
    assert not better_exclusive_winner("S1_a", 0.90, "S1_a", 0.90)
