"""Pure contracts shared by the streamed inference and final assembly paths.

The helpers deliberately preserve only candidate metadata consumed by the production
balanced merge score.  They contain no file, model, dataframe, or dataset logic so
that a shard can be checked before it reaches the expensive scoring/assembly stages.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Any


_COMPACT_METADATA_FIELDS = (
    "source",
    "channel_count",
    "exact_name",
    "exact_addr",
    "rare_token",
    "address_anchor",
    "tfidf_name_score",
    "tfidf_addr_score",
    "best_rank",
    "number_overlap",
)


def _finite_number(value: Any, field: str) -> float:
    """Return a finite float while rejecting booleans and non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _nonempty_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def compact_candidate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert blocker metadata to the primitive columns needed after indexing.

    ``channel_ranks`` and concrete channel names are intentionally not retained:
    :func:`src.blocking._candidate_score` only depends on the number of channels.
    The returned mapping is safe to write as flat Parquet columns.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("candidate metadata must be a mapping")
    channels = metadata.get("channels")
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Iterable):
        raise ValueError("channels must be a finite iterable")
    channel_values = tuple(channels)
    if not channel_values:
        raise ValueError("channels must contain at least one channel")
    if len(set(channel_values)) != len(channel_values):
        raise ValueError("channels must not contain duplicates")

    source = _nonempty_id(metadata.get("source"), "source")
    best_rank = metadata.get("best_rank")
    if isinstance(best_rank, bool) or not isinstance(best_rank, int) or best_rank < 1:
        raise ValueError("best_rank must be a positive integer")

    compact: dict[str, Any] = {
        "source": source,
        "channel_count": len(channel_values),
        "best_rank": best_rank,
    }
    for field in (
        "exact_name",
        "exact_addr",
        "rare_token",
        "address_anchor",
        "tfidf_name_score",
        "tfidf_addr_score",
        "number_overlap",
    ):
        compact[field] = _finite_number(metadata.get(field, 0.0), field)
    return compact


def reconstruct_candidate_metadata(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the exact score-relevant blocker shape from flat primitive fields."""
    if not isinstance(compact, Mapping):
        raise ValueError("compact metadata must be a mapping")
    missing = [field for field in _COMPACT_METADATA_FIELDS if field not in compact]
    if missing:
        raise ValueError(f"compact metadata is missing fields: {', '.join(missing)}")

    source = _nonempty_id(compact["source"], "source")
    channel_count = compact["channel_count"]
    if (
        isinstance(channel_count, bool)
        or not isinstance(channel_count, int)
        or channel_count < 1
    ):
        raise ValueError("channel_count must be a positive integer")
    best_rank = compact["best_rank"]
    if isinstance(best_rank, bool) or not isinstance(best_rank, int) or best_rank < 1:
        raise ValueError("best_rank must be a positive integer")

    result: dict[str, Any] = {
        "source": source,
        # Synthetic names are deliberately opaque.  Only set cardinality affects
        # the production candidate score.
        "channels": {f"__stream_channel_{index}" for index in range(channel_count)},
        "channel_ranks": {},
        "best_rank": best_rank,
    }
    for field in (
        "exact_name",
        "exact_addr",
        "rare_token",
        "address_anchor",
        "tfidf_name_score",
        "tfidf_addr_score",
        "number_overlap",
    ):
        result[field] = _finite_number(compact[field], field)
    return result


def _candidate_score(metadata: Mapping[str, Any]) -> float:
    """Byte-for-byte formula equivalent to ``src.blocking._candidate_score``."""
    return (
        4.0 * metadata["exact_name"]
        + 3.0 * metadata["exact_addr"]
        + 1.5 * metadata["rare_token"]
        + 1.8 * metadata["address_anchor"]
        + 3.0 * metadata["tfidf_name_score"]
        + 2.0 * metadata["tfidf_addr_score"]
        + 1.5 * metadata["number_overlap"]
        + 0.15 * len(metadata["channels"])
        + 0.05 / max(metadata["best_rank"], 1)
    )


def _rank_candidates(items: Iterable[tuple[str, Mapping[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    ranked: list[tuple[str, dict[str, Any]]] = []
    for candidate_id, compact in items:
        candidate_id = _nonempty_id(candidate_id, "candidate ID")
        ranked.append((candidate_id, reconstruct_candidate_metadata(compact)))
    return _rank_reconstructed_candidates(ranked)


def _rank_reconstructed_candidates(
    items: Iterable[tuple[str, Mapping[str, Any]]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Rank already-reconstructed metadata without applying conversion again."""
    return sorted(items, key=lambda item: (-_candidate_score(item[1]), item[0]))


def balanced_merge_compact_candidates(
    source2: Mapping[str, Mapping[str, Any]],
    source3: Mapping[str, Mapping[str, Any]],
    *,
    total_k: int,
) -> dict[str, dict[str, Any]]:
    """Apply ``SourceSeparatedBlocker._balanced_merge`` to compact metadata.

    The selected mapping retains the production blocker's score order.  Candidate
    TSV serialization must subsequently sort IDs independently; score order and TSV
    order are intentionally separate contracts.
    """
    if isinstance(total_k, bool) or not isinstance(total_k, int) or total_k < 0:
        raise ValueError("total_k must be a non-negative integer")
    if not isinstance(source2, Mapping) or not isinstance(source3, Mapping):
        raise ValueError("source candidate inputs must be mappings")
    overlapping_ids = set(source2).intersection(source3)
    if overlapping_ids:
        raise ValueError("candidate ID appears in both source inputs")

    ranked2 = _rank_candidates(source2.items())
    ranked3 = _rank_candidates(source3.items())
    quota2 = (total_k + 1) // 2
    quota3 = total_k // 2
    selected = ranked2[:quota2] + ranked3[:quota3]
    remaining = ranked2[quota2:] + ranked3[quota3:]
    if len(selected) < total_k:
        selected.extend(_rank_reconstructed_candidates(remaining)[: total_k - len(selected)])
    return dict(_rank_reconstructed_candidates(selected)[:total_k])


def _id_list(value: Any, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence of IDs")
    result = [_nonempty_id(item, field[:-1] if field.endswith("s") else field) for item in value]
    if len(set(result)) != len(result):
        singular = field[:-1] if field.endswith("s") else field
        raise ValueError(f"duplicate {singular}")
    if result != sorted(result):
        raise ValueError(f"{field} must be sorted")
    return result


def _normalise_expected_rows(expected_rows: Sequence[tuple[int, str]]) -> list[tuple[int, str]]:
    expected: list[tuple[int, str]] = []
    seen_ordinals: set[int] = set()
    seen_ids: set[str] = set()
    for item in expected_rows:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("expected rows must contain (ordinal, S1 ID) tuples")
        ordinal, s1_id = item
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("expected ordinal must be a non-negative integer")
        s1_id = _nonempty_id(s1_id, "expected S1 ID")
        if ordinal in seen_ordinals:
            raise ValueError("duplicate expected ordinal")
        if s1_id in seen_ids:
            raise ValueError("duplicate expected S1 ID")
        seen_ordinals.add(ordinal)
        seen_ids.add(s1_id)
        expected.append((ordinal, s1_id))
    if expected:
        expected_ordinals = [ordinal for ordinal, _ in expected]
        contiguous_ordinals = list(
            range(expected_ordinals[0], expected_ordinals[0] + len(expected_ordinals))
        )
        if expected_ordinals != contiguous_ordinals:
            raise ValueError("expected rows have a missing ordinal or are not in ordinal order")
    return expected


def validate_shard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: Sequence[tuple[int, str]] | None = None,
) -> None:
    """Fail closed unless row data has exact, ordered, subset-safe coverage.

    Each input row is a compact representation of the two final output rows:
    ``ordinal``, ``s1_id``, sorted ``candidate_ids``, and sorted
    ``prediction_ids``.  An optional expected sequence pins validation to original
    Source-1 file order; without it, ordinals must still be contiguous within the
    shard (and may start at any global ordinal).
    """
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("rows must be a sequence")
    observed: list[tuple[int, str]] = []
    seen_ordinals: set[int] = set()
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each shard row must be a mapping")
        ordinal = row.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        s1_id = _nonempty_id(row.get("s1_id"), "S1 ID")
        if ordinal in seen_ordinals:
            raise ValueError("duplicate ordinal")
        if s1_id in seen_ids:
            raise ValueError("duplicate S1 ID")
        candidates = _id_list(row.get("candidate_ids"), "candidate IDs")
        predictions = _id_list(row.get("prediction_ids"), "prediction IDs")
        missing_predictions = set(predictions).difference(candidates)
        if missing_predictions:
            raise ValueError("prediction ID not present in candidates")
        seen_ordinals.add(ordinal)
        seen_ids.add(s1_id)
        observed.append((ordinal, s1_id))

    if expected_rows is not None:
        expected = _normalise_expected_rows(expected_rows)
        if observed != expected:
            raise ValueError("rows do not match expected Source-1 order and coverage")
        return

    if observed:
        first_ordinal = observed[0][0]
        required = list(range(first_ordinal, first_ordinal + len(observed)))
        actual = [ordinal for ordinal, _ in observed]
        if actual != required:
            raise ValueError("missing ordinal or rows are not in ordinal order")


def better_exclusive_winner(
    candidate_s1_id: str,
    candidate_probability: Real,
    incumbent_s1_id: str,
    incumbent_probability: Real,
) -> bool:
    """Return whether a candidate wins target exclusivity deterministically."""
    candidate_s1_id = _nonempty_id(candidate_s1_id, "candidate S1 ID")
    incumbent_s1_id = _nonempty_id(incumbent_s1_id, "incumbent S1 ID")
    candidate_probability = _finite_number(candidate_probability, "candidate probability")
    incumbent_probability = _finite_number(incumbent_probability, "incumbent probability")
    return candidate_probability > incumbent_probability or (
        candidate_probability == incumbent_probability and candidate_s1_id < incumbent_s1_id
    )
