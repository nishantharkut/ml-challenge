"""Behavioral tests for compact, source-separated candidate retrieval."""
from __future__ import annotations

import re
from unittest import mock

from src.blocking import SourceSeparatedBlocker, TargetSearchIndex


def record(entity_id: str, name: str, address: str = "", country: str = "FRANCE") -> dict:
    name_norm = " ".join(name.casefold().split())
    addr_norm = " ".join(address.casefold().split())
    return {
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country_norm": country,
        "name_norm": name_norm,
        "name_folded": name_norm,
        "name_core": name_norm,
        "name_tokens": name_norm.split(),
        "addr_norm": addr_norm,
        "addr_folded": addr_norm,
        "addr_empty": not bool(addr_norm),
        "addr_numbers": set(re.findall(r"\d+", addr_norm)),
        "addr_postal": None,
    }


def test_character_channel_recovers_multi_token_typo() -> None:
    index = TargetSearchIndex(
        [record("S2_right", "Boulangerie Etoile"), record("S2_wrong", "Garage du Port")],
        source_label="S2",
        k_per_source=2,
    )

    result = index.retrieve_for_s1_chunk([record("S1_q", "Boulangeriee Etolie")], k=2)

    candidate = result["S1_q"]["S2_right"]
    assert "char_name" in candidate["channels"]
    assert candidate["best_rank"] == 1


def test_fuzzy_address_reranks_oversized_exact_name_bucket() -> None:
    targets = [record(f"S2_{i:02d}", "Acme", f"{i} Rue Victor Hugo") for i in range(20)]
    index = TargetSearchIndex(targets, source_label="S2", k_per_source=1)

    result = index.retrieve_for_s1_chunk(
        [record("S1_q", "Acme", "17 Rue Victor Hugo Apt 2")],
        k=1,
    )

    assert list(result["S1_q"]) == ["S2_17"]
    assert result["S1_q"]["S2_17"]["tfidf_addr_score"] > 0


def test_gallery_keeps_only_one_sparse_matrix() -> None:
    index = TargetSearchIndex(
        [record("S2_a", "Cafe Etoile", "1 Rue Victor Hugo")],
        source_label="S2",
        k_per_source=1,
    )

    assert index.sparse_channel_count == 1
    assert not hasattr(index, "address_channel")


def test_oversized_exact_bucket_uses_number_discriminator() -> None:
    targets = [record(f"S2_{i:03d}", "Acme", f"{i} Rue de Paris") for i in range(120)]
    index = TargetSearchIndex(targets, source_label="S2", k_per_source=1)
    index.combined_channel.enabled = False

    result = index.retrieve_for_s1_chunk(
        [record("S1_q", "Acme", "117 Rue de Paris Batiment B")],
        k=1,
    )

    assert list(result["S1_q"]) == ["S2_117"]


def test_rare_address_anchor_retrieves_when_names_do_not_overlap() -> None:
    targets = [
        record("S2_right", "Northwind Trading", "12 azalea avenue lyon"),
        record("S2_other", "Coastal Imports", "55 azalea avenue nice"),
    ]
    index = TargetSearchIndex(targets, source_label="S2", k_per_source=2)
    index.combined_channel.enabled = False

    result = index.retrieve_for_s1_chunk(
        [record("S1_q", "Unrelated Alias", "77 azalea avenue paris")], k=2
    )

    assert "S2_right" in result["S1_q"]
    assert "address_anchor" in result["S1_q"]["S2_right"]["channels"]


class _StaticIndex:
    def __init__(self, source: str, candidates: list[tuple[str, float]]):
        self.source = source
        self.candidates = candidates

    def retrieve_for_s1_chunk(self, s1_chunk, k=None):
        sid = s1_chunk[0]["entity_id"]
        return {
            sid: {
                target_id: {
                    "source": self.source,
                    "channels": {"char_name"},
                    "exact_name": 0,
                    "exact_addr": 0,
                    "rare_token": 0,
                    "tfidf_name_score": score,
                    "tfidf_addr_score": 0.0,
                    "best_rank": rank,
                }
                for rank, (target_id, score) in enumerate(self.candidates[:k], start=1)
            }
        }


def test_total_cap_preserves_both_source_quotas() -> None:
    blocker = SourceSeparatedBlocker(k_per_source=2, total_k=2)
    s2 = _StaticIndex("S2", [("S2_a", 1.0), ("S2_b", 0.9)])
    s3 = _StaticIndex("S3", [("S3_a", 0.2), ("S3_b", 0.1)])

    structured, _ = blocker.generate_candidates_for_chunk([record("S1_q", "Acme")], s2, s3)

    assert {meta["source"] for meta in structured["S1_q"].values()} == {"S2", "S3"}


def test_batch_generation_never_keeps_two_sparse_indices_alive() -> None:
    state = {"active": 0, "peak": 0}

    class _TrackedIndex:
        def __init__(self, records, source_label, k_per_source):
            self.source = source_label
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])

        def retrieve_for_s1_chunk(self, s1_chunk, k=None):
            return {row["entity_id"]: {} for row in s1_chunk}

        def close(self):
            state["active"] -= 1

    blocker = SourceSeparatedBlocker(k_per_source=2, total_k=2)
    with mock.patch("src.blocking.TargetSearchIndex", _TrackedIndex):
        blocker.generate_candidates([record("S1_q", "Acme")], [], [])

    assert state == {"active": 0, "peak": 1}
