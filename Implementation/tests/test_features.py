"""Logical pair-feature contracts independent of model training."""
from __future__ import annotations

from src.features import extract_pair_features
from src.normalization import preprocess_record


def rec(entity_id: str, name: str, address: str = "") -> dict:
    return preprocess_record(
        {
            "entity_id": entity_id,
            "business_name": name,
            "business_address": address,
            "country": "France",
        }
    )


def test_accent_folded_name_is_explicit_perfect_signal() -> None:
    features = extract_pair_features(rec("S1-a", "Café Étoile"), rec("S2-a", "Cafe Etoile"))

    assert features["name_folded_ratio"] == 1.0
    assert features["name_folded_exact"] == 1.0


def test_partial_number_overlap_is_not_treated_as_exact_agreement() -> None:
    features = extract_pair_features(
        rec("S1-a", "Acme", "12 34 Rue de Paris"),
        rec("S2-a", "Acme", "12 99 Rue de Paris"),
    )

    assert features["num_any_overlap"] == 1.0
    assert features["num_exact_set"] == 0.0
    assert 0.0 < features["num_jaccard"] < 1.0
    assert features["num_conflict"] == 1.0


def test_empty_fields_do_not_create_positive_similarity() -> None:
    features = extract_pair_features(rec("S1-a", "", ""), rec("S2-a", "", ""))

    assert features["name_exact"] == 0.0
    assert features["name_folded_exact"] == 0.0
    assert features["addr_exact"] == 0.0
    assert features["num_exact_set"] == 0.0


def test_address_anchor_metadata_is_available_to_the_matcher() -> None:
    features = extract_pair_features(
        rec("S1-a", "Northwind", "12 Azalea Avenue"),
        rec("S2-a", "Alias", "77 Azalea Avenue"),
        {"source": "S2", "address_anchor": 1},
    )

    assert features["address_anchor_hit"] == 1.0
