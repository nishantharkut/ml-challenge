"""Contracts for deterministic multilingual normalization views."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def test_accent_folded_and_original_views_are_distinct() -> None:
    from src.normalization import text_views

    views = text_views("Café Étoile")

    assert views.normalized == "café étoile"
    assert views.folded == "cafe etoile"


def test_nfkc_and_casefold_are_applied_before_tokenization() -> None:
    from src.normalization import text_views

    views = text_views("Ｓｔｒａße  AG")

    assert views.normalized == "strasse ag"
    assert views.tokens == ("strasse", "ag")


def test_compact_core_and_numeric_views_are_immutable() -> None:
    from src.normalization import text_views

    views = text_views("Unit 12 / Café 50001")

    assert views.compact == "unit12café50001"
    assert views.core == views.compact
    assert views.numeric_tokens == frozenset({"12", "50001"})
    with pytest.raises(FrozenInstanceError):
        views.normalized = "changed"  # type: ignore[misc]


def test_latin_accent_folding_preserves_indic_marks() -> None:
    from src.normalization import text_views

    views = text_views("कंपनी Café")

    assert views.normalized == "कंपनी café"
    assert views.folded == "कंपनी cafe"


@pytest.mark.parametrize("missing", [None, "", "   ", "null", "<NULL>", "None", "NaN"])
def test_missing_values_never_create_positive_text_evidence(missing: object) -> None:
    from src.normalization import text_views

    views = text_views(missing)

    assert views.normalized == ""
    assert views.folded == ""
    assert views.compact == ""
    assert views.tokens == ()
    assert views.numeric_tokens == frozenset()


def test_numeric_tokens_of_missing_text_are_empty() -> None:
    from src.normalization import numeric_tokens

    assert numeric_tokens("") == frozenset()


def test_preprocess_record_keeps_legacy_fields_and_adds_views() -> None:
    from src.normalization import preprocess_record

    record = preprocess_record(
        {
            "entity_id": "S1-1",
            "business_name": "Café Étoile Ltd",
            "business_address": "Unit 12",
            "country": "france",
        }
    )

    assert record["name_norm"] == "café étoile ltd"
    assert record["name_core"] == "café étoile"
    assert record["name_tokens"] == ["café", "étoile", "ltd"]
    assert record["addr_numbers"] == {"12"}
    assert record["name_folded"] == "cafe etoile ltd"
    assert record["name_compact"] == "caféétoileltd"
    assert record["addr_numeric_tokens"] == frozenset({"12"})
