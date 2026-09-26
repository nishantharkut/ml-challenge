"""Shared decoder behavior and unbiased policy tuning."""
from __future__ import annotations

from src.decoder import SetDecoder, tune_decoder_policies


def test_exclusivity_ties_are_resolved_deterministically() -> None:
    decoder = SetDecoder(prob_threshold=0.5, use_query_exclusivity=True)
    scored = {"S1-b": [("S2-x", 0.9)], "S1-a": [("S2-x", 0.9)]}

    predictions = decoder.decode(scored, s1_all_ids=["S1-b", "S1-a"])

    assert predictions["S1-a"] == {"S2-x"}
    assert predictions["S1-b"] == set()


def test_each_exclusivity_policy_gets_its_own_threshold_search() -> None:
    scored = {
        "S1-match": [("S2-x", 0.80)],
        "S1-singleton": [("S2-x", 0.70)],
    }
    truth = {"S1-match": {"S2-x"}, "S1-singleton": set()}

    result = tune_decoder_policies(scored, truth, thresholds=[0.60, 0.75])

    assert result[True]["score"] == 1.0
    assert result[True]["threshold"] == 0.60
    assert result[False]["score"] == 1.0
    assert result[False]["threshold"] == 0.75
