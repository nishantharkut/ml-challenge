from __future__ import annotations

from pathlib import Path

import pytest

from src.output_contract import SubmissionContractError, validate_submission_files


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _valid_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    source1 = _write(
        tmp_path / "test_source1.tsv",
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "s1-a\tA\tOne\tUS\n"
        "s1-b\tB\tTwo\tUS\n",
    )
    candidates = _write(
        tmp_path / "candidate_pairs.tsv",
        "source1_entity_id\tcandidate_entity_ids\n"
        "s1-a\tS2-1,S3-2\n"
        "s1-b\t\n",
    )
    matching = _write(
        tmp_path / "matching_results.tsv",
        "source1_entity_id\tmatched_entity_ids\n"
        "s1-a\tS3-2\n"
        "s1-b\t\n",
    )
    return source1, candidates, matching


def test_submission_contract_accepts_exact_ordered_subset(tmp_path: Path):
    source1, candidates, matching = _valid_paths(tmp_path)

    result = validate_submission_files(source1, candidates, matching)

    assert result == {"rows": 2, "candidate_pairs": 2, "matched_pairs": 1}


@pytest.mark.parametrize(
    ("candidate_body", "matching_body", "expected_message"),
    [
        ("s1-a\tS2-1,S3-2\n", "s1-a\tS3-2\ns1-b\t\n", "row count"),
        ("s1-b\t\ns1-a\tS2-1,S3-2\n", "s1-a\tS3-2\ns1-b\t\n", "order"),
        ("s1-a\tS2-1,S2-1\ns1-b\t\n", "s1-a\tS2-1\ns1-b\t\n", "duplicate"),
        ("s1-a\tS2-1,S3-2\ns1-b\t\n", "s1-a\tS3-9\ns1-b\t\n", "not present"),
    ],
)
def test_submission_contract_rejects_incomplete_malformed_or_non_subset_rows(
    tmp_path: Path,
    candidate_body: str,
    matching_body: str,
    expected_message: str,
):
    source1, candidates, matching = _valid_paths(tmp_path)
    candidates.write_text(
        "source1_entity_id\tcandidate_entity_ids\n" + candidate_body,
        encoding="utf-8",
        newline="\n",
    )
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n" + matching_body,
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(SubmissionContractError, match=expected_message):
        validate_submission_files(source1, candidates, matching)


@pytest.mark.parametrize("malformed", ["S2-1,", ",S2-1", "S2-1,,S3-2", ",,"])
def test_submission_contract_rejects_empty_ids_inside_nonempty_lists(tmp_path: Path, malformed: str):
    source1, candidates, matching = _valid_paths(tmp_path)
    candidates.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        f"s1-a\t{malformed}\n"
        "s1-b\t\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(SubmissionContractError, match="empty ID"):
        validate_submission_files(source1, candidates, matching)


@pytest.mark.parametrize("invalid_target_id", ["S1-100", "target-100", " S2-100"])
def test_submission_contract_rejects_non_target_source_ids(tmp_path: Path, invalid_target_id: str):
    source1, candidates, matching = _valid_paths(tmp_path)
    candidates.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        f"s1-a\t{invalid_target_id}\n"
        "s1-b\t\n",
        encoding="utf-8",
        newline="\n",
    )
    matching.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        f"s1-a\t{invalid_target_id}\n"
        "s1-b\t\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(SubmissionContractError, match="S2- or S3-"):
        validate_submission_files(source1, candidates, matching)
