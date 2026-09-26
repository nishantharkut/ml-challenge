"""Independent, streaming validation for final competition TSV files.

The official validator checks the published format.  This module additionally
enforces the provenance-critical contract that each final row corresponds to
the exact Source-1 test-file row and that every prediction is in the candidate
row supplied with the submission.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, TextIO


class SubmissionContractError(ValueError):
    """Raised when submission files cannot represent one valid final run."""


_CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids"
_MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"


def _open_tsv(path: Path | str, expected_header: str) -> TextIO:
    handle = Path(path).open("r", encoding="utf-8", newline="")
    header = handle.readline().rstrip("\r\n")
    if header != expected_header:
        handle.close()
        raise SubmissionContractError(
            f"{Path(path).name} header is invalid: expected {expected_header!r}"
        )
    return handle


def _parse_submission_row(line: str, path: Path, row_number: int) -> tuple[str, tuple[str, ...]]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 2 or not fields[0]:
        raise SubmissionContractError(
            f"{path.name} row {row_number} must contain a nonempty ID and one value field"
        )
    values = tuple(fields[1].split(",")) if fields[1] else ()
    if any(not item for item in values):
        raise SubmissionContractError(f"{path.name} row {row_number} has an empty ID")
    if any(
        item != item.strip() or not item.startswith(("S2-", "S3-"))
        for item in values
    ):
        raise SubmissionContractError(
            f"{path.name} row {row_number} IDs must be unpadded S2- or S3- target IDs"
        )
    if len(values) != len(set(values)):
        raise SubmissionContractError(f"{path.name} row {row_number} has duplicate IDs")
    if values != tuple(sorted(values)):
        raise SubmissionContractError(f"{path.name} row {row_number} IDs are not sorted")
    return fields[0], values


def _source1_ids(path: Path | str) -> Iterator[tuple[int, str]]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if not header or header[0] not in {"entity_id", "source1_entity_id"}:
            raise SubmissionContractError(
                f"{source_path.name} must have entity_id as its first column"
            )
        seen: set[str] = set()
        for row_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\r\n").split("\t")
            source_id = fields[0] if fields else ""
            if not source_id:
                raise SubmissionContractError(
                    f"{source_path.name} row {row_number} has an empty Source-1 ID"
                )
            if source_id in seen:
                raise SubmissionContractError(
                    f"{source_path.name} has duplicate Source-1 ID {source_id!r}"
                )
            seen.add(source_id)
            yield row_number - 1, source_id


def validate_submission_files(
    source1_path: Path | str,
    candidate_path: Path | str,
    matching_path: Path | str,
) -> dict[str, int]:
    """Validate final files in one pass without materializing a country map.

    Rows in both files must exactly match the Source-1 test file order.  IDs in
    a candidate or match cell are unique and lexicographically sorted.  Each
    matched ID must be present in the corresponding candidate cell.
    """
    candidate_file = Path(candidate_path)
    matching_file = Path(matching_path)
    candidate_pairs = 0
    matched_pairs = 0
    rows = 0

    with _open_tsv(candidate_file, _CANDIDATE_HEADER) as candidate_handle, _open_tsv(
        matching_file, _MATCHING_HEADER
    ) as matching_handle:
        for ordinal, expected_id in _source1_ids(source1_path):
            candidate_line = candidate_handle.readline()
            matching_line = matching_handle.readline()
            if not candidate_line:
                raise SubmissionContractError(
                    f"candidate row count is short: missing Source-1 row {ordinal} ({expected_id!r})"
                )
            if not matching_line:
                raise SubmissionContractError(
                    f"matching row count is short: missing Source-1 row {ordinal} ({expected_id!r})"
                )
            candidate_id, candidate_ids = _parse_submission_row(
                candidate_line, candidate_file, ordinal + 1
            )
            matching_id, matching_ids = _parse_submission_row(
                matching_line, matching_file, ordinal + 1
            )
            if candidate_id != expected_id or matching_id != expected_id:
                raise SubmissionContractError(
                    f"Source-1 order mismatch at ordinal {ordinal}: expected {expected_id!r}, "
                    f"got candidate={candidate_id!r}, matching={matching_id!r}"
                )
            candidate_set = set(candidate_ids)
            outside = set(matching_ids).difference(candidate_set)
            if outside:
                raise SubmissionContractError(
                    f"matching row {ordinal} has IDs not present in its candidate row: {sorted(outside)!r}"
                )
            rows += 1
            candidate_pairs += len(candidate_ids)
            matched_pairs += len(matching_ids)

        if candidate_handle.readline():
            raise SubmissionContractError("candidate row count is long: extra rows after Source-1 EOF")
        if matching_handle.readline():
            raise SubmissionContractError("matching row count is long: extra rows after Source-1 EOF")

    return {"rows": rows, "candidate_pairs": candidate_pairs, "matched_pairs": matched_pairs}
