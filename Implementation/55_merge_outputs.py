"""Stage 55: Merge Country Partition Outputs into Final Submissions.

Fails closed: requires all country partitions to exist and be complete.
Validates submission files against official test_source1.tsv row order,
ID sorting, uniqueness, and candidate-subset integrity via output_contract.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

# Ensure Implementation root is in sys.path
IMPL_ROOT = Path(__file__).resolve().parent
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from src.checkpoint import save_manifest
from src.config import OUTPUT_DIR, TEST_DIR
from src.output_contract import SubmissionContractError, validate_submission_files
from src.outputs import run_submission_validator


def _find_country_files(output_dir: Path, country: str) -> tuple[Path, Path]:
    """Find candidate and match partition files for a country, or raise error."""
    cand_candidates = [
        output_dir / f"part_cands_{country}.tsv",
        output_dir / f"candidates_{country}.tsv",
    ]
    match_candidates = [
        output_dir / f"part_match_{country}.tsv",
        output_dir / f"matching_{country}.tsv",
    ]

    cand_path = next((p for p in cand_candidates if p.is_file()), None)
    match_path = next((p for p in match_candidates if p.is_file()), None)

    if not cand_path or not match_path:
        missing = []
        if not cand_path:
            missing.append(f"candidates for {country}")
        if not match_path:
            missing.append(f"matches for {country}")
        raise FileNotFoundError(
            f"Cannot merge: missing partition files in {output_dir}: {', '.join(missing)}"
        )

    return cand_path, match_path


def merge_country_outputs(
    test_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    countries: tuple[str, ...] = ("FRANCE", "INDIA", "US"),
) -> bool:
    print("=" * 70, flush=True)
    print("STAGE 55: MERGING COUNTRY PARTITION OUTPUTS", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    test_path = Path(test_dir or TEST_DIR)
    out_path = Path(output_dir or OUTPUT_DIR)
    source1_file = test_path / "test_source1.tsv"

    if not source1_file.is_file():
        raise FileNotFoundError(f"Source-1 file not found at {source1_file}")

    # 1. Fail closed if any country partition is missing
    country_files: dict[str, tuple[Path, Path]] = {}
    for country in countries:
        c_upper = country.strip().upper()
        country_files[c_upper] = _find_country_files(out_path, c_upper)
        print(f"  Found verified partitions for {c_upper}:", flush=True)
        print(f"    Candidates: {country_files[c_upper][0].name}", flush=True)
        print(f"    Matches:    {country_files[c_upper][1].name}", flush=True)

    # 2. Read per-country mappings
    cand_lookup: dict[str, str] = {}
    match_lookup: dict[str, str] = {}

    for country, (cand_p, match_p) in country_files.items():
        t_c = time.time()
        c_count = 0
        with cand_p.open("r", encoding="utf-8") as f:
            for line in f:
                sid, _, val = line.partition("\t")
                sid_clean = sid.strip()
                if sid_clean and sid_clean != "source1_entity_id":
                    cand_lookup[sid_clean] = val.rstrip("\r\n")
                    c_count += 1

        m_count = 0
        with match_p.open("r", encoding="utf-8") as f:
            for line in f:
                sid, _, val = line.partition("\t")
                sid_clean = sid.strip()
                if sid_clean and sid_clean != "source1_entity_id":
                    match_lookup[sid_clean] = val.rstrip("\r\n")
                    m_count += 1

        print(f"  {country}: loaded {c_count:,} candidate rows, {m_count:,} match rows in {time.time() - t_c:.1f}s", flush=True)

    # 3. Stream final TSVs in exact test_source1.tsv order
    final_cand_path = out_path / "candidate_pairs.tsv"
    final_match_path = out_path / "matching_results.tsv"

    temp_cand_path = final_cand_path.with_suffix(".tmp")
    temp_match_path = final_match_path.with_suffix(".tmp")

    print(f"\nWriting final submission TSVs preserving exact test_source1.tsv order...", flush=True)
    n_written = 0
    missing_cands: list[str] = []
    missing_matches: list[str] = []

    with source1_file.open("r", encoding="utf-8") as f_s1, \
         temp_cand_path.open("w", encoding="utf-8", newline="\n") as f_cand, \
         temp_match_path.open("w", encoding="utf-8", newline="\n") as f_match:

        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        header = f_s1.readline()
        for line in f_s1:
            sid = line.split("\t", 1)[0].strip()
            if not sid:
                continue

            if sid not in cand_lookup:
                missing_cands.append(sid)
            if sid not in match_lookup:
                missing_matches.append(sid)

            c_val = cand_lookup.get(sid, "")
            m_val = match_lookup.get(sid, "")

            f_cand.write(f"{sid}\t{c_val}\n")
            f_match.write(f"{sid}\t{m_val}\n")
            n_written += 1

    # If any Source-1 rows are missing from partition outputs, fail closed!
    if missing_cands or missing_matches:
        temp_cand_path.unlink(missing_ok=True)
        temp_match_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Coverage check failed: {len(missing_cands)} S1 missing from candidates, "
            f"{len(missing_matches)} S1 missing from matches. Partitions are incomplete!"
        )

    temp_cand_path.replace(final_cand_path)
    temp_match_path.replace(final_match_path)
    print(f"  Successfully wrote {n_written:,} rows to candidate_pairs.tsv and matching_results.tsv", flush=True)

    # 4. Strict Contract Validation
    print("\nRunning independent streaming contract validation...", flush=True)
    metrics = validate_submission_files(source1_file, final_cand_path, final_match_path)
    print("  [OK] Submission contract check PASSED!")
    print(f"    Total S1 rows:      {metrics['rows']:,}")
    print(f"    Candidate pairs:    {metrics['candidate_pairs']:,} ({metrics['candidate_pairs']/metrics['rows']:.2f}/S1)")
    print(f"    Matched pairs:      {metrics['matched_pairs']:,} ({metrics['matched_pairs']/metrics['rows']:.2f}/S1)")

    # 5. Run official validate_submission.py tool
    print("\nRunning official submission validator...", flush=True)
    validator_passed = run_submission_validator(
        matching_path=str(final_match_path),
        candidate_path=str(final_cand_path),
        test_dir=str(test_path),
    )

    save_manifest("stage_55", {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "n_rows": metrics["rows"],
        "candidate_pairs": metrics["candidate_pairs"],
        "matched_pairs": metrics["matched_pairs"],
        "countries_merged": list(countries),
        "validator_passed": validator_passed,
    })
    print(f"\n[OK] Stage 55 complete in {time.time() - t0:.1f}s!")
    return True



if __name__ == "__main__":
    merge_country_outputs()
