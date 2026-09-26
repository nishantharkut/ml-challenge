"""Stage 51: Shard-Streamed Country Inference and Contract Validation.

Executes test inference for a single country partition with bounded memory:
1. Shards S1 queries by global file ordinal.
2. Builds S2 TargetSearchIndex, computes and persists shard feature Parquets, then frees S2.
3. Builds S3 TargetSearchIndex, computes and persists shard feature Parquets, then frees S3.
4. Performs shard-local candidate balanced merge and model scoring using compact metadata.
5. Resolves query exclusivity deterministically via on-disk SQLite.
6. Writes partitioned candidate and match TSVs and validates row-level contracts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import polars as pl
import psutil

# Ensure Implementation root is in sys.path
IMPL_ROOT = Path(__file__).resolve().parent
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from src.blocking import TargetSearchIndex
from src.checkpoint import save_manifest
from src.config import (
    CHECKPOINT_DIR,
    MAX_CANDIDATES_PER_SOURCE,
    OUTPUT_DIR,
    RUN_CONTEXT,
    TEST_DIR,
    TOTAL_CANDIDATE_BUDGET,
)
from src.features import extract_pair_features
from src.matcher import PairwiseMatcher
from src.normalization import preprocess_record
from src.streaming_contracts import (
    balanced_merge_compact_candidates,
    compact_candidate_metadata,
    validate_shard_rows,
)
from src.streaming_runner import (
    SqliteWinnerResolver,
    is_shard_manifest_valid,
    parse_source1_records_by_country,
    validate_feature_table_columns,
    write_shard_manifest,
)


def _get_rss_gib() -> float:
    """Return current process peak/current RSS in GiB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _compact_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Retain only the fields needed by blocking, feature extraction, and metadata."""
    return {
        "entity_id": rec["entity_id"],
        "name_norm": rec.get("name_norm", ""),
        "name_folded": rec.get("name_folded", rec.get("name_norm", "")),
        "name_core": rec.get("name_core", ""),
        "name_translit": rec.get("name_translit", ""),
        "name_suffix": rec.get("name_suffix", ""),
        "name_tokens": rec.get("name_tokens", ()),
        "script": rec.get("script", "Latin"),
        "addr_norm": rec.get("addr_norm", ""),
        "addr_folded": rec.get("addr_folded", rec.get("addr_norm", "")),
        "addr_empty": rec.get("addr_empty", True),
        "addr_numbers": rec.get("addr_numbers", set()),
        "addr_postal": rec.get("addr_postal"),
    }


def load_target_records_by_country(
    filepath: str | Path,
    country_filter: str,
) -> list[dict[str, Any]]:
    """Stream and preprocess target TSV records for a single country."""
    records: list[dict[str, Any]] = []
    target_country = (country_filter or "").strip().upper()
    path = Path(filepath)

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            country = parts[3].strip().upper()
            if country == target_country:
                raw_rec = preprocess_record({
                    "entity_id": parts[0],
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": country,
                })
                records.append(_compact_record(raw_rec))

    return records


def run_source_feature_pass(
    source_label: str,
    target_tsv_path: Path,
    country: str,
    s1_shards: list[list[dict[str, Any]]],
    intermediate_dir: Path,
    k_per_source: int,
    expected_fingerprints: dict[str, Any],
    force: bool = False,
) -> None:
    """Build TargetSearchIndex for one source, retrieve each S1 shard, and write feature Parquet."""
    print(f"\n--- [Phase {source_label}] Processing target source {source_label} for {country} ---", flush=True)
    t0 = time.time()

    # Check if all shards already have valid manifests
    all_valid = True
    for shard_idx in range(len(s1_shards)):
        m_path = intermediate_dir / f"shard_{shard_idx:04d}_{source_label}_manifest.json"
        p_path = intermediate_dir / f"shard_{shard_idx:04d}_{source_label}.parquet"
        if not p_path.is_file() or not is_shard_manifest_valid(m_path, expected_fingerprints):
            all_valid = False
            break

    if all_valid and not force:
        print(f"  All {len(s1_shards)} shards for {source_label} already exist and match manifest. Skipping indexing.", flush=True)
        return

    print(f"  Loading {source_label} records for {country} from {target_tsv_path.name}...", flush=True)
    target_records = load_target_records_by_country(target_tsv_path, country)
    n_targets = len(target_records)
    print(f"  Loaded {n_targets:,} {source_label} records; RSS={_get_rss_gib():.2f} GiB", flush=True)

    print(f"  Building {source_label} TargetSearchIndex...", flush=True)
    t_idx = time.time()
    target_index = TargetSearchIndex(target_records, source_label=source_label, k_per_source=k_per_source)
    target_lookup = {r["entity_id"]: r for r in target_records}
    print(f"  Built {source_label} index in {time.time() - t_idx:.1f}s; RSS={_get_rss_gib():.2f} GiB", flush=True)

    # Process shards
    for shard_idx, shard in enumerate(s1_shards):
        m_path = intermediate_dir / f"shard_{shard_idx:04d}_{source_label}_manifest.json"
        p_path = intermediate_dir / f"shard_{shard_idx:04d}_{source_label}.parquet"

        if not force and p_path.is_file() and is_shard_manifest_valid(m_path, expected_fingerprints):
            print(f"    Shard {shard_idx + 1}/{len(s1_shards)}: reused existing artifact", flush=True)
            continue

        t_shard = time.time()
        candidates = target_index.retrieve_for_s1_chunk(shard, k=k_per_source)

        rows = []
        for s1_rec in shard:
            s1_id = s1_rec["entity_id"]
            ordinal = s1_rec["ordinal"]
            cand_dict = candidates.get(s1_id, {})

            for target_id, meta in cand_dict.items():
                if target_id not in target_lookup:
                    continue
                cand_rec = target_lookup[target_id]
                feats = extract_pair_features(s1_rec, cand_rec, meta)
                compact = compact_candidate_metadata(meta)

                row: dict[str, Any] = {
                    "ordinal": ordinal,
                    "s1_id": s1_id,
                    "target_id": target_id,
                    "source": source_label,
                }
                row.update(feats)
                row.update(compact)
                rows.append(row)

        if rows:
            df = pl.DataFrame(rows)
            # Downcast floats to float32 to minimize disk space and I/O
            float_cols = [c for c, dtype in zip(df.columns, df.dtypes) if dtype == pl.Float64]
            if float_cols:
                df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
            temp_parquet = p_path.with_suffix(f".tmp.{os.getpid()}.parquet")
            df.write_parquet(temp_parquet, compression="zstd")
            temp_parquet.replace(p_path)
            row_count = len(df)
        else:
            # Empty candidate shard (zero retrieval candidates)
            empty_df = pl.DataFrame({
                "ordinal": pl.Series(dtype=pl.Int64),
                "s1_id": pl.Series(dtype=pl.Utf8),
                "target_id": pl.Series(dtype=pl.Utf8),
                "source": pl.Series(dtype=pl.Utf8),
            })
            p_path.parent.mkdir(parents=True, exist_ok=True)
            empty_df.write_parquet(p_path)
            row_count = 0

        manifest_data = dict(expected_fingerprints, **{
            "shard_idx": shard_idx,
            "source_label": source_label,
            "query_count": len(shard),
            "pair_count": row_count,
            "elapsed_seconds": time.time() - t_shard,
            "status": "completed",
        })
        write_shard_manifest(m_path, manifest_data)
        print(f"    Shard {shard_idx + 1}/{len(s1_shards)}: {len(shard):,} queries -> {row_count:,} pairs in {time.time() - t_shard:.1f}s", flush=True)

    # Release target index and reclaim memory
    target_index.close()
    del target_index, target_lookup, target_records
    gc.collect()
    print(f"  Finished {source_label} pass in {time.time() - t0:.1f}s. Freed gallery; RSS={_get_rss_gib():.2f} GiB", flush=True)


def stream_country_inference(
    country: str,
    shard_size: int = 25000,
    k_per_source: int = MAX_CANDIDATES_PER_SOURCE,
    total_k: int = TOTAL_CANDIDATE_BUDGET,
    smoke_limit: int | None = None,
    force: bool = False,
) -> bool:
    """Run memory-bounded, shard-streamed country inference conforming to AGENTS.md."""
    country = country.strip().upper()
    print("=" * 70, flush=True)
    print(f"STREAMED COUNTRY INFERENCE: {country}", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    # 1. Load Matcher Model
    model_path = Path(CHECKPOINT_DIR) / "lgbm_matcher.joblib"
    matcher = PairwiseMatcher(model_path=model_path)
    if not matcher.load():
        print(f"Error: Model not found at {model_path}. Run Stage 30 first.", flush=True)
        return False
    feature_names = matcher.feature_names
    print(f"Loaded matcher model with {len(feature_names)} features.", flush=True)

    # 2. Load Decoder Policy
    decoder_path = Path(CHECKPOINT_DIR) / "optimal_decoder.json"
    prob_threshold = 0.65
    use_exclusivity = True
    if decoder_path.is_file():
        with decoder_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        prob_threshold = cfg.get("prob_threshold", prob_threshold)
        use_exclusivity = cfg.get("use_query_exclusivity", use_exclusivity)
    print(f"Decoder policy: threshold={prob_threshold:.3f}, exclusivity={use_exclusivity}", flush=True)

    # 3. Setup paths and directories
    s1_path = Path(TEST_DIR) / "test_source1.tsv"
    s2_path = Path(TEST_DIR) / "test_source2.tsv"
    s3_path = Path(TEST_DIR) / "test_source3.tsv"

    intermediate_dir = RUN_CONTEXT.run_root / "intermediates" / country
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    part_cands_path = Path(OUTPUT_DIR) / f"part_cands_{country}.tsv"
    part_match_path = Path(OUTPUT_DIR) / f"part_match_{country}.tsv"

    expected_fingerprints = {
        "country": country,
        "feature_names": feature_names,
        "k_per_source": k_per_source,
        "total_k": total_k,
    }

    # 4. Parse and shard Source-1
    print(f"Loading Source-1 records for {country}...", flush=True)
    s1_raw_records = parse_source1_records_by_country(s1_path, country)
    if smoke_limit and smoke_limit > 0:
        print(f"  [SMOKE PROBE] Restricting S1 to first {smoke_limit} entities", flush=True)
        s1_raw_records = s1_raw_records[:smoke_limit]

    n_s1 = len(s1_raw_records)
    print(f"Total S1 entities for {country}: {n_s1:,}; RSS={_get_rss_gib():.2f} GiB", flush=True)

    s1_preprocessed = []
    for r in s1_raw_records:
        pre = preprocess_record(r)
        compact = _compact_record(pre)
        compact["ordinal"] = r["ordinal"]
        s1_preprocessed.append(compact)
    del s1_raw_records
    gc.collect()

    s1_shards = [
        s1_preprocessed[i : i + shard_size]
        for i in range(0, len(s1_preprocessed), shard_size)
    ]
    print(f"Partitioned into {len(s1_shards)} shards (shard size {shard_size:,}).", flush=True)

    # 5. One-source feature passes
    run_source_feature_pass(
        source_label="S2",
        target_tsv_path=s2_path,
        country=country,
        s1_shards=s1_shards,
        intermediate_dir=intermediate_dir,
        k_per_source=k_per_source,
        expected_fingerprints=expected_fingerprints,
        force=force,
    )

    run_source_feature_pass(
        source_label="S3",
        target_tsv_path=s3_path,
        country=country,
        s1_shards=s1_shards,
        intermediate_dir=intermediate_dir,
        k_per_source=k_per_source,
        expected_fingerprints=expected_fingerprints,
        force=force,
    )

    # 6. Shard-Local Merge, Model Scoring, and Global Exclusivity Resolution
    print(f"\n--- [Phase Merge & Score] Merging candidates and scoring with LightGBM ---", flush=True)
    db_path = intermediate_dir / "winners.sqlite3"
    if db_path.exists():
        db_path.unlink()
    resolver = SqliteWinnerResolver(db_path) if use_exclusivity else None
    nonexclusive_matches: dict[str, list[str]] = defaultdict(list)

    total_candidates_written = 0
    total_pairs_scored = 0

    # Open candidate TSV part
    temp_cand_path = part_cands_path.with_suffix(".tmp")
    with temp_cand_path.open("w", encoding="utf-8", newline="\n") as f_cand:
        for shard_idx, shard in enumerate(s1_shards):
            t_s = time.time()
            s2_parquet = intermediate_dir / f"shard_{shard_idx:04d}_S2.parquet"
            s3_parquet = intermediate_dir / f"shard_{shard_idx:04d}_S3.parquet"

            df2 = pl.read_parquet(s2_parquet) if s2_parquet.exists() else pl.DataFrame()
            df3 = pl.read_parquet(s3_parquet) if s3_parquet.exists() else pl.DataFrame()

            # Group candidates by S1 ID
            cands_s2: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            cands_s3: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

            if len(df2) > 0:
                validate_feature_table_columns(df2, feature_names)
                for row in df2.iter_rows(named=True):
                    cands_s2[row["s1_id"]][row["target_id"]] = row

            if len(df3) > 0:
                validate_feature_table_columns(df3, feature_names)
                for row in df3.iter_rows(named=True):
                    cands_s3[row["s1_id"]][row["target_id"]] = row

            # Merge and select top-k candidates for each S1 query
            selected_pairs_set: set[tuple[str, str]] = set()
            for s1_rec in shard:
                sid = s1_rec["entity_id"]
                sub2 = cands_s2.get(sid, {})
                sub3 = cands_s3.get(sid, {})
                merged = balanced_merge_compact_candidates(sub2, sub3, total_k=total_k)

                sorted_cids = sorted(merged.keys())
                f_cand.write(f"{sid}\t{','.join(sorted_cids)}\n")
                total_candidates_written += len(sorted_cids)

                for cid in sorted_cids:
                    selected_pairs_set.add((sid, cid))

            # Combine DataFrames and filter to only selected candidates
            dfs = [df for df in (df2, df3) if len(df) > 0]
            if dfs:
                df_all = pl.concat(dfs)
                # Filter to selected pairs
                if len(selected_pairs_set) > 0:
                    filter_mask = [
                        (s, t) in selected_pairs_set
                        for s, t in zip(df_all["s1_id"], df_all["target_id"])
                    ]
                    df_selected = df_all.filter(filter_mask)
                else:
                    df_selected = pl.DataFrame()

                del df_all, dfs
            else:
                df_selected = pl.DataFrame()

            del df2, df3
            gc.collect()

            # Score selected candidate pairs
            if len(df_selected) > 0:
                X = df_selected.select(feature_names).to_numpy()
                probs = matcher.predict_proba(X)
                total_pairs_scored += len(probs)

                # Passing threshold filter
                pass_mask = probs >= prob_threshold
                if np.any(pass_mask):
                    passing_ordinals = df_selected["ordinal"].filter(pass_mask).to_list()
                    passing_s1_ids = df_selected["s1_id"].filter(pass_mask).to_list()
                    passing_targets = df_selected["target_id"].filter(pass_mask).to_list()
                    passing_probs = probs[pass_mask]

                    if use_exclusivity and resolver is not None:
                        resolver.add_pairs(
                            zip(passing_ordinals, passing_s1_ids, passing_targets, passing_probs)
                        )
                    else:
                        for sid, tid in zip(passing_s1_ids, passing_targets):
                            nonexclusive_matches[sid].append(tid)

                del X, probs, df_selected
                gc.collect()

            print(f"    Scored shard {shard_idx + 1}/{len(s1_shards)} in {time.time() - t_s:.1f}s; RSS={_get_rss_gib():.2f} GiB", flush=True)

    temp_cand_path.replace(part_cands_path)
    print(f"  Successfully wrote {part_cands_path} ({total_candidates_written:,} candidates across {n_s1:,} queries)", flush=True)

    # 7. Write Partition Matches
    print(f"\n--- [Phase Output & Validation] Emitting matches and verifying row contracts ---", flush=True)
    if use_exclusivity and resolver is not None:
        matches_by_s1 = resolver.get_matches_by_s1()
        resolver.close()
    else:
        matches_by_s1 = {k: sorted(v) for k, v in nonexclusive_matches.items()}

    temp_match_path = part_match_path.with_suffix(".tmp")
    verification_rows: list[dict[str, Any]] = []

    # Read back candidates to verify row contracts
    cands_map: dict[str, list[str]] = {}
    with part_cands_path.open("r", encoding="utf-8") as f_c:
        for line in f_c:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            c_ids = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            cands_map[sid] = c_ids

    with temp_match_path.open("w", encoding="utf-8", newline="\n") as f_m:
        for idx, s1_rec in enumerate(s1_preprocessed):
            sid = s1_rec["entity_id"]
            m_ids = sorted(matches_by_s1.get(sid, []))
            f_m.write(f"{sid}\t{','.join(m_ids)}\n")

            verification_rows.append({
                "ordinal": idx,
                "s1_id": sid,
                "candidate_ids": cands_map.get(sid, []),
                "prediction_ids": m_ids,
            })

    temp_match_path.replace(part_match_path)

    # 8. Strict Contract Validation
    print("  Validating partition contract (coverage, ordering, candidate subset)...", flush=True)
    expected_rows = [(idx, r["entity_id"]) for idx, r in enumerate(s1_preprocessed)]
    validate_shard_rows(verification_rows, expected_rows=expected_rows)
    print("  [OK] Partition contract check PASSED!", flush=True)



    # Summary metrics
    n_with_match = sum(1 for r in verification_rows if len(r["prediction_ids"]) > 0)
    total_matches = sum(len(r["prediction_ids"]) for r in verification_rows)
    elapsed = time.time() - t_start
    peak_rss = _get_rss_gib()

    print("\n" + "=" * 70, flush=True)
    print(f"PARTITION {country} COMPLETED IN {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"  S1 Queries: {n_s1:,}", flush=True)
    print(f"  Total Candidates: {total_candidates_written:,} (avg {total_candidates_written/max(1,n_s1):.2f}/S1)", flush=True)
    print(f"  Total Pairs Scored: {total_pairs_scored:,}", flush=True)
    print(f"  S1 with >=1 Match: {n_with_match:,} ({n_with_match/max(1,n_s1)*100:.1f}%)", flush=True)
    print(f"  Total Predictions: {total_matches:,}", flush=True)
    print(f"  Peak RSS: {peak_rss:.2f} GiB", flush=True)
    print("=" * 70, flush=True)

    save_manifest(f"stream_{country}", {
        "status": "completed",
        "country": country,
        "n_s1": n_s1,
        "candidates_count": total_candidates_written,
        "pairs_scored": total_pairs_scored,
        "matches_count": total_matches,
        "s1_matched_count": n_with_match,
        "prob_threshold": prob_threshold,
        "use_exclusivity": use_exclusivity,
        "elapsed_seconds": elapsed,
        "peak_rss_gib": peak_rss,
    })
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamed Country Inference")
    parser.add_argument("--country", type=str, required=True, choices=["FRANCE", "INDIA", "US"])
    parser.add_argument("--shard-size", type=int, default=25000)
    parser.add_argument("--k-per-source", type=int, default=MAX_CANDIDATES_PER_SOURCE)
    parser.add_argument("--total-k", type=int, default=TOTAL_CANDIDATE_BUDGET)
    parser.add_argument("--smoke", type=int, default=None, help="Probe first N queries only")
    parser.add_argument("--force", action="store_true", help="Recompute existing shards")
    args = parser.parse_args()

    success = stream_country_inference(
        country=args.country,
        shard_size=args.shard_size,
        k_per_source=args.k_per_source,
        total_k=args.total_k,
        smoke_limit=args.smoke,
        force=args.force,
    )
    sys.exit(0 if success else 1)
