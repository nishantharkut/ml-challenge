"""
Stage 50 split runner — runs a SINGLE country partition.
Usage:
  python 50_run_country.py --country FRANCE
  python 50_run_country.py --country US
  python 50_run_country.py --country INDIA

Designed for parallel execution across machines.
"""
import os
import sys
import time
import json
import gc
import argparse
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    TEST_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    MAX_CANDIDATES_PER_SOURCE, TOTAL_CANDIDATE_BUDGET
)
from src.normalization import preprocess_record
from src.blocking import TargetSearchIndex, SourceSeparatedBlocker
from src.features import extract_pair_features
from src.matcher import PairwiseMatcher


def compact_record(rec):
    """Retain fields required by features and blocking."""
    return {
        'entity_id': rec['entity_id'],
        'name_norm': rec.get('name_norm', ''),
        'name_folded': rec.get('name_folded', rec.get('name_norm', '')),
        'name_core': rec.get('name_core', ''),
        'name_translit': rec.get('name_translit', ''),
        'name_suffix': rec.get('name_suffix', ''),
        'name_tokens': rec.get('name_tokens', []),
        'script': rec.get('script', 'Latin'),
        'addr_norm': rec.get('addr_norm', ''),
        'addr_folded': rec.get('addr_folded', rec.get('addr_norm', '')),
        'addr_empty': rec.get('addr_empty', True),
        'addr_numbers': rec.get('addr_numbers', set()),
        'addr_postal': rec.get('addr_postal'),
    }


def load_test_records_by_country(filename, country_filter):
    path = os.path.join(TEST_DIR, filename)
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            country = parts[3].strip().upper() if len(parts) > 3 else ''
            if country == country_filter:
                raw_rec = preprocess_record({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': country
                })
                records.append(compact_record(raw_rec))
    return records


def run_country(country, prob_thresh, use_excl, k_per_source, total_k):
    """Run inference for a single country."""
    print(f"{'='*70}")
    print(f"INFERENCE: {country}")
    print(f"  threshold={prob_thresh}, exclusivity={use_excl}")
    print(f"  k_per_source={k_per_source}, total_k={total_k}")
    print(f"{'='*70}")
    t0 = time.time()

    # Load model
    matcher = PairwiseMatcher(model_path=os.path.join(CHECKPOINT_DIR, "lgbm_matcher.joblib"))
    if not matcher.load():
        print("ERROR: Model not found!")
        return False
    print(f"Model loaded ({len(matcher.feature_names)} features)")

    # Load S1 for this country
    print(f"Loading S1 records for {country}...", flush=True)
    s1_records = []
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            c = parts[3].strip().upper() if len(parts) > 3 else ''
            if c == country:
                raw = preprocess_record({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': c
                })
                s1_records.append(compact_record(raw))
    print(f"  {len(s1_records):,} S1 records")

    # ── Step 1: Retrieve from S2 ──────────────────────────────────────────
    print(f"Loading S2 for {country}...", flush=True)
    s2_records = load_test_records_by_country("test_source2.tsv", country)
    print(f"  {len(s2_records):,} S2 records", flush=True)
    print("Building S2 index...", flush=True)
    s2_index = TargetSearchIndex(s2_records, "S2", k_per_source)

    chunk_size = 25000
    n_s1 = len(s1_records)
    n_chunks = (n_s1 + chunk_size - 1) // chunk_size

    cands_s2 = {}
    print(f"Retrieving S2 candidates across {n_chunks} chunks...", flush=True)
    for c_idx in range(0, n_s1, chunk_size):
        t_c = time.time()
        s1_chunk = s1_records[c_idx:c_idx + chunk_size]
        res = s2_index.retrieve_for_s1_chunk(s1_chunk, k_per_source)
        cands_s2.update(res)
        print(f"  [S2] Chunk {c_idx//chunk_size + 1}/{n_chunks} in {time.time()-t_c:.1f}s", flush=True)

    needed_s2 = set()
    for cdict in cands_s2.values():
        needed_s2.update(cdict.keys())
    s2_lookup = {r['entity_id']: r for r in s2_records if r['entity_id'] in needed_s2}
    s2_index.close()
    del s2_index, s2_records
    gc.collect()
    print(f"Retained {len(s2_lookup):,} active S2 candidates; freed S2 gallery", flush=True)

    # ── Step 2: Retrieve from S3 ──────────────────────────────────────────
    print(f"Loading S3 for {country}...", flush=True)
    s3_records = load_test_records_by_country("test_source3.tsv", country)
    print(f"  {len(s3_records):,} S3 records", flush=True)
    print("Building S3 index...", flush=True)
    s3_index = TargetSearchIndex(s3_records, "S3", k_per_source)

    cands_s3 = {}
    print(f"Retrieving S3 candidates across {n_chunks} chunks...", flush=True)
    for c_idx in range(0, n_s1, chunk_size):
        t_c = time.time()
        s1_chunk = s1_records[c_idx:c_idx + chunk_size]
        res = s3_index.retrieve_for_s1_chunk(s1_chunk, k_per_source)
        cands_s3.update(res)
        print(f"  [S3] Chunk {c_idx//chunk_size + 1}/{n_chunks} in {time.time()-t_c:.1f}s", flush=True)

    needed_s3 = set()
    for cdict in cands_s3.values():
        needed_s3.update(cdict.keys())
    s3_lookup = {r['entity_id']: r for r in s3_records if r['entity_id'] in needed_s3}
    s3_index.close()
    del s3_index, s3_records
    gc.collect()
    print(f"Retained {len(s3_lookup):,} active S3 candidates; freed S3 gallery", flush=True)

    target_lookup = {**s2_lookup, **s3_lookup}
    del s2_lookup, s3_lookup
    gc.collect()

    # ── Step 3: Balanced Merge & Pair Scoring ─────────────────────────────
    blocker = SourceSeparatedBlocker(k_per_source=k_per_source, total_k=total_k)
    country_candidates_str = {}
    candidate_hits = defaultdict(list)
    total_pairs = 0

    print("Scoring candidate pairs in chunks...", flush=True)
    for c_idx in range(0, n_s1, chunk_size):
        t_chunk = time.time()
        s1_chunk = s1_records[c_idx:c_idx + chunk_size]

        X_chunk = []
        pairs_chunk = []
        for s1_rec in s1_chunk:
            sid = s1_rec['entity_id']
            merged_dict = blocker._balanced_merge(
                cands_s2.get(sid, {}), cands_s3.get(sid, {})
            )
            country_candidates_str[sid] = ",".join(sorted(merged_dict.keys()))

            for cid, meta in merged_dict.items():
                if cid in target_lookup:
                    feats = extract_pair_features(s1_rec, target_lookup[cid], meta)
                    X_chunk.append(list(feats.values()))
                    pairs_chunk.append((sid, cid))

        if X_chunk:
            X_arr = np.array(X_chunk, dtype=np.float32)
            probs = matcher.predict_proba(X_arr)
            total_pairs += len(probs)

            for (sid, cid), prob in zip(pairs_chunk, probs):
                if prob >= prob_thresh:
                    candidate_hits[cid].append((sid, float(prob)))

            del X_arr, probs

        del X_chunk, pairs_chunk
        gc.collect()

        chunk_num = c_idx // chunk_size + 1
        pct = (c_idx + len(s1_chunk)) / n_s1 * 100
        print(f"  Chunk {chunk_num}/{n_chunks} ({pct:.1f}%): scored in {time.time() - t_chunk:.1f}s", flush=True)

    del cands_s2, cands_s3, target_lookup
    gc.collect()

    # Resolve exclusivity
    print(f"Resolving exclusivity ({len(candidate_hits):,} hits)...", flush=True)
    country_matches = defaultdict(list)
    if use_excl:
        for cid, hits in candidate_hits.items():
            hits_sorted = sorted(hits, key=lambda x: (-x[1], x[0]))
            country_matches[hits_sorted[0][0]].append(cid)
    else:
        for cid, hits in candidate_hits.items():
            for sid, _ in hits:
                country_matches[sid].append(cid)
    del candidate_hits
    gc.collect()

    # Write partition files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    part_cand = os.path.join(OUTPUT_DIR, f"candidates_{country}.tsv")
    part_match = os.path.join(OUTPUT_DIR, f"matching_{country}.tsv")

    with open(part_cand, 'w', encoding='utf-8', newline='\n') as f:
        for r in s1_records:
            sid = r['entity_id']
            f.write(f"{sid}\t{country_candidates_str.get(sid, '')}\n")

    with open(part_match, 'w', encoding='utf-8', newline='\n') as f:
        for r in s1_records:
            sid = r['entity_id']
            m_list = country_matches.get(sid, [])
            f.write(f"{sid}\t{','.join(sorted(m_list))}\n")

    n_matched = sum(1 for r in s1_records if country_matches.get(r['entity_id']))
    print(f"\n{country} complete in {time.time() - t0:.1f}s")
    print(f"  S1 entities: {len(s1_records):,}")
    print(f"  Matched: {n_matched:,} ({n_matched/len(s1_records)*100:.1f}%)")
    print(f"  Total pairs scored: {total_pairs:,}")
    print(f"  Output: {part_cand}")
    print(f"  Output: {part_match}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, choices=["FRANCE", "US", "INDIA"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--exclusivity", type=int, default=None)
    parser.add_argument("--total-k", type=int, default=None)
    parser.add_argument("--k-per-source", type=int, default=None)
    args = parser.parse_args()

    # Load decoder settings if available
    decoder_path = os.path.join(CHECKPOINT_DIR, "optimal_decoder.json")
    if os.path.exists(decoder_path) and (args.threshold is None or args.exclusivity is None):
        with open(decoder_path) as f:
            settings = json.load(f)
        prob_thresh = args.threshold or settings.get("prob_threshold", 0.45)
        use_excl = bool(args.exclusivity) if args.exclusivity is not None else settings.get("use_query_exclusivity", True)
    else:
        prob_thresh = args.threshold or 0.45
        use_excl = bool(args.exclusivity) if args.exclusivity is not None else True

    total_k = args.total_k or TOTAL_CANDIDATE_BUDGET
    k_per_source = args.k_per_source or MAX_CANDIDATES_PER_SOURCE

    run_country(args.country, prob_thresh, use_excl, k_per_source, total_k)
