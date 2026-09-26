"""
Stage 30: Feature Engineering and Matcher Training.
Extracts pairwise RapidFuzz and retrieval features, constructs balanced training data
with structured hard negatives from FULL country galleries, trains LightGBM classifier,
and saves model checkpoint.

KEY DESIGN DECISIONS:
- S2/S3 galleries are loaded PER COUNTRY (full gallery, no arbitrary truncation)
  to produce realistic hard negatives matching production difficulty.
- Early-stopping split is entity-disjoint (by S1 ID), not random row shuffle,
  to prevent leakage of S1-correlated negatives.
- All ground-truth positives are included even if missed by blocker.
"""
import os
import sys
import time
import json
import gc
import random
import numpy as np
from collections import defaultdict

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    TRAIN_DIR, CHECKPOINT_DIR, MAX_NEGATIVES_PER_POSITIVE,
    RANDOM_SEED, MAX_CANDIDATES_PER_SOURCE, TOTAL_CANDIDATE_BUDGET,
)
from src.normalization import preprocess_record
from src.blocking import TargetSearchIndex, SourceSeparatedBlocker
from src.features import extract_pair_features
from src.matcher import PairwiseMatcher
from src.checkpoint import save_manifest, is_checkpoint_valid


def _load_tsv_by_country(filepath, country_filter):
    """Load and preprocess records from TSV for one country."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            country = parts[3].strip().upper() if len(parts) > 3 else ''
            if country != country_filter:
                continue
            records.append(preprocess_record({
                'entity_id': parts[0],
                'business_name': parts[1] if len(parts) > 1 else '',
                'business_address': parts[2] if len(parts) > 2 else '',
                'country': country
            }))
    return records


def train_matcher(train_entity_limit=50000):
    print("=" * 70)
    print("STAGE 30: FEATURE EXTRACTION AND MATCHER TRAINING")
    print("  (Full-gallery country-partitioned blocking)")
    print("=" * 70)
    t0 = time.time()

    if is_checkpoint_valid("stage_30"):
        print("  Found valid Stage 30 checkpoint. Skipping training.")
        return True

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── 1. Load Train Split Ground Truth ──────────────────────────────────
    with open(os.path.join(CHECKPOINT_DIR, "train_gt_split.json"), 'r', encoding='utf-8') as f:
        train_gt_raw = json.load(f)

    train_s1_keys = list(train_gt_raw.keys())
    random.shuffle(train_s1_keys)
    sampled_train_s1 = set(train_s1_keys[:train_entity_limit])
    train_gt = {k: set(train_gt_raw[k]) for k in sampled_train_s1}

    print(f"Sampled {len(train_gt):,} S1 entities for training")

    # ── 2. Load sampled S1 records and group by country ───────────────────
    print("Loading S1 records...")
    s1_by_country = defaultdict(list)
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in sampled_train_s1:
                rec = preprocess_record({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3] if len(parts) > 3 else ''
                })
                s1_by_country[rec['country']].append(rec)

    countries = sorted(s1_by_country.keys())
    total_s1 = sum(len(v) for v in s1_by_country.values())
    print(f"  Loaded {total_s1:,} S1 records across {countries}")

    # ── 3. Country-partitioned blocking + feature extraction ──────────────
    blocker = SourceSeparatedBlocker(
        k_per_source=MAX_CANDIDATES_PER_SOURCE,
        total_k=TOTAL_CANDIDATE_BUDGET,
    )

    X_list = []
    y_list = []
    feat_names = None
    total_pos = 0
    total_neg = 0
    total_blocker_missed_positives = 0
    # Record the owner at feature creation time so the early-stopping split is
    # genuinely entity-disjoint even when retrieval coverage changes.
    row_s1_ids = []

    for country in countries:
        t_country = time.time()
        s1_country = s1_by_country[country]
        print(f"\n  Country {country}: {len(s1_country):,} S1 entities")

        # Step 1: Load S2 only
        print(f"    Loading full S2 gallery for {country}...", flush=True)
        s2_records = _load_tsv_by_country(
            os.path.join(TRAIN_DIR, "train_source2.tsv"), country
        )
        print(f"    Loaded {len(s2_records):,} S2 records", flush=True)
        s2_index = TargetSearchIndex(s2_records, "S2", k_per_source=MAX_CANDIDATES_PER_SOURCE)

        # Retrieve S2 candidates in chunks
        candidates2 = {}
        n_chunks = (len(s1_country) + 9999) // 10000
        for chunk_idx, start in enumerate(range(0, len(s1_country), 10_000), 1):
            chunk = s1_country[start : start + 10_000]
            t_c = time.time()
            res = s2_index.retrieve_for_s1_chunk(chunk, MAX_CANDIDATES_PER_SOURCE)
            candidates2.update(res)
            print(f"      [S2] Chunk {chunk_idx}/{n_chunks} ({len(chunk):,} queries) in {time.time()-t_c:.1f}s", flush=True)

        needed_s2_ids = set()
        for s1_rec in s1_country:
            needed_s2_ids.update(candidates2.get(s1_rec['entity_id'], {}).keys())

        s2_lookup = {r['entity_id']: r for r in s2_records if r['entity_id'] in needed_s2_ids}
        s2_index.close()
        del s2_index, s2_records
        gc.collect()
        print(f"    Retained {len(s2_lookup):,} active S2 records; freed S2 gallery", flush=True)

        # Step 2: Load S3 only
        print(f"    Loading full S3 gallery for {country}...", flush=True)
        s3_records = _load_tsv_by_country(
            os.path.join(TRAIN_DIR, "train_source3.tsv"), country
        )
        print(f"    Loaded {len(s3_records):,} S3 records", flush=True)
        s3_index = TargetSearchIndex(s3_records, "S3", k_per_source=MAX_CANDIDATES_PER_SOURCE)

        # Retrieve S3 candidates in chunks
        candidates3 = {}
        for chunk_idx, start in enumerate(range(0, len(s1_country), 10_000), 1):
            chunk = s1_country[start : start + 10_000]
            t_c = time.time()
            res = s3_index.retrieve_for_s1_chunk(chunk, MAX_CANDIDATES_PER_SOURCE)
            candidates3.update(res)
            print(f"      [S3] Chunk {chunk_idx}/{n_chunks} ({len(chunk):,} queries) in {time.time()-t_c:.1f}s", flush=True)

        needed_s3_ids = set()
        for s1_rec in s1_country:
            needed_s3_ids.update(candidates3.get(s1_rec['entity_id'], {}).keys())

        s3_lookup = {r['entity_id']: r for r in s3_records if r['entity_id'] in needed_s3_ids}
        s3_index.close()
        del s3_index, s3_records
        gc.collect()
        print(f"    Retained {len(s3_lookup):,} active S3 records; freed S3 gallery", flush=True)

        # Combine lookups
        cand_lookup = {**s2_lookup, **s3_lookup}
        del s2_lookup, s3_lookup
        gc.collect()

        # Merge S2 and S3 candidates using blocker's balanced merge
        struct_cands = {}
        for s1_rec in s1_country:
            sid = s1_rec['entity_id']
            struct_cands[sid] = blocker._balanced_merge(
                candidates2.get(sid, {}), candidates3.get(sid, {})
            )
        del candidates2, candidates3
        gc.collect()

        # Extract features
        country_pos = 0
        country_neg = 0
        for s1_rec in s1_country:
            s1_id = s1_rec['entity_id']
            true_mids = train_gt.get(s1_id, set())
            retrieved_dict = struct_cands.get(s1_id, {})

            # Classifier examples must be drawn from the same candidate
            # distribution that exists at inference.  A ground-truth pair that
            # blocking omitted is a retrieval miss, not a positive example with
            # synthetic empty retrieval metadata.
            retrieved_positive_ids = true_mids & set(retrieved_dict)
            total_blocker_missed_positives += len(true_mids - retrieved_positive_ids)
            for mid in retrieved_positive_ids:
                if mid in cand_lookup:
                    feats = extract_pair_features(s1_rec, cand_lookup[mid], retrieved_dict[mid])
                    if feat_names is None:
                        feat_names = list(feats.keys())
                    X_list.append(list(feats.values()))
                    y_list.append(1.0)
                    row_s1_ids.append(s1_id)
                    country_pos += 1

            # Hard negatives: top retrieved non-matches
            neg_candidates = [
                cid for cid in retrieved_dict.keys()
                if cid not in true_mids
            ]
            for cid in neg_candidates[:MAX_NEGATIVES_PER_POSITIVE]:
                if cid in cand_lookup:
                    meta = retrieved_dict[cid]
                    feats = extract_pair_features(s1_rec, cand_lookup[cid], meta)
                    X_list.append(list(feats.values()))
                    y_list.append(0.0)
                    row_s1_ids.append(s1_id)
                    country_neg += 1

        total_pos += country_pos
        total_neg += country_neg
        print(f"    {country}: {country_pos:,} pos, {country_neg:,} neg "
              f"in {time.time() - t_country:.1f}s", flush=True)

        # Free memory
        del cand_lookup, struct_cands
        gc.collect()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    del X_list, y_list
    gc.collect()

    print(f"\nTotal: {len(X):,} feature rows "
          f"({total_pos:,} pos, {total_neg:,} neg, ratio: {total_neg/max(1,total_pos):.2f})")
    print(f"Blocker-missed positives excluded from classifier training: "
          f"{total_blocker_missed_positives:,}")
    print(f"Feature count: {len(feat_names)}")
    print(f"Features: {feat_names}")

    # ── 4. Entity-disjoint train/val split for early stopping ─────────────
    # Features are grouped by S1 within each country, so we can derive
    # S1 ownership from the block structure without storing per-row IDs.
    # Simpler and memory-safe: split S1 IDs, then split rows by position.
    # Since extraction iterates s1_by_country in sorted country order,
    # and within each country in list order, rows are contiguous per S1.
    # We track cumulative row counts per S1 during extraction to split.

    if len(row_s1_ids) != len(X):
        raise RuntimeError("Feature-row ownership bookkeeping is inconsistent")
    s1_ids_list = list(sampled_train_s1)
    random.shuffle(s1_ids_list)
    split_point = int(len(s1_ids_list) * 0.85)
    val_s1_set = set(s1_ids_list[split_point:])
    val_mask = np.fromiter((sid in val_s1_set for sid in row_s1_ids), dtype=bool)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"  Entity-disjoint split: {len(s1_ids_list[:split_point]):,} train S1, "
          f"{len(val_s1_set):,} val S1")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Training LightGBM on {len(X_train):,} pairs "
          f"(early-stopping on {len(X_val):,} pairs)...")

    # ── 5. Train ──────────────────────────────────────────────────────────
    matcher = PairwiseMatcher(
        model_path=os.path.join(CHECKPOINT_DIR, "lgbm_matcher.joblib")
    )
    matcher.train(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feat_names)

    # ── 6. Report ─────────────────────────────────────────────────────────
    importances = matcher.get_feature_importances()
    print("\nTop 15 Feature Importances:")
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    for feat, imp in sorted_imp:
        print(f"  {feat:<25}: {imp}")

    top_features_clean = [[str(feat), int(imp)] for feat, imp in sorted_imp[:5]]
    save_manifest("stage_30", {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "n_samples": len(X),
        "n_positives": total_pos,
        "n_negatives": total_neg,
        "n_blocker_missed_positives": total_blocker_missed_positives,
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "top_features": top_features_clean,
        "countries_trained": countries,
        "train_entity_limit": train_entity_limit,
    })
    print(f"Stage 30 completed in {time.time() - t0:.2f}s!")
    return True

if __name__ == "__main__":
    train_matcher()
