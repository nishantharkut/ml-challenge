"""
Stage 40: Decision Decoder Optimization.
Evaluates trained LightGBM model on validation candidate pairs using FULL country
galleries, performs fine-grained threshold grid search, tests query exclusivity
independently, and freezes optimal decoder settings.

KEY DESIGN DECISIONS:
- Uses FULL S2/S3 galleries per country (not toy 80k distractors)
  to produce realistic candidate distributions matching production.
- Fine-grained threshold sweep (0.005 steps) for precision.
- Tests exclusivity ON/OFF independently with separate optimal thresholds.
"""
import os
import sys
import time
import json
import gc
import numpy as np
from collections import defaultdict

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    TRAIN_DIR, CHECKPOINT_DIR,
    MAX_CANDIDATES_PER_SOURCE, TOTAL_CANDIDATE_BUDGET,
)
from src.normalization import preprocess_record
from src.features import extract_pair_features
from src.matcher import PairwiseMatcher
from src.decoder import tune_decoder_policies
from src.metrics import competition_macro_f05, oracle_macro_f05
from src.checkpoint import save_manifest, is_checkpoint_valid


def _load_tsv_by_country(filepath, country_filter):
    """Load and preprocess records from TSV for one country."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
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


def tune_decoder(val_eval_limit=25000):
    print("=" * 70)
    print("STAGE 40: DECISION DECODER OPTIMIZATION")
    print("  (Full-gallery country-partitioned evaluation)")
    print("=" * 70)
    t0 = time.time()

    # ── 1. Load Validation Split ──────────────────────────────────────────
    with open(os.path.join(CHECKPOINT_DIR, "val_s1_ids.json"), 'r', encoding='utf-8') as f:
        val_s1_ids = json.load(f)[:val_eval_limit]
    val_s1_set = set(val_s1_ids)

    with open(os.path.join(CHECKPOINT_DIR, "val_gt_split.json"), 'r', encoding='utf-8') as f:
        val_gt_full = json.load(f)
    val_gt = {k: set(val_gt_full.get(k, [])) for k in val_s1_ids}

    print(f"Loaded {len(val_s1_ids):,} validation S1 entities")

    # ── 2. Load Trained Matcher ───────────────────────────────────────────
    matcher = PairwiseMatcher(model_path=os.path.join(CHECKPOINT_DIR, "lgbm_matcher.joblib"))
    if not matcher.load():
        print("  Error: Trained matcher not found! Run Stage 30 first.")
        return False
    print(f"Loaded matcher with {len(matcher.feature_names)} features")

    # ── 3. Load Validation S1 Records, grouped by country ─────────────────
    print("Loading validation S1 records...")
    s1_by_country = defaultdict(list)
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in val_s1_set:
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

    # ── 4. Country-partitioned candidate retrieval + scoring ──────────────
    from src.blocking import TargetSearchIndex, SourceSeparatedBlocker

    blocker = SourceSeparatedBlocker(
        k_per_source=MAX_CANDIDATES_PER_SOURCE,
        total_k=TOTAL_CANDIDATE_BUDGET,
    )

    scored_cands = {}  # {s1_id: [(cid, prob), ...]}
    total_pairs = 0

    for country in countries:
        t_c = time.time()
        s1_country = s1_by_country[country]
        print(f"\n  Country {country}: {len(s1_country):,} validation S1", flush=True)

        # Step 1: Load S2 only
        print(f"    Loading full S2 gallery for {country}...", flush=True)
        s2_records = _load_tsv_by_country(
            os.path.join(TRAIN_DIR, "train_source2.tsv"), country
        )
        print(f"    Loaded {len(s2_records):,} S2 records", flush=True)
        s2_index = TargetSearchIndex(s2_records, "S2", k_per_source=MAX_CANDIDATES_PER_SOURCE)

        candidates2 = {}
        n_chunks = (len(s1_country) + 9999) // 10000
        for chunk_idx, start in enumerate(range(0, len(s1_country), 10_000), 1):
            chunk = s1_country[start : start + 10_000]
            t_chunk = time.time()
            res = s2_index.retrieve_for_s1_chunk(chunk, MAX_CANDIDATES_PER_SOURCE)
            candidates2.update(res)
            print(f"      [S2] Chunk {chunk_idx}/{n_chunks} in {time.time()-t_chunk:.1f}s", flush=True)

        needed_s2 = set()
        for cands in candidates2.values():
            needed_s2.update(cands.keys())
        s2_lookup = {r['entity_id']: r for r in s2_records if r['entity_id'] in needed_s2}
        s2_index.close()
        del s2_index, s2_records
        gc.collect()

        # Step 2: Load S3 only
        print(f"    Loading full S3 gallery for {country}...", flush=True)
        s3_records = _load_tsv_by_country(
            os.path.join(TRAIN_DIR, "train_source3.tsv"), country
        )
        print(f"    Loaded {len(s3_records):,} S3 records", flush=True)
        s3_index = TargetSearchIndex(s3_records, "S3", k_per_source=MAX_CANDIDATES_PER_SOURCE)

        candidates3 = {}
        for chunk_idx, start in enumerate(range(0, len(s1_country), 10_000), 1):
            chunk = s1_country[start : start + 10_000]
            t_chunk = time.time()
            res = s3_index.retrieve_for_s1_chunk(chunk, MAX_CANDIDATES_PER_SOURCE)
            candidates3.update(res)
            print(f"      [S3] Chunk {chunk_idx}/{n_chunks} in {time.time()-t_chunk:.1f}s", flush=True)

        needed_s3 = set()
        for cands in candidates3.values():
            needed_s3.update(cands.keys())
        s3_lookup = {r['entity_id']: r for r in s3_records if r['entity_id'] in needed_s3}
        s3_index.close()
        del s3_index, s3_records
        gc.collect()

        cand_lookup = {**s2_lookup, **s3_lookup}
        del s2_lookup, s3_lookup
        gc.collect()

        # Merge candidates
        struct_cands = {}
        for s1_rec in s1_country:
            sid = s1_rec['entity_id']
            struct_cands[sid] = blocker._balanced_merge(
                candidates2.get(sid, {}), candidates3.get(sid, {})
            )
        del candidates2, candidates3
        gc.collect()

        # Score all candidate pairs
        for s1_rec in s1_country:
            sid = s1_rec['entity_id']
            cand_dict = struct_cands.get(sid, {})
            X_chunk = []
            pairs_chunk = []
            for cid, meta in cand_dict.items():
                if cid in cand_lookup:
                    feats = extract_pair_features(s1_rec, cand_lookup[cid], meta)
                    X_chunk.append(list(feats.values()))
                    pairs_chunk.append(cid)

            if X_chunk:
                X_arr = np.array(X_chunk, dtype=np.float32)
                probs = matcher.predict_proba(X_arr)
                scored_cands[sid] = [(cid, float(p)) for cid, p in zip(pairs_chunk, probs)]
                total_pairs += len(probs)

        print(f"    {country}: scored in {time.time() - t_c:.1f}s", flush=True)

        del cand_lookup, struct_cands
        gc.collect()

    print(f"\nTotal scored pairs: {total_pairs:,}")

    # ── 5. Fine-grained threshold sweep with independent exclusivity ──────
    print("\nRunning fine-grained decoder policy sweep...")
    thresholds = [round(0.10 + step * 0.005, 3) for step in range(171)]
    results = tune_decoder_policies(scored_cands, val_gt, thresholds=thresholds)

    excl_on = results[True]
    excl_off = results[False]

    print(f"\n  Exclusivity ON:  threshold={excl_on['threshold']:.3f}, "
          f"F0.5={excl_on['score']:.6f}")
    print(f"  Exclusivity OFF: threshold={excl_off['threshold']:.3f}, "
          f"F0.5={excl_off['score']:.6f}")

    # Pick the better policy
    if excl_on['score'] >= excl_off['score']:
        best_threshold = excl_on['threshold']
        best_score = excl_on['score']
        use_excl = True
    else:
        best_threshold = excl_off['threshold']
        best_score = excl_off['score']
        use_excl = False

    print(f"\n  --> SELECTED: threshold={best_threshold:.3f}, "
          f"exclusivity={use_excl}, F0.5={best_score:.6f}")

    # ── 6. Save frozen decoder settings ───────────────────────────────────
    optimal_settings = {
        "prob_threshold": best_threshold,
        "use_query_exclusivity": use_excl,
        "val_macro_f05": best_score,
        "val_entities_evaluated": len(val_s1_ids),
        "total_pairs_scored": total_pairs,
        "excl_on_result": excl_on,
        "excl_off_result": excl_off,
    }
    with open(os.path.join(CHECKPOINT_DIR, "optimal_decoder.json"), 'w', encoding='utf-8') as f:
        json.dump(optimal_settings, f, indent=2)

    save_manifest("stage_40", {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "optimal_settings": optimal_settings,
    })
    print(f"\nStage 40 completed in {time.time() - t0:.2f}s!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 40: Decision Decoder Optimization")
    parser.add_argument("--limit", type=int, default=25000, help="Number of validation S1 entities to evaluate")
    args = parser.parse_args()
    tune_decoder(val_eval_limit=args.limit)
