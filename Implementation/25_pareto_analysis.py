"""
Candidate Budget Pareto Analysis — ultra-fast single-pass version.
Retrieves candidates at max K once per source, then slices for K in {4, 6, 8, 10, 12, 16, 20, 25}.
Reports:
  - Oracle F0.5 at each K
  - Pair recall at each K
  - Mean/P50/P95 candidates per S1
  - ΔOracle gain from K-1 to K
"""
import os
import sys
import time
import json
import gc
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import TRAIN_DIR, CHECKPOINT_DIR
from src.normalization import preprocess_record
from src.blocking import TargetSearchIndex, SourceSeparatedBlocker
from src.metrics import evaluate_blocking_quality
import numpy as np


def load_tsv_by_country(filepath, country_filter):
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


def run_pareto_analysis(
    val_limit=5000,
    max_k_per_source=15,
    k_values=None,
    countries_override=None,
):
    print("=" * 70)
    print("FAST CANDIDATE BUDGET PARETO ANALYSIS")
    print(f"  (Using {val_limit:,} validation S1 entities)")
    print("=" * 70)
    t0 = time.time()

    # Load validation split
    with open(os.path.join(CHECKPOINT_DIR, "val_s1_ids.json"), 'r', encoding='utf-8') as f:
        val_s1_ids = json.load(f)[:val_limit]
    val_s1_set = set(val_s1_ids)

    with open(os.path.join(CHECKPOINT_DIR, "val_gt_split.json"), 'r', encoding='utf-8') as f:
        val_gt_full = json.load(f)
    val_gt = {k: set(val_gt_full.get(k, [])) for k in val_s1_ids}

    # Load S1 records grouped by country
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
    if countries_override:
        allowed = {country.strip().upper() for country in countries_override}
        countries = [country for country in countries if country in allowed]
        active_ids = {
            record["entity_id"]
            for country in countries
            for record in s1_by_country[country]
        }
        val_s1_ids = [s1_id for s1_id in val_s1_ids if s1_id in active_ids]
        val_gt = {s1_id: set(val_gt_full.get(s1_id, [])) for s1_id in val_s1_ids}
    print(f"Countries: {countries}")
    print(f"Total validation S1: {sum(len(v) for v in s1_by_country.values()):,}")

    max_k_per_source = int(max_k_per_source)
    if max_k_per_source < 1:
        raise ValueError("max_k_per_source must be >= 1")
    cands_s2_all = {}
    cands_s3_all = {}

    for country in countries:
        s1_country = s1_by_country[country]
        if not s1_country:
            continue

        print(f"\nProcessing {country} ({len(s1_country):,} S1 queries)...", flush=True)

        # Step 1: S2
        print(f"  Loading S2 for {country}...", flush=True)
        s2 = load_tsv_by_country(os.path.join(TRAIN_DIR, "train_source2.tsv"), country)
        print(f"  Building S2 index ({len(s2):,} records)...", flush=True)
        s2_index = TargetSearchIndex(s2, "S2", max_k_per_source)
        del s2
        gc.collect()

        print(f"  Retrieving S2 candidates...", flush=True)
        res2 = s2_index.retrieve_for_s1_chunk(s1_country, max_k_per_source)
        cands_s2_all.update(res2)
        s2_index.close()
        del s2_index
        gc.collect()

        # Step 2: S3
        print(f"  Loading S3 for {country}...", flush=True)
        s3 = load_tsv_by_country(os.path.join(TRAIN_DIR, "train_source3.tsv"), country)
        print(f"  Building S3 index ({len(s3):,} records)...", flush=True)
        s3_index = TargetSearchIndex(s3, "S3", max_k_per_source)
        del s3
        gc.collect()

        print(f"  Retrieving S3 candidates...", flush=True)
        res3 = s3_index.retrieve_for_s1_chunk(s1_country, max_k_per_source)
        cands_s3_all.update(res3)
        s3_index.close()
        del s3_index
        gc.collect()

    print(f"\nSingle-pass retrieval completed in {time.time()-t0:.1f}s. Evaluating Pareto frontier...", flush=True)

    K_VALUES = k_values or [4, 6, 8, 10, 12, 16, 20, 25]
    results = []

    for total_k in K_VALUES:
        blocker = SourceSeparatedBlocker(
            k_per_source=max_k_per_source, total_k=total_k
        )

        simple_cands = {}
        for sid in val_s1_ids:
            merged = blocker._balanced_merge(
                cands_s2_all.get(sid, {}),
                cands_s3_all.get(sid, {})
            )
            simple_cands[sid] = set(merged.keys())

        country_gt = {sid: val_gt.get(sid, set()) for sid in simple_cands}
        report = evaluate_blocking_quality(simple_cands, country_gt)
        cand_counts = [len(v) for v in simple_cands.values()]

        result = {
            'K': total_k,
            'retrieval_depth_per_source': max_k_per_source,
            'final_s2_quota': (total_k + 1) // 2,
            'final_s3_quota': total_k // 2,
            'pair_recall': report['pair_recall'],
            'complete_entity_recall': report['complete_entity_recall'],
            'oracle_f05': report['oracle_macro_f05'],
            'mean_cands': float(np.mean(cand_counts)),
            'p95_cands': float(np.percentile(cand_counts, 95)),
            'max_cands': int(max(cand_counts)),
        }
        results.append(result)

    with open(os.path.join(CHECKPOINT_DIR, "pareto_analysis.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("PARETO ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"{'K':>4} {'raw/src':>7} {'OracleF05':>10} {'PairRecall':>11} {'MeanCands':>10} {'P95Cands':>9} {'dOracle':>10}")
    print("-" * 70)

    for i, r in enumerate(results):
        delta = (r['oracle_f05'] - results[i-1]['oracle_f05']) if i > 0 else 0.0
        print(f"{r['K']:>4} {r['retrieval_depth_per_source']:>7} {r['oracle_f05']:>10.6f} {r['pair_recall']:>11.4f} {r['mean_cands']:>10.1f} {r['p95_cands']:>9.0f} {delta:>+10.6f}")

    max_oracle = max(r['oracle_f05'] for r in results)
    best_k = results[-1]['K']
    for r in results:
        if r['oracle_f05'] >= 0.995 * max_oracle:
            best_k = r['K']
            print(f"\n--> RECOMMENDATION: K={r['K']} (preserves {r['oracle_f05']/max_oracle*100:.2f}% of oracle ceiling)")
            break

    print(f"Total analysis elapsed: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Candidate-budget Pareto analysis")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-k-per-source", type=int, default=15)
    parser.add_argument(
        "--k-values",
        default="4,6,8,10,12,16,20,25",
        help="comma-separated final candidate budgets",
    )
    parser.add_argument(
        "--countries",
        default="",
        help="optional comma-separated country partition(s), e.g. INDIA",
    )
    args = parser.parse_args()
    budgets = [int(value) for value in args.k_values.split(",") if value.strip()]
    run_pareto_analysis(
        val_limit=args.limit,
        max_k_per_source=args.max_k_per_source,
        k_values=budgets,
        countries_override=[value for value in args.countries.split(",") if value.strip()],
    )
