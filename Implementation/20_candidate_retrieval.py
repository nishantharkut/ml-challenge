"""
Stage 20: Candidate Retrieval and Blocking Evaluation.
Performs source-separated multi-channel blocking (exact name/addr, rare tokens, char TF-IDF).
Evaluates candidate recall and Oracle Macro F0.5 on the validation split.
"""
import os
import sys
import time
import json
import gc
from collections import defaultdict

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import TRAIN_DIR, CHECKPOINT_DIR, MAX_CANDIDATES_PER_SOURCE, TOTAL_CANDIDATE_BUDGET
from src.normalization import preprocess_record
from src.blocking import SourceSeparatedBlocker
from src.metrics import evaluate_blocking_quality
from src.checkpoint import save_manifest, is_checkpoint_valid

def load_preprocessed_tsv(filepath, country_filter=None):
    """Load and preprocess records from TSV, optionally filtering by country."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            country = parts[3].strip().upper() if len(parts) > 3 else ''
            if country_filter and country != country_filter:
                continue
            records.append(preprocess_record({
                'entity_id': parts[0],
                'business_name': parts[1] if len(parts) > 1 else '',
                'business_address': parts[2] if len(parts) > 2 else '',
                'country': country
            }))
    return records

def run_candidate_retrieval(val_only=True):
    print("=" * 70)
    print("STAGE 20: CANDIDATE RETRIEVAL AND BLOCKING EVALUATION")
    print("=" * 70)
    t0 = time.time()
    
    if is_checkpoint_valid("stage_20"):
        print("  Found valid Stage 20 checkpoint. Skipping candidate retrieval.")
        return True
        
    # Load Validation S1 IDs and Ground Truth
    with open(os.path.join(CHECKPOINT_DIR, "val_s1_ids.json"), 'r', encoding='utf-8') as f:
        val_s1_ids = set(json.load(f))
    with open(os.path.join(CHECKPOINT_DIR, "val_gt_split.json"), 'r', encoding='utf-8') as f:
        val_gt_raw = json.load(f)
        val_gt = {k: set(v) for k, v in val_gt_raw.items()}
        
    print(f"Loaded {len(val_s1_ids):,} Validation S1 IDs")
    
    # Load Validation S1 records
    s1_all_path = os.path.join(TRAIN_DIR, "train_source1.tsv")
    val_s1_records = []
    countries = set()
    with open(s1_all_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in val_s1_ids:
                rec = preprocess_record({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3] if len(parts) > 3 else ''
                })
                val_s1_records.append(rec)
                countries.add(rec['country'])
                
    print(f"Extracted {len(val_s1_records):,} validation S1 records across countries: {sorted(list(countries))}")
    
    # Run blocking per country
    blocker = SourceSeparatedBlocker(
        k_per_source=MAX_CANDIDATES_PER_SOURCE,
        total_k=TOTAL_CANDIDATE_BUDGET
    )
    
    all_val_candidates_struct = {}
    all_val_candidates_simple = {}
    
    for country in sorted(list(countries)):
        print(f"\nProcessing Country Partition: {country}")
        val_s1_country = [r for r in val_s1_records if r['country'] == country]
        print(f"  Validation S1 in {country}: {len(val_s1_country):,}")
        
        # Load S2 and S3 for this country
        print(f"  Loading S2 for {country}...")
        s2_country = load_preprocessed_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"), country_filter=country)
        print(f"    Loaded {len(s2_country):,} S2 records")
        
        print(f"  Loading S3 for {country}...")
        s3_country = load_preprocessed_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"), country_filter=country)
        print(f"    Loaded {len(s3_country):,} S3 records")
        
        print(f"  Running source-separated blocking for {country}...")
        country_struct, country_simple = blocker.generate_candidates(val_s1_country, s2_country, s3_country)
        
        all_val_candidates_struct.update(country_struct)
        all_val_candidates_simple.update(country_simple)
        
        # Immediate evaluation for this country
        country_gt = {r['entity_id']: val_gt.get(r['entity_id'], set()) for r in val_s1_country}
        c_report = evaluate_blocking_quality(country_simple, country_gt)
        print(f"  --> {country} Blocking Report:")
        print(f"      Pair Recall:            {c_report['pair_recall']:.4f}")
        print(f"      Complete Entity Recall: {c_report['complete_entity_recall']:.4f}")
        print(f"      Oracle Macro F0.5:      {c_report['oracle_macro_f05']:.4f}")
        print(f"      Mean candidates / S1:   {c_report['mean_candidates_per_s1']:.1f}")
        
        del s2_country, s3_country, country_struct, country_simple
        gc.collect()
        
    # Overall Validation Blocking Report
    overall_report = evaluate_blocking_quality(all_val_candidates_simple, val_gt)
    print("\n" + "=" * 60)
    print("OVERALL VALIDATION CANDIDATE BLOCKING REPORT")
    print("=" * 60)
    print(f"  Total Validation S1:        {overall_report['total_s1']:,}")
    print(f"  Pair Recall:                {overall_report['pair_recall']:.4f}")
    print(f"  S2 Pair Recall:             {overall_report['s2_pair_recall']:.4f}")
    print(f"  S3 Pair Recall:             {overall_report['s3_pair_recall']:.4f}")
    print(f"  Complete Entity Recall:     {overall_report['complete_entity_recall']:.4f}")
    print(f"  Oracle Macro F0.5 Ceiling:  {overall_report['oracle_macro_f05']:.4f}")
    print(f"  Candidate Count per S1:     mean={overall_report['mean_candidates_per_s1']:.1f}, "
          f"median={overall_report['median_candidates']}, p95={overall_report['p95_candidates']}, "
          f"p99={overall_report['p99_candidates']}, max={overall_report['max_candidates']}")
    print(f"  Total Candidate Pairs:      {overall_report['total_candidate_pairs']:,}")
    print("=" * 60)
    
    # Save validation candidates
    val_cand_path = os.path.join(CHECKPOINT_DIR, "val_candidates.json")
    # Store simplified candidates for fast evaluation
    with open(val_cand_path, 'w', encoding='utf-8') as f:
        json.dump({k: list(v) for k, v in all_val_candidates_simple.items()}, f)
        
    save_manifest("stage_20", {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "overall_report": overall_report
    })
    print(f"Stage 20 completed in {time.time() - t0:.2f}s!")
    return True

if __name__ == "__main__":
    run_candidate_retrieval()
