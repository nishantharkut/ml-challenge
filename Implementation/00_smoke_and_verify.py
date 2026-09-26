"""
Phase 0: Smoke Test and Contract Verification.
Validates the entire pipeline end-to-end on a fast 1,000-entity sample.
"""
import sys
import os
import time

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import TRAIN_DIR, OUTPUT_DIR, VALIDATOR_SCRIPT, RANDOM_SEED
from src.normalization import preprocess_record
from src.transliteration import TransliterationEngine
from src.blocking import SourceSeparatedBlocker
from src.features import extract_pair_features
from src.matcher import PairwiseMatcher, smoke_model_path
from src.decoder import SetDecoder
from src.metrics import competition_macro_f05, oracle_macro_f05, evaluate_blocking_quality
from src.outputs import export_candidate_pairs, export_matching_results, run_submission_validator
import numpy as np


def build_smoke_matcher(artifact_dir):
    """Construct a matcher whose writes are isolated from production artifacts."""
    return PairwiseMatcher(model_path=smoke_model_path(artifact_dir))

def run_smoke_test(sample_size=1000):
    print("=" * 70)
    print("RUNNING PIPELINE SMOKE TEST ON 1,000-ENTITY SAMPLE")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Load Ground Truth sample
    gt_file = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    sample_gt = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        next(f) # header
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            parts = line.strip().split('\t')
            s1_id = parts[0]
            matched = set(parts[1].split(',')) if len(parts) > 1 and parts[1] else set()
            sample_gt[s1_id] = matched
            
    sample_s1_ids = set(sample_gt.keys())
    target_match_ids = set()
    for mids in sample_gt.values():
        target_match_ids.update(mids)
        
    print(f"[1] Loaded {len(sample_gt)} Ground Truth sample entities ({len(target_match_ids)} target matches)")
    
    # 2. Load matching S1 records
    s1_file = os.path.join(TRAIN_DIR, "train_source1.tsv")
    s1_records = []
    with open(s1_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in sample_s1_ids:
                s1_records.append(preprocess_record({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3] if len(parts) > 3 else ''
                }))
                
    # 3. Load target S2 and S3 records (including all target matches + distractors)
    s2_records = []
    s2_file = os.path.join(TRAIN_DIR, "train_source2.tsv")
    with open(s2_file, 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if parts[0] in target_match_ids or i < 3000:
                s2_records.append(preprocess_record({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3] if len(parts) > 3 else ''
                }))
                
    s3_records = []
    s3_file = os.path.join(TRAIN_DIR, "train_source3.tsv")
    with open(s3_file, 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if parts[0] in target_match_ids or i < 3000:
                s3_records.append(preprocess_record({
                    'entity_id': parts[0],
                    'business_name': parts[1] if len(parts) > 1 else '',
                    'business_address': parts[2] if len(parts) > 2 else '',
                    'country': parts[3] if len(parts) > 3 else ''
                }))
                
    print(f"[2] Preprocessed {len(s1_records)} S1 records, {len(s2_records)} S2 records, {len(s3_records)} S3 records")
    
    # 4. Run Blocking
    blocker = SourceSeparatedBlocker(k_per_source=10, total_k=18)
    struct_cands, simple_cands = blocker.generate_candidates(s1_records, s2_records, s3_records)
    
    report = evaluate_blocking_quality(simple_cands, sample_gt)
    print(f"\n[3] Candidate Recall Report (Smoke Sample):")
    print(f"    Pair Recall:              {report['pair_recall']:.4f}")
    print(f"    Complete Entity Recall:   {report['complete_entity_recall']:.4f}")
    print(f"    Oracle Macro F0.5:        {report['oracle_macro_f05']:.4f}")
    print(f"    Mean candidates per S1:   {report['mean_candidates_per_s1']:.1f} (max: {report['max_candidates']})")
    
    # 5. Extract Features for Candidate Pairs
    cand_lookup = {r['entity_id']: r for r in (s2_records + s3_records)}
    X = []
    y = []
    pair_index = []
    
    for s1_rec in s1_records:
        s1_id = s1_rec['entity_id']
        true_matches = sample_gt.get(s1_id, set())
        for cid, meta in struct_cands.get(s1_id, {}).items():
            if cid in cand_lookup:
                cand_rec = cand_lookup[cid]
                feats = extract_pair_features(s1_rec, cand_rec, meta)
                feat_values = list(feats.values())
                feat_names = list(feats.keys())
                
                is_match = 1.0 if cid in true_matches else 0.0
                X.append(feat_values)
                y.append(is_match)
                pair_index.append((s1_id, cid))
                
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"\n[4] Extracted features for {len(X)} candidate pairs ({int(sum(y))} positives, {int(len(y)-sum(y))} negatives)")
    
    # 6. Train Pairwise Matcher
    matcher = build_smoke_matcher(os.path.join(OUTPUT_DIR, "smoke_artifacts"))
    matcher.train(X, y, feature_names=feat_names)
    probs = matcher.predict_proba(X)
    
    scored_cands = {}
    for (s1_id, cid), prob in zip(pair_index, probs):
        if s1_id not in scored_cands:
            scored_cands[s1_id] = []
        scored_cands[s1_id].append((cid, float(prob)))
        
    # 7. Decode Predictions
    decoder = SetDecoder(prob_threshold=0.45, use_query_exclusivity=True)
    predictions = decoder.decode(scored_cands, s1_all_ids=list(sample_s1_ids))
    
    score = competition_macro_f05(predictions, sample_gt)
    print(f"\n[5] Decoder Evaluation on Smoke Sample:")
    print(f"    Macro F0.5 Score:         {score:.4f}")
    print(f"    Oracle Gap:               {(report['oracle_macro_f05'] - score):.4f}")
    
    # 8. Export and Validate Output Formatting
    smoke_cand_path = os.path.join(OUTPUT_DIR, "smoke_candidate_pairs.tsv")
    smoke_match_path = os.path.join(OUTPUT_DIR, "smoke_matching_results.tsv")
    
    all_s1_list = sorted(list(sample_s1_ids))
    export_candidate_pairs(simple_cands, all_s1_list, smoke_cand_path)
    export_matching_results(predictions, all_s1_list, smoke_match_path)
    
    print("\n[6] Validating Exported TSVs formatting...")
    # Verify exact header and line count
    with open(smoke_match_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert lines[0] == "source1_entity_id\tmatched_entity_ids\n", f"Header error: {lines[0]}"
        assert len(lines) == len(all_s1_list) + 1, f"Line count mismatch: {len(lines)} vs {len(all_s1_list) + 1}"
        
    with open(smoke_cand_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert lines[0] == "source1_entity_id\tcandidate_entity_ids\n", f"Header error: {lines[0]}"
        assert len(lines) == len(all_s1_list) + 1, f"Line count mismatch: {len(lines)} vs {len(all_s1_list) + 1}"
        
    print("    Format checks PASSED! Exactly 1 header + 1 row per S1 entity.")
    print(f"\nSmoke test successfully completed in {time.time() - t0:.2f}s!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_smoke_test()
