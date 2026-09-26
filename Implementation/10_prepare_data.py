"""
Stage 10: Data Preparation, Multi-View Normalization, and Entity-Disjoint Split.
Preprocesses raw TSVs, mines transliteration dictionary on training labels, and writes checkpoints.
"""
import os
import sys
import time
import json
import random
import polars as pl
from collections import defaultdict

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import TRAIN_DIR, TEST_DIR, CHECKPOINT_DIR, RANDOM_SEED
from src.normalization import preprocess_record
from src.transliteration import TransliterationEngine
from src.checkpoint import save_manifest, is_checkpoint_valid

def prepare_data(val_sample_size=50000):
    print("=" * 70)
    print("STAGE 10: DATA PREPARATION AND ENTITY-DISJOINT SPLIT")
    print("=" * 70)
    t0 = time.time()
    
    stage_manifest = os.path.join(CHECKPOINT_DIR, "stage_10_manifest.json")
    if is_checkpoint_valid("stage_10"):
        print("  Found valid Stage 10 checkpoint. Skipping preparation.")
        return True
        
    random.seed(RANDOM_SEED)
    
    # 1. Load Ground Truth
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    print(f"Loading Ground Truth from {gt_path}...")
    ground_truth = defaultdict(set)
    with open(gt_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            if len(parts) > 1 and parts[1]:
                ground_truth[s1_id] = set(parts[1].split(','))
            else:
                ground_truth[s1_id] = set()
                
    # 2. Load Train S1
    s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv")
    print(f"Loading Train Source 1 from {s1_path}...")
    s1_all = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            s1_all.append({
                'entity_id': parts[0],
                'business_name': parts[1] if len(parts) > 1 else '',
                'business_address': parts[2] if len(parts) > 2 else '',
                'country': parts[3] if len(parts) > 3 else ''
            })
            if parts[0] not in ground_truth:
                ground_truth[parts[0]] = set() # singleton
                
    print(f"  Total S1 entities: {len(s1_all):,} ({sum(1 for v in ground_truth.values() if not v):,} singletons)")
    
    # 3. Create Entity-Disjoint Validation Split
    print(f"Creating Entity-Disjoint Validation Split ({val_sample_size:,} S1 entities)...")
    all_s1_ids = [r['entity_id'] for r in s1_all]
    random.shuffle(all_s1_ids)
    val_s1_id_set = set(all_s1_ids[:val_sample_size])
    train_s1_id_set = set(all_s1_ids[val_sample_size:])
    
    # Save split ID mapping
    split_meta = {
        'total_s1': len(all_s1_ids),
        'train_s1_count': len(train_s1_id_set),
        'val_s1_count': len(val_s1_id_set),
        'random_seed': RANDOM_SEED
    }
    with open(os.path.join(CHECKPOINT_DIR, "val_s1_ids.json"), 'w', encoding='utf-8') as f:
        json.dump(list(val_s1_id_set), f)
        
    # Save split ground truth
    train_gt = {k: list(v) for k, v in ground_truth.items() if k in train_s1_id_set}
    val_gt = {k: list(v) for k, v in ground_truth.items() if k in val_s1_id_set}
    
    with open(os.path.join(CHECKPOINT_DIR, "train_gt_split.json"), 'w', encoding='utf-8') as f:
        json.dump(train_gt, f)
    with open(os.path.join(CHECKPOINT_DIR, "val_gt_split.json"), 'w', encoding='utf-8') as f:
        json.dump(val_gt, f)
        
    print(f"  Train S1: {len(train_gt):,}, Val S1: {len(val_gt):,}")
    
    # Save manifest
    save_manifest("stage_10", {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "split_meta": split_meta
    })
    print(f"Stage 10 completed in {time.time() - t0:.2f}s!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    prepare_data()
