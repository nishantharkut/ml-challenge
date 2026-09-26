"""
Amazon ML Challenge 2026 - E0-E2: Multi-Channel Blocking
Experiment: exact + rare-token + char-TF-IDF blocking
Produces candidate sets and measures recall/oracle-F0.5

Following v5.1 §5 (retrieval channels), §6 (evaluation), v5.2 patches
"""
import sys
import os
import csv
import time
import gc
import re
import unicodedata
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"C:\N Drive\Amazon ML Challenge\pipeline")
from phase0_infra import (
    load_tsv, load_ground_truth, normalize_text, extract_suffix,
    detect_script, preprocess_record, candidate_recall_report,
    print_recall_report, competition_macro_f05, oracle_macro_f05,
    TRAIN_DIR, TEST_DIR, OUTPUT_DIR, LEGAL_SUFFIXES
)

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz

# ============================================================
# E0: EXACT + RARE TOKEN BLOCKING
# ============================================================

def build_exact_name_index(records):
    """Channel C1: exact normalized name hash → bucket of entity_ids."""
    idx = defaultdict(list)
    for r in records:
        key = r['name_norm']
        if key:
            idx[key].append(r['entity_id'])
    return dict(idx)


def build_exact_addr_index(records):
    """Channel C2: exact normalized address hash → bucket of entity_ids."""
    idx = defaultdict(list)
    for r in records:
        key = r['addr_norm']
        if key:
            idx[key].append(r['entity_id'])
    return dict(idx)


def build_rare_token_index(records, min_idf_rank=0.8):
    """
    Channel C3: rare/distinctive token index.
    For each token, count document frequency. Index tokens in top 20% IDF.
    """
    # Count document frequency
    df = defaultdict(int)
    for r in records:
        seen = set()
        for t in r['name_tokens']:
            if t not in seen:
                df[t] += 1
                seen.add(t)
    
    n = len(records)
    # Compute IDF, find rare tokens (high IDF)
    idf = {t: np.log(n / (c + 1)) for t, c in df.items()}
    idf_threshold = sorted(idf.values(), reverse=True)[int(len(idf) * (1 - min_idf_rank))] if idf else 0
    
    # Build inverted index for rare tokens only
    idx = defaultdict(list)
    for r in records:
        for t in set(r['name_tokens']):
            if idf.get(t, 0) >= idf_threshold:
                idx[t].append(r['entity_id'])
    
    return dict(idx), idf


def exact_blocking(s1_records, s2s3_records, max_bucket=200):
    """
    E0: Exact name + exact address + rare token blocking.
    Returns candidates per S1.
    v5.2 Patch 2: large blocks get secondary discrimination, not arbitrary truncation.
    """
    candidates = defaultdict(set)
    
    # Build S2/S3 indices
    print("    Building exact name index...")
    s2s3_name_idx = build_exact_name_index(s2s3_records)
    print(f"    Name buckets: {len(s2s3_name_idx)}")
    
    print("    Building exact address index...")
    s2s3_addr_idx = build_exact_addr_index(s2s3_records)
    print(f"    Address buckets: {len(s2s3_addr_idx)}")
    
    print("    Building rare token index...")
    s2s3_rare_idx, idf = build_rare_token_index(s2s3_records)
    print(f"    Rare token entries: {len(s2s3_rare_idx)}")
    
    # Query each S1
    for r in s1_records:
        s1_id = r['entity_id']
        
        # C1: exact name
        if r['name_norm'] in s2s3_name_idx:
            bucket = s2s3_name_idx[r['name_norm']]
            if len(bucket) <= max_bucket:
                candidates[s1_id].update(bucket)
            # v5.2 Patch 2: large buckets skipped for now (handled by TF-IDF)
        
        # C2: exact address
        if r['addr_norm'] and r['addr_norm'] in s2s3_addr_idx:
            bucket = s2s3_addr_idx[r['addr_norm']]
            if len(bucket) <= max_bucket:
                candidates[s1_id].update(bucket)
        
        # C3: rare token matching (any shared rare token)
        for t in set(r.get('name_tokens', [])):
            if t in s2s3_rare_idx:
                bucket = s2s3_rare_idx[t]
                if len(bucket) <= max_bucket:
                    candidates[s1_id].update(bucket)
    
    return dict(candidates)


# ============================================================
# E2: CHARACTER TF-IDF BLOCKING
# ============================================================

def tfidf_blocking(s1_records, s2s3_records, top_k=20, field='name_norm',
                   ngram_range=(3, 4), max_features=80000, threshold=0.15):
    """
    E2: Char n-gram TF-IDF blocking using sparse_dot_topn.
    Returns candidates per S1.
    """
    print(f"    Building TF-IDF on {field} ({len(s2s3_records)} docs)...")
    
    # Prepare texts
    s1_texts = [r.get(field, '') or '' for r in s1_records]
    s2s3_texts = [r.get(field, '') or '' for r in s2s3_records]
    s1_ids = [r['entity_id'] for r in s1_records]
    s2s3_ids = [r['entity_id'] for r in s2s3_records]
    
    # Fit TF-IDF on combined corpus for shared vocabulary
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=ngram_range,
        max_features=max_features, dtype=np.float32,
        sublinear_tf=True, min_df=2
    )
    
    # Fit on S2/S3, transform both
    s2s3_tfidf = vectorizer.fit_transform(s2s3_texts)
    print(f"    S2S3 TF-IDF shape: {s2s3_tfidf.shape}")
    
    del s2s3_texts
    gc.collect()
    
    s1_tfidf = vectorizer.transform(s1_texts)
    print(f"    S1 TF-IDF shape: {s1_tfidf.shape}")
    
    del s1_texts, vectorizer
    gc.collect()
    
    # Compute top-K similarities using sparse_dot_topn
    print(f"    Computing sparse top-{top_k} similarities (threshold={threshold})...")
    t0 = time.time()
    
    # Process in chunks to avoid OOM
    chunk_size = 50000
    candidates = defaultdict(set)
    
    for start in range(0, s1_tfidf.shape[0], chunk_size):
        end = min(start + chunk_size, s1_tfidf.shape[0])
        chunk = s1_tfidf[start:end]
        
        result = sp_matmul_topn(
            chunk, s2s3_tfidf.T,
            top_n=top_k, threshold=threshold,
            sort=True, n_threads=4
        )
        
        # Extract results
        coo = result.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            s1_id = s1_ids[start + i]
            s2s3_id = s2s3_ids[j]
            candidates[s1_id].add(s2s3_id)
        
        if (start // chunk_size) % 5 == 0:
            print(f"      Chunk {start//chunk_size}: {end}/{s1_tfidf.shape[0]} "
                  f"({time.time()-t0:.0f}s)")
        
        del chunk, result, coo
        gc.collect()
    
    del s1_tfidf, s2s3_tfidf
    gc.collect()
    
    print(f"    TF-IDF blocking done in {time.time()-t0:.1f}s")
    return dict(candidates)


# ============================================================
# MERGE CANDIDATES
# ============================================================

def merge_candidates(*candidate_dicts):
    """Union all candidate sets."""
    merged = defaultdict(set)
    for cands in candidate_dicts:
        for s1_id, s2s3_ids in cands.items():
            merged[s1_id].update(s2s3_ids)
    return dict(merged)


# ============================================================
# MAIN
# ============================================================

def main():
    T0 = time.time()
    print("=" * 60)
    print("E0-E2: Multi-Channel Blocking Pipeline")
    print("=" * 60)
    
    # Load GT
    print("\n[1] Loading ground truth...")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    gt = load_ground_truth(gt_path)
    
    # Load S1
    print("[2] Loading and preprocessing S1...")
    s1_raw = load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    s1_records = [preprocess_record(r) for r in s1_raw]
    del s1_raw
    gc.collect()
    print(f"    S1: {len(s1_records)} records")
    
    # Add singletons to GT
    for r in s1_records:
        if r['entity_id'] not in gt:
            gt[r['entity_id']] = set()
    
    # Split by country for memory efficiency
    countries = sorted(set(r['country'] for r in s1_records))
    print(f"    Countries: {countries}")
    
    all_candidates = defaultdict(set)
    
    for country in countries:
        print(f"\n{'='*60}")
        print(f"PROCESSING COUNTRY: {country}")
        print(f"{'='*60}")
        
        # Filter S1 for this country
        s1_country = [r for r in s1_records if r['country'] == country]
        print(f"  S1: {len(s1_country)} records")
        
        # Load and preprocess S2 for this country
        print(f"  Loading S2...")
        s2_raw = load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"))
        s2_country = [preprocess_record(r) for r in s2_raw if r['country'].strip().upper() == country]
        del s2_raw
        gc.collect()
        print(f"  S2: {len(s2_country)} records")
        
        # Load and preprocess S3 for this country
        print(f"  Loading S3...")
        s3_raw = load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"))
        s3_country = [preprocess_record(r) for r in s3_raw if r['country'].strip().upper() == country]
        del s3_raw
        gc.collect()
        print(f"  S3: {len(s3_country)} records")
        
        s2s3_country = s2_country + s3_country
        print(f"  S2+S3 combined: {len(s2s3_country)} records")
        
        # --- E0: Exact + rare token blocking ---
        print(f"\n  [E0] Exact + rare token blocking...")
        e0_cands = exact_blocking(s1_country, s2s3_country)
        e0_report = candidate_recall_report(
            {k: v for k, v in e0_cands.items()}, 
            {k: v for k, v in gt.items() if any(r['entity_id'] == k for r in s1_country)}
        )
        gt_country = {r['entity_id']: gt.get(r['entity_id'], set()) for r in s1_country}
        e0_report = candidate_recall_report(e0_cands, gt_country)
        print_recall_report(e0_report, f"E0 ({country})")
        
        # --- E2: Char TF-IDF blocking (name) ---
        print(f"\n  [E2a] Char TF-IDF blocking (name)...")
        tfidf_name_cands = tfidf_blocking(
            s1_country, s2s3_country, 
            top_k=15, field='name_norm',
            ngram_range=(3, 4), max_features=80000, threshold=0.1
        )
        
        # --- E2b: Char TF-IDF blocking (address) ---
        print(f"\n  [E2b] Char TF-IDF blocking (address)...")
        tfidf_addr_cands = tfidf_blocking(
            s1_country, s2s3_country,
            top_k=10, field='addr_norm',
            ngram_range=(3, 4), max_features=60000, threshold=0.1
        )
        
        # Merge all channels
        country_cands = merge_candidates(e0_cands, tfidf_name_cands, tfidf_addr_cands)
        
        # Full report
        merged_report = candidate_recall_report(country_cands, gt_country)
        print_recall_report(merged_report, f"E0+E2 Merged ({country})")
        print(f"  OracleGap = {merged_report['oracle_f05']:.6f} - FinalF05 (TBD)")
        
        # Accumulate
        for s1_id, cids in country_cands.items():
            all_candidates[s1_id].update(cids)
        
        del s2_country, s3_country, s2s3_country
        del e0_cands, tfidf_name_cands, tfidf_addr_cands, country_cands
        gc.collect()
    
    # Overall report
    print(f"\n{'='*60}")
    print("OVERALL BLOCKING RESULTS")
    print(f"{'='*60}")
    overall_report = candidate_recall_report(dict(all_candidates), gt)
    print_recall_report(overall_report, "ALL COUNTRIES MERGED")
    
    # Save candidates for next phase
    cand_path = os.path.join(OUTPUT_DIR, "train_candidates.tsv")
    print(f"\nSaving candidates to {cand_path}...")
    with open(cand_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source1_entity_id', 'candidate_entity_id'])
        for s1_id, cids in sorted(all_candidates.items()):
            for cid in sorted(cids):
                writer.writerow([s1_id, cid])
    
    total_pairs = sum(len(v) for v in all_candidates.values())
    print(f"  Total candidate pairs saved: {total_pairs:,}")
    
    T1 = time.time()
    print(f"\nTotal blocking time: {T1-T0:.1f}s ({(T1-T0)/60:.1f} min)")


if __name__ == '__main__':
    main()
