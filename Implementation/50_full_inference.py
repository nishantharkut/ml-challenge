"""
Stage 50: Full Test Inference and Output Generation.
Supports modular partitioned execution per country (France, US, India) and final merger.
Guarantees zero memory thrashing, progress logging with flush=True, checkpointed partition TSVs,
and exact preservation of test_source1.tsv row order.
"""
import os
import sys
import time
import json
import gc
import argparse
from collections import defaultdict
import numpy as np

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    TEST_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    MAX_CANDIDATES_PER_SOURCE, TOTAL_CANDIDATE_BUDGET
)
from src.normalization import preprocess_record
from src.blocking import TargetSearchIndex, SourceSeparatedBlocker
from src.features import extract_pair_features
from src.matcher import PairwiseMatcher
from src.checkpoint import save_manifest

def compact_record(rec):
    """Retain only the fields required by extract_pair_features and blocking.

    IMPORTANT: name_folded and addr_folded MUST be preserved for accent-aware
    features (critical for French names). name_tokens must NOT be truncated
    because rare-token blocking uses all tokens.
    """
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
    """Load and preprocess records from test TSV for a specific country."""
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

def run_country_partition(country):
    print("=" * 70, flush=True)
    print(f"PROCESSING TEST PARTITION: {country}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    
    # 1. Load Trained Matcher
    matcher = PairwiseMatcher(model_path=os.path.join(CHECKPOINT_DIR, "lgbm_matcher.joblib"))
    if not matcher.load():
        print("  Error: Trained matcher not found! Please run Stage 30 first.", flush=True)
        return False
        
    # 2. Load Decoder Settings
    decoder_settings_path = os.path.join(CHECKPOINT_DIR, "optimal_decoder.json")
    if os.path.exists(decoder_settings_path):
        with open(decoder_settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        prob_thresh = settings.get("prob_threshold", 0.65)
        use_excl = settings.get("use_query_exclusivity", True)
    else:
        prob_thresh = 0.65
        use_excl = True
    print(f"Decoder policy: prob_threshold={prob_thresh:.2f}, exclusivity={use_excl}", flush=True)
    
    # 3. Load S1 for this country
    print(f"  [1/4] Loading S1 records for {country}...", flush=True)
    s1_records = load_test_records_by_country("test_source1.tsv", country)
    n_s1 = len(s1_records)
    print(f"        Loaded {n_s1:,} S1 records for {country}", flush=True)
    
    # 4. Load & Index S2
    print(f"  [2/4] Loading and indexing S2 for {country}...", flush=True)
    t_idx = time.time()
    s2_records = load_test_records_by_country("test_source2.tsv", country)
    print(f"        Loaded {len(s2_records):,} S2 records", flush=True)
    s2_index = TargetSearchIndex(s2_records, 'S2', k_per_source=MAX_CANDIDATES_PER_SOURCE)
    
    # 5. Load & Index S3
    print(f"  [3/4] Loading and indexing S3 for {country}...", flush=True)
    s3_records = load_test_records_by_country("test_source3.tsv", country)
    print(f"        Loaded {len(s3_records):,} S3 records", flush=True)
    s3_index = TargetSearchIndex(s3_records, 'S3', k_per_source=MAX_CANDIDATES_PER_SOURCE)
    print(f"        Indices built in {time.time() - t_idx:.2f}s", flush=True)
    
    # Build unified lookup map
    target_lookup = {r['entity_id']: r for r in (s2_records + s3_records)}
    del s2_records, s3_records
    gc.collect()
    
    blocker = SourceSeparatedBlocker(
        k_per_source=MAX_CANDIDATES_PER_SOURCE,
        total_k=TOTAL_CANDIDATE_BUDGET
    )
    
    # Output file paths for this partition
    part_cand_file = os.path.join(OUTPUT_DIR, f"part_cands_{country}.tsv")
    part_match_file = os.path.join(OUTPUT_DIR, f"part_match_{country}.tsv")
    
    # 6. Stream candidate retrieval and scoring in chunks
    print(f"  [4/4] Streaming inference in chunks of 25,000...", flush=True)
    chunk_size = 25000
    candidate_hits = defaultdict(list)  # {cid: [(s1_id, prob), ...]}
    country_candidates_str = {}         # {s1_id: "S2-xxx,S3-yyy"}
    total_pairs = 0
    
    n_chunks = (n_s1 + chunk_size - 1) // chunk_size
    for c_idx in range(0, n_s1, chunk_size):
        t_chunk = time.time()
        s1_chunk = s1_records[c_idx:c_idx + chunk_size]
        struct_chunk, simple_chunk = blocker.generate_candidates_for_chunk(s1_chunk, s2_index, s3_index)
        
        # Save candidate strings
        for sid, c_set in simple_chunk.items():
            country_candidates_str[sid] = ",".join(sorted(c_set))
            
        # Flatten candidate pairs
        X_chunk = []
        pairs_chunk = []
        for s1_rec in s1_chunk:
            sid = s1_rec['entity_id']
            cand_dict = struct_chunk.get(sid, {})
            for cid, meta in cand_dict.items():
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
            
        del struct_chunk, simple_chunk, X_chunk, pairs_chunk
        gc.collect()
        
        chunk_num = c_idx // chunk_size + 1
        pct = (c_idx + len(s1_chunk)) / n_s1 * 100
        print(f"        Chunk {chunk_num}/{n_chunks} ({pct:.1f}%): {len(s1_chunk):,} queries scored in {time.time() - t_chunk:.2f}s", flush=True)
        
    print(f"  Resolving query exclusivity across {len(candidate_hits):,} candidate hits...", flush=True)
    country_matches = defaultdict(list)
    if use_excl:
        for cid, hits in candidate_hits.items():
            # Deterministic: highest prob wins; ties broken by smallest s1_id
            hits_sorted = sorted(hits, key=lambda x: (-x[1], x[0]))
            country_matches[hits_sorted[0][0]].append(cid)
    else:
        for cid, hits in candidate_hits.items():
            for sid, _ in hits:
                country_matches[sid].append(cid)
                
    # Write Partition Candidate TSV
    print(f"  Writing {part_cand_file}...", flush=True)
    with open(part_cand_file, 'w', encoding='utf-8', newline='\n') as f_out:
        for r in s1_records:
            sid = r['entity_id']
            c_str = country_candidates_str.get(sid, '')
            f_out.write(f"{sid}\t{c_str}\n")
            
    # Write Partition Matching TSV
    print(f"  Writing {part_match_file}...", flush=True)
    with open(part_match_file, 'w', encoding='utf-8', newline='\n') as f_out:
        for r in s1_records:
            sid = r['entity_id']
            m_list = country_matches.get(sid, [])
            m_str = ",".join(sorted(m_list))
            f_out.write(f"{sid}\t{m_str}\n")
            
    n_matched = sum(1 for m in country_matches.values() if m)
    total_target_matched = sum(len(m) for m in country_matches.values())
    print(f"\nPartition {country} finished in {time.time() - t0:.2f}s!", flush=True)
    print(f"  S1 entities with >=1 match: {n_matched:,} / {n_s1:,} ({n_matched/n_s1*100:.1f}%)", flush=True)
    print(f"  Total target records matched: {total_target_matched:,}", flush=True)
    print(f"  Total pairs scored: {total_pairs:,}", flush=True)
    return True

def merge_final_submissions():
    print("=" * 70, flush=True)
    print("MERGING FINAL TSVs PRESERVING TEST ROW ORDER", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    
    # Check all partition files exist
    countries = ['FRANCE', 'INDIA', 'US']
    for c in countries:
        cand_p = os.path.join(OUTPUT_DIR, f"part_cands_{c}.tsv")
        match_p = os.path.join(OUTPUT_DIR, f"part_match_{c}.tsv")
        if not os.path.exists(cand_p) or not os.path.exists(match_p):
            print(f"Error: Missing partition files for {c} ({cand_p} or {match_p})!", flush=True)
            return False
            
    print("Loading partition candidate mappings...", flush=True)
    cands_lookup = {}
    for c in countries:
        cand_p = os.path.join(OUTPUT_DIR, f"part_cands_{c}.tsv")
        with open(cand_p, 'r', encoding='utf-8') as f:
            for line in f:
                sid, _, val = line.partition('\t')
                cands_lookup[sid.strip()] = val.strip()
    print(f"  Loaded {len(cands_lookup):,} candidate rows", flush=True)
    
    print("Loading partition matching mappings...", flush=True)
    match_lookup = {}
    for c in countries:
        match_p = os.path.join(OUTPUT_DIR, f"part_match_{c}.tsv")
        with open(match_p, 'r', encoding='utf-8') as f:
            for line in f:
                sid, _, val = line.partition('\t')
                match_lookup[sid.strip()] = val.strip()
    print(f"  Loaded {len(match_lookup):,} match rows", flush=True)
    
    final_cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    final_match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    s1_test_path = os.path.join(TEST_DIR, "test_source1.tsv")
    
    print(f"Streaming final TSVs in exact test_source1.tsv order...", flush=True)
    n_written = 0
    with open(s1_test_path, 'r', encoding='utf-8') as f_in, \
         open(final_cand_path, 'w', encoding='utf-8', newline='\n') as f_cand, \
         open(final_match_path, 'w', encoding='utf-8', newline='\n') as f_match:
        
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        
        next(f_in)  # skip header
        for line in f_in:
            sid = line.split('\t', 1)[0].strip()
            c_val = cands_lookup.get(sid, '')
            m_val = match_lookup.get(sid, '')
            f_cand.write(f"{sid}\t{c_val}\n")
            f_match.write(f"{sid}\t{m_val}\n")
            n_written += 1
            
    print(f"  Successfully wrote {n_written:,} rows to both output TSVs in {time.time() - t0:.2f}s!", flush=True)
    
    # Line count verification
    with open(final_cand_path, 'r', encoding='utf-8') as f:
        c_lines = sum(1 for _ in f)
    with open(final_match_path, 'r', encoding='utf-8') as f:
        m_lines = sum(1 for _ in f)
        
    expected = n_written + 1
    print(f"  candidate_pairs.tsv:  {c_lines:,} lines (Expected: {expected:,})", flush=True)
    print(f"  matching_results.tsv: {m_lines:,} lines (Expected: {expected:,})", flush=True)
    assert c_lines == expected and m_lines == expected, "Line count mismatch!"
    
    save_manifest("stage_50", {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "n_test_s1": n_written,
        "cand_lines": c_lines,
        "match_lines": m_lines
    })
    return True

def run_test_inference():
    """Run all partitions and merge final submissions."""
    for c in ['FRANCE', 'US', 'INDIA']:
        run_country_partition(c)
    return merge_final_submissions()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, choices=['FRANCE', 'US', 'INDIA', 'ALL'], default='ALL')
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    
    if args.merge:
        merge_final_submissions()
    elif args.country == 'ALL':
        run_test_inference()
    else:
        run_country_partition(args.country)
