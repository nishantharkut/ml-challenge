"""
Leakage-Free Transliteration Dictionary Miner.
Mines Indic -> Latin token mappings strictly from train_gt_split.json (excluding all validation S1s).
"""
import sys, os, csv, json, time
from collections import Counter, defaultdict

INDIC_RANGES = {
    'Devanagari': (0x0900, 0x097F),
    'Bengali': (0x0980, 0x09FF),
    'Gurmukhi': (0x0A00, 0x0A7F),
    'Gujarati': (0x0A80, 0x0AFF),
    'Odia': (0x0B00, 0x0B7F),
    'Tamil': (0x0B80, 0x0BFF),
    'Telugu': (0x0C00, 0x0C7F),
    'Kannada': (0x0C80, 0x0CFF),
    'Malayalam': (0x0D00, 0x0D7F),
}

def has_indic(text):
    for ch in text:
        cp = ord(ch)
        for lo, hi in INDIC_RANGES.values():
            if lo <= cp <= hi:
                return True
    return False

def get_script(text):
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for script, (lo, hi) in INDIC_RANGES.items():
            if lo <= cp <= hi:
                counts[script] += 1
                break
        else:
            if 0x0041 <= cp <= 0x007A or 0x0061 <= cp <= 0x007A:
                counts['Latin'] += 1
    if not counts:
        return 'Unknown'
    return counts.most_common(1)[0][0]

def mine_dictionary(dataset_dir, checkpoint_dir):
    print("=" * 70)
    print("MINING LEAKAGE-FREE INDIC TRANSLITERATION DICTIONARY")
    print("=" * 70)
    t0 = time.time()
    
    train_gt_path = os.path.join(checkpoint_dir, "train_gt_split.json")
    print(f"Loading train ground truth from {train_gt_path}...")
    with open(train_gt_path, 'r', encoding='utf-8') as f:
        train_gt_raw = json.load(f)
    print(f"  Loaded {len(train_gt_raw):,} training S1 entities (validation strictly excluded)")
    
    # Load India S1 records
    s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    print(f"Loading India S1 records from {s1_path}...")
    s1_records = {}
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            sid = parts[0]
            if sid in train_gt_raw:
                country = parts[3].strip().upper() if len(parts) > 3 else ''
                if country == 'INDIA':
                    s1_records[sid] = parts[1] if len(parts) > 1 else ''
    print(f"  Retained {len(s1_records):,} training India S1 records")
    
    # Build reverse lookup for matches
    gt_target_to_s1 = {}
    for sid in s1_records:
        for mid in train_gt_raw.get(sid, []):
            gt_target_to_s1[mid] = sid
            
    print(f"  Total target GT match links to scan: {len(gt_target_to_s1):,}")
    
    # Scan S2 and S3 for cross-script pairs
    cross_script_pairs = []
    for sname in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(dataset_dir, "train", sname)
        print(f"Scanning {sname} for Indic records...")
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                eid = parts[0]
                if eid in gt_target_to_s1:
                    bname = parts[1] if len(parts) > 1 else ''
                    if has_indic(bname):
                        sid = gt_target_to_s1[eid]
                        s1_name = s1_records[sid]
                        script = get_script(bname)
                        cross_script_pairs.append((s1_name, bname, script))
                        count += 1
        print(f"  Found {count:,} cross-script matches in {sname}")
        
    print(f"\nTotal cross-script pairs: {len(cross_script_pairs):,}")
    
    # Token alignment
    token_alignments = defaultdict(Counter)
    for s1_name, indic_name, script in cross_script_pairs:
        s1_toks = s1_name.lower().split()
        ind_toks = indic_name.split()
        if len(s1_toks) == len(ind_toks):
            for stok, itok in zip(s1_toks, ind_toks):
                if has_indic(itok):
                    token_alignments[itok][stok] += 1
                    
    # Build dictionary
    dictionary = {}
    for itok, eng_counts in token_alignments.items():
        top_eng, top_count = eng_counts.most_common(1)[0]
        total_count = sum(eng_counts.values())
        if top_count / total_count >= 0.75 and total_count >= 3:
            dictionary[itok] = top_eng
            
    print(f"\nMined Dictionary: {len(dictionary):,} high-confidence Indic->Latin token entries")
    
    # Save dictionary
    out_path = os.path.join(checkpoint_dir, "indic_translit_dict.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path} in {time.time()-t0:.1f}s")
    
    # Sample translations
    sample = list(dictionary.items())[:15]
    print("\nSample Mappings:")
    for k, v in sample:
        print(f"  {k} -> {v}")
    return dictionary

if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/covid/amazon_ml/dataset"
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/covid/amazon_ml/Implementation/runs/default/checkpoints"
    mine_dictionary(dataset_dir, checkpoint_dir)
