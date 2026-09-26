"""
Amazon ML Challenge 2026 - Entity Resolution Pipeline
Phase 0: Competition metric + data loading + validation infrastructure
Following Master Plan v5.1/v5.2
"""
import sys
import os
import csv
import time
import gc
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# PATHS
# ============================================================
BASE = r"C:\N Drive\Amazon ML Challenge\dataset\6ab10eb3b23ba_student_resource\student_resource\dataset"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
OUTPUT_DIR = r"C:\N Drive\Amazon ML Challenge\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# EXACT COMPETITION METRIC [MUST — implemented first per v5.1 §12]
# ============================================================
def competition_macro_f05(predictions, ground_truth, beta=0.5):
    """
    Exact competition scoring.
    predictions: {s1_id: set of matched_ids}
    ground_truth: {s1_id: set of true_ids}
    """
    scores = []
    for s1_id in ground_truth:
        true = ground_truth[s1_id]
        pred = predictions.get(s1_id, set())
        if len(true) == 0 and len(pred) == 0:
            scores.append(1.0)
        elif len(true) == 0 and len(pred) > 0:
            scores.append(0.0)
        elif len(pred) == 0:
            scores.append(0.0)
        else:
            tp = len(true & pred)
            p = tp / len(pred) if len(pred) > 0 else 0
            r = tp / len(true) if len(true) > 0 else 0
            if p + r == 0:
                scores.append(0.0)
            else:
                scores.append((1 + beta**2) * p * r / (beta**2 * p + r))
    return sum(scores) / len(scores)


def oracle_macro_f05(candidates, ground_truth, beta=0.5):
    """
    Oracle F0.5: perfect matcher on given candidate set.
    candidates: {s1_id: set of candidate_ids}
    ground_truth: {s1_id: set of true_ids}
    Returns the ceiling score the matcher can never exceed.
    """
    # For each S1, the oracle prediction is GT ∩ candidates
    oracle_preds = {}
    for s1_id in ground_truth:
        cands = candidates.get(s1_id, set())
        true = ground_truth[s1_id]
        oracle_preds[s1_id] = true & cands
    return competition_macro_f05(oracle_preds, ground_truth, beta)


# ============================================================
# DATA LOADING
# ============================================================
def load_tsv(filepath):
    """Load TSV returning list of dicts. Memory-efficient streaming."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            records.append(row)
    return records


def load_ground_truth(filepath):
    """Load GT into {s1_id: set(matched_ids)}"""
    gt = defaultdict(set)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_id = row['source1_entity_id']
            matched = row['matched_entity_ids']
            if matched:
                for mid in matched.split(','):
                    mid = mid.strip()
                    if mid:
                        gt[s1_id].add(mid)
    return dict(gt)


def build_entity_index(records, id_field='entity_id'):
    """Build {entity_id: record_dict} index."""
    idx = {}
    for r in records:
        idx[r[id_field]] = r
    return idx


# ============================================================
# CANDIDATE RECALL METRICS [v5.1 §6]
# ============================================================
def candidate_recall_report(candidates, ground_truth, s2_index=None, s3_index=None):
    """
    Full recall certificate for a candidate set.
    candidates: {s1_id: set(candidate_ids)}
    ground_truth: {s1_id: set(true_ids)}
    """
    total_pairs = 0
    found_pairs = 0
    s2_total = 0; s2_found = 0
    s3_total = 0; s3_found = 0
    complete_entities = 0
    total_entities = 0

    for s1_id, true_ids in ground_truth.items():
        if len(true_ids) == 0:
            complete_entities += 1
            total_entities += 1
            continue
        total_entities += 1
        cands = candidates.get(s1_id, set())
        found = true_ids & cands
        total_pairs += len(true_ids)
        found_pairs += len(found)

        if found == true_ids:
            complete_entities += 1

        for tid in true_ids:
            if tid.startswith('S2'):
                s2_total += 1
                if tid in cands:
                    s2_found += 1
            elif tid.startswith('S3'):
                s3_total += 1
                if tid in cands:
                    s3_found += 1

    # Candidate count stats
    counts = [len(candidates.get(s1_id, set())) for s1_id in ground_truth]
    counts.sort()
    n = len(counts)

    report = {
        'pair_recall': found_pairs / total_pairs if total_pairs > 0 else 1.0,
        's2_recall': s2_found / s2_total if s2_total > 0 else 1.0,
        's3_recall': s3_found / s3_total if s3_total > 0 else 1.0,
        'complete_entity_recall': complete_entities / total_entities if total_entities > 0 else 1.0,
        'mean_cands': sum(counts) / n if n > 0 else 0,
        'median_cands': counts[n // 2] if n > 0 else 0,
        'p95_cands': counts[int(n * 0.95)] if n > 0 else 0,
        'p99_cands': counts[int(n * 0.99)] if n > 0 else 0,
        'max_cands': counts[-1] if counts else 0,
    }

    # Oracle F0.5
    report['oracle_f05'] = oracle_macro_f05(candidates, ground_truth)

    return report


def print_recall_report(report, label=""):
    print(f"\n{'='*60}")
    print(f"CANDIDATE RECALL REPORT: {label}")
    print(f"{'='*60}")
    print(f"  Pair recall:              {report['pair_recall']:.6f}")
    print(f"  S2 recall:                {report['s2_recall']:.6f}")
    print(f"  S3 recall:                {report['s3_recall']:.6f}")
    print(f"  Complete-entity recall:   {report['complete_entity_recall']:.6f}")
    print(f"  Oracle Macro F0.5:        {report['oracle_f05']:.6f}")
    print(f"  Candidates/S1: mean={report['mean_cands']:.1f} "
          f"median={report['median_cands']} p95={report['p95_cands']} "
          f"p99={report['p99_cands']} max={report['max_cands']}")


# ============================================================
# PREPROCESSING [v5.1 §4]
# ============================================================
import unicodedata
import re

def detect_script(text):
    """Detect dominant script in text."""
    scripts = defaultdict(int)
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, '')
            if 'DEVANAGARI' in name: scripts['Devanagari'] += 1
            elif 'TAMIL' in name: scripts['Tamil'] += 1
            elif 'TELUGU' in name: scripts['Telugu'] += 1
            elif 'KANNADA' in name: scripts['Kannada'] += 1
            elif 'BENGALI' in name: scripts['Bengali'] += 1
            elif 'GUJARATI' in name: scripts['Gujarati'] += 1
            elif 'MALAYALAM' in name: scripts['Malayalam'] += 1
            elif 'GURMUKHI' in name: scripts['Gurmukhi'] += 1
            elif 'ORIYA' in name: scripts['Odia'] += 1
            else: scripts['Latin'] += 1
    if not scripts:
        return 'Latin'
    return max(scripts, key=scripts.get)


def normalize_text(text):
    """
    Script-safe normalization [v5.1 §4.1, v5.2 Patch 2].
    NFKC (not NFKD). Preserve Mn/Mc marks for Indic.
    """
    if not text or text.lower() in ('null', '<null>', 'none', ''):
        return ''
    # NFKC — safe for Indic
    text = unicodedata.normalize('NFKC', text)
    text = text.lower().strip()
    # Remove punctuation but keep Mn/Mc marks (Indic vowels)
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('L') or cat.startswith('N') or cat in ('Mn', 'Mc', 'Zs'):
            cleaned.append(ch)
        elif cat == 'Zs' or ch in (' ', '\t'):
            cleaned.append(' ')
        # Skip punctuation (P*), symbols (S*), etc.
    text = ''.join(cleaned)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Legal suffix standardization
LEGAL_SUFFIXES = {
    'pvt': 'private', 'ltd': 'limited', 'llp': 'llp',
    'inc': 'incorporated', 'corp': 'corporation', 'co': 'company',
    'llc': 'llc', 'plc': 'plc', 'sa': 'sa', 'sarl': 'sarl',
    'sas': 'sas', 'gmbh': 'gmbh', 'ag': 'ag', 'nv': 'nv',
    'bv': 'bv', 'pty': 'proprietary', 'pte': 'private',
    'intl': 'international', 'svc': 'services', 'svcs': 'services',
    'mfg': 'manufacturing', 'assoc': 'associates',
    'bros': 'brothers', 'dept': 'department',
    'enterprises': 'enterprises', 'enterprise': 'enterprises',
    'private': 'private', 'limited': 'limited',
    'incorporated': 'incorporated', 'corporation': 'corporation',
    'company': 'company',
}

def extract_suffix(name_tokens):
    """Extract and standardize legal suffix from name tokens."""
    if not name_tokens:
        return name_tokens, ''
    # Check last 1-3 tokens for known suffixes
    suffix_parts = []
    core = list(name_tokens)
    for i in range(min(3, len(core)), 0, -1):
        token = core[-1]
        std = LEGAL_SUFFIXES.get(token.rstrip('.'), None)
        if std:
            suffix_parts.insert(0, std)
            core.pop()
        else:
            break
    return core, ' '.join(suffix_parts)


def preprocess_record(record):
    """Preprocess a single record into multi-view representation [v5.2 Patch 6]."""
    name_raw = record.get('business_name', '') or ''
    addr_raw = record.get('business_address', '') or ''
    country = record.get('country', '') or ''
    eid = record.get('entity_id', '')

    name_norm = normalize_text(name_raw)
    addr_norm = normalize_text(addr_raw)

    name_tokens = name_norm.split()
    name_core, name_suffix = extract_suffix(name_tokens)

    script = detect_script(name_raw)
    is_indic = script != 'Latin'

    return {
        'entity_id': eid,
        'country': country.strip().upper(),
        'name_raw': name_raw,
        'name_norm': name_norm,
        'name_core': ' '.join(name_core),
        'name_suffix': name_suffix,
        'name_tokens': name_tokens,
        'addr_raw': addr_raw,
        'addr_norm': addr_norm,
        'addr_empty': len(addr_norm) == 0,
        'script': script,
        'is_indic': is_indic,
    }


# ============================================================
# MAIN — Phase 0 validation
# ============================================================
if __name__ == '__main__':
    t0 = time.time()
    print("="*60)
    print("PHASE 0: Infrastructure validation")
    print("="*60)

    # Test metric on known examples
    gt = {'S1-1': {'S2-A', 'S3-B'}, 'S1-2': set(), 'S1-3': {'S2-C'}}
    pred = {'S1-1': {'S2-A'}, 'S1-3': {'S2-C', 'S2-D'}}
    score = competition_macro_f05(pred, gt)
    print(f"\nMetric sanity check: {score:.4f}")
    # S1-1: TP=1, pred=1, true=2 → P=1.0, R=0.5, F0.5=1.25*1.0*0.5/(0.25*1.0+0.5)=0.8333
    # S1-2: true=0, pred=0 → 1.0
    # S1-3: TP=1, pred=2, true=1 → P=0.5, R=1.0, F0.5=1.25*0.5*1.0/(0.25*0.5+1.0)=0.5556
    # Macro = (0.8333 + 1.0 + 0.5556) / 3 = 0.7963
    expected = (0.8333 + 1.0 + 0.5556) / 3
    print(f"  Expected ~{expected:.4f}, got {score:.4f}")
    assert abs(score - expected) < 0.01, f"Metric bug! Expected {expected}, got {score}"
    print("  ✅ Metric validated")

    # Load GT
    print("\nLoading ground truth...")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    gt = load_ground_truth(gt_path)
    # Add singletons (S1 not in GT = singleton)
    print(f"  GT loaded: {len(gt)} S1 with matches")

    # Load S1
    print("Loading train S1...")
    s1_records = load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    print(f"  S1 records: {len(s1_records)}")

    # Add singletons to GT
    for r in s1_records:
        if r['entity_id'] not in gt:
            gt[r['entity_id']] = set()
    print(f"  Total S1 (with singletons): {len(gt)}")
    singletons = sum(1 for v in gt.values() if len(v) == 0)
    print(f"  Singletons: {singletons} ({100*singletons/len(gt):.2f}%)")

    # Country blocking validation [v5.2: validate by measuring recall, not just GT count]
    print("\nValidating country blocking safety...")
    s1_countries = {r['entity_id']: r['country'].strip().upper() for r in s1_records}
    # Load a sample of S2/S3 to check
    s2_sample = load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"))
    s3_sample = load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"))
    s2s3_countries = {}
    for r in s2_sample:
        s2s3_countries[r['entity_id']] = r['country'].strip().upper()
    for r in s3_sample:
        s2s3_countries[r['entity_id']] = r['country'].strip().upper()
    del s2_sample, s3_sample
    gc.collect()

    cross_country = 0
    total_checked = 0
    for s1_id, match_ids in gt.items():
        s1_c = s1_countries.get(s1_id, '')
        for mid in match_ids:
            total_checked += 1
            m_c = s2s3_countries.get(mid, '')
            if s1_c and m_c and s1_c != m_c:
                cross_country += 1
    print(f"  Cross-country GT pairs: {cross_country} / {total_checked}")
    if cross_country == 0:
        print("  ✅ Country blocking achieves 100% candidate recall — SAFE")
    else:
        print(f"  ⚠️ {cross_country} cross-country pairs! Country blocking loses recall!")

    del s2s3_countries
    gc.collect()

    t1 = time.time()
    print(f"\nPhase 0 complete in {t1-t0:.1f}s")
    print("\n" + "="*60)
    print("READY FOR E0: Exact/rare token blocking")
    print("="*60)
