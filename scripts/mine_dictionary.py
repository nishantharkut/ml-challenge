"""
Critical experiment: Auto-mine transliteration dictionary from GT cross-script pairs.
Measures: How many cross-script pairs exist? Can we auto-align tokens?
"""
import sys, os, csv
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"C:\N Drive\Amazon ML Challenge\dataset\6ab10eb3b23ba_student_resource\student_resource\dataset"
train_dir = os.path.join(base_dir, "train")

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

def get_script(text):
    """Return dominant Indic script or 'Latin'"""
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

def has_indic(text):
    for ch in text:
        cp = ord(ch)
        for lo, hi in INDIC_RANGES.values():
            if lo <= cp <= hi:
                return True
    return False

# Load S1 records (all Latin)
print("Loading S1 records...")
s1_records = {}
with open(os.path.join(train_dir, 'train_source1.tsv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if row[3] == 'India':
            s1_records[row[0]] = {'name': row[1], 'addr': row[2], 'country': row[3]}

print(f"  S1 India records: {len(s1_records):,}")

# Load GT
print("Loading ground truth...")
gt = {}
with open(os.path.join(train_dir, 'train_ground_truth.tsv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if row[0] in s1_records:
            matches_str = row[1] if len(row) > 1 else ''
            gt[row[0]] = [m.strip() for m in matches_str.split(',') if m.strip()]

# Find cross-script GT pairs
print("Scanning S2+S3 for cross-script GT matches...")
cross_script_pairs = []  # (s1_name, indic_name, script)

for sname in ['train_source2.tsv', 'train_source3.tsv']:
    # Build set of India GT IDs for this source
    prefix = 'S2-' if 'source2' in sname else 'S3-'
    gt_ids = set()
    s1_for_match = {}  # match_id -> s1_id
    for s1_id, matches in gt.items():
        for m in matches:
            if m.startswith(prefix):
                gt_ids.add(m)
                s1_for_match[m] = s1_id
    
    print(f"  {sname}: scanning for {len(gt_ids):,} India GT IDs...")
    
    count = 0
    with open(os.path.join(train_dir, sname), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0]
            if eid in gt_ids and has_indic(row[1]):
                s1_id = s1_for_match[eid]
                s1_name = s1_records[s1_id]['name']
                indic_name = row[1]
                script = get_script(indic_name)
                cross_script_pairs.append((s1_name, indic_name, script, eid, s1_id))
                count += 1
    
    print(f"    Found {count:,} cross-script name matches")

print(f"\nTotal cross-script name GT pairs: {len(cross_script_pairs):,}")

# Stats by script
script_counts = Counter(p[2] for p in cross_script_pairs)
print("\nBreakdown by script:")
for script, count in script_counts.most_common():
    print(f"  {script}: {count:,} pairs ({count/len(cross_script_pairs)*100:.1f}%)")

# Token alignment: try to auto-build dictionary
print("\n" + "=" * 80)
print("AUTO-MINING TRANSLITERATION DICTIONARY")
print("=" * 80)

token_alignments = defaultdict(Counter)  # indic_token -> {english_token: count}
total_aligned = 0
total_unaligned = 0

for s1_name, indic_name, script, _, _ in cross_script_pairs:
    s1_tokens = s1_name.lower().split()
    indic_tokens = indic_name.split()
    
    # Simple positional alignment (when token counts match)
    if len(s1_tokens) == len(indic_tokens):
        for s1_tok, ind_tok in zip(s1_tokens, indic_tokens):
            if has_indic(ind_tok):
                token_alignments[ind_tok][s1_tok] += 1
                total_aligned += 1
    else:
        total_unaligned += 1

print(f"\nExactly-aligned pairs: {total_aligned:,}")
print(f"Unaligned (different token count): {total_unaligned:,}")
print(f"Unique Indic tokens in dictionary: {len(token_alignments):,}")

# Show top 30 most frequent alignments
print("\nTop 30 most frequent Indic→English mappings:")
freq_list = []
for indic_tok, eng_counts in token_alignments.items():
    top_eng, top_count = eng_counts.most_common(1)[0]
    total_count = sum(eng_counts.values())
    confidence = top_count / total_count
    freq_list.append((indic_tok, top_eng, top_count, total_count, confidence))

freq_list.sort(key=lambda x: -x[3])
for i, (ind, eng, top_c, tot_c, conf) in enumerate(freq_list[:30]):
    print(f"  {i+1:2d}. {ind:30s} → {eng:20s} ({tot_c:6,} pairs, confidence={conf:.2%})")

# Show coverage: what % of cross-script pairs could be translated by dictionary?
print("\n" + "=" * 80)
print("DICTIONARY COVERAGE ANALYSIS")
print("=" * 80)

# Build dictionary from all alignments with confidence > 80%
dictionary = {}
for indic_tok, eng_counts in token_alignments.items():
    top_eng, top_count = eng_counts.most_common(1)[0]
    total_count = sum(eng_counts.values())
    if top_count / total_count >= 0.8 and total_count >= 5:
        dictionary[indic_tok] = top_eng

print(f"Dictionary size (confidence≥80%, count≥5): {len(dictionary):,} entries")

# Test: how many cross-script names can we fully translate?
fully_translated = 0
partially_translated = 0
no_translation = 0

for s1_name, indic_name, script, _, _ in cross_script_pairs[:10000]:
    indic_tokens = indic_name.split()
    translated = []
    for tok in indic_tokens:
        if tok in dictionary:
            translated.append(dictionary[tok])
        elif not has_indic(tok):
            translated.append(tok.lower())
        else:
            translated.append(None)
    
    if all(t is not None for t in translated):
        fully_translated += 1
    elif any(t is not None for t in translated):
        partially_translated += 1
    else:
        no_translation += 1

total = fully_translated + partially_translated + no_translation
print(f"\nFirst 10K cross-script pairs translation coverage:")
print(f"  Fully translated:     {fully_translated:,} ({fully_translated/total*100:.1f}%)")
print(f"  Partially translated: {partially_translated:,} ({partially_translated/total*100:.1f}%)")
print(f"  No translation:       {no_translation:,} ({no_translation/total*100:.1f}%)")

print("\nDone!")
