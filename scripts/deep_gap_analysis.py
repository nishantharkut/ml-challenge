"""
Deep EDA Analysis - Gaps in current plan that need validation
"""
import sys, os, csv
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"C:\N Drive\Amazon ML Challenge\dataset\6ab10eb3b23ba_student_resource\student_resource\dataset"
train_dir = os.path.join(base_dir, "train")

print("=" * 80)
print("ANALYSIS 1: S2 vs S3 noise differences")
print("=" * 80)

# Load S1 country map
s1_country = {}
with open(os.path.join(train_dir, 'train_source1.tsv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        s1_country[row[0]] = row[3]

# Load ground truth
gt = {}  # s1_id -> list of matched ids
with open(os.path.join(train_dir, 'train_ground_truth.tsv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        s1_id = row[0]
        matches_str = row[1] if len(row) > 1 else ''
        if matches_str.strip():
            gt[s1_id] = [m.strip() for m in matches_str.split(',') if m.strip()]
        else:
            gt[s1_id] = []

# Analyze match count distribution by source
s2_per_s1 = Counter()  # how many S2 matches does each S1 have
s3_per_s1 = Counter()
for s1_id, matches in gt.items():
    s2_count = sum(1 for m in matches if m.startswith('S2-'))
    s3_count = sum(1 for m in matches if m.startswith('S3-'))
    s2_per_s1[s2_count] += 1
    s3_per_s1[s3_count] += 1

print("\nS2 matches per S1 distribution:")
for k in sorted(s2_per_s1.keys()):
    print(f"  {k} S2 matches: {s2_per_s1[k]:>10,} S1 entities ({s2_per_s1[k]/len(gt)*100:.2f}%)")

print("\nS3 matches per S1 distribution:")
for k in sorted(s3_per_s1.keys()):
    print(f"  {k} S3 matches: {s3_per_s1[k]:>10,} S1 entities ({s3_per_s1[k]/len(gt)*100:.2f}%)")

# S1 entities with ONLY S2 matches, ONLY S3 matches, both, neither
only_s2 = sum(1 for s1, m in gt.items() if any(x.startswith('S2-') for x in m) and not any(x.startswith('S3-') for x in m))
only_s3 = sum(1 for s1, m in gt.items() if any(x.startswith('S3-') for x in m) and not any(x.startswith('S2-') for x in m))
both = sum(1 for s1, m in gt.items() if any(x.startswith('S2-') for x in m) and any(x.startswith('S3-') for x in m))
neither = sum(1 for s1, m in gt.items() if len(m) == 0)

print(f"\nS1 entities with ONLY S2 matches: {only_s2:,} ({only_s2/len(gt)*100:.2f}%)")
print(f"S1 entities with ONLY S3 matches: {only_s3:,} ({only_s3/len(gt)*100:.2f}%)")
print(f"S1 entities with BOTH S2+S3 matches: {both:,} ({both/len(gt)*100:.2f}%)")
print(f"S1 entities with NO matches (singletons): {neither:,} ({neither/len(gt)*100:.2f}%)")

print("\n" + "=" * 80)
print("ANALYSIS 2: Empty address breakdown by country and source")
print("=" * 80)

# Check which matched S2/S3 records have empty addresses
gt_s2_ids = set()
gt_s3_ids = set()
for matches in gt.values():
    for m in matches:
        if m.startswith('S2-'):
            gt_s2_ids.add(m)
        else:
            gt_s3_ids.add(m)

# Read S2 and check empty address among GT matches
s2_empty_in_gt = 0
s2_total_in_gt = 0
s2_empty_by_country = Counter()
for sname, gt_ids in [('train_source2.tsv', gt_s2_ids), ('train_source3.tsv', gt_s3_ids)]:
    empty_in_gt = 0
    total_in_gt = 0
    empty_by_country = Counter()
    with open(os.path.join(train_dir, sname), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            eid = row[0]
            if eid in gt_ids:
                total_in_gt += 1
                if not row[2] or row[2].strip() == '':
                    empty_in_gt += 1
                    empty_by_country[row[3]] += 1
    print(f"\n{sname}: GT-matched records with empty address: {empty_in_gt:,} / {total_in_gt:,} ({empty_in_gt/total_in_gt*100:.2f}%)")
    print(f"  Empty by country: {dict(empty_by_country)}")

print("\n" + "=" * 80)
print("ANALYSIS 3: Script detection in Indian records")
print("=" * 80)

def detect_script(text):
    """Detect the primary script of text"""
    scripts = Counter()
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            scripts['Devanagari'] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            scripts['Tamil'] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            scripts['Telugu'] += 1
        elif 0x0041 <= cp <= 0x007A:
            scripts['Latin'] += 1
        elif 0x0980 <= cp <= 0x09FF:
            scripts['Bengali'] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            scripts['Gurmukhi'] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            scripts['Gujarati'] += 1
        elif 0x0B00 <= cp <= 0x0B7F:
            scripts['Odia'] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            scripts['Kannada'] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            scripts['Malayalam'] += 1
    return scripts

# Sample 500K Indian records from S2 and S3 to see script distribution
for sname in ['train_source2.tsv', 'train_source3.tsv']:
    script_counts = Counter()
    has_indic_name = 0
    has_indic_addr = 0
    total_india = 0
    
    with open(os.path.join(train_dir, sname), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if row[3] != 'India':
                continue
            total_india += 1
            
            name_scripts = detect_script(row[1])
            addr_scripts = detect_script(row[2])
            
            indic_scripts = {'Devanagari', 'Tamil', 'Telugu', 'Bengali', 'Gurmukhi', 'Gujarati', 'Odia', 'Kannada', 'Malayalam'}
            
            name_indic = sum(name_scripts[s] for s in indic_scripts)
            addr_indic = sum(addr_scripts[s] for s in indic_scripts)
            
            if name_indic > 0:
                has_indic_name += 1
                for s in indic_scripts:
                    if name_scripts[s] > 0:
                        script_counts[f"name_{s}"] += 1
            
            if addr_indic > 0:
                has_indic_addr += 1
                for s in indic_scripts:
                    if addr_scripts[s] > 0:
                        script_counts[f"addr_{s}"] += 1
    
    print(f"\n{sname} India records: {total_india:,}")
    print(f"  Records with Indic script in NAME: {has_indic_name:,} ({has_indic_name/total_india*100:.2f}%)")
    print(f"  Records with Indic script in ADDR: {has_indic_addr:,} ({has_indic_addr/total_india*100:.2f}%)")
    print(f"  Script breakdown:")
    for k in sorted(script_counts.keys()):
        print(f"    {k}: {script_counts[k]:,}")

print("\n" + "=" * 80)
print("ANALYSIS 4: S1 India records - any Indic script?")
print("=" * 80)

s1_indic = 0
s1_india_total = 0
with open(os.path.join(train_dir, 'train_source1.tsv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in reader:
        if row[3] != 'India':
            continue
        s1_india_total += 1
        scripts = detect_script(row[1] + ' ' + row[2])
        indic_scripts = {'Devanagari', 'Tamil', 'Telugu', 'Bengali', 'Gurmukhi', 'Gujarati', 'Odia', 'Kannada', 'Malayalam'}
        if sum(scripts[s] for s in indic_scripts) > 0:
            s1_indic += 1

print(f"S1 India records with ANY Indic script: {s1_indic:,} / {s1_india_total:,} ({s1_indic/s1_india_total*100:.4f}%)")

print("\n" + "=" * 80)
print("ANALYSIS 5: How many S2/S3 records are NOT in any GT match? (unmatched pool)")
print("=" * 80)

# Total S2 IDs in train
with open(os.path.join(train_dir, 'train_source2.tsv'), 'r', encoding='utf-8') as f:
    next(f)
    total_s2 = sum(1 for _ in f)

with open(os.path.join(train_dir, 'train_source3.tsv'), 'r', encoding='utf-8') as f:
    next(f)
    total_s3 = sum(1 for _ in f)

print(f"Total S2 records: {total_s2:,}")
print(f"S2 in GT matches: {len(gt_s2_ids):,}")
print(f"S2 NOT in any GT match: {total_s2 - len(gt_s2_ids):,} ({(total_s2 - len(gt_s2_ids))/total_s2*100:.2f}%)")
print(f"\nTotal S3 records: {total_s3:,}")
print(f"S3 in GT matches: {len(gt_s3_ids):,}")
print(f"S3 NOT in any GT match: {total_s3 - len(gt_s3_ids):,} ({(total_s3 - len(gt_s3_ids))/total_s3*100:.2f}%)")

print("\nDone!")
