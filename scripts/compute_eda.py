import sys
import os
import csv
import json
from collections import Counter, defaultdict

base_dir = r"C:\N Drive\Amazon ML Challenge\dataset\6ab10eb3b23ba_student_resource\student_resource\dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

print("Starting EDA analysis...")

# 1. First, analyze train_source1.tsv to get s1_countries and source1 stats
source_files = {
    'train_s1': os.path.join(train_dir, 'train_source1.tsv'),
    'train_s2': os.path.join(train_dir, 'train_source2.tsv'),
    'train_s3': os.path.join(train_dir, 'train_source3.tsv'),
    'test_s1': os.path.join(test_dir, 'test_source1.tsv'),
    'test_s2': os.path.join(test_dir, 'test_source2.tsv'),
    'test_s3': os.path.join(test_dir, 'test_source3.tsv'),
}

file_stats = {}
country_variants_all = defaultdict(Counter)

# Read train_source1.tsv first
print("Reading train_source1.tsv...")
s1_country_map = {} # s1_id -> country string
s1_counts = Counter()
s1_empty_addr = 0
s1_total = 0

with open(source_files['train_s1'], 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    # entity_id, business_name, business_address, country
    for row in reader:
        s1_total += 1
        eid, name, addr, country = row[0], row[1], row[2], row[3]
        s1_counts[country] += 1
        country_variants_all['train_s1'][country] += 1
        s1_country_map[eid] = country
        if not addr or addr.strip() == '':
            s1_empty_addr += 1

file_stats['train_s1'] = {
    'total_rows': s1_total,
    'country_counts': dict(s1_counts),
    'empty_address_count': s1_empty_addr,
    'empty_address_rate': s1_empty_addr / s1_total if s1_total else 0,
}
print(f"train_s1 parsed: {s1_total:,} rows. Countries: {dict(s1_counts)}")

# 2. Analyze train_ground_truth.tsv
print("Reading train_ground_truth.tsv...")
gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')

gt_total = 0
singletons = 0
match_counts = []
s2_per_s1 = []
s3_per_s1 = []

# Map matched S2 and S3 to their S1 ID and expected S1 country
s2_to_s1 = {} # s2_id -> (s1_id, s1_country)
s3_to_s1 = {} # s3_id -> (s1_id, s1_country)

s2_multi_match = 0
s3_multi_match = 0

# Sample 15 interesting S1 entities for raw matched groups
# We want: 
# - ~8 from US, ~7 from India
# - varying number of matches (1 match, 2 matches, 3 matches, 4+ matches)
# - both S2 and S3 matches
sample_groups_s1_ids = []
sample_group_data = {} # s1_id -> {'s1': None, 'matches': {s2_or_s3_id: None}}

gt_rows_for_sampling = []

with open(gt_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        gt_total += 1
        s1_id = row[0]
        matches_str = row[1] if len(row) > 1 else ''
        if not matches_str.strip():
            singletons += 1
            match_counts.append(0)
            s2_per_s1.append(0)
            s3_per_s1.append(0)
            continue
        
        matches = [m.strip() for m in matches_str.split(',') if m.strip()]
        match_counts.append(len(matches))
        s1_c = s1_country_map.get(s1_id, 'UNKNOWN')
        
        s2_this = 0
        s3_this = 0
        for m in matches:
            if m.startswith('S2-'):
                s2_this += 1
                if m in s2_to_s1:
                    s2_multi_match += 1
                s2_to_s1[m] = (s1_id, s1_c)
            elif m.startswith('S3-'):
                s3_this += 1
                if m in s3_to_s1:
                    s3_multi_match += 1
                s3_to_s1[m] = (s1_id, s1_c)
        s2_per_s1.append(s2_this)
        s3_per_s1.append(s3_this)
        
        # Collect candidates for the 15 raw groups
        if len(gt_rows_for_sampling) < 2000 and s2_this > 0 and s3_this > 0:
            gt_rows_for_sampling.append((s1_id, s1_c, matches, s2_this, s3_this))

# Select exactly 15 diverse groups
us_candidates = [x for x in gt_rows_for_sampling if x[1] == 'US']
in_candidates = [x for x in gt_rows_for_sampling if x[1] == 'India']

selected_groups = []
# 8 US candidates with match sizes 2, 3, 4, 5, 6...
us_by_size = defaultdict(list)
for c in us_candidates:
    us_by_size[len(c[2])].append(c)

for sz in [2, 3, 4, 5, 2, 3, 4, 6]:
    if us_by_size[sz]:
        selected_groups.append(us_by_size[sz].pop(0))

in_by_size = defaultdict(list)
for c in in_candidates:
    in_by_size[len(c[2])].append(c)

for sz in [2, 3, 4, 5, 2, 3, 4]:
    if in_by_size[sz]:
        selected_groups.append(in_by_size[sz].pop(0))

while len(selected_groups) < 15 and gt_rows_for_sampling:
    selected_groups.append(gt_rows_for_sampling.pop(0))

selected_groups = selected_groups[:15]
sample_target_ids = set()
for s1_id, c, matches, _, _ in selected_groups:
    sample_target_ids.add(s1_id)
    for m in matches:
        sample_target_ids.add(m)

print(f"Selected 15 sample groups with {len(sample_target_ids)} total target IDs to fetch.")

# match counts sorting and percentiles
match_counts.sort()
s2_per_s1.sort()
s3_per_s1.sort()

def get_percentiles(arr):
    n = len(arr)
    if n == 0:
        return {}
    return {
        'p10': arr[int(0.10 * n)],
        'p25': arr[int(0.25 * n)],
        'p50 (median)': arr[int(0.50 * n)],
        'p75': arr[int(0.75 * n)],
        'p90': arr[int(0.90 * n)],
        'p95': arr[int(0.95 * n)],
        'p99': arr[int(0.99 * n)],
        'max': arr[-1],
        'min': arr[0],
        'mean': sum(arr) / n,
    }

gt_stats = {
    'total_s1': gt_total,
    'singletons': singletons,
    'singleton_share': singletons / gt_total if gt_total else 0,
    'non_singletons': gt_total - singletons,
    'total_matches': sum(match_counts),
    'matches_per_s1_all': get_percentiles(match_counts),
    'matches_per_s1_non_singletons': get_percentiles([x for x in match_counts if x > 0]),
    's2_matches_per_s1': get_percentiles(s2_per_s1),
    's3_matches_per_s1': get_percentiles(s3_per_s1),
    's2_ids_shared_across_s1': s2_multi_match,
    's3_ids_shared_across_s1': s3_multi_match,
    'match_distribution': dict(Counter(match_counts).most_common(15)),
}

print("GT stats computed.")

# 3. Read train_source2 and train_source3 to:
# - compute row counts, empty address, country counts
# - check cross-country GT pairs!
# - extract raw records for the 15 sample groups

cross_country_pairs = []
raw_records = {} # eid -> (business_name, business_address, country)

for sname, sfile, s_to_s1 in [('train_s2', source_files['train_s2'], s2_to_s1), 
                              ('train_s3', source_files['train_s3'], s3_to_s1)]:
    print(f"Reading {sname} ({sfile})...")
    tot = 0
    cntry = Counter()
    empty_addr = 0
    with open(sfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            tot += 1
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            cntry[country] += 1
            country_variants_all[sname][country] += 1
            if not addr or addr.strip() == '':
                empty_addr += 1
            
            # Check cross country
            if eid in s_to_s1:
                s1_id, s1_country = s_to_s1[eid]
                if country != s1_country:
                    cross_country_pairs.append({
                        's1_id': s1_id,
                        's1_country': s1_country,
                        'match_id': eid,
                        'match_country': country,
                    })
            
            # Check if needed for sample
            if eid in sample_target_ids:
                raw_records[eid] = {
                    'entity_id': eid,
                    'business_name': name,
                    'business_address': addr,
                    'country': country,
                }
    
    file_stats[sname] = {
        'total_rows': tot,
        'country_counts': dict(cntry),
        'empty_address_count': empty_addr,
        'empty_address_rate': empty_addr / tot if tot else 0,
    }
    print(f"{sname} parsed: {tot:,} rows. Countries: {dict(cntry)}")

# Also get sample records for train_source1
with open(source_files['train_s1'], 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        eid, name, addr, country = row[0], row[1], row[2], row[3]
        if eid in sample_target_ids:
            raw_records[eid] = {
                'entity_id': eid,
                'business_name': name,
                'business_address': addr,
                'country': country,
            }

# 4. Now process test files (test_s1, test_s2, test_s3)
for sname in ['test_s1', 'test_s2', 'test_s3']:
    sfile = source_files[sname]
    print(f"Reading {sname} ({sfile})...")
    tot = 0
    cntry = Counter()
    empty_addr = 0
    with open(sfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            tot += 1
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            cntry[country] += 1
            country_variants_all[sname][country] += 1
            if not addr or addr.strip() == '':
                empty_addr += 1
    
    file_stats[sname] = {
        'total_rows': tot,
        'country_counts': dict(cntry),
        'empty_address_count': empty_addr,
        'empty_address_rate': empty_addr / tot if tot else 0,
    }
    print(f"{sname} parsed: {tot:,} rows. Countries: {dict(cntry)}")

# Build final output structure
final_sample_groups = []
for s1_id, c, matches, s2_this, s3_this in selected_groups:
    group_entry = {
        's1': raw_records.get(s1_id, {'entity_id': s1_id, 'business_name': 'N/A', 'business_address': 'N/A', 'country': c}),
        'matches': [raw_records.get(m, {'entity_id': m, 'business_name': 'N/A', 'business_address': 'N/A', 'country': 'N/A'}) for m in matches]
    }
    final_sample_groups.append(group_entry)

result = {
    'file_stats': file_stats,
    'country_variants': {k: dict(v) for k, v in country_variants_all.items()},
    'ground_truth_stats': gt_stats,
    'cross_country_gt_pairs_count': len(cross_country_pairs),
    'cross_country_gt_pairs_sample': cross_country_pairs[:20],
    'sample_15_matched_groups': final_sample_groups,
}

out_path = r"C:\N Drive\Amazon ML Challenge\scripts\eda_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"EDA analysis complete! Results saved to {out_path}")
