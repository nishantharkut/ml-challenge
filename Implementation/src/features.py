"""
High-Performance Feature Engineering for Candidate Pairs.
Computes deterministic string, token, address, and retrieval meta-features via RapidFuzz.
"""
from rapidfuzz import fuzz, distance

def extract_pair_features(s1_rec, cand_rec, meta=None):
    """
    Compute comprehensive feature dictionary for a candidate pair (s1, cand).
    meta: optional retrieval metadata dict from blocking stage.
    """
    s1_name = s1_rec.get('name_norm', '')
    cand_name = cand_rec.get('name_norm', '')
    
    s1_core = s1_rec.get('name_core', '')
    cand_core = cand_rec.get('name_core', '')
    s1_folded = s1_rec.get('name_folded', s1_name)
    cand_folded = cand_rec.get('name_folded', cand_name)
    
    s1_addr = s1_rec.get('addr_norm', '')
    cand_addr = cand_rec.get('addr_norm', '')
    s1_addr_folded = s1_rec.get('addr_folded', s1_addr)
    cand_addr_folded = cand_rec.get('addr_folded', cand_addr)
    
    # 1. Name Features
    name_jw = distance.JaroWinkler.similarity(s1_name, cand_name) if s1_name and cand_name else 0.0
    name_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_fuzz = fuzz.ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_core_set = fuzz.token_set_ratio(s1_core, cand_core) / 100.0 if s1_core and cand_core else name_set
    
    len1 = len(s1_name)
    len2 = len(cand_name)
    name_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    name_folded_ratio = (
        fuzz.ratio(s1_folded, cand_folded) / 100.0
        if s1_folded and cand_folded else 0.0
    )
    name_folded_exact = 1.0 if s1_folded and s1_folded == cand_folded else 0.0
    name_partial = (
        fuzz.partial_ratio(s1_folded, cand_folded) / 100.0
        if s1_folded and cand_folded else 0.0
    )
    
    # Suffix agreement
    s1_suf = s1_rec.get('name_suffix', '')
    cand_suf = cand_rec.get('name_suffix', '')
    suffix_match = 1.0 if (s1_suf and cand_suf and s1_suf == cand_suf) else 0.0
    suffix_mismatch = 1.0 if (s1_suf and cand_suf and s1_suf != cand_suf) else 0.0
    
    # First token match (Brand root)
    s1_toks = s1_rec.get('name_tokens', [])
    cand_toks = cand_rec.get('name_tokens', [])
    first_tok_match = 1.0 if (s1_toks and cand_toks and s1_toks[0] == cand_toks[0]) else 0.0
    
    # Script mismatch
    is_cross_script = 1.0 if (s1_rec.get('script') != cand_rec.get('script')) else 0.0
    
    # 2. Address Features
    addr_empty1 = s1_rec.get('addr_empty', True)
    addr_empty2 = cand_rec.get('addr_empty', True)
    either_addr_empty = 1.0 if (addr_empty1 or addr_empty2) else 0.0
    both_addr_empty = 1.0 if (addr_empty1 and addr_empty2) else 0.0
    
    if not addr_empty1 and not addr_empty2:
        addr_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
        addr_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        addr_exact = 1.0 if s1_addr == cand_addr else 0.0
        addr_folded_ratio = (
            fuzz.ratio(s1_addr_folded, cand_addr_folded) / 100.0
            if s1_addr_folded and cand_addr_folded else 0.0
        )
        
        # Number agreement vs conflict
        nums1 = s1_rec.get('addr_numbers', set())
        nums2 = cand_rec.get('addr_numbers', set())
        if nums1 and nums2:
            intersection = nums1 & nums2
            num_any_overlap = 1.0 if intersection else 0.0
            num_match = num_any_overlap
            num_exact_set = 1.0 if nums1 == nums2 else 0.0
            num_jaccard = len(intersection) / len(nums1 | nums2)
            num_conflict = 1.0 if nums1 != nums2 else 0.0
        else:
            num_match = 0.0
            num_any_overlap = 0.0
            num_exact_set = 0.0
            num_jaccard = 0.0
            num_conflict = 0.0
            
        # Postal code agreement
        post1 = s1_rec.get('addr_postal')
        post2 = cand_rec.get('addr_postal')
        post_match = 1.0 if (post1 and post2 and post1 == post2) else 0.0
        post_conflict = 1.0 if (post1 and post2 and post1 != post2) else 0.0
    else:
        addr_set = 0.0
        addr_sort = 0.0
        addr_exact = 0.0
        addr_folded_ratio = 0.0
        num_match = 0.0
        num_any_overlap = 0.0
        num_exact_set = 0.0
        num_jaccard = 0.0
        num_conflict = 0.0
        post_match = 0.0
        post_conflict = 0.0
        
    # 3. Retrieval Meta-Features
    meta = meta or {}
    source_label = meta.get('source', '') or ('S2' if cand_rec.get('entity_id', '').startswith('S2') else 'S3')
    is_s2 = 1.0 if source_label == 'S2' else 0.0
    
    channels = meta.get('channels', set())
    channel_count = float(len(channels))
    exact_name_hit = float(meta.get('exact_name', 0))
    exact_addr_hit = float(meta.get('exact_addr', 0))
    rare_token_hit = float(meta.get('rare_token', 0))
    address_anchor_hit = float(meta.get('address_anchor', 0))
    
    tfidf_name_score = float(meta.get('tfidf_name_score', 0.0))
    tfidf_addr_score = float(meta.get('tfidf_addr_score', 0.0))
    best_rank = float(meta.get('best_rank', 999))
    reciprocal_rank = 1.0 / (best_rank + 1.0) if best_rank < 999 else 0.0
    
    feats = {
        'name_jw': name_jw,
        'name_sort': name_sort,
        'name_set': name_set,
        'name_fuzz': name_fuzz,
        'name_core_set': name_core_set,
        'name_len_ratio': name_len_ratio,
        'name_exact': name_exact,
        'name_folded_ratio': name_folded_ratio,
        'name_folded_exact': name_folded_exact,
        'name_partial': name_partial,
        'suffix_match': suffix_match,
        'suffix_mismatch': suffix_mismatch,
        'first_tok_match': first_tok_match,
        'is_cross_script': is_cross_script,
        'either_addr_empty': either_addr_empty,
        'both_addr_empty': both_addr_empty,
        'addr_set': addr_set,
        'addr_sort': addr_sort,
        'addr_exact': addr_exact,
        'addr_folded_ratio': addr_folded_ratio,
        'num_match': num_match,
        'num_any_overlap': num_any_overlap,
        'num_exact_set': num_exact_set,
        'num_jaccard': num_jaccard,
        'num_conflict': num_conflict,
        'post_match': post_match,
        'post_conflict': post_conflict,
        'is_s2': is_s2,
        'channel_count': channel_count,
        'exact_name_hit': exact_name_hit,
        'exact_addr_hit': exact_addr_hit,
        'rare_token_hit': rare_token_hit,
        'address_anchor_hit': address_anchor_hit,
        'tfidf_name_score': tfidf_name_score,
        'tfidf_addr_score': tfidf_addr_score,
        'reciprocal_rank': reciprocal_rank,
        'name_addr_interaction': name_set * addr_set
    }
    return feats
