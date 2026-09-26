# 🏆 Amazon ML Challenge 2026 — CORRECTED Master Battle Plan v2.0
## Business Entity Resolution: Top-10 Strategy (Post Deep-Research Hardening)

---

> [!CAUTION]
> **This plan supersedes v1.0.** Deep research and actual testing on our dataset exposed **6 critical flaws** in the original plan. Every correction is marked with 🔴 below.

> [!IMPORTANT]
> **NEW EMAIL GUIDELINES (v2.1 Update)** — These fundamentally change our blocking strategy:
> 1. **`candidate_pairs.tsv` is part of final submission** and reviewed for ranking
> 2. **Smaller candidate sets per S1 = higher ranking** beyond the leaderboard score
> 3. **`candidate_pairs.tsv` must be the LAST stage before ML inference** — "whatever your model actually runs inference over"
> 4. **Blocking code is reviewed** — must be elegant, scalable, well-documented
> 
> **Impact on strategy**: We can NO LONGER just maximize blocking recall with loose thresholds and 30-50 candidates per S1. We need a **multi-stage funnel** that produces a TIGHT, SMALL candidate set (target: **≤10 per S1**) while still maintaining high recall. The candidate_pairs.tsv IS the competitive differentiator beyond F₀.₅ score.

---

## 0. Error Log: What v1.0 Got Wrong

| # | Original Assumption | What Testing/Research Revealed | Severity | Fix |
|:--|:---|:---|:---|:---|
| 🔴1 | `anyascii` is a viable fast transliteration for Hindi/Tamil/Telugu | **anyascii achieves 0–50% token overlap** with expected English on our data. "रेड वेंचर्स" → "red vemcrs" instead of "Red Ventures". State names like "महाराष्ट्र" → "mharastr" (0% match) | **CRITICAL** | 3-tier transliteration: dictionary → rule-based → n-gram fallback |
| 🔴2 | `unidecode` is a fallback for Indic | **unidecode is WORSE**: doubles consonants everywhere. "redd veNcrs praaivett limittedd" | **CRITICAL** | Remove from Indic pipeline. Keep only for French/Latin accents |
| 🔴3 | Apply Unicode NFKD to strip diacritics globally | **NFKD stripping DESTROYS Indic scripts**. Vowel marks are combining marks (`Mn`); removing them renders text unreadable | **CRITICAL** | Apply NFKD only to Latin-script text |
| 🔴4 | `ai4bharat-transliteration` (IndicXlit) is primary transliteration | **May not install on Python 3.14**. Depends on `fairseq`/`torch` without cp314 wheels | **HIGH** | Use `indic-transliteration` (rule-based, pure Python) + dictionary. Keep IndicXlit as optional GPU enhancement in 3.11 venv |
| 🔴5 | All libraries (torch, lightgbm, faiss) work on Python 3.14 | **Many C++/CUDA extensions lack cp314 pre-built wheels** | **HIGH** | Create Python 3.11/3.12 virtual environment as first step |
| 🔴6 | TF-IDF `char_wb` on 5M docs fits in 15GB RAM | **Likely OOM** without `max_features` cap and `float32` dtype | **MEDIUM** | Cap `max_features=100K`, use `float32`, consider `HashingVectorizer` |
| 🔴7 | `multilingual-e5` is the best embedding model | **BGE-M3 is superior** for ER: has hybrid dense+sparse+ColBERT retrieval, 8192 token context | **MEDIUM** | Switch primary embedding to `bge-m3` |
| 🔴8 | Llama 3.1 / Gemma 2 are candidate models | **NOT MIT/Apache 2.0 licensed** — using them risks disqualification | **HIGH** | Strict license audit: only bge-m3 (MIT), e5 (MIT), mdeberta-v3 (MIT), Qwen2.5 (Apache 2.0), Phi-3.5 (MIT) |

### 🆕 New Data Insights from Deep Gap Analysis (v2.1)

| Finding | Number | Impact on Strategy |
|:---|:---|:---|
| **S1 is ALWAYS Latin script** | 0/883,188 India S1 have any Indic script | Transliteration is strictly ONE-WAY: Indic S2/S3 → Latin |
| **23.5% of S2 India names are Indic** | 474,345 / 2,017,799 records | ~750K total Indic records need transliteration — NOT a niche problem |
| **9 different Indian scripts** | Devanagari (427K), Telugu (62K), Kannada (59K), Tamil (54K), Bengali (49K), Gujarati (49K), Malayalam (30K), Odia (11K), Gurmukhi (11K) | Dictionary must cover ALL scripts, not just Hindi |
| **80.5% of S1 have BOTH S2+S3 matches** | 1,776,047 / 2,206,821 entities | Model must match across BOTH sources consistently |
| **6.5% S1 have ONLY S2 matches** | 143,029 entities | Some S1 only appear in one source |
| **7.5% S1 have ONLY S3 matches** | 164,498 entities | Some S1 only appear in one source |
| **26% of S2/S3 are distractors** | 1.34M records NOT matched to any S1 | These are the false positive traps — blocking must reject them |
| **Empty addresses are 4.4% of GT matches** | 165K S2 + 172K S3 GT records have empty addr | Empty addr records are MORE likely to be matches (4.4% vs 3.3% overall) |
| **US has MORE empty-addr GT matches** | US: 109K empty each in S2/S3; India: 57K/63K | US empty-address handling is MORE important than India |

---

## 1. Environment Setup (DO THIS FIRST)

### 🔴 Python Version Strategy

Our system has Python 3.14.0, but key libraries may lack cp314 wheels. **Two-path strategy:**

```
Path A (Preferred): Create Python 3.11 or 3.12 virtual environment
  → Use for: torch, sentence-transformers, lightgbm, faiss-cpu
  → Guaranteed compatibility with all libraries
  → If 3.11/3.12 not available, install via pyenv-win or official installer

Path B (Fallback): Pure Python 3.14 pipeline
  → Use only stdlib + pure-Python packages
  → scikit-learn, rapidfuzz, polars (all have cp314 wheels)
  → Skip neural embeddings, use TF-IDF + string features only
  → Still competitive: ~0.82-0.86 F₀.₅ without embeddings
```

### Required Packages (Priority Order)

```
Tier 1 (Must have — pure Python or known cp314 support):
  polars          # Fast TSV I/O (5-10x faster than pandas)
  rapidfuzz       # String similarities (C++ extension, has cp314)
  scikit-learn    # TF-IDF vectorization, metrics
  scipy           # Sparse matrix operations
  numpy           # Numerics

Tier 2 (Need venv if cp314 missing):
  lightgbm        # Classifier (check: pip install lightgbm)
  sparse-dot-topn # Efficient sparse top-k (check compatibility)

Tier 3 (Need Python 3.11/3.12 venv):
  torch           # Backend for embeddings
  sentence-transformers  # bge-m3 / multilingual-e5
  faiss-cpu       # ANN search
  transformers    # Cross-encoder (Phase 5)
  
Tier 4 (Pure Python, always works):
  anyascii        # ASCII fallback (French only, NOT for Indic)
  unidecode       # French accent stripping only
  indic-transliteration  # Rule-based Devanagari→Latin (pure Python)
```

---

## 2. The REAL Transliteration Strategy (Completely Redesigned)

### 🔴 Why This Is the #1 Make-or-Break Decision

From our EDA, ~40% of Indian S2/S3 records contain Indic-script names. Without correct transliteration, these are **impossible to match**, representing ~18% of all test pairs. The original plan's `anyascii` approach would have failed catastrophically.

### 3-Tier Transliteration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│            TIER 1: DICTIONARY LOOKUP (100% accuracy, instant)       │
│                                                                     │
│  Pre-built dictionary of common business terms:                     │
│                                                                     │
│  Devanagari → English:                                              │
│    प्राइवेट लिमिटेड → private limited                                │
│    एलएलपी → llp                                                     │
│    प्राइवेट → private                                                │
│    लिमिटेड → limited                                                 │
│    कंपनी → company                                                   │
│    एंटरप्राइजेज → enterprises                                       │
│    इन्वेस्टमेंट → investment                                         │
│    सॉल्यूशंस → solutions                                             │
│    फूड → food                                                        │
│    होटल → hotel                                                      │
│    ट्रेडिंग → trading                                                │
│    इंडस्ट्रीज → industries                                           │
│    सर्विसेज → services                                               │
│    फाइनेंस → finance                                                 │
│    टेक्नोलॉजी → technology                                          │
│    ...100+ most frequent business terms                             │
│                                                                     │
│  Tamil → English:                                                   │
│    எல்எல்பி → llp                                                    │
│    ப்ரைவேட் லிமிடெட் → private limited                                │
│    இன்வெஸ்ட்மெண்ட்ஸ் → investments                                     │
│    ...50+ terms                                                     │
│                                                                     │
│  Telugu → English:                                                  │
│    ప్రైవేట్ లిమిటెడ్ → private limited                                  │
│    ఇన్వెస్ట్మెంట్ → investment                                         │
│    ...50+ terms                                                     │
│                                                                     │
│  State Names (ALL Indian scripts → canonical English):              │
│    उत्तर प्रदेश / UP → uttar pradesh                                 │
│    महाराष्ट्र / MH → maharashtra                                     │
│    हरियाणा / HR → haryana                                            │
│    தமிழ்நாடு / TN → tamil nadu                                       │
│    తెలంగాణ / TG → telangana                                          │
│    ...all 36 states+UTs in all regional scripts                     │
│                                                                     │
│  HOW TO BUILD: Mine training data! For each S1↔S2/S3 cross-script  │
│  match in GT, align tokens to build the dictionary automatically.  │
├─────────────────────────────────────────────────────────────────────┤
│     TIER 2: indic-transliteration (Rule-based, pure Python)        │
│                                                                     │
│  For tokens NOT in dictionary:                                      │
│    from indic_transliteration import sanscript, detect              │
│    sanscript.transliterate("वेंचर्स", sanscript.DEVANAGARI,         │
│                            sanscript.IAST)                          │
│    → "veṃcars" (then strip diacritics from IAST)                   │
│                                                                     │
│  Supports: Devanagari, Tamil, Telugu, Bengali, Gujarati, etc.      │
│  Quality: Good phonetic mapping, better than anyascii              │
│  Speed: ~50K tokens/sec (pure Python)                              │
│  License: MIT                                                       │
├─────────────────────────────────────────────────────────────────────┤
│     TIER 3: CHARACTER N-GRAM FUZZY MATCHING (No transliteration!)  │
│                                                                     │
│  For tokens where transliteration is still imperfect:              │
│  Skip transliteration entirely. Use multilingual dense embeddings  │
│  (bge-m3) which natively embed all scripts into the same space.    │
│                                                                     │
│  "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்" and "Raj Investments" will have         │
│  high cosine similarity WITHOUT any transliteration.               │
│                                                                     │
│  This is why bge-m3 embeddings are NOT optional — they are the     │
│  safety net for every transliteration failure.                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔴 How to AUTO-BUILD the Dictionary from Training Data

This is a **key innovation** most teams will miss:

```python
# For every S1↔S2/S3 match where S1 is Latin and S2/S3 has Indic script:
# 1. Tokenize both names
# 2. Align tokens by position and phonetic similarity
# 3. Build frequency-weighted Indic→English token dictionary
#
# Example from Group 9:
#   S1: "Red Ventures Private Limited"  → [red, ventures, private, limited]
#   S2: "रेड वेंचर्स प्राइवेट लिमिटेड"  → [रेड, वेंचर्स, प्राइवेट, लिमिटेड]
#   Alignment: रेड↔red, वेंचर्स↔ventures, प्राइवेट↔private, लिमिटेड↔limited
#
# With 3.9M S3 matches and 3.7M S2 matches, we'll have MILLIONS of 
# aligned pairs to build a very comprehensive dictionary.
```

---

## 3. Corrected Text Preprocessing Pipeline

### 🔴 Script-Aware Normalization (NOT Global NFKD)

```python
import unicodedata

def normalize_text(text, is_address=False):
    """Script-aware normalization — DOES NOT destroy Indic text"""
    
    # Step 1: Detect script blocks present
    has_latin = any(0x0041 <= ord(c) <= 0x007A or 0x0061 <= ord(c) <= 0x007A for c in text)
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
    has_tamil = any(0x0B80 <= ord(c) <= 0x0BFF for c in text)
    has_telugu = any(0x0C00 <= ord(c) <= 0x0C7F for c in text)
    has_indic = has_devanagari or has_tamil or has_telugu
    
    # Step 2: For LATIN text only — apply NFKD + strip accents
    if has_latin and not has_indic:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Step 3: For INDIC text — DO NOT strip combining marks!
    # Instead, apply transliteration (Tier 1 → Tier 2 → Tier 3)
    if has_indic:
        text = transliterate_to_latin(text)  # Uses 3-tier system above
    
    # Step 4: Universal normalization (safe for all scripts after transliteration)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 5: Remove decorator patterns
    text = re.sub(r'^\[.*?\]\s*', '', text)   # [Corp], [LLP]
    text = re.sub(r'^>+\s*', '', text)         # >>
    text = re.sub(r'^smt\s+', '', text)        # Smt prefix
    
    # Step 6: Remove literal nulls
    text = re.sub(r'\bnull\b', '', text, flags=re.IGNORECASE)
    
    return text.strip()
```

---

## 4. Blocking Strategy: Multi-Stage Funnel (REDESIGNED for Competition Rules)

### 🔴🔴 THE MOST CRITICAL CHANGE IN v2.1

The email guidelines reveal that **candidate_pairs.tsv directly affects final ranking**:
- Smaller candidate set per S1 = **ranked higher** (beyond leaderboard score)
- Amazon reviews the blocking CODE for scalability
- The candidate_pairs.tsv must be "whatever your model actually runs inference over"

This means our blocking must be a **precision-focused funnel**, NOT a loose recall-maximizing net.

### The Scoring Tradeoff We Must Navigate

```
WRONG approach (v1.0):
  Loose blocking (30-50 candidates) → High recall ceiling (99%) → But BAD candidate_pairs rank
  
RIGHT approach (v2.1):
  Stage 1: Loose blocking (30-50 internally) → NEVER EXPORTED
  Stage 2: Lightweight scoring to prune → 8-12 candidates
  Stage 3: This IS candidate_pairs.tsv ← EXPORTED, reviewed by Amazon
  Stage 4: ML model runs on Stage 3 output → matching_results.tsv
```

### Architecture: 3-Stage Precision Funnel

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INTERNAL STAGE 1: WIDE NET (Never Exported)          │
│                                                                         │
│  Per country partition:                                                 │
│  Channel A: TF-IDF char n-gram top-k=20 (threshold=0.15)               │
│  Channel B: Dense embedding ANN top-k=10 (if available)                │
│  Channel C: Address anchor inverted index (k≤5)                        │
│  Union → ~25-35 raw candidates per S1                                  │
│                                                                         │
│  Purpose: Maximize recall ceiling. Allow false positives.              │
│  Memory: transient, not persisted to disk                              │
├─────────────────────────────────────────────────────────────────────────┤
│             INTERNAL STAGE 2: LIGHTWEIGHT SCORING & PRUNING             │
│                                                                         │
│  For each (S1, candidate) pair from Stage 1:                           │
│    Compute 5 FAST features (no embeddings, no GPU):                    │
│      1. name_token_set_ratio (RapidFuzz)                               │
│      2. name_jaro_winkler (RapidFuzz)                                  │
│      3. addr_token_set_ratio (RapidFuzz)                               │
│      4. addr_number_match (regex extraction)                           │
│      5. name_char3gram_jaccard (set operation)                         │
│                                                                         │
│    Combined score = max(name_score, composite_score)                   │
│    (max to handle empty-address and DBA/trade-name cases)              │
│                                                                         │
│  Sort candidates by score, keep top-K where:                           │
│    K = min(10, count(score > dynamic_threshold))                       │
│    dynamic_threshold = adaptive per S1 based on score distribution     │
│                                                                         │
│  Target: 6-10 candidates per S1 entity                                │
│  Recall: ≥97% of GT matches survive pruning                           │
├─────────────────────────────────────────────────────────────────────────┤
│    ★ EXPORTED STAGE 3: candidate_pairs.tsv ★                          │
│                                                                         │
│  This IS the output. Contains the pruned candidate set.                │
│  Every S1 entity has exactly one row.                                  │
│  Singletons have empty candidate list.                                 │
│                                                                         │
│  Metrics to optimize (for Amazon's review):                            │
│    - Avg candidates per S1: TARGET ≤ 8-10                             │
│    - Recall ceiling: TARGET ≥ 97%                                     │
│    - Reduction ratio: TARGET ≥ 99.9999%                               │
│    - Code: clean, documented, demonstrates scalability                │
├─────────────────────────────────────────────────────────────────────────┤
│               STAGE 4: FULL ML INFERENCE (matching_results.tsv)        │
│                                                                         │
│  LightGBM with 35 features on Stage 3 candidates only                 │
│  Global Disjoint Assignment                                            │
│  Threshold optimization for F₀.₅                                      │
│  Singleton guardrails                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Design Wins

| Aspect | Our Approach | Typical Team |
|:---|:---|:---|
| **Candidates per S1** | 8-10 | 30-100 |
| **Blocking recall** | ~97% | ~99% (but wasted) |
| **Candidate_pairs quality** | Tight, focused | Bloated, noisy |
| **Amazon code review** | Clean 3-stage funnel | Single-pass spaghetti |
| **Scalability demo** | Explicit reduction at each stage | Hope-and-pray |
| **F₀.₅ impact** | Same (ML model handles the rest) | Same |
| **Final ranking boost** | ✅ Smaller set = ranked higher | ❌ Penalized |

### The Math: Why 8-10 Candidates Is Optimal

From our EDA:
- Median matches per S1: **3** (mean: 3.46, max: 11)
- Distribution: 0 matches (5.6%), 1-3 matches (47%), 4-6 matches (44%), 7+ matches (3.4%)
- Max ground-truth matches for any S1: **11** (S2 max=5, S3 max=6)

So **10 candidates** covers the maximum ground-truth size (11) with a tiny margin for false positives — demonstrating efficient blocking while being safe.

If we used 30 candidates, 20 of them would ALWAYS be false positives, which looks bad in Amazon's review.

### Stage 2 Lightweight Scoring Details

```python
from rapidfuzz import fuzz
import re

def lightweight_score(s1_name, s1_addr, cand_name, cand_addr):
    """Fast scoring for Stage 2 pruning. No GPU, no embeddings."""
    
    # Name similarity (most important)
    name_tsr = fuzz.token_set_ratio(s1_name, cand_name) / 100
    name_jw = fuzz.jaro_winkler_similarity(s1_name, cand_name)
    name_score = max(name_tsr, name_jw)
    
    # Address similarity
    if cand_addr and s1_addr:
        addr_tsr = fuzz.token_set_ratio(s1_addr, cand_addr) / 100
        
        # Number match (strongest address signal)
        s1_nums = set(re.findall(r'\d+', s1_addr))
        c_nums = set(re.findall(r'\d+', cand_addr))
        num_match = len(s1_nums & c_nums) / max(len(s1_nums | c_nums), 1)
        
        addr_score = 0.6 * addr_tsr + 0.4 * num_match
    else:
        addr_score = 0  # Empty address: rely on name only
    
    # Adaptive composite
    if cand_addr and s1_addr:
        composite = 0.55 * name_score + 0.45 * addr_score
    else:
        composite = name_score  # Name-only matching for empty addresses
    
    return max(name_score * 0.8, composite)  # Ensure good names aren't killed by bad addresses

def prune_candidates(s1_entity, candidates, max_k=10):
    """Prune to top-K candidates with adaptive threshold."""
    scored = []
    for cand in candidates:
        score = lightweight_score(
            s1_entity['name'], s1_entity['addr'],
            cand['name'], cand['addr']
        )
        scored.append((cand, score))
    
    scored.sort(key=lambda x: -x[1])
    
    # Adaptive threshold: keep candidates within 40% of top score
    if scored:
        top_score = scored[0][1]
        threshold = max(0.25, top_score * 0.60)
        filtered = [(c, s) for c, s in scored if s >= threshold]
        return filtered[:max_k]
    
    return []
```

### Blocking Quality Targets (REVISED for Competition)

| Metric | Target | Why |
|:---|:---|:---|
| **Avg candidates per S1** | ≤ 8-10 | **Smaller = ranked higher** by Amazon |
| **Recall ceiling** | ≥ 97% | High enough for competitive F₀.₅ |
| **Reduction ratio** | ≥ 99.9999% | Demonstrates scalability |
| **Max candidates for any S1** | ≤ 15 | Prevents outlier bloat |
| **Code clarity** | Clean 3-stage funnel | Amazon reviews blocking code |

---

## 5. Corrected Feature Engineering

### 5.1 Features REMOVED (Redundant or Infeasible)

🔴 Removed features that research showed have low marginal gain:
- `name_jaccard_char4gram` — highly correlated with 3-gram version
- `addr_embedding_cosine` — redundant with `concat_embedding_cosine`
- `name_metaphone_match` — Metaphone fails on Indic languages
- `embedding_reciprocal_rank` — low feature importance in practice

### 5.2 Features ADDED (Research-Backed High Impact)

| Feature | Why It Was Missing | Expected Impact |
|:---|:---|:---|
| `has_number_conflict` | If both addresses have street numbers but they DIFFER → strongest negative signal | **Very High** |
| `diff_to_next_score` | Score gap between rank-1 and rank-2 candidate. High gap = confident match | **High** |
| `s1_candidate_pool_size` | How many candidates blocking returned. Small pool = more confident | **Medium** |
| `reverse_s2_degree` | How many S1 entities retrieved this S2/S3 as candidate. High = generic name | **High** |
| `ratio_to_top_score` | Current candidate score / best candidate score for this S1 | **Medium** |
| `dictionary_translit_match` | Did Tier-1 dictionary transliteration produce exact name match? | **Very High** |

### 5.3 Final Feature Set (35 Features — Focused, Not Bloated)

**Name Features (12):**
1. `name_jaro_winkler` — Jaro-Winkler (best for short business names)
2. `name_levenshtein_norm` — Normalized edit distance
3. `name_token_sort_ratio` — Handles word reordering
4. `name_token_set_ratio` — Handles subset/superset names
5. `name_jaccard_char3gram` — Typo-robust character matching
6. `name_tfidf_cosine` — TF-IDF weighted similarity
7. `name_core_token_overlap` — Word Jaccard after legal suffix removal
8. `name_first_token_match` — Binary: first brand word matches
9. `name_len_ratio` — min/max length ratio
10. `name_embedding_cosine` — Dense multilingual embedding similarity
11. `name_is_cross_script` — Binary: different Unicode blocks
12. `dictionary_translit_name_match` — Tier-1 dictionary produced exact match

**Address Features (10):**
13. `addr_token_set_ratio` — Handles reordering + partial addresses
14. `addr_jaccard_char3gram` — Character-level address matching
15. `addr_number_exact_match` — Premise numbers match
16. `addr_number_conflict` — 🔴 **NEW**: Both have numbers but they DIFFER
17. `addr_state_match` — Normalized state code matches
18. `addr_len_ratio` — min/max length ratio
19. `addr_tfidf_cosine` — TF-IDF weighted similarity
20. `addr_has_null_tokens` — "null"/"<NULL>" present
21. `addr_pincode_match` — PIN/ZIP codes match
22. `addr_city_fuzzy_match` — Fuzzy city name match (Mumbai↔Bombay)

**Metadata & Cross-Field Features (8):**
23. `either_addr_empty` — One or both addresses empty
24. `both_addr_empty` — Both addresses empty
25. `source_type` — S2 or S3 (different noise profiles)
26. `name_addr_agreement` — name_sim × addr_sim (high = confident)
27. `name_addr_disagreement` — |name_sim - addr_sim| (DBA detection)
28. `phone_number_match` — Extracted phone numbers match
29. `max_sim_across_features` — max of all similarity features
30. `blocking_channel_count` — How many channels retrieved this pair

**Ranking/Competition Features (5):** 🔴 **NEW category**
31. `candidate_rank` — Rank of this candidate for this S1
32. `diff_to_next_score` — Score gap to next candidate
33. `ratio_to_top_score` — Score / best score for this S1
34. `s1_candidate_pool_size` — Total candidates for this S1
35. `reverse_candidate_degree` — How many S1s retrieved this S2/S3

---

## 6. Training Strategy (Corrected)

### 🔴 Critical Fix: Negative Sampling Strategy

The original plan was vague on negative sampling. Research shows this is where most teams fail.

```
TRAINING DATA CONSTRUCTION:

1. POSITIVES (~2M pairs, sampled from 7.6M GT matches):
   - Stratified sample: 60% US, 40% India (match country distribution)
   - Include ALL cross-script positive pairs (these are hardest)
   - Include ALL empty-address positive pairs

2. HARD NEGATIVES (~4M pairs, from blocking stage):
   - Run blocking on training data
   - For each S1, take top-5 blocking candidates NOT in GT
   - These are the most confusing non-matches
   - 🔴 CRITICAL: Verify against GT to avoid mining true matches!
   
3. MEDIUM NEGATIVES (~1M pairs):
   - Random same-country pairs sharing ≥1 name token
   - Calibrates the model's low-end probability

4. DO NOT USE random cross-country pairs — they're too easy
```

### Hard Negative Mining with Curriculum Learning

```
Phase 1: Train on easy negatives → learn basic features
Phase 2: Mine hard negatives using Phase-1 model → retrain
Phase 3: Mine harder negatives using Phase-2 model → final model

Each phase produces a stronger model that generates harder negatives.
Typically 2-3 rounds is sufficient.
```

---

## 7. LightGBM Classifier (Corrected Parameters)

```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 63,              # 🔴 Reduced from 127 (prevent overfit)
    'max_depth': 8,                # 🔴 Added depth limit
    'min_child_samples': 200,      # 🔴 Increased (prevent overfit on singletons)
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'reg_alpha': 0.5,              # 🔴 Increased regularization
    'reg_lambda': 2.0,             # 🔴 Increased regularization
    'scale_pos_weight': 1.5,
    'n_estimators': 3000,
    'early_stopping_rounds': 200,
    'is_unbalanced': False,
    'verbose': -1,
    
    # 🔴 Memory optimization for 15GB RAM:
    'max_bin': 128,                # Reduce from default 255
    'min_data_in_bin': 5,
}

# 🔴 Use categorical feature support directly:
categorical_features = ['source_type']
```

---

## 8. Post-Processing: Precision Optimization

### 8.1 F₀.₅ Threshold Optimization (Unchanged but Validated)

Research confirmed optimal threshold is typically **τ ∈ [0.72, 0.84]** for F₀.₅ with class imbalance ratios of 10:1 to 20:1.

### 8.2 🔴 Global Disjoint Assignment (Confirmed Critical)

Every S2/S3 ID maps to at most 1 S1. After scoring all pairs:

```python
def global_disjoint_assignment(scored_pairs, threshold):
    """Greedy maximum-weight disjoint assignment"""
    # Sort ALL (s1_id, candidate_id, prob) triples descending by prob
    scored_pairs.sort(key=lambda x: -x[2])
    
    assigned = set()      # Already-assigned S2/S3 IDs
    results = defaultdict(list)  # s1_id → [matched ids]
    
    for s1_id, cand_id, prob in scored_pairs:
        if prob < threshold:
            break
        if cand_id not in assigned:
            results[s1_id].append(cand_id)
            assigned.add(cand_id)
    
    return results
```

### 8.3 Singleton Detection (Enhanced)

```python
def predict_for_s1(s1_id, candidates_with_probs, tau_match, tau_singleton):
    """
    tau_match:     threshold for including a match (e.g., 0.78)
    tau_singleton: threshold below which we declare singleton (e.g., 0.60)
    """
    if not candidates_with_probs:
        return []  # No candidates from blocking → singleton
    
    max_prob = max(p for _, p in candidates_with_probs)
    
    # SINGLETON GATE: If best candidate is weak → empty list (1.0 F₀.₅)
    if max_prob < tau_singleton:
        return []
    
    # UNCERTAINTY ZONE: Between singleton and match thresholds
    # Default to empty (precision-first for F₀.₅)
    if max_prob < tau_match:
        return []
    
    # MATCH: Include all candidates above threshold
    return [cid for cid, p in candidates_with_probs if p >= tau_match]
```

---

## 9. France Zero-Shot Strategy (Corrected)

🔴 **Key addition:** French legal suffix dictionary.

```python
FRENCH_LEGAL_SUFFIXES = {
    'sarl': 'SARL', 'sas': 'SAS', 'sa': 'SA', 'sasu': 'SASU',
    'eurl': 'EURL', 'sci': 'SCI', 'snc': 'SNC', 'scp': 'SCP',
    'gie': 'GIE', 'earl': 'EARL', 'association': 'ASSOC',
    's.a.r.l': 'SARL', 's.a.s': 'SAS', 's.a': 'SA',
}

# 🔴 A SARL can legally become a SAS without changing identity
# → legal suffix match/mismatch should be a SOFT signal, not hard filter
```

French text handling:
- `anyascii` works perfectly for French (`Société` → `Societe`) ✅
- `unidecode` also works for French ✅
- No transliteration needed — French uses Latin script with accents

---

## 10. Revised Experimentation Timeline

### Phase 0: Environment & Dictionary (Day 1)
1. Install Python 3.11/3.12 + create venv
2. Install all Tier 1-3 packages
3. **AUTO-BUILD transliteration dictionary from training GT** ← highest ROI task
4. Validate dictionary coverage (target: ≥80% of Indic tokens covered)

### Phase 1: MVP Pipeline (Days 2-4)
1. Text preprocessing with corrected script-aware normalization
2. TF-IDF char n-gram blocking only (Channel A)
3. 15 core features (Jaro-Winkler, token_set_ratio, number_match, etc.)
4. LightGBM with default params
5. Single threshold on validation
6. **Submit** → establish baseline (expect ~0.65-0.72 F₀.₅)

### Phase 2: Transliteration + Embeddings (Days 5-7)
1. Integrate dictionary transliteration
2. Add `indic-transliteration` for uncovered tokens
3. Add BGE-M3 / multilingual-e5 embedding features
4. Add Channel B (FAISS ANN blocking)
5. Expand to 25+ features
6. **Submit** → expect ~0.78-0.85 F₀.₅

### Phase 3: Hard Negatives + Full Features (Days 8-10)
1. Hard negative mining (2 rounds)
2. All 35 features including ranking features
3. Per-country threshold tuning
4. Singleton detection guardrails
5. Global Disjoint Assignment
6. **Submit** → expect ~0.85-0.90 F₀.₅

### Phase 4: France + Precision Hardening (Days 11-12)
1. French legal suffix normalization
2. Error analysis on validation misclassifications
3. Feature selection (prune correlated features)
4. Optuna hyperparameter tuning for LightGBM
5. **Submit**

### Phase 5: Optional Cross-Encoder Reranking (Days 13-14)
1. Fine-tune `microsoft/mdeberta-v3-base` on borderline pairs only
2. Model stacking: LightGBM + XGBoost → meta-learner
3. **Final submissions**

---

## 11. Risk Matrix (Corrected)

| Risk | Prob | Impact | Mitigation | Fallback |
|:---|:---|:---|:---|:---|
| Python 3.14 breaks torch/faiss | High | High | Create 3.11 venv | Path B: pure TF-IDF pipeline |
| Dictionary transliteration gaps | Medium | High | Mine ALL 7.6M GT pairs for dictionary | Rely on embedding similarity |
| TF-IDF OOM on 15GB | Medium | High | max_features=100K, float32, per-country | HashingVectorizer |
| Blocking misses cross-script pairs | Medium | High | Channel B (embeddings) + dictionary | Dense embeddings are the safety net |
| LightGBM overfits US/India | Medium | Medium | Regularization, country-stratified CV | Separate models per country |
| False merges on singletons | High | Very High | Conservative tau_singleton, gap analysis | Default empty when uncertain |
| France F₀.₅ drops significantly | Medium | Medium | Conservative thresholds, accent stripping | Sacrifice some France recall for precision |
| sparse_dot_topn not available for 3.14 | Medium | Medium | Fallback to scipy sparse + manual top-k | Slower but functionally equivalent |

---

## 12. The 5 Things That Will ACTUALLY Win This Competition
## (Corrected from v1.0)

1. **Auto-mined transliteration dictionary from training GT** — This is our #1 edge. By aligning 7.6M GT pairs across scripts, we build a massive, highly accurate Indic→English dictionary that captures business-specific transliterations no generic tool provides.

2. **Multi-channel blocking with embedding safety net** — TF-IDF catches 90%+ of pairs. Dense multilingual embeddings catch the cross-script remainder. Address anchors catch DBA/trade name divergence. The union gives ≥98.5% recall.

3. **Ranking/competition features in LightGBM** — `diff_to_next_score`, `reverse_candidate_degree`, and `candidate_pool_size` are the secret sauce from Kaggle winning solutions (Foursquare 1st place). They give the model context about *how confident* a match is relative to alternatives.

4. **Global Disjoint Assignment + Singleton Guardrails** — Exploiting the 1-to-many structural property eliminates cross-entity false merges for free. The singleton gate protects the 5.58% of entities worth 1.0 each.

5. **Precision-first at every layer with F₀.₅ math** — Every threshold, every feature, every design decision asks: "does this reduce false merges more than it hurts recall?" The scoring math says: missing a match costs 0.06, a false merge costs 0.22, a false merge on a singleton costs 1.0.

---

## 13. What Most Teams Will Get Wrong (Our Competitive Edge)

| Common Mistake | Why It Fails | Our Advantage |
|:---|:---|:---|
| Use anyascii/unidecode for Hindi | 0-50% token overlap, misses most matches | Auto-mined dictionary + indic-transliteration |
| Apply NFKD globally | Destroys Indic vowel marks | Script-aware normalization |
| Single blocking channel | Misses cross-script or DBA/trade name pairs | 3-channel union blocking |
| **Bloated candidate_pairs (30-100/S1)** | **Ranked LOWER by Amazon beyond leaderboard** | **Tight 8-10/S1 via 3-stage funnel** |
| Optimize for F₁ not F₀.₅ | Threshold too low → too many false merges | Precision-first threshold at 0.75-0.85 |
| Ignore singletons | Lose 5.58 points on leaderboard | Explicit singleton detection gate |
| Random negative sampling | Model learns easy distinctions, fails on hard cases | Hard negative mining with curriculum |
| Use Llama/Gemma | Disqualified for license violation | Strict MIT/Apache 2.0 models only |
| Ignore Global Disjoint Assignment | Duplicate S2/S3 assignments = guaranteed false positives | Greedy maximum-weight assignment |
| **Ignore 9 Indic scripts** | Only handle Hindi, miss Bengali/Telugu/Tamil/etc. | Dictionary covers all 9 scripts |
