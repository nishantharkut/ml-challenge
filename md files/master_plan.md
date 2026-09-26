# 🏆 Amazon ML Challenge 2026 — Master Battle Plan
## Business Entity Resolution: Top-10 Strategy

---

## 0. Executive Summary

We are solving a **large-scale multilingual entity resolution** problem: matching ~1.7M Source 1 business entities against ~5M Source 2 and ~5M Source 3 records across **US, India, and France** (France is unseen in training). The metric is **macro-averaged F₀.₅** — precision is weighted **2× over recall**. Each correct singleton (no-match) contributes a full 1.0 to the score. The model constraint is **≤ 8B parameters, MIT/Apache 2.0 license**.

**Our winning formula:** A 4-stage cascading pipeline with **multi-channel blocking** (TF-IDF + dense embeddings + phonetic) → **rich feature engineering** (50+ features) → **LightGBM cross-pair classifier** → **precision-tuned threshold with singleton guardrails**.

---

## 1. Critical Problem Constraints & Insights

### 1.1 What Makes This Hard (From EDA)

| Challenge | Evidence | Impact |
|:---|:---|:---|
| **Scale** | 1.7M S1 × 10M (S2+S3) = 17 trillion raw pairs | Blocking is the #1 bottleneck |
| **Cross-script matching** | Hindi (Devanagari), Tamil, Telugu transliterations of English business names | Standard string similarity fails completely |
| **Heavy name corruption** | Typos ("Etrepndiels"), word reordering ("Hendricks and Inc Flowers"), DBA trade names ("Dréxkor"), URLs as names, phone numbers in names | Need multiple similarity signals |
| **Address chaos** | Component reordering, literal `null`/`<NULL>`, missing components, landmark references, ordinal errors ("45th" → "45ND"), "St" → "SAINT" | Token-set matching, not exact match |
| **Empty addresses** | 3.3% of S2/S3 have empty addresses — and these ARE ground-truth matches | Address cannot be a mandatory blocking key |
| **Singletons matter** | 5.58% of S1 have zero matches → each correct singleton = 1.0 F₀.₅ | Must actively detect singletons |
| **France (zero-shot)** | 15% of test is France — never seen in training | Pipeline must be language-agnostic |
| **Precision > Recall** | F₀.₅ weights precision 2×. A single false merge on a singleton costs 1.0 → 0.0 | Conservative thresholds, high-confidence only |

### 1.2 Key Structural Properties (From Verified EDA)

- **Country partitions matching perfectly**: 0 cross-country GT pairs → **hard-block by country** (0% recall loss, ~60% search reduction)
- **Each S2/S3 maps to at most 1 S1**: No overlapping clusters → simplified matching
- **Median 3-4 matches per S1**: Most entities have 2–6 matches across S2+S3
- **No missing business names**: 0 empty names across all 24.2M records
- **Country labels are clean**: No typos, no ISO variants, just `{US, India, France}`

### 1.3 The Scoring Math That Drives Our Strategy

For F₀.₅ = (1.25 × P × R) / (0.25 × P + R):

| Scenario | Precision | Recall | F₀.₅ | Lesson |
|:---|:---|:---|:---|:---|
| Perfect match | 1.00 | 1.00 | 1.000 | — |
| 1 false positive out of 4 | 0.75 | 1.00 | 0.781 | One bad merge hurts badly |
| Miss 1 out of 4 true matches | 1.00 | 0.75 | 0.938 | Missing a match is cheap |
| Correct singleton (empty prediction) | — | — | 1.000 | Free points |
| False merge on singleton (predict 1 match) | 0.00 | — | 0.000 | Catastrophic: 1.0 → 0.0 |

> **Key Insight**: Missing a true match costs ~0.06. A false merge costs ~0.22. A false merge on a singleton costs 1.0. **Our pipeline must be precision-first at every stage.**

---

## 2. Architecture: 4-Stage Cascading Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STAGE 0: PREPROCESSING                       │
│  Text normalization · Cross-script transliteration · Feature prep   │
├─────────────────────────────────────────────────────────────────────┤
│                   STAGE 1: MULTI-CHANNEL BLOCKING                   │
│  Country partition → Union of 3 independent blocking channels       │
│  Channel A: TF-IDF char n-gram (name+addr) → sparse top-k          │
│  Channel B: Dense embedding ANN (multilingual-e5) → FAISS top-k    │
│  Channel C: Phonetic/token blocking (Soundex + exact token overlap) │
│  Target: ≤50 candidates per S1 entity, ≥99% recall                 │
├─────────────────────────────────────────────────────────────────────┤
│                    STAGE 2: FEATURE ENGINEERING                      │
│  50+ pairwise features across name, address, and metadata           │
│  String similarities · Token overlaps · Phonetic · Embedding cosine │
│  Adaptive weighting for empty-address and cross-script cases        │
├─────────────────────────────────────────────────────────────────────┤
│               STAGE 3: LIGHTGBM PAIRWISE CLASSIFIER                │
│  Binary classifier: match / no-match                                │
│  Trained on train GT pairs (positives) + hard negatives from Stage 1│
│  Output: calibrated match probability per candidate pair            │
├─────────────────────────────────────────────────────────────────────┤
│             STAGE 4: PRECISION-TUNED DECISION LAYER                 │
│  F₀.₅-optimized threshold on validation set                        │
│  Singleton detection guardrails                                     │
│  Per-country threshold tuning (US, India, France)                   │
│  Output: matching_results.tsv + candidate_pairs.tsv                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 0: Preprocessing — The Foundation

### 3.1 Text Normalization Pipeline

Applied to **both name and address** fields across all sources:

```
Input text
  │
  ├── Unicode NFKD normalization (strip diacritics: é→e, ó→o, á→a, É→E)
  ├── Lowercase everything
  ├── Strip decorator patterns: [Corp], [LLP], >>, Smt, Dr prefix
  ├── Remove literal null tokens: "null", "<null>", "<NULL>", "<none>"
  ├── Normalize whitespace (collapse multiple spaces, strip leading/trailing)
  ├── Normalize punctuation (&→and, +→and, .→ , -→ )
  ├── Extract and remove phone numbers (10-digit patterns → store as feature)
  ├── Extract and remove URLs (.com, .in, .fr → store domain as feature)
  │
  └── Output: cleaned_text
```

### 3.2 Business Name Normalization

Additional name-specific processing:

```
cleaned_name
  │
  ├── Legal suffix normalization dictionary:
  │     inc/incorporated/incorporation → INC
  │     corp/corporation → CORP
  │     llc/l.l.c → LLC
  │     llp/l.l.p → LLP
  │     ltd/limited → LTD
  │     pvt/private → PVT
  │     co/company → CO
  │
  ├── Extract and store legal suffix separately (for suffix-match feature)
  ├── Remove legal suffix from core name tokens
  ├── Sort remaining tokens alphabetically → "canonical_name_tokens"
  │     (handles word reordering: "Hendricks and Inc Flowers" → "and flowers hendricks")
  │
  └── Output: (core_name, legal_suffix, canonical_tokens, original_tokens)
```

### 3.3 Address Normalization

```
cleaned_address
  │
  ├── Street suffix dictionary:
  │     st/street/saint(when after number) → ST
  │     ave/avenue → AVE
  │     rd/road → RD
  │     dr/drive → DR
  │     blvd/boulevard → BLVD
  │     ter/terrace → TER
  │
  ├── State normalization dictionary (US + India):
  │     US: oh→ohio, ny→new york, ca→california, etc. (both directions → canonical form)
  │     India: up→uttar pradesh, tn→tamil nadu, tg→telangana, rj→rajasthan,
  │            dl→delhi, mh→maharashtra, hr→haryana, etc.
  │     India (Indic script states): उत्तर प्रदेश→uttar pradesh, तमिழ்நாடு→tamil nadu,
  │            महाराष्ट्र→maharashtra, हरियाणा→haryana, etc.
  │     France: standard département/region normalization
  │
  ├── Historical city names: bombay→mumbai, madras→chennai, calcutta→kolkata, etc.
  │
  ├── Extract numeric tokens (house/premise numbers) → store separately
  ├── Remove ordinal suffixes: 1st/2nd/3rd/4th/45th/45nd → just the number
  ├── Remove leading zeros from numbers: 0684→684, 01604→1604
  ├── Remove PO BOX / PMB tokens → store as feature
  ├── Remove premise prefixes: "H.NO", "DOOR NO", "BLOCK", "PLOT NO" → keep just the number
  │
  └── Output: (normalized_addr_tokens, extracted_numbers, state, city)
```

### 3.4 Cross-Script Transliteration Pipeline (CRITICAL for India + France)

This is the **single highest-impact preprocessing step** for India records.

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSLITERATION STRATEGY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Detect script of each token using Unicode blocks        │
│    - Devanagari: U+0900–U+097F                                   │
│    - Tamil:      U+0B80–U+0BFF                                   │
│    - Telugu:     U+0C00–U+0C7F                                   │
│    - Latin:      U+0000–U+007F (ASCII)                           │
│    - French:     Latin + accented chars (handled by NFKD)        │
│                                                                  │
│  Step 2: Transliterate non-Latin tokens to Latin                 │
│    Option A (Preferred for quality): ai4bharat IndicXlit model      │
│      - MIT license, supports 21 Indian languages                 │
│      - Handles Hindi, Tamil, Telugu phonetic transliteration     │
│      - "एसएस फूड प्राइवेट लिमिटेड" → "ss food private limited"  │
│      - "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி" → "raj investments llp"│
│      - Slower: requires model loading, ~1K words/sec per CPU     │
│                                                                  │
│    Option A-alt (Preferred for speed): anyascii library           │
│      - Pure Python, zero deps, >150K strings/sec per CPU core    │
│      - Character-level Unicode→ASCII mapping (no ML model)       │
│      - "रेड वेंचर्स" → "red venchars" (approximate but fast)     │
│      - Quality lower than IndicXlit but 150× faster              │
│      - USE THIS for blocking; use IndicXlit for features         │
│                                                                  │
│    Option B (Fallback): indic-transliteration (rule-based)       │
│      - Lightweight, no GPU needed                                │
│      - Lower quality but faster and more predictable             │
│                                                                  │
│    Option C (Complementary): unidecode library                   │
│      - Handles French accented characters                        │
│      - "Société Générale" → "Societe Generale"                   │
│                                                                  │
│  Step 3: Build "romanized" version of every record               │
│    - All downstream features computed on romanized text           │
│    - Original script preserved for script-match feature          │
│                                                                  │
│  IMPORTANT: Transliterate BOTH name AND address                  │
│    - Address state names appear in Devanagari (हरियाणा, महाराष्ट्र)│
│    - Address locality names appear in Indic scripts              │
└─────────────────────────────────────────────────────────────────┘
```

> **Why this is #1 priority**: In our sample, 6 out of 7 Indian groups had cross-script matches. Without transliteration, these are **impossible** to match using any string similarity. This single step likely separates top-10 from top-50.

---

## 4. Stage 1: Multi-Channel Blocking

### 4.1 Design Principle: Union of Channels

No single blocking strategy catches everything. Our EDA reveals:
- **Empty addresses** → TF-IDF on name alone must work
- **DBA/trade names** (e.g., "Dréxkor" = "Maure Williams Colombier") → Only address or embedding similarity catches these
- **Cross-script** → Only works after transliteration + embedding

We use the **union** of 3 independent blocking channels. A candidate pair enters the feature engineering stage if **any channel** retrieves it.

### 4.2 Channel A: TF-IDF Character N-Gram Blocking (Primary)

**Why**: Character n-grams are robust to typos, word reordering, and partial matches. This is the workhorse channel with the best precision/speed tradeoff.

```python
# Pseudocode
from sklearn.feature_extraction.text import TfidfVectorizer

# Build separate TF-IDF matrices for name and address
# Use character n-grams (3,5) — robust to typos
name_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=500000)
addr_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=500000)

# For each country partition:
#   1. Fit vectorizer on S2+S3 records
#   2. Transform S1 records
#   3. Compute sparse cosine similarity: S1_matrix @ S2S3_matrix.T
#   4. For each S1 row, take top-k (k=30) candidates above threshold 0.15

# Combined blocking text = romanized_name + " | " + romanized_address
# This handles empty addresses gracefully (name tokens still present)
```

**Parameters to tune**:
- `ngram_range`: (3, 5) for names, (3, 4) for addresses
- `max_features`: 500K (balance memory vs. discriminative power)
- `top_k`: 30 candidates per S1 per source (S2, S3)
- `min_similarity`: 0.15 (very loose — high recall)

**Scalability**: For 1.7M × 5M, process in country-partitioned chunks:
- US: 663K × 1.9M = ~1.3T pairs → sparse multiplication handles this
- India: 810K × 2.3M = ~1.9T pairs
- France: 259K × 700K = ~181B pairs
Use `sklearn`'s sparse matrix multiplication + top-k extraction via `sparse_dot_topn` library (10-100× faster than naive).

### 4.3 Channel B: Dense Embedding ANN Blocking (Cross-Script Safety Net)

**Why**: TF-IDF character n-grams **cannot** match "Raj Investments" with "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி" even after transliteration, if transliteration quality is imperfect. Dense multilingual embeddings capture semantic similarity across scripts natively.

```python
# Use multilingual-e5-base or multilingual-e5-large (MIT license, <1B params)
# These embed text into a shared vector space across 100+ languages

from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('intfloat/multilingual-e5-base')  # 278M params, MIT

# Encode business records: "query: " + name + " " + address
# Build FAISS index per country partition
# Search top-k=20 nearest neighbors for each S1 record

# Key advantage: works across scripts WITHOUT transliteration
# "Raj Investments LLP" and "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்" map to nearby vectors
```

**Alternative models** (all MIT/Apache 2.0, under 8B):
- `intfloat/multilingual-e5-large` (560M params) — best quality
- `BAAI/bge-m3` (567M, **MIT**) — hybrid dense+sparse+ColBERT
- `intfloat/multilingual-e5-small` (118M, **MIT**) — ultra-fast
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (278M) — good fallback

> [!CAUTION]
> **License Compliance Alert**: The following popular models are **NOT** MIT/Apache 2.0 and risk disqualification:
> - **Llama 3.1 8B**: Llama Community License (custom Meta terms) → ❌
> - **Gemma 2 2B/9B**: Gemma Terms of Use (custom Google terms) → ❌
> - **Mistral 7B**: Apache 2.0 ✅ but too slow for 18M pair inference
>
> **Safe choices**: `bge-m3` (MIT), `multilingual-e5-*` (MIT), `mdeberta-v3-base` (MIT), `Qwen2.5-7B` (Apache 2.0), `Phi-3.5-mini` (MIT)

**FAISS Configuration**:
- `IndexIVFFlat` with `nlist=4096`, `nprobe=64` for speed
- L2-normalize vectors → inner product = cosine similarity
- top-k=20 per S1

### 4.4 Channel C: Phonetic + Token Blocking (Precision Booster)

**Why**: Catches cases where character n-grams and embeddings both miss, especially for short names with specific matching tokens.

```
For each S1 record:
  1. Generate Soundex/Metaphone codes for first 2 name tokens
  2. Build inverted index: {phonetic_code + country} → [entity_ids]
  3. Retrieve all S2/S3 records sharing ≥1 phonetic code in same country

  4. Also: exact token overlap blocking
     - Build inverted index on rare name tokens (TF-IDF token weight > threshold)
     - Retrieve S2/S3 records sharing ≥1 rare token

  5. Also: numeric blocking for addresses
     - Build inverted index on extracted house/premise numbers
     - Retrieve S2/S3 records sharing same premise number in same country
```

### 4.5 Blocking Quality Targets

| Metric | Target | Rationale |
|:---|:---|:---|
| **Recall ceiling** | ≥ 99.0% | Missing a true pair = permanent recall loss |
| **Candidates per S1** | ≤ 50 | Keep Stage 2 feature computation tractable |
| **Reduction ratio** | > 99.99% | From 10M to ~50 candidates = 5 orders of magnitude |

**Validation**: Measure blocking recall on a held-out 20% of training GT.

---

## 5. Stage 2: Feature Engineering — 50+ Pairwise Features

For each (S1, candidate) pair surviving blocking, compute:

### 5.1 Name Similarity Features (~20 features)

| # | Feature | Method | Why |
|:--|:---|:---|:---|
| 1 | `name_jaccard_char3gram` | Jaccard on character 3-gram sets | Robust to typos |
| 2 | `name_jaccard_char4gram` | Jaccard on character 4-gram sets | Slightly more specific |
| 3 | `name_cosine_tfidf` | Cosine similarity on TF-IDF char n-gram vectors | Weighted by token rarity |
| 4 | `name_levenshtein_norm` | Normalized Levenshtein distance (RapidFuzz) | Edit distance on full name |
| 5 | `name_jaro_winkler` | Jaro-Winkler similarity (RapidFuzz) | Prefix-weighted edit distance |
| 6 | `name_token_sort_ratio` | RapidFuzz `token_sort_ratio` | Handles word reordering |
| 7 | `name_token_set_ratio` | RapidFuzz `token_set_ratio` | Handles subset names |
| 8 | `name_partial_ratio` | RapidFuzz `partial_ratio` | Handles truncated names |
| 9 | `name_embedding_cosine` | Cosine of dense embeddings (multilingual-e5) | Cross-script semantic similarity |
| 10 | `name_soundex_match` | Boolean: Soundex codes match for ≥1 token | Phonetic similarity |
| 11 | `name_metaphone_match` | Boolean: Metaphone codes match | Alternative phonetic |
| 12 | `core_name_exact_match` | Boolean: core name tokens identical after normalization | High-precision signal |
| 13 | `name_token_overlap_ratio` | |intersection| / |union| of word tokens | Word-level Jaccard |
| 14 | `name_first_token_match` | Boolean: first word token matches | Companies often share first word |
| 15 | `legal_suffix_match` | Boolean: legal suffixes match (Inc=Inc, LLC=LLC) | Soft signal |
| 16 | `legal_suffix_missing` | Boolean: one record has suffix, other doesn't | Common noise pattern |
| 17 | `name_len_ratio` | min(len)/max(len) of name strings | Catch truncated names |
| 18 | `name_rare_token_overlap` | Count of shared tokens with high TF-IDF weight | Discriminative matching |
| 19 | `name_is_cross_script` | Boolean: records use different Unicode blocks | Flag for model |
| 20 | `romanized_name_levenshtein` | Levenshtein on transliterated names | Cross-script edit distance |

### 5.2 Address Similarity Features (~20 features)

| # | Feature | Method | Why |
|:--|:---|:---|:---|
| 21 | `addr_jaccard_char3gram` | Jaccard on char 3-gram sets | Typo-robust address matching |
| 22 | `addr_cosine_tfidf` | Cosine on TF-IDF vectors | Weighted similarity |
| 23 | `addr_token_set_ratio` | RapidFuzz `token_set_ratio` | Handles reordering + partial addresses |
| 24 | `addr_token_sort_ratio` | RapidFuzz `token_sort_ratio` | Ignores component order |
| 25 | `addr_levenshtein_norm` | Normalized Levenshtein | Full-string edit distance |
| 26 | `addr_number_match` | Boolean: extracted premise numbers match | Strongest address signal |
| 27 | `addr_number_partial` | Boolean: one number is substring of other (1056 in 1056c) | Handle number suffixes |
| 28 | `addr_state_match` | Boolean: normalized state matches | State partitioning signal |
| 29 | `addr_city_match` | Fuzzy match on extracted city tokens | Handle Mumbai↔Bombay |
| 30 | `addr_embedding_cosine` | Cosine of dense address embeddings | Semantic address similarity |
| 31 | `addr_token_overlap_ratio` | Word token Jaccard | Handles reordering |
| 32 | `addr_len_ratio` | min(len)/max(len) | Detect truncated addresses |
| 33 | `addr_has_null_tokens` | Boolean: address contained "null"/<NULL> | Quality flag |
| 34 | `addr_pincode_match` | Boolean: PIN/ZIP codes match (if extractable) | High-precision signal for India |
| 35 | `addr_rare_token_overlap` | Count of shared rare address tokens | Discriminative matching |

### 5.3 Cross-Field & Metadata Features (~10 features)

| # | Feature | Method | Why |
|:--|:---|:---|:---|
| 36 | `country_match` | Boolean (always True after blocking, but encode for safety) | Partition signal |
| 37 | `source_type` | Categorical: S2 or S3 | S2 and S3 have different noise profiles |
| 38 | `either_addr_empty` | Boolean: one or both addresses are empty | Adaptive weighting trigger |
| 39 | `both_addr_empty` | Boolean: both addresses empty | Extreme case flag |
| 40 | `combined_score` | α×name_sim + (1-α)×addr_sim, α adaptive | Classic ER composite |
| 41 | `name_addr_agreement` | name_sim × addr_sim | Both signals strong = high confidence |
| 42 | `name_addr_disagreement` | |name_sim - addr_sim| | Detect DBA/trade name cases |
| 43 | `phone_number_match` | Boolean: extracted phone numbers match | Rare but very high precision |
| 44 | `blocking_channel_count` | How many blocking channels retrieved this pair (1, 2, or 3) | Confidence proxy |
| 45 | `max_name_sim` | max of all name similarity features | Summary statistic |
| 46 | `max_addr_sim` | max of all address similarity features | Summary statistic |

### 5.4 Embedding Features (~5 features)

| # | Feature | Method | Why |
|:--|:---|:---|:---|
| 47 | `concat_embedding_cosine` | Cosine of name+address embedding | Joint similarity |
| 48 | `name_embedding_rank` | Rank of this candidate in S1's top-k by name embedding | How unusual is this match |
| 49 | `addr_embedding_rank` | Rank of this candidate in S1's top-k by address embedding | How unusual is this match |
| 50 | `embedding_reciprocal_rank` | 1/rank in ANN search | Features from retrieval stage |

---

## 6. Stage 3: LightGBM Pairwise Classifier

### 6.1 Why LightGBM Over Deep Learning for Final Classification

| Criterion | LightGBM | Deep Learning (Cross-Encoder) |
|:---|:---|:---|
| **Speed at scale** | Score ~100M pairs in minutes | Score ~100M pairs in hours/days |
| **Feature interpretability** | Full feature importance | Black box |
| **Handles mixed feature types** | Natively | Requires all-text serialization |
| **Handles missing values** | Natively | Requires imputation |
| **Training data efficiency** | Excellent with 7.6M positive pairs | Would need fine-tuning |
| **F₀.₅ optimization** | Direct threshold tuning on probabilities | Same |
| **License** | MIT | — |

### 6.2 Training Data Construction

```
POSITIVE PAIRS (from train_ground_truth.tsv):
  - 7,638,365 true (S1, S2/S3) match pairs
  - Sample ~2M for training (stratified by country, match-count)

HARD NEGATIVE PAIRS (from blocking stage):
  - For each S1, take blocking candidates that are NOT in ground truth
  - These are the most informative negatives (similar but wrong)
  - Sample ~4M hard negatives (2:1 negative:positive ratio)

EASY NEGATIVE PAIRS (random):
  - Small set of random country-matched non-pairs
  - Help calibrate the low end of the probability scale

VALIDATION SET:
  - Hold out 20% of S1 entities (not pairs!) for validation
  - Compute macro-F₀.₅ on held-out S1 entities
  - This is the ONLY metric that matters for threshold tuning
```

### 6.3 LightGBM Hyperparameters

```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',    # Train on logloss, tune threshold for F₀.₅
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 2.0,       # Slight upweight for positives
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'verbose': -1,
}
```

### 6.4 Probability Calibration

- Use **Isotonic Regression** calibration on validation set
- Ensures predicted probabilities are well-calibrated
- Critical for threshold tuning to be meaningful

---

## 7. Stage 4: Precision-Tuned Decision Layer

### 7.1 Threshold Optimization

```python
# For each threshold t in np.arange(0.3, 0.95, 0.01):
#   For each S1 in validation set:
#     predicted_matches = candidates with P(match) >= t
#     Compute per-entity F₀.₅
#   macro_f05 = mean of per-entity F₀.₅ scores
#   Select t* that maximizes macro_f05

# Expected: optimal threshold will be HIGH (0.6–0.8) because of precision focus
```

### 7.2 Singleton Detection Strategy

Singletons are 5.58% of entities but contribute disproportionately to the score.

```
For each S1 entity:
  IF max(P(match)) across all candidates < singleton_threshold:
    Predict empty match list → score 1.0 if true singleton
  
  singleton_threshold should be LOWER than match_threshold
  This creates a "gap zone" where we're uncertain → default to empty
  
  The gap zone exploits F₀.₅'s precision preference:
    - False singleton (missed match) costs ~0.06 per entity
    - False merge costs ~0.22 per entity  
    - False merge on true singleton costs 1.0
```

### 7.3 Per-Country Threshold Tuning

- Tune separate thresholds for US and India on training data
- For France (unseen): use the more conservative of the two thresholds
- Rationale: noise patterns and name conventions differ by country

### 7.4 Global Disjoint Assignment (Critical Post-Processing)

**Exploit verified dataset property**: Every S2/S3 ID matches **at most 1** S1 entity.

If independent thresholding assigns `S2-999` to both `S1-A` (score 0.88) and `S1-B` (score 0.74), one is a **guaranteed false positive**. We use greedy competitive assignment:

```python
# Greedy Maximum Weight Disjoint Assignment
# 1. Pool all (s1_id, candidate_id, probability) triples per country
# 2. Sort descending by probability
# 3. Iterate: assign candidate to s1 only if candidate not yet assigned
#    assigned_s2 = set()
#    assigned_s3 = set()
#    for s1_id, cand_id, prob in sorted_pairs:
#        if prob < threshold: break
#        target_set = assigned_s2 if cand_id.startswith('S2-') else assigned_s3
#        if cand_id not in target_set:
#            add cand_id to s1_id's matches
#            target_set.add(cand_id)
```

> [!IMPORTANT]
> This step is **free precision improvement**. It eliminates cross-entity false merges with zero recall cost (the higher-confidence assignment is always kept). Expected F₀.₅ boost: **+0.5 to 1.5 points**.

### 7.5 Confidence Tiers

```
Tier 1 (Auto-match):   P(match) > 0.90 → Always include
Tier 2 (Likely match):  0.70 < P(match) ≤ 0.90 → Include if address evidence strong
Tier 3 (Uncertain):     0.50 < P(match) ≤ 0.70 → Include only if name AND address agree
Tier 4 (Unlikely):      P(match) ≤ 0.50 → Exclude (default to precision)
```

---

## 8. Handling France (Zero-Shot Country)

France appears only in the test set (15% of test S1). Our pipeline handles this by design:

### 8.1 Why Our Pipeline Transfers

| Component | France Compatibility |
|:---|:---|
| **Country blocking** | ✅ Just another partition value |
| **TF-IDF char n-grams** | ✅ Language-agnostic character patterns |
| **Multilingual-e5 embeddings** | ✅ French is a high-resource language for this model |
| **RapidFuzz string similarities** | ✅ Language-agnostic |
| **LightGBM features** | ✅ All features are similarity scores, not language-dependent |
| **Transliteration** | ✅ French uses Latin script with accents → NFKD handles it |

### 8.2 France-Specific Prep (Minimal)

- Add French legal suffix normalization: `SARL`, `SAS`, `SA`, `EURL`, `SCI`, `SASU`
- Add French address abbreviations: `Rue`, `Avenue`, `Boulevard`, `Place`, `Allée`
- French department codes and region names
- Use `unidecode` for accent stripping: `Société` → `Societe`

### 8.3 France Threshold Strategy

- Default to the **stricter** of US/India thresholds
- Rationale: with no training data, err on side of precision
- We can submit once, see the public leaderboard F₀.₅, then adjust

---

## 9. Computational Resource Strategy

### 9.1 Memory Budget (15 GB RAM Available)

| Stage | Peak Memory | Mitigation |
|:---|:---|:---|
| Preprocessing | ~4 GB | Process one source file at a time |
| TF-IDF blocking | ~8 GB | Process per-country; use `sparse_dot_topn` |
| Embedding generation | ~6 GB | Batch encode (batch_size=512); use float16 |
| FAISS search | ~4 GB | Per-country index; use IVF not flat |
| Feature engineering | ~6 GB | Process in chunks of 100K pairs |
| LightGBM training | ~4 GB | 2M training pairs × 50 features |

### 9.2 GPU Strategy

- If GPU available: Use for embedding generation (10× faster)
- If CPU only: Use `multilingual-e5-base` (smaller) and batch carefully
- Embedding generation is the only GPU-intensive step

### 9.3 Estimated Pipeline Runtime

| Stage | CPU Time | GPU Time |
|:---|:---|:---|
| Preprocessing + transliteration | 2–3 hours | 1 hour |
| TF-IDF blocking | 1–2 hours | N/A |
| Embedding generation (10M records) | 8–12 hours | 1–2 hours |
| FAISS blocking | 30 min | 10 min |
| Feature engineering | 2–3 hours | N/A |
| LightGBM training + tuning | 30 min | N/A |
| Inference on test | 1–2 hours | N/A |
| **Total** | **~18 hours** | **~6 hours** |

---

## 10. Experimentation & Iteration Plan

### Phase 1: Minimum Viable Pipeline (MVP) — Days 1-3

**Goal**: Get a working end-to-end pipeline with a baseline F₀.₅ score.

1. Basic preprocessing (lowercase, strip punctuation, state normalization)
2. TF-IDF char n-gram blocking only (Channel A)
3. 10 core features (Jaccard, Levenshtein, token_set_ratio for name+address)
4. LightGBM with default hyperparameters
5. Single threshold tuned on validation set
6. **Submit to leaderboard** → establish baseline

### Phase 2: Cross-Script + Embeddings — Days 4-6

**Goal**: Handle India cross-script matching (the biggest recall gap).

1. Integrate IndicXlit transliteration for Devanagari/Tamil/Telugu
2. Add multilingual-e5 embedding features
3. Add Channel B (FAISS ANN blocking)
4. Expand to 30+ features
5. **Submit** → expect significant F₀.₅ improvement

### Phase 3: Feature Engineering Deep Dive — Days 7-9

**Goal**: Maximize precision with rich features.

1. Add all 50+ features from Section 5
2. Add phonetic blocking (Channel C)
3. Implement adaptive empty-address weighting
4. Add hard negative mining from blocking stage
5. Hyperparameter tuning (Optuna for LightGBM + threshold)
6. **Submit** → expect incremental improvement

### Phase 4: France + Precision Hardening — Days 10-12

**Goal**: Handle France zero-shot and squeeze final precision.

1. French legal suffix normalization
2. Per-country threshold tuning
3. Singleton detection guardrails
4. Error analysis on validation misclassifications
5. Ensemble: try training separate models per country
6. **Final submissions**

### Phase 5: Ensemble & Polish — Days 13-14

**Goal**: Final optimizations.

1. **Cross-encoder re-ranking on borderline pairs only** (~300K pairs where 0.40 ≤ P < 0.85):
   - Use `microsoft/mdeberta-v3-base` (278M, MIT) as a Ditto-style cross-encoder
   - Serialize pairs: `[COL] name [VAL] <S1_name> [COL] addr [VAL] <S1_addr> [SEP] [COL] name [VAL] <S2_name> [COL] addr [VAL] <S2_addr>`
   - Fine-tune on train GT hard positives + hard negatives
   - Only applied to borderline pairs (Stage 2 high-confidence and reject decisions are untouched)
   - Expected: +1-2 points F₀.₅ on borderline cases
2. Model stacking: LightGBM + XGBoost → meta-learner (logistic regression)
3. Global Disjoint Assignment post-processing
4. Final validation, format check with `utils/validate_submission.py`, package submission

---

## 11. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|:---|:---|:---|:---|
| Transliteration quality low | Medium | High | Use dense embeddings as fallback; build manual dictionary for common terms |
| TF-IDF blocking misses cross-script pairs | High | High | Channel B (embeddings) specifically covers this |
| France performance drops | Medium | Medium | Conservative thresholds; French is high-resource for multilingual models |
| Memory OOM on large TF-IDF matrices | Medium | Medium | Per-country chunking; use `sparse_dot_topn` |
| LightGBM overfits to US/India | Medium | Medium | Validate on country-stratified folds; feature-based (not entity-specific) |
| Pipeline too slow | Low | High | Prioritize channels by ROI; Channel C is optional |
| False merges on singletons tank F₀.₅ | High | Very High | Conservative singleton threshold; default to empty when uncertain |

---

## 12. Key Technical Decisions & Rationale

### Why NOT an End-to-End LLM Approach?

1. **Scale**: 1.7M × 10M pairs = ~17 trillion. Even with blocking to 50 candidates each, that's 85M pairs. An 8B LLM at 50 tokens/pair would take weeks.
2. **No fine-tuning data for France**: LLM-based approaches would overfit to US/India patterns.
3. **Precision control**: LightGBM with calibrated probabilities gives fine-grained threshold control. LLMs give binary yes/no or uncalibrated scores.
4. **Feature interpretability**: We can diagnose exactly which features drive false merges and fix them.

### Why Multi-Channel Blocking Instead of One?

Because no single method catches all noise patterns:

| Noise Pattern | TF-IDF (A) | Embedding (B) | Phonetic (C) |
|:---|:---|:---|:---|
| Typos | ✅ | ✅ | ✅ |
| Word reordering | ✅ | ✅ | ❌ |
| Cross-script | ❌ | ✅ | ❌ |
| DBA/trade names | ❌ | ❌ | ❌ (address saves) |
| Empty addresses | ✅ (name only) | ✅ | ✅ |
| Severely garbled | ❌ | ✅ | ❌ |

### Why LightGBM Over XGBoost?

- **Faster training**: 2–5× on large datasets
- **Native categorical support**: Handles `source_type`, `country` directly
- **Lower memory**: Histogram-based splitting
- **Comparable accuracy**: Marginal difference on tabular data
- (We'll try XGBoost too as an ensemble member)

---

## 13. Expected Performance Targets

| Metric | MVP (Phase 1) | Phase 2 | Phase 3 | Final |
|:---|:---|:---|:---|:---|
| **Blocking recall** | 92% | 97% | 99% | 99.5% |
| **Validation F₀.₅ (US)** | 0.70 | 0.82 | 0.88 | 0.92+ |
| **Validation F₀.₅ (India)** | 0.60 | 0.78 | 0.86 | 0.90+ |
| **Validation F₀.₅ (Overall)** | 0.66 | 0.80 | 0.87 | 0.91+ |
| **Singleton accuracy** | 80% | 88% | 93% | 96% |

> Top-10 teams in past Amazon ML challenges typically score 0.85–0.95 F₀.₅ on the private leaderboard.

---

## 14. Software Stack

```
Core:
  Python 3.10+
  polars (fast data I/O — 5-10× faster than pandas on large TSVs)
  pandas (compatibility where needed)
  scikit-learn (TF-IDF, preprocessing, metrics)
  lightgbm (classifier)
  rapidfuzz (string similarities — 10× faster than fuzzywuzzy)
  sparse_dot_topn (efficient sparse matrix top-k — 40× faster than scipy)

Embeddings:
  sentence-transformers (multilingual-e5 / bge-m3)
  faiss-cpu or faiss-gpu (ANN search)
  torch (embedding model backend)

Transliteration:
  anyascii (ultra-fast Unicode→ASCII, 150K strings/sec — for blocking)
  ai4bharat-transliteration (IndicXlit — higher quality, for features)
  unidecode (French accent stripping)

Optional (Phase 5):
  optuna (hyperparameter optimization)
  xgboost (ensemble member)
  transformers (mDeBERTa cross-encoder for re-ranking borderline pairs)

Infrastructure:
  joblib / multiprocessing (parallelization)
  tqdm (progress tracking)
  numpy, scipy (linear algebra)
```

---

## 15. Summary: The 5 Things That Will Win This Competition

1. **Cross-script transliteration** — Most teams will fail on India's Devanagari/Tamil/Telugu records. Solving this is a 5–10 point F₀.₅ advantage.

2. **Multi-channel blocking with ≥99% recall** — The recall ceiling of your blocking IS the ceiling of your final score. No downstream model can recover missed pairs.

3. **Precision-first philosophy** — Every architectural decision optimizes for precision over recall. Conservative thresholds, singleton guardrails, high-confidence tiers.

4. **Rich feature engineering (50+ features)** — The classifier is only as good as its features. Character n-grams, phonetic codes, dense embeddings, address number matching, and cross-field interaction features together capture every noise pattern.

5. **France zero-shot readiness** — A language-agnostic pipeline with multilingual embeddings and conservative France-specific thresholds ensures we don't lose 15% of the test set.
