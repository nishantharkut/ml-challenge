# 🏆 Amazon ML Challenge 2026 — Master Plan v5.1
## Experimental Blueprint: Evidence-Driven Entity Resolution
### All 20 Audit Points Incorporated (v5 + ChatGPT Deep Audit)

---

> [!IMPORTANT]
> **Evidence Hierarchy:**
> - **[FACT-STMT]** = Official problem statement / email guidelines
> - **[FACT-DATA]** = Measured from actual supplied training TSVs by our scripts
> - **[RESEARCH]** = Established peer-reviewed methodology
> - **[HYPOTHESIS]** = Must be validated by ablation before inclusion
>
> **Nothing is called "proven", "best", or "must adopt" unless our own
> leakage-safe experiments establish it.**
>
> **This plan is NOT final.** It becomes final only when Section 22's open
> questions have experimental answers.

---

## 0. Competition Specification [FACT-STMT]

| Rule | Source |
|:---|:---|
| Metric: entity-level macro F₀.₅ (β=0.5, precision weighted 2× recall) | Problem statement |
| `candidate_pairs.tsv` reviewed in final ranking; smaller sets preferred | Email update |
| `candidate_pairs.tsv` = exact records fed to matcher | Problem statement |
| Schema: `entity_id \t business_name \t business_address \t country` | Problem statement |
| **Only 4 columns.** No phone, email, URL, or other fields. | Problem statement |
| Models ≤8B params, MIT/Apache 2.0 only | Problem statement |
| No external data/APIs/lookups | Problem statement |
| S1 may match zero, one, or many S2/S3 [FACT-STMT] | Problem statement |
| France appears only in test (zero-shot) [FACT-STMT] | Problem statement |

### F₀.₅ Scoring [FACT-STMT]

| True Matches | Prediction | F₀.₅ Score |
|:---|:---|:---|
| 0 (singleton) | Empty | **1.0** |
| 0 (singleton) | Any non-empty | **0.0** |
| >0 | Empty | **0.0** |
| >0 | Partially correct | Per formula |

FP penalty: β²=0.25 → FP penalized **4× more** than FN in denominator.

---

## 1. Dataset Facts [FACT-DATA]

| Fact | Value | Source |
|:---|:---|:---|
| Train S1 | 2,206,821 | `compute_eda.py` |
| Train S2 | 5,034,616 | `compute_eda.py` |
| Train S3 | 5,285,603 | `compute_eda.py` |
| Test S1 | 1,732,544 | Verified directly |
| Test S2 | 4,887,273 | `compute_eda.py` |
| Test S3 | 5,082,316 | `compute_eda.py` |
| GT match pairs | 7,638,365 | `compute_eda.py` |
| Singletons | 123,247 (5.58%) | `compute_eda.py` |
| Mean matches/S1 | 3.46 | `compute_eda.py` |
| Max matches/S1 (train) [FACT-DATA] | 11 (S2 max=5, S3 max=6) | `compute_eda.py` |
| Cross-country GT pairs | 0 | `compute_eda.py` |
| Each S2/S3 → at most 1 S1 | 0 multi-links | `compute_eda.py` |
| S1 India always Latin | 100% | `deep_gap_analysis.py` |
| S2 India Indic names | 474,345 (23.5%) | `deep_gap_analysis.py` |
| S3 India Indic names | 278,524 (13.2%) | `deep_gap_analysis.py` |
| Cross-script GT pairs | 551,240 | `mine_dictionary.py` |
| Mined dictionary entries | 1,316 (conf≥80%) | `mine_dictionary.py` |
| Dictionary coverage | 93.5% on 10K sample ⚠️ same-data | `mine_dictionary.py` |
| Empty-address GT matches | 4.4% | `deep_gap_analysis.py` |
| S1 with BOTH S2+S3 | 80.5% | `deep_gap_analysis.py` |
| Distractor S2/S3 | ~26% | `deep_gap_analysis.py` |
| France test S1 | ~259K (15%) | `compute_eda.py` |

> [!NOTE]
> Max train matches=11 is **[FACT-DATA]**, not a guaranteed test maximum.
> "S1 may match zero, one, or many" is **[FACT-STMT]**.

---

## 2. Hardware [FACT-DATA]

| Resource | Value |
|:---|:---|
| RAM | 15.25 GB total (~5.5 GB free) |
| GPU | Assume none until confirmed |
| System Python | 3.14.0 |
| Timeline | ~2 weeks |
| Team | 1 person |

**Action**: Test dependency installation on system Python first. Pin 3.11/3.12 venv only if required.

---

## 3. Architecture Search Space

The final architecture **emerges from ablation**. No component is mandatory.

```
                    S1
                     │
             difficulty routing        ← [HYPOTHESIS §5.4]
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
       EASY        MEDIUM       HARD
         │           │           │
     exact/rare   lexical     lexical
     + structured + address   + dense
                              + specialist
                              + bridge
         └───────────┬───────────┘
                     ▼
        source-aware retrieval         ← [HYPOTHESIS §5.1]
             S2 / S3
                     │
                     ▼
      deterministic evidence pool      ← [HYPOTHESIS §5.3]
      + ranked channel RRF fusion
                     │
                     ▼
         high-recall safe pool
                     │
                     ▼
   risk-controlled K₂ / K₃ budget     ← [HYPOTHESIS §7]
                     │
                     ▼
           candidate_pairs.tsv
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     classifier              ranker    ← [HYPOTHESIS §9]
          └──────────┬──────────┘
                     ▼
            OOF calibration            ← [HYPOTHESIS, inner fold]
                     │
                     ▼
      cardinality + S1-level decoder   ← [HYPOTHESIS §11]
                     │
                     ▼
        optional global constraints
                     │
                     ▼
           matching_results.tsv
```

---

## 4. Preprocessing [HYPOTHESIS]

### 4.1 Script-Aware Unicode Normalization
- NFKC (NOT NFKD) — NFKD destroys Indic vowel marks [FACT-DATA: tested]
- Preserve Mn/Mc Unicode marks for Indic scripts
- Strip accents only for Latin-range characters (French accented names)

### 4.2 Legal Suffix Extraction
- Dual representation: `(core_name, standardized_suffix)`
- Suffix match/mismatch as feature, NOT filter

### 4.3 Multi-Dictionary Mining [HYPOTHESIS]
Three dictionaries, all fold-safe:

| Dictionary | Pairs Mined From | Example |
|:---|:---|:---|
| **Cross-script transliteration** | Indic ↔ Latin GT pairs | लिमिटेड → limited |
| **Abbreviation/variant** | Same-script GT pairs with token count/surface mismatch | pvt ↔ private, intl ↔ international |
| **Spelling variant** | Near-duplicate GT pairs | centre ↔ center |

Apply to BOTH names AND addresses [FACT-STMT: address transliteration listed as noise type].

### 4.4 Multi-View Representations [RESEARCH]
For each record, maintain:
```
raw_text          — original Unicode
normalized_text   — lowered, whitespace-collapsed, NFKC
translit_text     — dictionary-translated (if cross-script)
phonetic_text     — ITRANS romanized (if Indic, for fuzzy matching)
structured_tokens — (name_core, suffix, addr_components, country)
```
Do NOT force into one canonical form.

---

## 5. Retrieval [HYPOTHESIS — every choice validated]

### 5.1 Source-Aware Retrieval [HYPOTHESIS, not "must"]

Source-separated (S1→S2 independent from S1→S3) may prevent source crowding when one source dominates similarity. But it is NOT established by the problem statement.

**Compare at equal total candidate budget**:
- Joint retrieval: top-K from S2∪S3
- Independent: top-K₂ from S2 + top-K₃ from S3

If source separation only appears better because it produces 2× candidates, the comparison is unfair.

### 5.2 Retrieval Channels [HYPOTHESIS — each must earn inclusion]

| Channel | Method | Type |
|:---|:---|:---|
| C1 | Exact normalized name hash | Deterministic |
| C2 | Exact normalized address hash | Deterministic |
| C3 | Rare/distinctive token index | Deterministic |
| C4 | Word-level BM25 (name) | Ranked |
| C5 | Char TF-IDF 3-5gram (name) | Ranked |
| C6 | Char TF-IDF 3-5gram (address) | Ranked |
| C7 | Challenge-trained dense ANN | Ranked (Phase 3+) |

Per-country processing [HYPOTHESIS — validate by measuring candidate recall].

### 5.3 Channel Fusion: Deterministic + Ranked Separation

> [!WARNING]
> Exact hash buckets do NOT produce meaningful rankings. Giving arbitrary
> RRF ranks to unordered bucket members corrupts the fused ranking.

**Two-stage fusion:**

**Stage 1: Deterministic evidence pool**
- All exact-name matches, exact-address matches, rare-token matches
- These candidates enter the pool without rank — they are "guaranteed candidates"

**Stage 2: Ranked channel fusion (RRF)**
- Apply RRF only to channels with meaningful score-based rankings (C4-C7)
- RRF constant `k` is a hyperparameter — do not assume 60 is optimal

```python
def rrf_fusion(ranked_lists, k=60):
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, cand_id in enumerate(ranked_list):
            scores[cand_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**Merge**: Deterministic pool ∪ RRF-ranked pool → high-recall candidate pool.

### 5.4 Difficulty-Based Selective Retrieval [HYPOTHESIS]

Not every S1 needs every channel. Selective routing improves runtime + candidate efficiency + precision simultaneously.

```
              S1 entity
                 │
         difficulty estimator
         (based on cheap features:
          exact-match found? rare token?
          script mismatch? address empty?)
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
   EASY        MEDIUM       HARD
     │           │           │
  exact/rare  + BM25      + dense
  + structured  + char     + specialist
    address     + address  + bridge
     │           │         + phonetic
     │           │           │
     └───────────┴───────────┘
```

| Route | Trigger [HYPOTHESIS] | Channels |
|:---|:---|:---|
| EASY | Exact name match found + address coherent | C1,C2,C3 only |
| MEDIUM | No exact match but lexical evidence strong | C1-C6 |
| HARD | Cross-script, empty address, weak lexical | C1-C7 + bridge |

The routing thresholds are learned/validated, not arbitrary.

### 5.5 Channel Evaluation [MUST for each]
Every channel reports **marginal contribution**:
- Δ(candidate recall) when added
- Δ(candidate count) when added
- Runtime cost
- RAM cost

A channel adding 0.1% recall for 5 extra candidates/S1 may not be worth keeping.

---

## 6. Candidate Generation Evaluation [MUST IMPLEMENT]

Every candidate configuration reports:

```
RECALL METRICS
  Pair recall (overall / S2 / S3 / cross-script / empty-addr / high-match):
  Complete-entity recall (% S1 with ALL GT in candidates):

ORACLE MACRO F₀.₅
  Use GT_i ∩ C_i as perfect prediction → exact competition metric
  This is the CEILING the matcher can never exceed.

CANDIDATE DISTRIBUTION
  Mean / Median / P95 / P99 / Max  (per S1)

EFFICIENCY
  Reduction ratio
  Runtime (wall-clock)
  Peak RAM
```

**Oracle F₀.₅ is a central project metric.** It answers: "How much can the matcher possibly achieve given this candidate generator?"

### Country Blocking Validation [MUST]
Do not assume "0 cross-country GT pairs" automatically justifies hard blocking.

**Validate directly**: Does enforcing same-country blocking achieve 100.000% training candidate recall? If yes, it is a safe compression rule.

---

## 7. Candidate Budget: Risk-Controlled, Source-Specific [HYPOTHESIS]

### 7.1 Source-Specific Budgets

Predict **two budgets** per S1:

$$K_2,\ K_3$$

not a single K. A global top-K can still crowd one source.

```
S1_A → K₂=5,  K₃=5   (balanced)
S1_B → K₂=12, K₃=3   (mostly S2 matches)
S1_C → K₂=4,  K₃=15  (mostly S3 matches)
```

Optionally apply a total budget cap: K₂ + K₃ ≤ K_total.

### 7.2 Risk-Controlled Formulation

Instead of predicting exact K, estimate:

$$P(K_{\text{required}} \le k \mid X)$$

Then choose the **smallest k satisfying a target coverage probability**:

```
P(all S2 matches captured | K₂=4)  = 0.99
P(all S2 matches captured | K₂=6)  = 0.998
P(all S2 matches captured | K₂=8)  = 0.9995
```

Choose K₂ and K₃ at the validated coverage level on the candidate-size ↔ recall ↔ oracle-F₀.₅ Pareto frontier.

### 7.3 Budget Difficulty Features [HYPOTHESIS]
- Best retrieval score per channel
- Score concentration (entropy/Gini)
- Score gap (1st minus 2nd)
- Number of channels agreeing
- Exact-match evidence present
- Rare-token evidence present
- Name/address length, missingness
- Script mismatch detected
- Candidate pool size per source
- Lexical/dense channel disagreement

### 7.4 Formal Candidate Objective

$$\min\ \mathbb{E}[K_2 + K_3]$$

subject to:

$$R_{\text{entity}} \ge R_{\min}$$

$$F_{0.5}^{\text{oracle}} \ge F_{\min}$$

$$\text{runtime} \le T_{\max}$$

$$\text{RAM} \le M_{\max}$$

---

## 8. Feature Engineering [HYPOTHESIS — validate via importance]

### Name Features (13)
| # | Feature | Purpose |
|:---|:---|:---|
| 1 | `name_jaro_winkler` | Short-name comparison |
| 2 | `name_levenshtein_norm` | Edit distance |
| 3 | `name_token_sort_ratio` | Word reordering |
| 4 | `name_token_set_ratio` | Subset/superset |
| 5 | `name_jaccard_char3gram` | Typo-robust |
| 6 | `name_overlap_coefficient` | Short-in-long detection |
| 7 | `name_tfidf_cosine` | IDF-weighted similarity |
| 8 | `name_core_jaccard` | After suffix removal |
| 9 | `name_suffix_match` | Legal suffixes agree? |
| 10 | `name_first_token_match` | Brand word agreement |
| 11 | `name_len_ratio` | min/max length |
| 12 | `name_is_cross_script` | Different script blocks |
| 13 | `name_dict_translit_score` | Dictionary-translated overlap |

### IDF-Weighted Discriminative Token Features (5)
| # | Feature | Purpose |
|:---|:---|:---|
| 14 | `shared_token_idf_sum` | Total IDF of shared name tokens |
| 15 | `max_shared_token_idf` | IDF of rarest shared token |
| 16 | `distinctive_token_fraction_shared` | % of high-IDF tokens shared |
| 17 | `rare_token_exact_match` | Any IDF>threshold token matches? |
| 18 | `generic_only_match` | Only low-IDF tokens shared? |

Rationale: Sharing "hydroflux" is radically different from sharing "company". IDF statistics are computed on the full S1+S2+S3 corpus within each country (allowed — no labels used) [RESEARCH].

### Address Features (11)
| # | Feature | Purpose |
|:---|:---|:---|
| 19 | `addr_token_set_ratio` | Handles reordering + partial |
| 20 | `addr_jaccard_char3gram` | Character-level |
| 21 | `addr_number_exact_match` | Premise numbers match |
| 22 | `addr_number_conflict` | Both have DIFFERENT numbers |
| 23 | `addr_number_missing_vs_conflict` | **Missing ≠ disagreement** |
| 24 | `addr_state_match` | Normalized state codes |
| 25 | `addr_len_ratio` | min/max length |
| 26 | `addr_tfidf_cosine` | IDF-weighted |
| 27 | `addr_pincode_match` | ZIP/PIN match |
| 28 | `addr_dict_translit_score` | Dictionary-translated address overlap |
| 29 | `addr_component_agreement` | Structured: city/state/postal agree? |

### Cross-Field & Meta (4)
| # | Feature | Purpose |
|:---|:---|:---|
| 30 | `either_addr_empty` | One/both empty |
| 31 | `both_addr_empty` | Both empty |
| 32 | `source_type` | S2 vs S3 |
| 33 | `name_addr_disagreement` | |name_sim - addr_sim| |

### Retrieval-Derived (7)
| # | Feature | Purpose |
|:---|:---|:---|
| 34 | `rrf_score` | Ranked-channel fusion score |
| 35 | `n_channels_retrieved` | Channel agreement count |
| 36 | `candidate_rank` | Rank in merged list |
| 37 | `score_gap_to_next` | Gap to rank+1 |
| 38 | `s1_pool_size` | Total candidates for this S1 |
| 39 | `reciprocal_rank` | 1/best_channel_rank |
| 40 | `bidirectional_rank` | S2→S1 retrieval rank (if computed) |

### Optional [HYPOTHESIS]
| # | Feature | Purpose |
|:---|:---|:---|
| 41 | `is_acronym_match` | Short name ↔ initials of full |
| 42 | `embedding_cosine` | If dense retrieval active |

> [!NOTE]
> `phone_match` is **permanently removed**. The schema provides only 4 columns:
> `entity_id`, `business_name`, `business_address`, `country` [FACT-STMT].

---

## 9. Pairwise Matcher [HYPOTHESIS — compare alternatives]

| Option | Description |
|:---|:---|
| M1 | LightGBM binary classifier (baseline) |
| M2 | LambdaMART/listwise ranking (S1 as group) |
| M3 | Source-specific models (separate S2/S3 models) |
| M4 | Shared model + source-specific calibration |

### Optional neural reranker [HYPOTHESIS, Phase 3+]
- Only on borderline pairs (routing band)
- Field-aware Ditto-style serialization: `COL name VAL ... COL addr VAL ... [SEP] ...`
- Candidates: bge-reranker-v2-m3 (Apache 2.0, 568M), mdeberta-v3-base (MIT, 278M)

---

## 10. Training Data [HYPOTHESIS]

### Entity-Balanced Sampling [RESEARCH]
- Per-S1 caps: max N positive + M hard negative per S1 per epoch
- Explicit singleton S1 representation

### Structured Hard Negatives [HYPOTHESIS]
| Cat | Description |
|:---|:---|
| H1 | Same/similar name, different address |
| H2 | Same/similar address, different name |
| H3 | Same transliteration, different entity |
| H4 | Same rare token, different business |
| H5 | High retrieval score but non-match |
| H6 | Dense retriever false positive |
| H7 | Same numeric component, different entity |
| H8 | High-scoring LightGBM false positive (Round 2+) |

### Leakage Safety [MUST]
- All GT-derived artifacts rebuilt inside each fold
- Protect ALL true positives for same S1 from negative pool
- Hard negatives verified against fold's GT

---

## 11. S1-Level Set Decoder [HYPOTHESIS — compare alternatives]

The prediction for each S1 is a **SET**. Independent pair decisions are suboptimal.

### Decoder candidates [HYPOTHESIS — compared on outer validation]:

| ID | Description |
|:---|:---|
| D1 | Global probability threshold (baseline) |
| D2 | Per-S1 threshold from difficulty features |
| D3 | 2D grid: absolute threshold + score-gap as feature |
| D4 | Learned prefix-selection policy (see §14 for nested validation) |
| D5 | Cardinality-aware: estimate P(N=0), P(N=1), ..., condition decoder |
| D6 | Learned S1-level utility model |

> [!CAUTION]
> **D4/D5/D6 require nested validation (§14).** A decoder that selects the
> prefix maximizing F₀.₅ using validation labels is an **oracle**, not a
> deployable system. The decoder must predict k WITHOUT seeing the S1's GT.

### Query Exclusivity [HYPOTHESIS — not automatically beneficial]
Each S2/S3 → at most 1 S1 [FACT-DATA]. But naive "assign to highest scorer" may not be optimal for macro entity-level F₀.₅.

Compare:
- Independent S1 decisions (no constraint)
- Greedy exclusivity (each query → highest-scoring S1)
- Globally constrained assignment

**Only keep if it improves outer validation macro F₀.₅.**

---

## 12. Exact Competition Metric [MUST IMPLEMENT FIRST]

```python
def competition_macro_f05(predictions, ground_truth, beta=0.5):
    """
    Exact competition scoring.
    predictions: {s1_id: set of matched_ids}
    ground_truth: {s1_id: set of true_ids}
    S1 IDs derived from test_source1.tsv, NOT hard-coded.
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
            p = tp / len(pred)
            r = tp / len(true)
            if p + r == 0:
                scores.append(0.0)
            else:
                scores.append((1 + beta**2) * p * r / (beta**2 * p + r))
    return sum(scores) / len(scores)
```

**Expected output rows**: Derived from `test_source1.tsv` directly, NOT hard-coded.

---

## 13. Validation Design [CRITICAL — fixes v5 leakage]

### 13.1 The Problem with Simple OOF

If OOF predictions are used to **both** tune downstream components (decoder, threshold, budget, calibration) **and** report the final score, the estimate is optimistic.

### 13.2 Nested Validation Architecture

```
FULL TRAINING DATA (2.2M S1)
     │
     ├── HOLDOUT SET (~10% = 220K S1)
     │   Never touched until final confirmation
     │
     └── WORKING SET (~90% = 1.98M S1)
              │
         3 OUTER FOLDS
              │
         ┌────┴────┐
     Fold train   Fold test (outer val)
         │              │
    ┌────┴────┐         │
  inner     inner       │
  train     val         │
    │         │         │
  train     tune:       honest
  model    threshold    evaluation
           decoder      (report
           budget       THIS score)
           calibration
           RRF k
```

### 13.3 Concrete Split

| Split | Size | Use |
|:---|:---|:---|
| Holdout | ~220K S1 (10%) | Final confirmation only |
| Outer fold 1 train | ~1.32M S1 | Train + inner tune |
| Outer fold 1 test | ~660K S1 | Honest evaluation |
| Outer fold 2 train | ~1.32M S1 | Train + inner tune |
| Outer fold 2 test | ~660K S1 | Honest evaluation |
| Outer fold 3 train | ~1.32M S1 | Train + inner tune |
| Outer fold 3 test | ~660K S1 | Honest evaluation |

### 13.4 Within Each Outer Fold Train, Rebuild:
- Transliteration + abbreviation dictionaries
- Retrieval indices
- Hard negatives
- LightGBM model
- Inner validation → calibration, threshold, decoder, budget
- Apply to outer test → honest metric

### 13.5 Why 3 Folds, Not 5
With 2.2M S1, each fold test set has ~660K entities — more than enough statistical power. 5 folds would waste ~40% more compute for marginal precision gain.

### 13.6 Statistical Significance [MUST for major ablations]

For each ablation, compute per-S1 ΔF₀.₅ and bootstrap:

```python
def paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=10000):
    """95% CI for mean(scores_b - scores_a)"""
    deltas = np.array(scores_b) - np.array(scores_a)
    boot_means = [np.mean(np.random.choice(deltas, len(deltas), replace=True))
                  for _ in range(n_bootstrap)]
    return np.percentile(boot_means, [2.5, 97.5])
```

Report: `mean ΔF₀.₅, 95% CI`. Do not invest days optimizing noise.

---

## 14. France Robustness [HYPOTHESIS]

### Leave-One-Country-Out Experiments

Since France is zero-shot in test [FACT-STMT], standard random CV does NOT test domain shift.

Add these experiments:

| Experiment | Train On | Validate On | Tests |
|:---|:---|:---|:---|
| LOC-US | India only | US | Can system generalize across countries? |
| LOC-India | US only | India | Can system generalize across scripts? |

These test whether features and thresholds generalize without country-specific memorization.

---

## 15. Experiment Design: Incremental + Ablation [MUST]

### 15.1 Sequential Incremental Path (development order)

| Exp | Added Component | Key Metrics |
|:---|:---|:---|
| E0 | Exact/rare token blocking only | cand_count, pair_recall, entity_recall, oracle_F05 |
| E1 | + Word BM25 | Δ from E0 |
| E2 | + Char TF-IDF | Δ from E1 |
| E3 | + Address retrieval | Δ from E2 |
| E4 | + Transliteration dictionaries | Δ from E3 |
| E5 | + Source-aware retrieval | Δ from E4, controlled at equal budget |
| E6 | + LightGBM with core features | final_F05 |
| E7 | + Structured hard negatives | Δ from E6 |
| E8 | + Full feature set (IDF, structured addr) | Δ from E7 |
| E9 | + OOF calibration (nested) | Δ from E8 |
| E10 | + S1-level decoder (compare D1-D6) | Δ from E9 |
| E11 | + Risk-controlled K₂/K₃ budget | cand_count + final_F05 |
| E12 | + Difficulty routing | runtime + Δ from E11 |
| E13 | + Dense learned blocker | cand_recall + Δ |
| E14 | + Neural cross-encoder reranker | Δ for borderline |
| E15 | + Cross-source bridge | cand_recall + cand_count |

### 15.2 Controlled Ablation Table (after strong baseline established)

Run after E8+ to measure TRUE contribution of each component:

| Config | Transliteration | Address retrieval | Source separation | IDF features | F₀.₅ | Δ from full |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| Full | ✓ | ✓ | ✓ | ✓ | ? | baseline |
| -translit | ✗ | ✓ | ✓ | ✓ | ? | ? |
| -address | ✓ | ✗ | ✓ | ✓ | ? | ? |
| -source_sep | ✓ | ✓ | ✗ | ✓ | ? | ? |
| -idf | ✓ | ✓ | ✓ | ✗ | ? | ? |
| Minimal | ✗ | ✗ | ✗ | ✗ | ? | ? |

### 15.3 Per-Experiment Report
```
candidate_count (mean/median/P95/P99/max)
pair_recall (overall / S2 / S3 / cross-script / empty-addr)
complete_entity_recall
oracle_macro_F05
final_macro_F05
runtime (wall-clock)
peak_RAM
ΔF05 + 95% CI vs previous  (for major comparisons)
```

---

## 16. Failure Analysis [MUST after E6-E8]

### False Negatives
| Code | Cause | Next Experiment |
|:---|:---|:---|
| FN-R | Retrieval miss | More/better channels |
| FN-T | Transliteration miss | Better dictionary/embeddings |
| FN-A | Address-only distinguishable | Better address features |
| FN-K | Ranking error | Better features/model |
| FN-D | Decoder too conservative | Better decoder |
| FN-S | Singleton misclassification | Cardinality model |

### False Positives
| Code | Cause | Next Experiment |
|:---|:---|:---|
| FP-N | Same-name collision | IDF/distinctive token features |
| FP-A | Same-address collision | Name discrimination |
| FP-L | Legal suffix only difference | Suffix features |
| FP-G | Generic name collision | IDF weighting |
| FP-T | Transliteration collision | Native-script features |
| FP-M | Multi-match set error | Better decoder |

**Every advanced component (E12-E15) must target an observed failure mode.**

---

## 17. Licensed Models [FACT-STMT + Verified]

| Model | Params | License | Use |
|:---|:---|:---|:---|
| multilingual-e5-small | 118M | MIT | Dense retrieval |
| bge-m3 | 567M | MIT | Dense + sparse |
| bge-reranker-v2-m3 | 568M | Apache 2.0 | Cross-encoder |
| mdeberta-v3-base | 278M | MIT | Cross-encoder |
| MuRIL | 236M | Apache 2.0 | Indian specialist |
| ❌ Llama 3.x | 8B | ❌ Llama License | DISQUALIFIED |
| ❌ Gemma 2 | 2-9B | ❌ Gemma Terms | DISQUALIFIED |

---

## 18. Prior-Art Hypotheses [HYPOTHESIS — not solutions to copy]

Observations from public repos. **No code, thresholds, hyperparameters, or architectures copied.**

- Multi-channel union blocking [HYPOTHESIS: test as E0-E5 baseline]
- RapidFuzz deterministic features [HYPOTHESIS: feature starting point]
- 5-category hard negatives [HYPOTHESIS: structured likely beats random]
- 2D threshold grid [HYPOTHESIS: test as decoder D3]
- Query exclusivity [HYPOTHESIS: test, may help or hurt]
- NFKC Devanagari normalization [HYPOTHESIS: test vs our approach]
- Acronym matching feature [HYPOTHESIS: test as feature #41]
- Streaming chunks with gc.collect() [HYPOTHESIS: RAM management]

---

## 19. Submission Protocol

### Validation Before Submit
```bash
python validate_submission.py --check-ids \
  --matching matching_results.tsv \
  --candidate candidate_pairs.tsv \
  --test-s1 test_source1.tsv \
  --test-s2 test_source2.tsv \
  --test-s3 test_source3.tsv
```

Derive expected S1 count from `test_source1.tsv` directly. Do NOT hard-code.

### Singleton format: `<s1_id>\t\n` (tab then newline)

---

## 20. Risk Matrix

| Risk | Prob | Impact | Mitigation |
|:---|:---|:---|:---|
| No GPU | High | High | Classical pipeline competitive [RESEARCH] |
| TF-IDF OOM | Medium | High | max_features cap, float32, per-country |
| Dict coverage drops on test | Low | Medium | ITRANS + embeddings fallback |
| Budget drops true matches | Medium | High | Risk-controlled budget + recall certificate |
| France F₀.₅ poor | Medium | Medium | Open-set + LOC validation |
| Public LB overfitting | Medium | Medium | Trust nested CV |
| Python 3.14 incompatibility | Medium | Medium | Test first, venv only if needed |

---

## 21. What This Plan Does NOT Include

- ❌ No guaranteed scores
- ❌ No "this will win"
- ❌ No copied code/thresholds/architectures
- ❌ No pre-decided final architecture
- ❌ No arbitrary fixed candidate counts
- ❌ No invalid mathematical formulas
- ❌ No features from nonexistent schema columns
- ❌ No unsupported runtime estimates

---

## 22. Open Questions (Answered by Experiments Only)

1. What is the smallest candidate set preserving sufficiently high candidate recall?
2. Which retrieval channels provide unique recall?
3. Is learned blocking better than strong lexical blocking here?
4. Does source-separated retrieval improve recall at equal candidate budget?
5. Does transliteration help after leakage-safe fold-level validation?
6. Does a ranking model beat binary classification?
7. Does the S1-level decoder beat a simple threshold?
8. Does a cardinality model improve singleton/multi-match decisions?
9. Does a neural reranker improve hard residual cases enough to justify cost?
10. Does every added component improve macro F₀.₅ or candidate efficiency?

**Do not call any plan "FINAL" until these have experimental answers.**
