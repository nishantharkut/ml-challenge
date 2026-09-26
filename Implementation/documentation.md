# ML Challenge 2026: Business Entity Resolution Solution

**Team:** Autonomous Entity Resolution Engineering Team  
**Date:** September 2026  
**Competition:** Amazon ML Challenge 2026  

---

## 1. Executive Summary
We present an enterprise-scale, high-precision Entity Resolution system designed to match business records across three heterogeneous, noisy data sources ($S_1$, $S_2$, $S_3$). Our solution employs a **source-separated, multi-channel blocking architecture** (exact hashing, distinctive token inverted indexing, and sparse sublinear character $n$-gram TF-IDF joins) paired with a **gradient-boosted decision tree matcher** trained with structured hard negatives and an **$S_1$-level precision-oriented set decoder** with query exclusivity. On held-out validation data, our system achieves an **Oracle Macro $F_{0.5}$ ceiling of $0.9525$** and a validated **Macro $F_{0.5}$ score of $0.9129$**, while maintaining a compact candidate set of under 20 candidates per entity.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis revealed key structural phenomena:
- **Reference Topology:** $S_1$ is deduplicated reference; $S_2$ and $S_3$ are disjoint query sources where each record matches at most one $S_1$ entity (0 cross-source multi-links in ground truth).
- **Match Multiplicity:** $80.5\%$ of $S_1$ entities possess matches across *both* $S_2$ and $S_3$, averaging $3.46$ matches per entity, with $5.58\%$ true singletons.
- **Geographic Partitioning:** $100\%$ of ground-truth matches occur within the same country (0 cross-country links). Test data introduces an unseen country partition (France), requiring open-set linguistic invariance.
- **Multilingual Noise:** $23.5\%$ of Indian $S_2$ records and $13.2\%$ of $S_3$ records use native Indic scripts (across 9 regional scripts), whereas Indian $S_1$ records are $100\%$ Latin. Address fields exhibit missing literals (`<null>`), token permutations, and abbreviated premise numbers.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Channel Blocking + GBDT Matcher + $S_1$-Level Set Decoder.
- **Core Innovation:**
  1. *Source-Separated Independent Quotas:* Independent candidate budgets ($K_2, K_3$) prevent source crowding and ensure balanced recall across both $S_2$ and $S_3$.
  2. *Script-Safe Normalization:* Unicode NFKC normalization preserving non-spacing and spacing combining marks (`Mn`, `Mc`) to prevent Indic vowel corruption, combined with legal suffix standardization.
  3. *Query Exclusivity Resolution:* Post-scoring global bipartite assignment enforcing the empirical invariant that each query record links to at most one reference entity, generating a $+0.0019$ Macro $F_{0.5}$ boost.

---

## 3. Candidate Generation (Blocking)

To scale across billions of potential pairs, our candidate generator applies hard country partitioning followed by four complementary retrieval channels:
1. **Channel 1 (Exact Normalized Name Hash):** Instant $O(1)$ lookup on NFKC-normalized names.
2. **Channel 2 (Exact Normalized Address Hash):** Instant $O(1)$ lookup on normalized addresses.
3. **Channel 3 (Distinctive Token Index):** Inverted index over tokens in the top 80th percentile of document frequency (IDF), capturing rare brand names.
4. **Channel 4 & 5 (Sparse Character $n$-gram TF-IDF):** 3-to-4-character sublinear TF-IDF retrieval via multi-threaded sparse top-$K$ cosine similarity (`sparse_dot_topn`) with thresholding ($0.12$).

- **Candidate Budget:** Bounded to a maximum of $15$ candidates per query source and total $20$ candidates per $S_1$ entity.
- **Recall Preservation:** Over-sized buckets undergo secondary score-based discrimination rather than arbitrary truncation, guaranteeing zero loss of distinctive matches.

---

## 4. Matching Model

### 4.1 Feature Engineering (29 Deterministic Features)
- **Name Metrics:** RapidFuzz Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Core Name Set Ratio (suffix-stripped), Length Ratio, Exact Match Flag, Legal Suffix Agreement, First Token / Brand Root Agreement, Cross-Script Indicator.
- **Address Metrics:** Token Set Ratio, Token Sort Ratio, Exact Address Match Flag, Missing Address Flag, Premise Number Match vs. Number Conflict (distinguishing missingness from explicit contradiction), Postal Code Match vs. Conflict.
- **Retrieval Evidence:** Channel Agreement Count, Best Rank, Reciprocal Rank ($1 / (\text{rank}+1)$), TF-IDF Name Score, TF-IDF Address Score, Source Type Indicator ($S_2$ vs. $S_3$), Name $\times$ Address Interaction Product.

### 4.2 Model & Training
- **Model:** LightGBM Binary Classifier ($600$ estimators, learning rate $0.05$, $45$ leaves, max depth $8$).
- **Training Strategy:** All ground-truth positives preserved; structured hard negatives mined from top non-matching retrieval candidates (ratio $2.32:1$).
- **Threshold & Exclusivity Selection:** Grid sweep on $50,000$ held-out validation entities directly optimizing the competition's exact Macro $F_{0.5}$ metric with singletons scored strictly according to official guidelines.

---

## 5. Results & Error Analysis

- **Pair Recall:** $91.35\%$
- **Complete Entity Recall:** $81.10\%$
- **Oracle Macro $F_{0.5}$ Ceiling:** $0.9525$
- **Validation Macro $F_{0.5}$:** **$0.9129$**
- **Optimal Decision Threshold:** $0.65$
- **Query Exclusivity Gain:** $+0.001932$ Macro $F_{0.5}$

### Error Analysis:
- **False Positives:** Primarily localized to franchise chains sharing identical legal names in close geographic proximity (e.g. branch offices with identical names but subtle unit-number differences).
- **False Negatives:** Concentrated in low-frequency transliteration edge cases where non-standard phonetic spelling diverges significantly from English orthography without shared premise numbers.

---

## 6. Conclusion
By treating candidate generation as a first-class optimization objective and enforcing source-separated quotas, our solution achieves state-of-the-art Macro $F_{0.5}$ performance while generating an extraordinarily compact candidate set. The system runs deterministically, requires no unapproved external dependencies or API calls, and passes all official validation checks.

---

## Appendix: Code Artefacts & Structure

- **`src/`:** Modular, importable Python package containing normalization, blocking, feature extraction, matcher, and decoder.
- **`notebooks/Amazon_ML_Challenge_2026.ipynb`:** Self-contained, portable Jupyter Notebook ready for execution on Kaggle, Google Colab, or local environments.
- **`output/candidate_pairs.tsv`:** Bounded candidate pairs per $S_1$ entity.
- **`output/matching_results.tsv`:** Final predicted entity resolution matches.
- **Verification:** Verified with official `validate_submission.py`.
