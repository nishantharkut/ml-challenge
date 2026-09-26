# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Nishanth Arkut, Kanika  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

We developed an industrial-grade, leakage-safe, high-precision Business Entity Resolution system designed for multi-source commercial data across diverse geographic and linguistic regions (US, India, and zero-shot France). Our solution couples a multi-channel inverted index blocker (reducing the $>8.3$ trillion cross-product by $>99.9999\%$ while maintaining $>96\%$ pair recall) with a 36-feature LightGBM Gradient Boosted Decision Tree optimized specifically for the precision-heavy macro $F_{0.5}$ metric ($\beta=0.5$). By combining calibrated probability thresholding ($0.750$) with disk-backed SQLite global query exclusivity, our model achieves a verified validation macro $F_{0.5}$ score of **0.889157** while strictly respecting sub-3GB RAM streaming limits.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across the reference (Source-1) and target sources (Source-2, Source-3) revealed five primary real-world noise modalities:
1. **Linguistic & Script Diversity:** Extensive Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia) and French accented characters with complex transliteration inconsistencies across sources.
2. **Legal Entity Suffix Variations:** Severe variations in corporate designators (`Pvt Ltd`, `Private Limited`, `LLP`, `Inc`, `Corp`, `SARL`, `SASU`, `GmbH`) obscuring core business names.
3. **Address Component Permutations & Abbreviations:** Landmark-based Indian references (*"Near SBI ATM, Opp Railway Station"*), missing PIN/postal codes, municipal numbering discrepancies, and standard street abbreviations (*"Rd"* vs *"Road"*, *"St"* vs *"Street"*, *"Ave"* vs *"Avenue"*).
4. **Extreme Asymmetric Cardinality:** 1.73M Source-1 reference queries against $>4.8$ million candidate targets across Source-2 and Source-3. An all-pairs comparison would require $>8.3\times 10^{12}$ comparisons, demanding an ultra-selective blocking architecture.
5. **Macro $F_{0.5}$ Evaluation Dynamics:** In macro $F_{0.5}$, Precision is weighted 4× higher than Recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$), and correctly identifying singleton entities (businesses with 0 matches) awards a perfect 1.0 score, while predicting a single false positive on a singleton yields 0.0. Conservative, high-precision matching is mathematically essential.

### 2.2 Solution Strategy
We structured our architecture into four modular, decoupled phases:
- **Approach Type:** Hybrid Multi-Channel Inverted Index Blocker + GBDT Pairwise Reranker + Global Exclusivity Match Decoder.
- **Core Innovations:**
  1. *Script-Safe NFKC Normalization & Legal Suffix Stripping:* Tokenizes and isolates brand roots while preserving Indic vowel marks and European diacritics.
  2. *Source-Separated Shard-Streamed Retrieval:* Indexes Source-2 and Source-3 galleries independently and frees gallery RAM after candidate caching, ensuring deterministic $<3\text{ GiB}$ peak RSS on local consumer hardware.
  3. *Balanced Quota Candidate Merging:* Enforces a 50/50 quota across Source-2 and Source-3 candidates (max 20 candidates per query) to guarantee source diversity.
  4. *Disk-Backed Global Query Exclusivity:* An atomic SQLite-backed winner resolver that resolves many-to-one conflicting matches globally based on LightGBM probability rankings without in-memory pair accumulation.

---

## 3. Candidate Generation (Blocking)

To prune the comparison space from $>8.3$ trillion pairs down to a maximum of 20 high-quality candidates per Source-1 entity, we implemented a multi-channel inverted index retrieval engine:

- **Blocking Channels Used:**
  1. *Exact Normalized Name & Address Matching:* Deterministic hash index over lowercased, punctuation-stripped, legal-suffix-standardized strings.
  2. *Token-Frequency Inverted Index (Rare Token Channel):* Indexes entities by their lowest document-frequency unigram/bigram tokens to efficiently connect rare, distinctive brand names while ignoring generic terms (*"Store"*, *"Enterprises"*, *"Solutions"*).
  3. *Address Anchor Matching:* Extracts numbers, PIN/postal codes, and locality anchors to index co-located establishments.
  4. *Sparse Character $n$-gram TF-IDF Cosine Matching:* Sub-linear sparse matrix multiplication over 3-gram and 4-gram character tokens, capturing misspellings, optical character recognition errors, and phonetically transliterated name variations.
  5. *Phonetic & Core Name Hashing:* Maps transliterated names into phonetic representations for robust cross-script candidate retrieval.
- **Candidate Pairs Generated:**
  - Capped strictly at a maximum budget of $K \le 20$ candidates per Source-1 entity (balanced up to 10 from Source-2 and 10 from Source-3).
  - Average candidates retrieved per entity: $\approx 14.8$ candidates.
  - Overall reduction ratio: $> 99.9999\%$ of non-matching pairs eliminated before model scoring.
- **Ensuring True Matches Were Not Lost:**
  - Verified on a frozen, entity-disjoint training split of 15,000 entities. Multi-channel retrieval achieved **$>96.2\%$ candidate recall** with an Oracle Macro $F_{0.5}$ ceiling of **$0.954$**.

---

## 4. Matching Model

### Features Used (36 Engineered Features)
Candidate pairs generated by blocking are featurized using 36 deterministic string, token, address, and interaction features computed via RapidFuzz:
1. **Name Similarity Features (14):**
   - Jaro-Winkler similarity, Token Sort Ratio, Token Set Ratio, Levenshtein Fuzzy Ratio, Core Name Set Ratio, Length Ratio, Exact Normalized Match flag, Folded Ratio, Folded Exact Match flag, Partial Token Ratio.
   - Suffix agreement and mismatch indicators (distinguishing e.g. *"Acme Corp"* vs *"Acme Ltd"*).
   - First token brand root match flag (identifying the defining brand anchor).
   - Cross-script mismatch indicator.
2. **Address Similarity Features (11):**
   - Address Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Address Fuzzy Ratio, Address Length Ratio, Exact Address Match flag, Address Folded Ratio, Address Folded Exact Match flag, Address Partial Ratio.
   - Numeric token overlap ratio and exact number matching indicators (distinguishing suite numbers, building numbers, and PIN codes).
3. **Cross-Domain Interaction Features (3):**
   - Multiplicative interaction: $\text{Name Sort Ratio} \times \text{Address Sort Ratio}$.
   - Interaction: $\text{Name JW} \times \text{Address JW}$.
   - Harmonic mean of name and address similarities.
4. **Retrieval Metadata Channel Features (8):**
   - Channel count (number of distinct retrieval channels that surfaced this pair).
   - Binary indicator flags for retrieval source: Exact Name, Exact Address, Rare Token, Address Anchor.
   - Continuous TF-IDF retrieval scores for name and address.

### Model Architecture & Training
- **Model Type:** LightGBM Gradient Boosted Decision Trees (GBDT).
- **Hyperparameters:** `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_samples=50`.
- **Training Set Design:** Strictly composed of candidates obtainable by the production blocker (no synthetic ground truth pairs that the blocker could never find). Leakage-safe, entity-disjoint train/validation split.

### Threshold Selection & Decoder Optimization
- **Optimization Criterion:** Direct maximization of Macro $F_{0.5}$ on 15,000 frozen validation entities.
- **Optimal Probability Threshold:** **`0.750`** (calibrated to suppress false merges on singleton entities).
- **Query Exclusivity:** Enabled (`use_query_exclusivity=True`). Ensures that if multiple Source-1 queries contend for the same physical Source-2/Source-3 record, only the pair with the strictly highest predicted probability retains the match.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Validation Score:** **`0.889157`** (Precision: $\approx 91.2\%$, Recall: $\approx 81.6\%$).
- **Common False Positives (Wrong Merges):**
  - Chain stores, bank branches, and retail franchises (e.g., *"State Bank of India"*, *"Carrefour"*, *"Subway"*) that share identical brand names and similar municipal city names but represent distinct physical branch locations.
- **Common False Negatives (Missed Matches):**
  - Severely truncated address records lacking street numbers or postal codes combined with phonetic transliteration drift across regional scripts where character $n$-gram overlap fell below the retrieval threshold.
- **Zero-Shot Transferability to France:**
  - Text normalization NFKC strip diacritics and legal suffix dictionaries (`SARL`, `SAS`, `SASU`) enabled seamless zero-shot execution on French test records without retraining or data leakage.

---

## 6. Conclusion

By unifying robust multi-channel inverted indexing, precision-engineered RapidFuzz string features, and an $F_{0.5}$-optimized LightGBM classifier with SQLite global exclusivity decoding, our solution delivers state-of-the-art entity resolution accuracy (0.889 macro $F_{0.5}$) while guaranteeing bounded memory execution ($<3\text{ GiB}$ peak RSS) and strict row-level contract conformance.

---

## Appendix

### A. Code Artifacts & Reproduction Pipeline
The runnable reproduction pipeline is located in `code/business_entity_resolution/`:
1. `src/normalization.py`: Script detection, NFKC transliteration, legal suffix standardizer.
2. `src/blocking.py`: Multi-channel inverted index blocker and balanced candidate merge.
3. `src/features.py`: RapidFuzz 36-feature vectorized pairwise extractor.
4. `src/matcher.py`: LightGBM model loader, inference wrapper, and score evaluator.
5. `src/decoder.py` & `src/streaming_runner.py`: Threshold filtering and SQLite global exclusivity resolver.
6. `51_stream_country.py`: Single-country shard-streamed retrieval, scoring, and partition emitter.
7. `55_merge_outputs.py`: Fail-closed partition merger and submission validator.

**End-to-End Reproduction Command:**
```bash
python 55_merge_outputs.py
```
This generates the final verified `matching_results.tsv` and `candidate_pairs.tsv` in exact `test_source1.tsv` row sequence.
