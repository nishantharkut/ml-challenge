# Master Plan v5.2 — Precision Patch on v5.1
## FINAL PLANNING DOCUMENT. Implementation begins now.

> v5.1 remains the full architecture reference. This document contains 12 precision
> clarifications only. No architectural changes.

---

### Patch 1: Candidate-budget objective (amends §7)
Do not optimize K only for P(containing ALL true matches).
Also measure marginal oracle-F₀.₅ gain: `ΔOracleF0.5(K) = OracleF0.5(K) - OracleF0.5(K-1)`.
Final K₂/K₃ policy chosen using: candidate count + recall + oracle-F₀.₅ + runtime/RAM.

### Patch 2: Deterministic-block overflow (amends §5.3)
Exact-name/address/rare-token blocks must not create unbounded candidate sets.
Small blocks: retain directly. Large/ambiguous blocks: secondary discrimination (e.g., re-rank by address similarity), NOT arbitrary truncation.

### Patch 3: Difficulty routing safety (amends §5.4)
EASY route must NOT trigger merely on exact-name match existence.
Must also consider: exact-name bucket size, distinctive-token evidence, address coherence, candidate ambiguity. "ABC Trading Company" matching exactly is NOT automatically easy.

### Patch 4: France / open-set inference (amends §14)
Learned parameters trained only on available countries. Inference must gracefully handle unseen country labels (France) without country-specific branches. Country used as input/partition where validated, but no hard-coded France model/rule.

### Patch 5: Inductive vs transductive preprocessing (NEW)
During validation: IDF/vocabulary/retrieval statistics fit on fold-train only.
During final inference: fit on full train corpus. Test-corpus statistics allowed only if competition rules explicitly permit (transductive IDF on test S2/S3 uses no labels, likely safe — but document the choice).

### Patch 6: Text-view definition (amends §8)
Every similarity feature must specify its input views:
- `name_token_set_ratio`: normalized vs normalized
- `name_dict_translit_score`: transliterated vs normalized (Latin)
- `addr_token_set_ratio`: normalized vs normalized
- `name_jaccard_char3gram`: normalized vs normalized
- etc.
Do not leave ambiguous. Document in feature engineering code.

### Patch 7: Dictionary collision safety (amends §4.3)
Retain mappings only when sufficiently reliable (forward prob, reverse prob, frequency, entropy).
Dictionary mappings provide **features/evidence** — do NOT blindly overwrite original text.
Keep both original and dictionary-translated views.

### Patch 8: Positive sampling (amends §10)
Do not use arbitrary per-S1 positive cap that underrepresents high-cardinality entities.
Retain ALL positives where computationally feasible. If sampling necessary, stratify by source, match count, cross-script status, missing address. Use entity weighting.

### Patch 9: ORACLE GAP metric (amends §6, §15)
Report: `OracleGap = OracleMacroF0.5 - FinalMacroF0.5`.
This distinguishes candidate-generation bottleneck from matcher/decoder bottleneck.
Add to every experiment report.

### Patch 10: Final holdout protocol (amends §13)
Explicit sequence:
1. Reserve untouched holdout (~10%)
2. Nested validation on working data → choose architecture/hyperparameters/decoder/budget
3. Freeze choices
4. Evaluate ONCE on holdout
5. After confirmation, retrain selected pipeline on 100% training data
6. Generate final test outputs

### Patch 11: F₀.₅ wording (amends §0)
The 4× FP/FN denominator weighting is a property of the formula. The actual change in per-S1 F₀.₅ depends on the entity's TP/FP/FN configuration. Do not describe it as a fixed per-error score penalty.

### Patch 12: No further expansion
Keep everything else as v5.1 specifies. Do not add models or complexity unless experiments show measurable improvement.

---

> **Planning phase is CLOSED.** Implementation begins with E0.
