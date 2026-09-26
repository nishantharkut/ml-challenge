prompt : There is another instruction too in the problem statement : Update: candidate_pairs.tsv is part of your final submission

Blocking has to scale. Amazon resolves business entities across billions of records, so comparing every record with every other one is not an option. Your blocking / candidate-generation step must cut the search space to a small candidate set per Source 1 entity.
Candidate generation counts toward the final ranking. We will review your candidate_pairs.tsv and the code that produces it when deciding final rankings, alongside your matching_results.tsv score. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard

Yes. **This update materially changes how we should interpret the candidate-generation objective**, and it confirms one thing I previously underweighted:

> Candidate generation is not merely an internal recall-preserving preprocessing step. **It is itself part of the final evaluation/ranking.**

The important distinction is between **leaderboard F0.5** and the **final overall evaluation**, where they will also inspect candidate-set size and the blocking implementation.

### What this means for our plan

The objective is now explicitly **multi-objective**:

1. **High final matching F0.5**
2. **High candidate recall**
3. **Small candidate set per S1**
4. **Scalable blocking/code**
5. **No invalid candidates / no missed final matches**

The dangerous mistake would be to interpret this as:

> “Just make K = 10 or K = 5 because smaller is better.”

**That is still wrong.**

The statement says *small candidate set*, but it does **not specify a fixed K**. Since S1 can legitimately match many S2/S3 records, an arbitrary K can destroy recall.

### The correct optimization problem

For each S1 entity \(i\), our blocking stage should produce:

$$
C_i = \text{candidate S2/S3 records}
$$

and we need to minimize something like:

$$
\text{candidate burden}
$$

while maintaining:

$$
\text{candidate recall} =
\frac{\#\text{true matches contained in candidates}}
{\#\text{true matches}}
$$

as close to 100% as possible.

But because the final ranking explicitly rewards smaller candidate sets, we should **measure the entire recall-vs-candidate-size curve**.

For example, suppose experiments produce:

| Blocking configuration | Avg candidates/S1 | 99th percentile | GT candidate recall |
| ---------------------- | ----------------: | --------------: | ------------------: |
| A                      |                35 |              80 |               99.9% |
| B                      |                18 |              42 |               99.7% |
| C                      |                11 |              27 |               99.2% |
| D                      |                 7 |              18 |               96.1% |

We should not blindly choose D.

We should find the **smallest candidate regime that preserves an acceptable amount of true-match coverage**, then verify its effect on actual entity-level F0.5.

### And there is an even more important implication

The candidate set is defined by the problem as the set **actually fed into the final matching model**.

So we cannot cheat the evaluation by saying:

> “Internally I generated 100 candidates, but I'll only write 10 into candidate_pairs.tsv.”

That would be inconsistent with the required pipeline.

Likewise, if a true match is absent from `candidate_pairs.tsv`, the final matcher **cannot recover it**.

So the pipeline is:

**Blocking → candidate_pairs.tsv → matching model → matching_results.tsv**

not:

**Huge hidden candidate pool → matching model → write a small candidate_pairs.tsv afterward.**

The statement explicitly makes `candidate_pairs.tsv` the final candidate set.

### Therefore, I would change the master plan

Instead of the old:

> **Union retrieval → hard prune to K=10**

I would make it:

> **High-recall multi-channel retrieval → candidate-set compression → measure recall/size frontier → select validated operating point → candidate_pairs.tsv → pair matcher**

And the compression stage itself must be **recall-constrained**.

For every training fold, we should calculate at least:

* candidate recall
* complete-entity candidate recall
* average candidates/S1
* median candidates/S1
* P95/P99 candidates/S1
* maximum candidate count
* candidate reduction ratio
* S2 recall
* S3 recall
* singleton handling
* recall by country
* recall by script/cross-script status
* recall for missing-address entities
* recall for entities with many true matches

That last one is especially important because the problem explicitly allows **many matches per S1**.

### One correction to our previous thinking

The update means I would **not** describe candidate count merely as a “secondary engineering metric.”

It is a **competition objective**.

But I still would **not hard-code `max_k=10`** unless experiments on the actual training data demonstrate that it lies on the desirable recall/size frontier.

The winning strategy should therefore be:

$$
\boxed{
\text{maximize F0.5}
\quad+\quad
\text{maximize candidate efficiency}
\quad\text{subject to near-complete candidate recall}
}
$$

And importantly, **we should use the actual training TSVs to discover that operating point rather than guessing it from the statement or the previous v3 plan.**

The update you quoted is absolutely something that must be incorporated into the final master plan.


---
---

prompt : So what has to be done now? Should I give all your response to plan maker? Think deeply

Yes, but **do not dump all my previous responses into the plan maker as-is**. That will create conflicting instructions and make the plan harder to reason about.

What you need now is a **single consolidated correction brief** that tells the plan maker exactly what must be reconsidered before producing the next version.

The most important thing is that the plan maker should **not modify v3 by patching a few lines**. It should re-audit the architecture around the newly confirmed competition objective.

Send this:

Before producing the next version of the master plan, do a full re-audit of the existing plan against the official problem statement, including the latest update that candidate_pairs.tsv is part of the final submission and that smaller candidate sets are explicitly preferred in final ranking.

Do NOT simply patch v3. Treat v3 as a hypothesis document and rebuild any part that is not supported by the statement or by leakage-safe experiments.

Critical requirements:

1. Candidate generation is a competition objective, not merely preprocessing.
   The final candidate_pairs.tsv will be reviewed along with matching_results.tsv, and smaller candidate sets per Source 1 entity are explicitly preferred in final ranking, subject to the matching quality/relevance of those candidates.

2. candidate_pairs.tsv must represent the actual final candidate set fed into the matching model.
   Do not generate a large hidden candidate pool and then artificially write a smaller candidate_pairs.tsv. The submitted candidate_pairs.tsv must correspond to the candidates actually available to final matching.

3. Do NOT hard-code max_k=10 (or any arbitrary K).
   The official statement does not specify K, and Source 1 may have zero, one, or many matches. Determine the smallest empirically justified candidate regime using the training data.

4. Treat candidate generation as a constrained optimization problem:
   minimize candidate-set size / maximize reduction while preserving extremely high candidate recall and ultimately strong entity-level F0.5.

Measure the full recall-vs-candidate-size frontier rather than assuming a particular K.

At minimum evaluate:

* candidate recall
* complete-entity candidate recall
* average candidates per S1
* median candidates per S1
* P95/P99 candidate count
* maximum candidate count
* candidate reduction ratio
* S2 recall
* S3 recall
* recall by country
* recall by script/cross-script status
* recall with missing addresses
* recall for high-match-count S1 entities
* singleton behavior

5. The official final metric is entity-level macro F0.5, not ordinary pair-level F0.5.
   Threshold selection and final decoding must therefore be validated against the exact competition scoring procedure:

* per-S1 entity
* set of predicted matches
* beta=0.5
* macro averaging across S1
* singleton entities included.

6. Any candidate-ranking/pruning model must be recall-constrained.
   A learned ranker is allowed only if experiments demonstrate that it does not remove true matches. Candidate recall must be measured after every pruning stage.

7. Re-audit all v3 numerical claims using the actual training TSVs.
   Do not treat claims such as:

* 93.5% dictionary coverage
* 80.5% with both S2/S3 matches
* 5.58% singleton rate
* maximum 11 matches
* cross-script counts
* reverse-overlap / one-to-one assumptions
  as facts until independently reproduced from the actual data.

8. Verify whether any S2/S3 record can match multiple S1 records.
   The problem statement does not guarantee one-to-one mapping. Therefore do not use global disjoint assignment unless the training ground truth proves that the relevant mapping structure supports it and leakage-safe validation demonstrates it improves results.

9. Dictionary/transliteration learning must be leakage-safe.
   Any GT-derived transliteration dictionary, lexicon, token mapping, hard-negative mining artifact, threshold, etc. must be rebuilt using only the training portion available inside each validation fold. No validation-label leakage.

10. Country handling must remain open-set.
    Do not hard-code only US/India/France as the conceptual architecture. France is mentioned for test, but the system should not depend on a fixed country list.

11. Re-evaluate the retrieval architecture.
    The preferred conceptual structure should be:

multi-view normalization
→ separate name/address retrieval channels
→ high-recall multi-channel blocking
→ candidate-set compression under measured recall constraints
→ candidate_pairs.tsv
→ pairwise matching model
→ S1-level set decision
→ matching_results.tsv

Do not assume that one embedding model, one transliteration system, or one blocking method is sufficient.

12. Name and address retrieval should be considered separately.
    The official problem explicitly describes different noise patterns for names and addresses. Do not rely entirely on a single concatenated representation.

13. Dense multilingual embeddings, Indic transliteration, TF-IDF, character n-grams, token retrieval, etc. are engineering hypotheses, not facts.
    Benchmark them using entity-level validation and candidate recall. Do not declare one approach universally superior without experiments.

14. Optimize the entire pipeline against two linked objectives:
    A. candidate efficiency
    B. final entity-level F0.5

A blocking method that gives excellent F0.5 but unnecessarily huge candidate sets is not sufficient.
A blocking method that produces tiny candidate sets but misses true matches is also not sufficient.

15. Explicitly design experiments to find the Pareto frontier between:
    candidate set size ↔ candidate recall ↔ final F0.5.

The final candidate configuration should be selected from this evidence, not from an arbitrary candidate-count target.

16. Clearly separate three categories throughout the plan:
    FACT FROM OFFICIAL PROBLEM STATEMENT
    EMPIRICAL FACT REPRODUCED FROM THE DATA
    HYPOTHESIS / ENGINEERING CHOICE TO BE TESTED

Do not present assumptions or expected score ranges as established facts.

17. Do not include unsupported claims such as “this will win”, competitor-error percentages, guaranteed score ranges, or claims that a particular model is definitely optimal.

18. The revised plan must include an exact validation protocol and ablation table covering at least:

* blocking channels
* transliteration/normalization
* candidate pruning
* pairwise features/model
* thresholding
* S1-level decoding
* singleton handling
* candidate-set size vs recall
* source-specific behavior
* cross-script behavior
* missing-address behavior.

Most importantly:

The final plan must answer this question quantitatively:

“What is the smallest candidate set that we can produce for each S1 while preserving sufficiently high true-match recall and maximizing the actual entity-level F0.5?”

Do not lock the answer before measuring this on the actual training data.

### What should happen after you send this

The plan maker should produce something like **Master Plan v4**, but the first stage should **not be writing the polished plan**.

It should first do a **data audit + methodology audit** and report:

**Statement facts → verified dataset facts → experiments needed → architecture decisions.**

That is important because we still have not independently verified the numerical claims from v3 against the actual TSVs.

### Your immediate next move

Send the consolidated message above to the plan maker.

Then, ideally, **give the plan maker the actual training TSVs as well**. Without the real data, it can reason about the architecture, but it cannot honestly determine the optimal candidate-set size, recall frontier, cross-script behavior, one-to-many structure, or whether the v3 numerical claims are actually true.

The biggest mistake now would be to ask it to produce another polished “final plan” **before doing that empirical audit**.


---
---

prompt : Sure it's perfect and your view is perfect? You have more access to everything, internet and everything. Think deeply
And sure not anything is missed? 
See everything uploaded

---

No. After going back through **both uploaded files in full**, checking the PDF pages visually as well as the parsed text, and cross-checking the currently available public challenge/dataset material, I would **not** call my previous instruction to the plan maker “perfect” yet.

I found several additional issues that matter enough that I would **not send the previous message**. The two uploads I reviewed are the full 7-page official problem statement and `master_plan_v3.md`.  

One important point: the **later candidate-pairs update you pasted is not present in the PDF you uploaded**, so I am treating that update as an additional official competition rule supplied by you. The PDF itself already says that `candidate_pairs.tsv` is the exact last candidate set fed to the model and that Amazon will analyze its recall ceiling and reduction ratio. 

## The biggest things I would correct

### 1. The v3 blocking architecture is too aggressive

The biggest remaining error is still:

> `20–35 raw candidates → top 10 → candidate_pairs.tsv`

That is not justified.

The official task allows **zero, one, or many** matches per S1. 

Even if v3's claimed training statistic of maximum 11 matches is correct, that does **not** logically justify a maximum of 10 candidates. The test distribution can differ.

So the plan maker must discover:

$$
\text{candidate size} \leftrightarrow
\text{candidate recall} \leftrightarrow
\text{final F}_{0.5}
$$

from validation.

No `max_k=10` should exist in the architecture until experiments justify it.

---

### 2. Much more serious: blocking has to be separate for S2 and S3

The v3 plan retrieves from a combined `S2 ∪ S3` pool.

That creates a subtle failure:

Suppose the correct matches are:

* 4 S2 records
* 3 S3 records

If the global retrieval returns its top 10 and S2 happens to dominate similarity, you could get:

* 9 S2
* 1 S3

and lose true S3 matches **before the ML model ever sees them**.

This is especially important because v3 itself claims that a large majority of S1 entities have matches in both sources. 

The safer architecture is:

**S1 → S2 retrieval**

and independently

**S1 → S3 retrieval**

then merge the candidates.

Candidate budget can be allocated independently to S2/S3 and later made adaptive.

This is one of the things my previous brief should have made explicit.

---

### 3. The TF-IDF/embedding design may not actually scale

This is a major engineering concern that v3 underestimates.

The public dataset mirror currently reports roughly:

* 2.2M train S1
* 5.0M train S2
* 5.3M train S3
* 1.73M test S1
* 4.89M test S2
* 5.08M test S3

for about **26.4M total records**. ([Hugging Face][1])

So a design that conceptually says:

> build a 100K-feature character TF-IDF matrix over millions of records → matrix multiply S1 against S2/S3 → top-k

needs an actual **memory and runtime design**, not just a parameter choice.

Likewise, 10M records × 1024-dimensional float32 BGE-M3 vectors would be roughly **40 GB just for the raw vector data**, before the index and overhead.

Therefore the plan must explicitly decide:

* chunked retrieval
* source-separated indices
* on-disk/memmap representations
* sparse inverted retrieval
* compressed ANN if embeddings are used
* memory budget
* peak RAM measurement
* runtime measurement

A retrieval method that is theoretically good but cannot finish on the available machine is not a solution.

---

### 4. We need an S1-level decoder, not merely a pair classifier + threshold

This is probably the biggest modelling issue remaining.

The official score is:

> calculate F0.5 **for each S1 entity**, then macro-average.

It is not a pairwise classification metric. 

Yet v3 still essentially does:

> pair probability → global threshold → collect matches.

That is a surrogate, not the actual optimization target.

For each S1, the decision is:

$$
\hat M_i \subseteq C_i
$$

where \(C_i\) is its candidate set.

We should explicitly experiment with:

* probability threshold
* top-k prefix
* threshold + score-gap rule
* adaptive number of accepted matches
* possibly an expected-F0.5 decoder using calibrated probabilities

and select these rules by **actual per-S1 macro F0.5**.

This is much more important than obsessing over whether LightGBM has 35 or 40 features.

---

### 5. The “singleton math” in v3 is wrong

The plan says things such as:

> “Missing a match costs ~0.06 F0.5”
> “False merge costs 0.22”
> “Defaulting to singleton when uncertain is ALWAYS safer.”

Those are not generally valid statements.

The cost depends on:

* number of true matches
* number already predicted
* number of false positives
* the S1's current precision/recall

For example, the effect of adding one prediction to an entity with 1 true match is different from adding one to an entity with 8 true matches.

The only unconditional statement from the problem is that a **true singleton** gets 1.0 with an empty prediction and 0.0 with any false match. 

So the singleton logic must be learned/validated, not justified with those fixed penalty numbers.

---

### 6. Candidate recall must be measured at the entity level

The v3 plan uses something like:

> candidate recall ≥ 97%

That isn't enough.

We need at least these two distinct measurements:

**Pair recall**

$$
\frac{\text{true pairs present in candidates}}
{\text{all true pairs}}
$$

and

**Complete-entity recall**

$$
\frac{\#\{S1_i: GT_i \subseteq C_i\}}
{\#S1}
$$

The second one is extremely important.

A blocking system could achieve 99% pair recall while systematically missing one true match from thousands of entities. That creates a very different effect on macro F0.5 than the same number of missed pairs concentrated elsewhere.

The candidate-generation selection should therefore inspect both.

---

### 7. Name transliteration was deeply considered, but address transliteration is underdeveloped

The official statement explicitly says that **addresses can have transliteration variants**. 

V3 spends a lot of engineering effort on cross-script **names**, including the mined dictionary, but the same reasoning needs to exist for addresses.

Especially:

* house/street components
* locality names
* city names
* landmark text
* mixed-script addresses

We should have multiple address views too:

`raw script → normalized → transliterated/phonetic → structured tokens`

not just name-side transliteration.

---

### 8. “Different address number = strongest negative” must not become a hard rule

The feature itself is good.

The statement explicitly warns that addresses can have:

* missing components
* municipal numbering formats
* reordered components
* landmark-based references. 

Therefore:

> number conflict = feature

is reasonable.

But:

> number conflict = rejection

is unsafe.

The plan currently describes it as the “STRONGEST NEGATIVE.” That should be empirical language, not a hard assumption.

---

### 9. The pair-training data must be balanced at the S1/entity level

V3 samples millions of positive and negative pairs.

But if some S1 entities have many matches, those entities can dominate the training set.

That is misaligned with the competition metric, which gives each S1 equal macro weight.

So the training design should consider:

* per-S1 sampling caps
* per-S1 weights
* equal/controlled entity contribution
* separate handling of singleton S1s
* hard negatives per S1

This is another important piece missing from the prior brief.

---

### 10. Global one-to-one assignment is still not automatically safe

Interestingly, current public dataset material reports the same structural finding claimed by v3: every S2/S3 record maps to at most one S1 and there are no multi-links. ([Hugging Face][1])

So this hypothesis now has some external corroboration.

But there are two caveats:

First, that source is a public dataset mirror, not the official PDF.

Second, even if one-to-one is true, the proposed **greedy** algorithm isn't necessarily the correct optimizer.

If one-to-one is verified from the actual training ground truth, the constraint is roughly:

* each S2/S3 record → at most one S1
* each S1 → potentially many records

That is not ordinary one-to-one matching. Greedy assignment can steal a candidate from an S1 that has no good alternatives.

So if this constraint is used, compare:

* no assignment
* greedy assignment
* constrained weighted matching / b-matching style decoding

on validation.

---

### 11. The 93.5% dictionary result needs a much stricter audit

The plan calls the dictionary result:

> “93.5% full coverage — validated”

but the plan itself does not provide enough information about that validation protocol to trust the number as-is. 

We need to know:

* Were the 10K examples completely held out from dictionary mining?
* Were their GT pairs excluded?
* Were token mappings built from the same entities?
* Were rare business-specific tokens included?
* Was full translation measured on unseen entities?
* Was address translation included or only names?

If the dictionary was mined from all GT and then tested on examples drawn from those same GT-derived relationships, the 93.5% number is inflated by leakage.

This must be rebuilt fold-by-fold.

---

### 12. France strategy in v3 is too hand-wavy

This:

> “Use the MORE CONSERVATIVE of US/India thresholds for France.”

has no statistical basis.

France is unseen during training, and the official problem explicitly tells us to treat country as an **open set**, not hard-code a finite country list. 

So the better design is:

* country-agnostic core matcher
* country-independent text features
* optional country-conditioned features that gracefully handle unseen labels
* threshold/decoder selected for robust cross-country validation
* no `if country == France`

France should work because the architecture generalizes, not because France gets a special hand-written rule.

---

## So what should you send the plan maker?

**Not my previous message.**

Send a corrected instruction that says the previous v3 architecture is now under a **full technical audit**, with these mandatory additions.

Use this:

```text
Do NOT produce another patched version of v3 yet.

Rebuild the plan after auditing the full official problem statement and the entire v3 document.

The revised plan must explicitly solve these issues:

1. Candidate generation is a competition objective in addition to final F0.5.
   The latest official update says smaller candidate sets per S1 are preferred in final ranking.
   candidate_pairs.tsv must be exactly the last candidate set actually scored by the matcher.

2. Do not hard-code max_k=10.
   Determine the candidate-size operating point empirically from the training data.
   Optimize the tradeoff:
   candidate size <-> candidate recall <-> final macro F0.5.

3. Candidate generation MUST be evaluated with both:
   - pair-level candidate recall
   - complete-entity candidate recall:
     fraction of S1 entities whose entire GT match set is contained in candidates.

4. Blocking should be source-specific:
   retrieve S1->S2 and S1->S3 independently before merging candidates.
   Do not allow a combined S2+S3 top-k retrieval to crowd one source out.

5. The entire blocking system must actually scale to the dataset size.
   Give an explicit memory/runtime strategy for TF-IDF, inverted retrieval and ANN.
   Do not assume a 5-10M-record TF-IDF or dense embedding matrix fits in RAM.
   Include chunking, compression, sharding, indexing, and peak-memory measurements.

6. Treat S1-level prediction as a set-selection problem, not merely pair classification.
   The final decision for each S1 is a subset of its candidate set.
   Tune/validate entity-level decoders such as:
   - calibrated probability threshold
   - top-k/prefix
   - adaptive threshold
   - score-gap rule
   - expected-F0.5 or other set-level decoder
   using the EXACT competition macro-F0.5 evaluator.

7. Do not use pair-level sklearn fbeta_score as the main threshold objective.
   Implement the exact competition metric:
   per-S1 precision/recall/F0.5, then macro average.

8. Remove unsupported singleton penalty claims such as:
   "missing a match costs ~0.06" or "defaulting to singleton is always safer".
   Measure actual delta-F0.5 on validation.

9. Candidate pruning must be recall-constrained.
   Every pruning stage must report candidate recall and complete-entity recall.
   A learned pruning model may not be allowed to remove hard true matches without validated recall guarantees.

10. Expand transliteration analysis to ADDRESS as well as NAME.
    The official problem explicitly lists address transliteration as noise.
    Build raw-script, normalized, transliterated/phonetic and structured address views.

11. Treat address number disagreement as a feature, not an unconditional rejection rule.
    Validate its actual precision/recall effect because addresses can be partial,
    reformatted, reordered and landmark-based.

12. Rebuild all GT-derived artifacts leakage-safely inside each validation fold:
    transliteration dictionaries, token mappings, hard negatives, thresholds,
    ranker statistics, etc.
    No validation GT may influence training artifacts.

13. Training should account for entity-level macro averaging.
    Prevent high-match-count S1 entities from dominating pairwise training.
    Consider per-S1 sampling/weights and explicitly include singleton entities.

14. Do not assume global disjoint assignment.
    Verify whether each S2/S3 record maps to at most one S1 using the actual ground truth.
    If verified, compare no-assignment, greedy assignment and a proper constrained
    weighted-decoding approach. Do not assume greedy is optimal.

15. Do not hard-code US/India/France.
    Treat country as an open set as required by the official problem.
    France must work zero-shot without a France-specific rule.

16. Re-audit the claimed empirical facts from v3:
    - S1 script distribution
    - cross-script counts
    - dictionary coverage
    - singleton percentage
    - both-source match percentage
    - max/mean match count
    - distractor percentage
    - reverse overlap / one-to-one property
    - empty-address match statistics
    using the actual provided training data.

17. Every numerical claim marked "verified", "validated", "tested" or
    "expected" must identify exactly how it was measured.
    Separate:
      FACT FROM PROBLEM STATEMENT
      FACT VERIFIED FROM DATA
      HYPOTHESIS TO TEST.

18. Remove unsupported competitor percentages, guaranteed score ranges,
    "secret sauce", "this will win" and similar claims.

19. Benchmark the architecture empirically before selecting:
    TF-IDF
    character n-grams
    exact/inverted blocking
    transliteration dictionary
    phonetic views
    dense ANN
    source-specific retrieval
    pairwise LightGBM/XGBoost
    optional transformer reranking.

20. The final experiment should produce a Pareto table:
    candidate pairs/S1
    P50/P95/P99/max candidate count
    pair candidate recall
    complete-entity candidate recall
    final macro F0.5
    runtime
    peak memory

21. Do not optimize only average candidate count.
    Report the full candidate-count distribution because the competition evaluates candidate efficiency per S1.

22. The final architecture should be conceptually:

    preprocessing / multi-view representation
        -> independent S1->S2 and S1->S3 retrieval
        -> high-recall union
        -> validated candidate compression
        -> candidate_pairs.tsv
        -> pairwise matcher
        -> S1-level set decoder
        -> optional verified global constraints
        -> matching_results.tsv

23. The revised document must explicitly identify which decisions are:
    - proven from the problem statement
    - proven from the dataset
    - experimental choices.

Do not call the next document "FINAL" until the empirical audits and ablations above are complete.
```

### My current verdict

The **core direction** of v3 is still good:

**blocking → matching → entity-level decision**

and its emphasis on multilingual noise, training-derived transliteration, hard negatives, and precision-heavy evaluation is sensible. The official statement strongly supports the importance of blocking, string similarity, country-aware address handling, and singleton treatment. 

But I would **not trust v3 as a final implementation plan yet**.

The three things I would treat as highest priority before anything else are:

**(1) scalable source-separated candidate generation,
(2) exact S1-level F0.5 decoding,
(3) empirical candidate-size/recall Pareto analysis.**

Those are more fundamental than choosing BGE-M3 vs XLM-R vs LightGBM.

And importantly, **you have not uploaded the actual six train/test TSV files here**. The two uploads are only the problem statement and v3 plan. So neither I nor the plan maker should claim that the v3 numerical dataset facts have been independently verified from the actual challenge data yet.  

A current public dataset mirror does independently report the large dataset scale, zero cross-country ground-truth links, and at-most-one S1 linkage per S2/S3 record, which makes some of v3's structural claims plausible, but I would still treat the mirror as secondary evidence rather than a substitute for analyzing your actual training files. ([Hugging Face][1])

[1]: https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026?utm_source=chatgpt.com "akshatbakshi/amazon-ml-challenge-2026 · Datasets at Hugging Face"



---
---
prompt : You just saw the corrections and defects
Why not optimisation that would make the solution better? 
Think deeply
Explore more 

---

Yes. I was too focused on **finding defects in v3**. That is only half the job.

After looking again at the full plan, the official specification, current public copies of the dataset information, and recent ER/retrieval approaches, I think there is a **second layer of improvements** that could make the system materially stronger rather than merely “less wrong.”

The key insight is this:

> We should not just build a better matcher. We should build a system that **learns how difficult each S1 entity is, spends candidate budget where it is needed, and optimizes the final S1-level decision directly.**

That gives us several high-upside ideas.

---

# 1. Replace fixed candidate K with a learned candidate-budget controller

This is, in my view, the most important optimization beyond correcting v3.

Instead of:

```text
S1
 → retrieve 20
 → prune to 10
```

do:

```text
S1
 → retrieve candidate pools
 → estimate retrieval difficulty
 → allocate candidate budget
 → final candidate set
```

For example, an easy S1 might have:

* exact normalized name hit
* very rare name token
* strong name/address agreement
* multiple retrieval channels agreeing
* huge score gap after rank 1

It could safely receive perhaps 4–6 candidates.

A hard S1 might have:

* cross-script name
* missing address
* weak name similarity
* conflicting retrieval channels
* no exact/rare-token hit

It might need 15–30 candidates.

This is much better aligned with Amazon's candidate-efficiency requirement than a universal K.

### How to learn it

On each validation fold:

1. Generate a sufficiently large candidate pool.
2. For every S1, determine the **minimum candidate depth required to contain all its true matches**.
3. Record features describing retrieval difficulty:

   * best lexical score
   * best dense score
   * score gap
   * number of agreeing retrieval channels
   * name length
   * address length
   * missingness
   * cross-script
   * source
   * number of rare-token hits
   * S2/S3-specific retrieval statistics
4. Train a small budget model to predict a suitable candidate depth.

The target is not “predict match”.

The target is:

> **How much candidate budget does this S1 need?**

That is a competition-specific optimization.

---

# 2. Optimize the retrieval channels jointly, not independently

v3 says:

> TF-IDF + dense + token index → union

That's good, but it leaves a lot of efficiency on the table.

Suppose we have four retrieval channels:

| Channel      | Candidates | New GT recall |
| ------------ | ---------: | ------------: |
| rare-token   |          4 |           88% |
| char TF-IDF  |         10 |           +7% |
| address BM25 |          8 |           +3% |
| dense        |         10 |           +1% |

The correct question isn't:

> "How many candidates should each return?"

It's:

> **"Which combination of retrieval depths produces the maximum marginal recall per candidate?"**

For each channel, measure:

$$
\Delta Recall / \Delta Candidates
$$

conditional on the other channels already being active.

Then greedily allocate retrieval budget where the next candidates produce the largest marginal gain.

This creates a **candidate-generation Pareto optimizer** rather than a bag of blocking methods.

Ensemble blocking is well-established because different blocking methods recover different subsets of matches. ([arXiv][1])

---

# 3. Use Reciprocal Rank Fusion for retrieval, not arbitrary weighted averaging

Instead of trying to compare incompatible scores like:

* TF-IDF cosine = 0.73
* BM25 = 14.2
* embedding cosine = 0.81
* RapidFuzz = 0.91

use rank information.

For candidate \(c\):

$$
RRF(c)=
\sum_m \frac{1}{k+r_m(c)}
$$

where \(r_m(c)\) is the candidate's rank in retrieval channel \(m\).

This is attractive because:

* lexical retrieval can dominate exact cases
* dense retrieval can rescue semantic/cross-script cases
* address retrieval can rescue damaged names
* no need to calibrate raw retrieval-score scales

Current public implementations for this challenge are already exploring combinations such as BM25 + character TF-IDF + reranking, which reinforces that hybrid retrieval is a sensible direction. ([GitHub][2])

But I would go one step further:

**learn the channel contribution on validation**, rather than blindly use standard RRF.

---

# 4. Train a task-specific dense retriever rather than relying only on BGE-M3

This is a potentially major upgrade.

v3 treats BGE-M3 mainly as:

> generic multilingual safety net.

Instead:

### Train a small two-tower retriever from the challenge's own GT.

Training pairs:

$$
S1 \leftrightarrow S2
$$

and

$$
S1 \leftrightarrow S3
$$

Positive = actual GT match.

Hard negatives = highly similar non-matches retrieved by lexical/dense retrieval.

Then optimize:

$$
L=-\log
\frac{e^{sim(q,p^+)/\tau}}
{e^{sim(q,p^+)/\tau}+\sum_j e^{sim(q,n_j)/\tau}}
$$

This lets the embedding space learn **the exact distortions of this dataset**:

* Indian transliterations
* business abbreviations
* legal suffix changes
* address degradation
* source-specific corruption

The entity-retrieval literature supports learned dual encoders with ANN candidate generation and, importantly, hard-negative mining. ([arXiv][3])

This could outperform a frozen generic embedding model while remaining scalable.

### Even better

Don't train only one representation.

Train:

**Name retriever**

and

**Address retriever**

separately.

Then use both.

That directly addresses one of the biggest structural weaknesses in v3.

---

# 5. Learn transliteration instead of depending on a dictionary alone

This is another major potential improvement.

The v3 dictionary is clever, but it is essentially:

> observed Indic token → observed English token.

That breaks on unseen names.

Instead, use the matched training records to learn **script-specific character correspondences**.

For example:

```text
Indic token
      ↓
character alignment learned from GT
      ↓
Latin-like representation
```

For every script:

* Devanagari → Latin
* Telugu → Latin
* Kannada → Latin
* Tamil → Latin
* Bengali → Latin
* Gujarati → Latin
* Malayalam → Latin
* Odia → Latin
* Gurmukhi → Latin

You don't need a gigantic neural transliterator.

A lightweight learned character/substring mapping or small seq2seq transliterator trained only on the supplied GT can be sufficient.

Then your representation becomes:

```text
raw Indic
normalized Indic
dictionary translit
learned translit
phonetic representation
```

The dictionary handles frequent words.

The learned transliterator handles **unseen variants**.

That is substantially stronger than the current Layer-1/Layer-2 design.

---

# 6. Learn address transformations from the training pairs too

The plan concentrates heavily on name transliteration.

But the problem statement explicitly says addresses suffer from transliteration, abbreviation, missing components, landmark references, municipal-number variation and reordering. 

So build a training-derived address transformation system:

```text
street → st
road → rd
avenue → ave
...
```

but more importantly learn:

* token equivalences
* common abbreviation expansions
* component reorderings
* numeric formatting transformations
* local-place transliteration correspondences

from actual matched address pairs.

This gives us a **data-derived address canonicalizer**, not merely generic regex normalization.

---

# 7. Build a retrieval "agreement" signal

v3 has ranking features, but there is a better idea.

For every candidate, record:

```text
name_exact_hit
name_token_hit
name_char_hit
name_dense_hit
address_exact_hit
address_bm25_hit
address_dense_hit
phonetic_hit
transliteration_hit
```

Then:

$$
channel\_agreement = \#\text{independent retrieval channels supporting candidate}
$$

A candidate that appears:

```text
name retrieval rank 2
address retrieval rank 4
dense retrieval rank 3
transliteration retrieval rank 1
```

is qualitatively different from one that appears only because of a weak dense embedding.

This is especially useful for F0.5 because it gives the final matcher a strong way to distinguish **multi-signal evidence from one-channel coincidence**.

---

# 8. Add one-hop S2↔S3 bridge retrieval

This is an interesting optimization that v3 didn't exploit enough.

Suppose:

```text
S1 ── weak ── S2-A
S2-A ── strong ── S3-B
```

and S3-B has poor direct similarity to S1.

S3-B may nevertheless be a real match.

So after direct S1→S2 and S1→S3 retrieval:

```text
strong S1-S2 candidates
        ↓
retrieve similar S3 records
```

and symmetrically:

```text
strong S1-S3 candidates
        ↓
retrieve similar S2 records
```

But **only one hop**.

Do not form connected components.

Do not propagate recursively.

Use the bridge only as a **candidate-generation rescue channel**.

This can recover cases where the two noisy sources each preserve different pieces of the business identity.

The key is to validate whether the added recall justifies the extra candidate burden.

---

# 9. Use a pairwise ranking objective in addition to classification

v3 uses binary LightGBM.

That's fine, but our actual decision is:

> rank true matches above false candidates for each S1.

So test a second model trained with a group-wise ranking objective:

```text
S1_1:
   positive
   hard negative
   hard negative
   positive
   hard negative

S1_2:
   ...
```

The S1 entity becomes the ranking group.

Then compare:

* binary classifier
* LambdaMART/ranking model
* classifier + ranker ensemble

This is especially attractive because candidate selection itself is essentially a ranking problem.

I would expect this to matter more than endlessly adding small handcrafted features.

---

# 10. The biggest modelling upgrade: an S1-level F0.5 decoder

This is where we can go beyond ordinary thresholding.

Suppose for one S1, calibrated model probabilities are:

```text
0.97
0.92
0.88
0.61
0.31
0.16
0.08
```

A universal threshold of 0.70 gives 3 predictions.

But the competition score is a **set metric per S1**, not a pairwise metric.

For β=.5:

$$
F_{0.5}
=
\frac{1.25TP}
{0.25T+k}
$$

where:

* \(T\) = total true matches for that S1
* \(k\) = number predicted
* \(TP\) = true positives

That denominator simplification is extremely useful.

With calibrated probabilities, we can estimate expected TP for each prefix:

$$
E[TP_k]=\sum_{j=1}^{k}p_j
$$

and estimated expected true-match count:

$$
E[T]=\sum_j p_j
$$

giving a useful approximate decoder:

$$
\widehat F_{0.5}(k)
=
\frac{1.25\sum_{j=1}^{k}p_j}
{0.25\sum_j p_j+k}
$$

Then choose the prefix \(k\) that maximizes this estimate.

For \(k=0\), separately estimate the probability that the S1 is a singleton.

This is **far more aligned with the actual competition objective** than:

```python
probability > 0.72
```

I would absolutely test this.

---

# 11. Make singleton prediction probabilistic

Instead of:

> best probability < threshold → singleton

estimate:

$$
P(T=0)
$$

from the calibrated candidate probabilities.

Under an independence approximation:

$$
P(T=0) \approx \prod_j(1-p_j)
$$

Then compare:

```text
Expected F0.5(empty)
vs.
Expected F0.5(best non-empty set)
```

This creates a mathematically motivated **abstain/no-match decision**.

It is much better than v3's arbitrary two-threshold singleton gate.

Again, it must be evaluated empirically because the pair probabilities may not be independent or perfectly calibrated.

---

# 12. Add model uncertainty as a feature

Train several models:

```text
LightGBM seed 1
LightGBM seed 2
LightGBM source-specific
possibly neural scorer
```

For each candidate:

$$
\mu = mean(probabilities)
$$

$$
\sigma = std(probabilities)
$$

Then:

* high probability + low variance → strong
* high probability + high variance → suspicious
* medium probability + low variance → borderline but stable
* low probability + high variance → unreliable

The decoder can become more conservative when uncertainty is high.

This is particularly valuable under precision-heavy F0.5.

---

# 13. Hard-negative mining should become *structured*, not merely larger

The v3 plan says hard negatives.

Go one level deeper.

Create explicit negative categories:

### Type A — same name, different address

Very dangerous false merges.

### Type B — same address, different business name

### Type C — same transliteration, different native-script entity

### Type D — same legal suffix, different core

### Type E — same rare token but different business

### Type F — same phone/pincode/number, different name

### Type G — retrieval-channel disagreement

### Type H — S2/S3 bridge false matches

Then ensure each category is represented during training.

This teaches the matcher **why a candidate is wrong**, not merely that it is wrong.

---

# 14. Synthetic positive augmentation from observed noise

This is another underused opportunity.

The training GT tells us what real corruption looks like.

For a genuine pair:

```text
ABC PRIVATE LIMITED
```

we can create realistic augmented forms using transformations observed in the training data:

```text
ABC PVT LTD
ABC PRIVATE LTD
ABC & CO
abc private ltd
ABC PVT
word reordering
punctuation removal
component deletion
address token reorder
etc.
```

For Indian names, use transformations learned from actual transliteration pairs.

This creates additional positive training examples without external data.

The critical restriction:

> Only use transformation types/statistics discovered from the supplied training data.

That stays within the spirit of the challenge.

---

# 15. Use OOF-calibration everywhere

This deserves its own architecture layer.

Do not tune thresholds on predictions from the same training data used to fit the model.

Pipeline:

```text
Fold 1
 train → predict fold 1

Fold 2
 train → predict fold 2

...

OOF predictions
      ↓
calibration
      ↓
threshold/set-decoder optimization
```

Then:

```text
train final model on all train data
      ↓
apply frozen calibration/decoder
      ↓
test
```

This prevents a subtle but very damaging form of threshold overfitting.

---

# 16. Use test data transductively — but only unsupervised

This is another improvement that is allowed in most ER setups and does not require external lookup.

The test corpus is available.

Use it for things such as:

* IDF statistics
* vocabulary frequency
* script distribution
* character frequency
* address token distributions
* candidate index construction

For example, fitting the lexical retrieval IDF statistics on the actual test corpus can make rare French business terms more informative.

But:

**never use test labels, external business information, or internet enrichment.**

The problem prohibits external data lookup/augmentation. 

---

# 17. Use a "recall certificate" for candidate generation

This is another concept I'd add to the master plan.

Every candidate-production configuration should produce a report like:

```text
Overall pair recall             99.72%
Complete S1 recall              99.31%

S2 pair recall                  99.81%
S3 pair recall                  99.62%

Cross-script pair recall        99.10%
Empty-address recall            99.34%
High-match-count S1 recall      99.05%

Median candidates/S1             6
P95 candidates/S1               13
P99 candidates/S1               21
Max                              47
```

Then the candidate generator is not accepted unless it passes predefined recall gates.

This gives us a **blocking contract**:

> No later optimization can silently destroy retrieval recall.

---

# 18. Use a Pareto frontier rather than choosing one candidate-size target

This should replace v3's:

> average ≤ 8–10

completely.

Generate configurations such as:

| Variant | Avg | P95 | Complete S1 recall | F0.5 |
| ------- | --: | --: | -----------------: | ---: |
| A       |  25 |  47 |              99.8% | 0.91 |
| B       |  17 |  31 |              99.7% | 0.91 |
| C       |  11 |  22 |              99.4% | 0.90 |
| D       |   7 |  15 |              96.8% | 0.86 |

A is dominated by B if B has better/equal quality and fewer candidates.

So we keep the **Pareto-optimal configurations**.

Then decide based on the observed leaderboard and final review considerations.

This is much more defensible than guessing "10".

---

# 19. I would test a three-model matcher stack

Rather than betting everything on one model:

### Model A — fast GBDT

Features:

* RapidFuzz
* TF-IDF
* numbers
* address structure
* retrieval ranks
* channel agreement
* script/transliteration features

### Model B — learned pair encoder / cross encoder

Only difficult candidates.

### Model C — deterministic evidence layer

Very high-confidence rules:

* exact normalized identity
* extremely strong name + address agreement
* exact high-confidence phone/email/identifier when present

Then combine:

$$
score =
w_A P_A +
w_B P_B +
w_C Evidence
$$

where weights are learned on OOF validation.

This gives us:

**speed + structured evidence + deep semantic matching.**

A recent multilingual reranker such as BGE-reranker-v2-m3 is Apache-2.0, while mDeBERTa-v3-base is currently listed as MIT, so there are technically compliant reranking options to benchmark under the challenge's license constraint. ([Hugging Face][4])

BGE-M3 itself is currently listed as MIT. ([Hugging Face][5])

That doesn't mean we should automatically use them; they need actual validation.

---

# 20. One thing I would NOT do

I would **not** make the solution increasingly complicated just because the complexity sounds impressive.

The strongest architecture is likely:

```text
                    ┌─ lexical exact/rare-token
                    ├─ character retrieval
S1 ──> retrieval ───┼─ address retrieval
                    ├─ learned dense retrieval
                    └─ one-hop S2/S3 bridge

                       ↓
                 RRF / candidate fusion

                       ↓
             learned candidate budget

                       ↓
                 candidate_pairs

                       ↓
             GBDT + optional reranker

                       ↓
             calibrated probabilities

                       ↓
             S1-level F0.5 decoder

                       ↓
             matching_results
```

Not:

```text
27 models
+ 83 heuristics
+ 5 embeddings
+ graph clustering
+ random thresholds
```

The goal is **controlled complexity with measurable marginal gain**.

---

# My revised priority order

After thinking through the whole thing, I'd rank the optimization work like this:

### Tier 1 — highest potential

**1. Learned adaptive candidate budget**

**2. Source-separated retrieval**

**3. Task-trained dense retriever**

**4. Exact S1-level F0.5 decoder**

**5. Hybrid retrieval with channel agreement/RRF**

### Tier 2 — very worthwhile

**6. Learned transliteration for unseen Indic tokens**

**7. Training-derived address normalization/transformation**

**8. Structured hard-negative mining**

**9. OOF calibration + uncertainty**

**10. S2↔S3 one-hop bridge retrieval**

### Tier 3 — test after the above

**11. Cross-encoder on difficult cases**

**12. Ranking-loss model vs binary classifier**

**13. Synthetic noise augmentation**

**14. Ensemble of independently trained matchers**

---

## And there is one especially important new idea

I think the **candidate generator and matcher should share information**.

Traditional thinking is:

```text
blocking
   ↓
matcher
```

But here we can make it:

```text
retrieval
   ↓
candidate difficulty estimation
   ↓
candidate budget
   ↓
matcher
   ↓
S1 decision
   ↓
feedback to validate retrieval policy
```

The matcher tells us, on training folds, **which retrieval candidates were actually useful**.

That lets us learn:

> “For this kind of S1, how much retrieval budget is necessary?”

That directly attacks the competition's unusual second objective: **small candidate sets**.

Current public challenge projects already show people converging on BM25/TF-IDF hybrid retrieval, hard negatives and margin-based thresholding. ([GitHub][2]) The opportunity for our plan is to go beyond that baseline by **learning the retrieval budget and optimizing the S1-level set decision**, instead of simply stacking more retrieval methods.

So yes: **there is considerably more optimization to do than just fixing the defects I previously identified.**

And I would now tell the plan maker to create **v4 around these optimization principles**, rather than merely producing “v3 with bugs removed.”

[1]: https://arxiv.org/abs/1609.06265?utm_source=chatgpt.com "An Ensemble Blocking Scheme for Entity Resolution of Large and Sparse Datasets"
[2]: https://github.com/ChandrimaNandi/Amazon-ML-Hackathon-2026?utm_source=chatgpt.com "GitHub - ChandrimaNandi/Amazon-ML-Hackathon-2026 · GitHub"
[3]: https://arxiv.org/abs/1909.10506?utm_source=chatgpt.com "Learning Dense Representations for Entity Retrieval"
[4]: https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main?utm_source=chatgpt.com "BAAI/bge-reranker-v2-m3 at main"
[5]: https://huggingface.co/BAAI/bge-m3/tree/main?utm_source=chatgpt.com "BAAI/bge-m3 at main"


---
---

prompt : I think you should do deep research
See ai ml and relevant research papers too
Find the best things. Think deeply 

---


I did a deeper literature pass specifically for this challenge: modern entity resolution, learned blocking, multilingual retrieval, transliteration, hard-negative mining, transformer matching, set-level prediction, and scalable IR. I also checked the currently public copies/audits of the challenge data and public challenge implementations.

The conclusion is stronger than “fix v3”:

> **The best architecture is probably not a better version of v3's 3-stage TF-IDF → top-10 → LightGBM pipeline.**
>
> It should be a **learned blocking + hybrid retrieval + hard-negative-trained matcher + S1-level set decoder**, with specialized treatment for Indian-script/transliteration cases.

And there are several research-backed ideas v3 completely missed.

---

# 1. The biggest upgrade: make blocking itself learned

This is the first thing I would change.

v3 currently has:

```text
TF-IDF
dense ANN
token index
      ↓
union
      ↓
hand-designed RapidFuzz score
      ↓
top 10
```

Research has moved considerably beyond that.

### SC-Block

SC-Block uses **supervised contrastive learning specifically for entity blocking**, followed by nearest-neighbor search. In its experiments it produced substantially smaller candidate sets while maintaining very high pair completeness; on a large benchmark, pipelines using it were reported to be substantially faster than competing blockers. ([arXiv][1])

This is **almost perfectly aligned with this competition's unusual requirement** that candidate-set size itself matters.

Instead of:

$$
\text{generic embedding} \rightarrow kNN
$$

we learn:

$$
\text{S1/S2/S3 challenge records}
\rightarrow
\text{identity-aware embedding space}
\rightarrow
kNN
$$

using the actual training GT.

### UniBlocker

UniBlocker takes a similar idea into dense blocking and shows dense blocking can be complementary to sparse methods. ([arXiv][2])

### The important conclusion

We should have a dedicated model:

> **Challenge-trained blocking encoder**

not merely BGE-M3.

The training objective is something like:

$$
L =
-\log
\frac{
e^{sim(q,p^+)/\tau}
}{
e^{sim(q,p^+)/\tau}+
\sum_j e^{sim(q,n_j)/\tau}
}
$$

where:

* \(q\) = S1 record
* \(p^+\) = real S2/S3 match
* \(n_j\) = hard non-matches

Then ANN search operates over the learned space.

That is a much more principled solution to the candidate-pairs problem.

---

# 2. Keep lexical retrieval — don't replace it with embeddings

Another important conclusion from the literature:

**Do not make the mistake of going all-neural.**

Sparkly showed that relatively simple TF/IDF top-k blocking can be surprisingly strong and explicitly studied the trade-off between recall, output size and runtime. It outperformed eight blocking baselines in its evaluation. ([DOI][3])

That is especially relevant here because business names contain things embeddings are not necessarily good at:

```text
ABC PVT LTD
ABC PRIVATE LIMITED
ABC PVT. LTD.
ABC PRIVATE LTD
```

Character/token lexical retrieval is excellent at these.

So the learned blocker should be **one channel**, not the only channel.

I would use:

```text
Channel 1: exact / rare-token retrieval
Channel 2: character TF-IDF / BM25-like retrieval
Channel 3: challenge-trained dense blocker
Channel 4: multilingual specialist retrieval
Channel 5: address-specific retrieval
```

Then fuse them.

---

# 3. Use rank fusion instead of manually mixing incomparable scores

TF-IDF cosine, BM25, embedding cosine, RapidFuzz and neural scores live on very different scales.

Rather than saying:

```text
0.5 * tfidf + 0.3 * dense + 0.2 * fuzzy
```

use rank fusion.

Reciprocal Rank Fusion is an old but very robust IR technique for combining independent ranked lists and has repeatedly shown that combining rankings can outperform individual systems. ([DOI][4])

For candidate \(c\):

$$
RRF(c)
=
\sum_m
\frac{1}{k+r_m(c)}
$$

where \(r_m(c)\) is candidate rank in retrieval channel \(m\).

For this challenge that's attractive because different channels are solving different failure modes.

Even better:

### Phase 1

Use RRF as an unbiased fusion baseline.

### Phase 2

Train a tiny learned fusion model using:

* rank from each channel
* score from each channel
* number of channels retrieving candidate
* source
* script
* name/address availability

That gives us **learned retrieval fusion**.

---

# 4. BGE-M3 is useful, but v3's role for it is wrong

BGE-M3 is genuinely interesting for this dataset because it supports multilingual retrieval and provides dense, sparse and multi-vector retrieval capabilities in one model. Its current model card lists ~569M parameters and an MIT license. ([arXiv][5])

So instead of:

> “BGE-M3 = safety net”

I'd test:

```text
BGE-M3 dense retrieval
BGE-M3 sparse retrieval
BGE-M3 multi-vector retrieval
```

against:

```text
TF-IDF/BM25
challenge-trained dense blocker
```

But there is a crucial distinction:

> **Frozen BGE-M3 is the baseline. Fine-tuned challenge-specific retrieval is the serious model.**

The challenge gives us millions of positive links, so there is an unusually large amount of domain-specific supervision available.

---

# 5. We discovered a very relevant model that v3 completely missed: MuRIL

This is probably the most important multilingual addition.

Google's **MuRIL** was pretrained specifically for Indian languages and, critically, on both native-language and transliterated forms. The model card says it covers 17 Indian languages plus their transliterated counterparts, and its research paper specifically reports improved performance on transliterated Indian-language text. It is Apache-2.0 licensed. ([Hugging Face][6])

That matches our challenge extremely well.

The challenge explicitly tells us that names and addresses may contain transliterations, and v3's own claimed dataset analysis says large numbers of S2/S3 records use Indic scripts while S1 is Latin. The official statement confirms transliteration is a named noise pattern. 

### I would therefore test a specialist branch:

```text
IF either record contains Indic script:
    MuRIL representation
    +
    learned/fuzzy transliteration representation
    +
    lexical features
ELSE:
    general multilingual representation
```

Not as a hard country rule.

Not:

```text
if country == India
```

but:

```text
if script == Indic
```

This also lets the model handle code-mixed cases.

---

# 6. The final matcher should probably be a two-model system

The research strongly supports separating:

**retrieval**

from

**fine-grained matching**.

Ditto demonstrated that pretrained transformer sequence-pair matching can be extremely strong for entity matching and reported strong results on large company-matching data. ([arXiv][7])

Address matching research also uses a very similar structure:

> bi-encoder retrieval → cross-encoder reranking. ([arXiv][8])

So I would build:

### Fast matcher

LightGBM/XGBoost:

```text
name similarities
address similarities
numeric evidence
TF-IDF
retrieval ranks
channel agreement
script
missingness
source
etc.
```

Then:

### Neural matcher

Run only on a **small difficult candidate subset**.

Input should be field-aware:

```text
[NAME]
S1 name

[ADDRESS]
S1 address

[COUNTRY]
S1 country

[SEP]

[NAME]
candidate name

[ADDRESS]
candidate address

[COUNTRY]
candidate country
```

This is preferable to simply concatenating:

```text
name + address
```

because entity-matching research has repeatedly found that attribute relationships and field structure matter. Relation-aware matching methods explicitly model relationships between attributes instead of treating them as independent strings. ([ScienceDirect][9])

---

# 7. For the neural matcher, I would benchmark three candidates

Not pick one blindly.

### A. mDeBERTa-v3-base

MIT licensed, but its model card lists only 16 languages. ([Hugging Face][10])

Good general candidate, but I would **not assume** it is best for the nine Indian scripts.

### B. BGE-reranker-v2-m3

Apache-2.0 and multilingual. ([Hugging Face][11])

Very attractive as a reranking starting point.

### C. MuRIL

Apache-2.0, specifically trained for Indian native + transliterated text. ([Hugging Face][6])

I would use this particularly for Indic/script-mismatch cases.

So a plausible final ensemble becomes:

```text
General pair scorer
        +
Indic-specialist scorer
        ↓
OOF calibrated fusion
```

rather than forcing one model to be excellent at every phenomenon.

---

# 8. A challenge-trained dual encoder is more important than a giant model

The current public dataset mirror reports roughly **24.2M source records** across the six TSVs and about **9.97M S2/S3 records in the test gallery**. An independent public audit of the challenge files reports approximately **17.27 trillion raw S1×S2/S3 test comparisons before even considering country filtering**. ([Hugging Face][12])

That changes the model economics.

A giant cross-encoder cannot be the primary retrieval mechanism.

The best architecture is almost certainly:

```text
cheap retrieval
       ↓
learned retrieval
       ↓
small candidate set
       ↓
expensive interaction model
```

This is exactly why late-interaction retrieval systems such as ColBERT were developed: they preserve more detailed token-level interaction than single-vector retrieval while still allowing precomputed representations and scalable search. ColBERTv2 further compresses its representation substantially. ([ACL Anthology][13])

### ColBERT-style retrieval is therefore worth one experiment

Not necessarily as the final system.

But as:

> **high-recall rescue retrieval**

for difficult multilingual/cross-script cases.

---

# 9. Hard-negative mining should be redesigned around ER, not generic negatives

The literature repeatedly shows hard negatives matter substantially for dense retrieval. Multilingual retrieval research explicitly reports gains from stronger hard-negative construction, and more recent work continues to emphasize high-quality hard negatives and better data utilization. ([ACL Anthology][14])

For this challenge I would create **specific hard-negative families**:

### H1

Same normalized name, different entity.

### H2

Very similar name, different address.

### H3

Same address, different name.

### H4

Same rare business token, different entity.

### H5

Same transliteration, different entity.

### H6

Same house/PIN/city, different business.

### H7

Dense retriever false positive.

### H8

Lexical retriever false positive.

### H9

S2↔S3 bridge false positive.

### H10

Previously high-scoring LightGBM false positive.

That is far more valuable than merely collecting millions of random negatives.

---

# 10. Protect the dense retriever from false negatives

This is subtle and important.

Suppose an S1 has:

```text
S2-A = true
S2-B = true
S3-C = true
```

A naïve contrastive learning batch might treat S2-B as a negative when training the S1/S2-A pair.

That is disastrous.

The public audit of this challenge itself recommends masking other true variants from the contrastive denominator. ([GitHub][15])

Therefore:

> **All known positives for the same S1 must be excluded from negative sampling.**

And ideally:

* all linked IDs for that entity
* possibly same underlying duplicate text clusters

must be protected.

This should be explicitly mandated in v4.

---

# 11. Don't do pairwise-only inference

This is where the research becomes especially interesting for this specific problem.

The task is not really:

```text
Is S1 == S2?
```

It is:

```text
For this S1, which subset of S2/S3 belongs to it?
```

That's a **one-to-set** problem.

GNEM explicitly studies one-to-set neural entity matching and argues that seeing multiple related records can help resolve individual uncertain pairs. ([DBLP][16])

HierGAT similarly models dependencies between ER decisions rather than treating every candidate pair independently. ([ResearchGate][17])

### This suggests a powerful fourth-stage model:

```text
S1
 │
 ├── candidate A
 ├── candidate B
 ├── candidate C
 ├── candidate D
 └── candidate E
       ↓
 small Set Transformer / attention decoder
       ↓
 probabilities for A,B,C,D,E
       +
singleton probability
       +
optional cardinality estimate
```

We don't need a giant graph over millions of records.

We can do it **locally per S1 candidate set**.

That's computationally feasible.

This could learn things such as:

> “Candidate A is strong, Candidate B has almost the same identity evidence, Candidate C only shares a generic name, therefore A+B form a coherent match set while C should be rejected.”

That is much closer to the real task than independent binary classification.

---

# 12. Optimize the actual F0.5 set decision

This is probably the most important mathematical upgrade.

The official metric is explicitly per-S1:

$$
F_{0.5}(S1_i)
$$

then macro-averaged across S1 entities. 

Research on multilabel F-measure optimization shows that simply thresholding independent probabilities is generally **not the Bayes-optimal way to optimize a set F-measure**. Plug-in/Bayes decision approaches have been developed specifically for this problem. ([Proceedings of Machine Learning Research][18])

There is even a 2026 CVPR paper specifically revisiting efficient F-measure optimization in multilabel prediction. ([Open Access CVF][19])

That is directly relevant.

### Useful identity for this competition

For a non-singleton S1:

* \(T\) = number of true matches
* \(k\) = predicted matches
* \(t\) = true positives in those predictions

then:

$$
F_{0.5}
=
\frac{1.25t}{0.25T+k}
$$

That means prediction quality depends jointly on:

* how many matches we predict
* how many of them are right
* how many true matches the entity actually has

So:

> **One global probability threshold is a poor final policy.**

---

# 13. I would build a learned S1-level selection policy

Rather than blindly using:

```python
prob > 0.72
```

do this.

For each S1:

1. Sort candidates by match score.
2. Consider prefixes:

$$
k=0,1,2,\ldots,K
$$

3. For every prefix, compute:

```text
k
best score
last score
score gap
sum scores
mean score
number of agreeing retrieval channels
S2 count
S3 count
predicted cardinality
singleton probability
cross-script proportion
candidate pool size
```

4. Train a small S1-level selector from OOF data to choose the prefix that maximizes actual validation F0.5.

This is much more defensible than manually choosing a threshold.

And it naturally handles:

```text
S1 A → 1 match
S1 B → 4 matches
S1 C → 8 matches
S1 D → 0 matches
```

without a fixed K.

---

# 14. Add an explicit match-count model

Train:

$$
P(N=n \mid S1,\ retrieval\ evidence)
$$

where \(N\) is the number of true matches.

This should **never hard-cap candidate retrieval**.

It is only a soft signal to the final decoder.

That gives the decoder:

```text
"this S1 probably has 0-1 matches"
```

versus:

```text
"this S1 has evidence for 5-6 matches"
```

That is particularly useful because v3's own claimed training statistic showed a highly nonuniform match-count distribution.

A public full-data audit of the same challenge files reports:

* 123,247 singleton S1s
* 119,157 one-match S1s
* 1,964,417 multi-match S1s
* observed maximum 11 matches

and explicitly warns that 11 is an observed training maximum, **not a legal test-set cap**. ([GitHub][15])

That supports using cardinality as a **soft feature**, not a hard rule.

---

# 15. Use a candidate-budget controller

This is our competition-specific optimization.

For each S1, estimate retrieval difficulty.

Features:

```text
exact-name hit?
rare-token hit?
best lexical rank?
best dense rank?
channel agreement?
cross-script?
address missing?
name length?
number of plausible candidates?
score concentration?
```

Then learn:

$$
B(S1)=\text{candidate budget}
$$

So:

```text
easy entity → 5 candidates
medium entity → 10
hard entity → 18
extreme entity → 30
```

The exact numbers are determined experimentally.

This gives us the **small candidate sets Amazon explicitly wants**, without the enormous recall damage caused by universal `K=10`.

This concept is also consistent with the blocking literature's emphasis on the joint tradeoff among recall, output size and runtime. ([DOI][3])

---

# 16. Add a one-hop cross-source bridge

Suppose direct retrieval finds:

```text
S1
 ↓
S2-A  (strong)
```

and:

```text
S2-A
 ↓
S3-B  (extremely strong)
```

while S1→S3-B is weak because S3-B contains a different fragment of the identity.

Then:

```text
S3-B
```

is a useful rescue candidate.

So:

```text
S1 → S2
S1 → S3
S2 candidate → S3
S3 candidate → S2
```

Use **one hop only**.

Do not run connected components.

Do not recursively propagate identities.

Collective/relationship-aware ER research supports exploiting cross-record relationships, while graph-based methods show that independent pair decisions can miss useful context. ([ScienceDirect][20])

This should be a **recall rescue channel**, not the primary blocker.

---

# 17. Address should become its own retrieval problem

This is another place where v3 is too name-centric.

The official problem explicitly says addresses can undergo:

* transliteration
* abbreviations
* missing components
* landmark references
* municipal numbering changes
* reordering. 

So we need:

```text
NAME retrieval
        +
ADDRESS retrieval
        +
joint retrieval
```

not simply:

```text
"name | address" → one embedding
```

A strong architecture would have:

### Name encoder

Captures identity/brand similarity.

### Address encoder

Captures location/address similarity.

### Joint matcher

Learns how the two interact.

This also matches address-specific research showing bi-encoder retrieval followed by cross-encoder reranking can work well for noisy addresses. ([arXiv][8])

---

# 18. Don't turn numerical address conflicts into hard rejection

This one is important because v3 called number conflict the “strongest negative.”

The actual challenge data contains positives where apparent numeric components differ, according to a public full-file audit. ([GitHub][15])

So:

```text
different house numbers
```

should be:

> strong negative feature

not:

> hard non-match.

The model should learn exceptions.

---

# 19. Use OOF calibration and OOF stacking

This is essential if we combine:

```text
LightGBM
BGE/MuRIL
cross encoder
retrieval score
singleton model
```

Never train a meta-model from in-sample predictions.

Correct:

```text
fold 1 model → fold 1 predictions
fold 2 model → fold 2 predictions
...
                ↓
             OOF scores
                ↓
         calibration / stacking
```

Then refit the base models on all training data.

This becomes especially important because the final metric is sensitive to tiny threshold changes.

---

# 20. A specialist Indic branch is better than forcing a universal transliteration system

I'd actually revise our old transliteration strategy.

Instead of:

```text
Indic
 ↓
convert everything to English
 ↓
compare
```

use:

```text
Indic native representation
       +
training-derived transliteration
       +
MuRIL native/transliterated semantic representation
       +
raw character representation
```

Why?

Because transliteration itself can destroy distinctions.

A native-script similarity channel remains useful.

And MuRIL was explicitly trained to connect native Indian text with transliterated forms. ([arXiv][21])

This is much safer than trying to force every script through an English transliterator.

---

# 21. What about a giant LLM?

I would **not** make an 8B LLM the core.

There is interesting current research showing LLM-based entity matching can benefit from global candidate selection and that smaller models can be fine-tuned effectively, but these approaches remain expensive compared with retrieval + specialized matching. ([arXiv][22])

For this challenge:

```text
10M gallery records
millions of candidate pairs
```

makes an LLM-everywhere strategy unattractive.

A much better use would be:

```text
99.5% of candidates → LightGBM / small transformer
hardest 0.5% → expensive neural adjudication
```

And even that should earn its place through ablation.

---

# 22. The best architecture I now see

This is what I would make the plan maker investigate.

```text
                         RAW DATA
                            │
                            ▼
                 leakage-safe preprocessing
                            │
             ┌──────────────┼───────────────┐
             │              │               │
             ▼              ▼               ▼
          NAME views     ADDRESS views   metadata
             │              │
             └───────┬──────┘
                     │
                     ▼
              COUNTRY PARTITION
               (open-set safe)
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       lexical    dense       specialist
       retrieval  retrieval   retrieval
          │          │          │
     TFIDF/BM25   learned      MuRIL for
                  blocker      Indic cases
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
              RANK FUSION
             + channel agreement
                     │
                     ▼
        ONE-HOP CROSS-SOURCE BRIDGE
                     │
                     ▼
          LEARNED CANDIDATE BUDGET
                     │
                     ▼
              candidate_pairs.tsv
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
         LightGBM      neural reranker
              │             │
              └──────┬──────┘
                     ▼
                 OOF fusion
                     │
                     ▼
          S1-LOCAL SET DECODER
              │       │
              │       └── singleton probability
              │
              └────────── cardinality prior
                     │
                     ▼
       optional verified target uniqueness
                     │
                     ▼
             matching_results.tsv
```

---

# 23. What I would call the three core innovations

If we eventually have to explain the methodology to Amazon, I would want the real technical novelty to be:

### **1. Supervised compact blocking**

A challenge-trained multilingual retriever/blocker designed to maximize true-match recall **while minimizing candidate count**.

This is directly aligned with the competition and supported by SC-Block/DeepBlocker/Sparkly research. ([arXiv][1])

### **2. Multilingual specialist matching**

General multilingual retrieval + an Indian-script/transliteration specialist rather than crude universal ASCII transliteration.

MuRIL makes this particularly attractive. ([arXiv][21])

### **3. S1-level set decision**

Don't merely classify pair `(S1,S2)`.

Select the **whole match set for S1**, explicitly optimizing the competition's macro F0.5.

This connects the system to the multilabel/set-F-measure literature and one-to-set ER research. ([Proceedings of Machine Learning Research][18])

---

# 24. What gets removed from v3

I would now explicitly remove or demote these:

| v3 idea                                  | New status                           |
| ---------------------------------------- | ------------------------------------ |
| Universal `max_k=10`                     | **Remove**                           |
| 20–35 → 10 fixed funnel                  | **Remove**                           |
| Pair-level F0.5 threshold tuning         | **Remove**                           |
| “False merge always costs X” math        | **Remove**                           |
| Hard numeric-address conflict            | **Remove**                           |
| Frozen BGE-M3 as primary neural strategy | **Demote to baseline**               |
| Indic → one English transliteration      | **Replace with multi-view strategy** |
| Greedy global disjoint assignment        | **Optional validated decoder**       |
| Generic 35-feature list as final design  | **Baseline, not final**              |
| France-specific threshold                | **Remove**                           |
| LLM everywhere                           | **Reject as core architecture**      |
| Competitor-error percentages             | **Remove**                           |

---

# 25. What gets added

| New component                          | Priority                       |
| -------------------------------------- | ------------------------------ |
| Supervised contrastive blocker         | **Very high**                  |
| Source-separated S2/S3 retrieval       | **Very high**                  |
| Candidate-size/recall Pareto analysis  | **Very high**                  |
| Learned candidate budget               | **Very high**                  |
| Exact S1-level F0.5 decoder            | **Very high**                  |
| MuRIL specialist branch                | **High**                       |
| Hard-negative curriculum               | **High**                       |
| Field-aware cross-encoder              | **High**                       |
| OOF calibration/stacking               | **High**                       |
| Separate address retrieval             | **High**                       |
| S2↔S3 one-hop bridge                   | **Medium-high**                |
| Candidate-set/Set Transformer reranker | **Medium-high**                |
| ColBERT-style rescue retrieval         | **Medium**                     |
| Full graph model                       | **Low / experimental**         |
| 8B LLM judge                           | **Low / probably unnecessary** |

---

# The most important conclusion

The research changed my view in one major way.

I no longer think the central problem is:

> **“Which matcher should we use?”**

It is:

> **“How do we construct the smallest high-recall candidate set, and then make an S1-level decision over that set?”**

The literature strongly supports the two-stage retrieval/matching paradigm, learned blocking, hard-negative training, late interaction/reranking, and set-aware entity matching. ([DOI][3])

And the public audit of the actual challenge data makes the scale argument much stronger: there are millions of records, billions/trillions of potential comparisons, millions of positive links, substantial singleton/multi-match structure, and meaningful Indic-script mismatches. ([Hugging Face][12])

So **I would not ask the plan maker to “fix v3” anymore**.

I would instruct it to build a **Master Plan v4 based on these research findings**, with v3 treated as one baseline among several candidates.

The strongest V4 hypothesis is:

$$
\boxed{
\text{Supervised Compact Blocking}
+
\text{Hybrid Multilingual Retrieval}
+
\text{Hard-Negative Neural Matching}
+
\text{S1-Level F}_{0.5}\text{ Set Decoding}
}
$$

with every major decision selected by **actual grouped validation on the challenge data**, not by assuming that a paper's result transfers directly.

And one final caution: some public repositories now contain audits and proposed solutions for this exact September 2026 challenge. They are useful for discovering ideas, but their reported scores and methodological claims are **not authoritative competition results**. The official problem statement remains the governing source, and we should independently reproduce dataset statistics and validation results before treating any public claim as fact. ([GitHub][23])

[1]: https://arxiv.org/abs/2303.03132?utm_source=chatgpt.com "SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines"
[2]: https://arxiv.org/abs/2404.14831?utm_source=chatgpt.com "Towards Universal Dense Blocking for Entity Resolution"
[3]: https://doi.org/10.14778%2F3583140.3583163?utm_source=chatgpt.com "Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching | Proceedings of the VLDB Endowment"
[4]: https://doi.org/10.1145/1571941.1572114?utm_source=chatgpt.com "Reciprocal rank fusion outperforms condorcet and individual rank learning methods | Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval"
[5]: https://arxiv.org/abs/2402.03216?utm_source=chatgpt.com "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"
[6]: https://huggingface.co/google/muril-base-cased/blob/main/README.md?utm_source=chatgpt.com "README.md · google/muril-base-cased at main"
[7]: https://arxiv.org/abs/2004.00584?utm_source=chatgpt.com "Deep Entity Matching with Pre-Trained Language Models"
[8]: https://arxiv.org/abs/2307.02300?utm_source=chatgpt.com "Improving Address Matching using Siamese Transformer Networks"
[9]: https://www.sciencedirect.com/org/science/article/pii/S1546221821000734?utm_source=chatgpt.com "Relation-Aware Entity Matching Using Sentence-BERT - ScienceDirect"
[10]: https://huggingface.co/microsoft/mdeberta-v3-base/blob/main/config.json?utm_source=chatgpt.com "config.json · microsoft/mdeberta-v3-base at main"
[11]: https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/main?utm_source=chatgpt.com "BAAI/bge-reranker-v2-m3 at main"
[12]: https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026?utm_source=chatgpt.com "akshatbakshi/amazon-ml-challenge-2026 · Datasets at Hugging Face"
[13]: https://aclanthology.org/2022.naacl-main.272/?utm_source=chatgpt.com "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction - ACL Anthology"
[14]: https://aclanthology.org/2022.emnlp-main.730/?utm_source=chatgpt.com "Recovering Gold from Black Sand: Multilingual Dense Passage Retrieval with Hard and False Negative Samples - ACL Anthology"
[15]: https://github.com/AcID3r/RF_AMAZON_2026/blob/main/docs/implementation.md?utm_source=chatgpt.com "RF_AMAZON_2026/docs/implementation.md at main · AcID3r/RF_AMAZON_2026 · GitHub"
[16]: https://dblp.org/rec/conf/www/ChenSZ21?utm_source=chatgpt.com "dblp: GNEM: A Generic One-to-Set Neural Entity Matching Framework."
[17]: https://www.researchgate.net/publication/361246442_Entity_Resolution_with_Hierarchical_Graph_Attention_Networks?utm_source=chatgpt.com "(PDF) Entity Resolution with Hierarchical Graph Attention Networks"
[18]: https://proceedings.mlr.press/v28/dembczynski13?utm_source=chatgpt.com "Optimizing the F-Measure in Multi-Label Classification: Plug-in Rule Approach versus Structured Loss Minimization"
[19]: https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Revisiting_F-measure_Optimization_in_Multi-Label_Classification_A_Sampling-based_Approach_CVPR_2026_paper.pdf?utm_source=chatgpt.com "Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach"
[20]: https://www.sciencedirect.com/science/article/abs/pii/S002002551300618X?utm_source=chatgpt.com "Efficient entity matching using materialized lists - ScienceDirect"
[21]: https://arxiv.org/abs/2103.10730?utm_source=chatgpt.com "MuRIL: Multilingual Representations for Indian Languages"
[22]: https://arxiv.org/abs/2405.16884?utm_source=chatgpt.com "Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching"
[23]: https://github.com/orgs/Amazon-ML-Challenge-2026/repositories?utm_source=chatgpt.com "Amazon-ML-Challenge-2026 repositories · GitHub"
