# Streamed Inference and Safe Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a resumable, memory-bounded inference path whose final candidate file contains exactly the pairs scored by the final model and whose final outputs have complete, ordered Source-1 coverage.

**Architecture:** A country run is partitioned into deterministic Source-1 shards.  Each target source is indexed and consumed separately; feature rows are atomically persisted while its target index is live.  Per-shard merge/scoring then uses only the persisted compact primitive columns.  Global exclusivity, if enabled, is resolved on disk.  Final assembly streams ordered shard parts and rejects missing, duplicate, malformed, or non-subset output.

**Tech stack:** Python 3.10+, Polars Parquet/Zstandard, NumPy, LightGBM/joblib model checkpoint, SQLite standard library, pytest.

**Root-cause evidence:** the old `50_run_country.py` keeps S1, S2 candidates/lookup, then S3 candidates/lookup, then a combined target lookup and all candidate hits.  A local France profile measured 2.50 GiB RSS for S2 index alone (703,378 targets), proving target indexing is feasible but two-source/country-wide retained state is not a safe India architecture.  `55_merge_outputs.py` additionally silently substitutes empty rows for missing country files, which can make an incomplete run appear structurally valid.

---

### Task 1: Define and test compact shard contracts

**Files:**
- Create: `Implementation/src/streaming_contracts.py`
- Create: `Implementation/tests/test_streaming_contracts.py`

- [ ] **Step 1: Write failing tests** for: deterministic candidate merge from S2/S3 compact rows; candidate IDs are sorted/deduplicated; predictions are a subset of candidates; a duplicate S1 ordinal/ID and a missing ordinal are rejected; tie comparison is probability-descending then Source-1-ID ascending.
- [ ] **Step 2: Run the focused test file** and confirm it fails because the module is absent.
- [ ] **Step 3: Implement only the pure, dependency-free helpers**: compact metadata reconstruction, candidate selection using `SourceSeparatedBlocker._balanced_merge` semantics, row-level contract validation, and `better_exclusive_winner`.
- [ ] **Step 4: Run the focused tests and the complete test suite.**

### Task 2: Add a shard-streamed country runner

**Files:**
- Create: `Implementation/51_stream_country.py`
- Modify: `Implementation/src/checkpoint.py` only if a missing atomic helper is required
- Create: `Implementation/tests/test_stream_country_helpers.py`

- [ ] **Step 1: Write failing tests** for raw Source-1 parsing retaining global ordinal, an atomic shard-completion manifest rejected when the source-file fingerprint/model fingerprint/config differs, and source-feature Parquet column schema validation.
- [ ] **Step 2: Run focused tests** and confirm expected failures.
- [ ] **Step 3: Implement deterministic Source-1 shards** with global source-file order, raw input fields required by `preprocess_record`, source fingerprint/config hash, atomic Parquet and manifest writes.
- [ ] **Step 4: Implement one-source-at-a-time feature passes.** For S2 then S3: load only that country target, build one `TargetSearchIndex`, retrieve each Source-1 shard, calculate features before releasing the target index, write primitive feature Parquet rows, record count/RSS/elapsed time, then close index and force collection.  Do not write any final candidate file at this phase.
- [ ] **Step 5: Implement shard-local merge and scoring.** Recreate only the metadata needed by the exact production merge policy, cap candidates after merging, verify feature column order against `matcher.feature_names`, score exactly those rows, and atomically write score and candidate shard parts.  Candidate shard rows must include empty candidate fields for zero-candidate Source-1 rows.
- [ ] **Step 6: Implement global decoder.** For exclusive decoding, insert threshold-passing pairs into a local SQLite `WITHOUT ROWID` winner table with the specified probability/Source-1 tie ordering.  For nonexclusive decoding, write deterministic local matches.  Stream match shard parts in ordinal order.
- [ ] **Step 7: Run focused tests, py_compile, and a small France run (for example 5,000 records) with recorded RSS, elapsed time, source row count, candidate count, and model schema.**

### Task 3: Replace unsafe final assembly with strict streaming validation

**Files:**
- Modify: `Implementation/55_merge_outputs.py`
- Modify: `Implementation/src/outputs.py`
- Create: `Implementation/tests/test_output_contract.py`

- [ ] **Step 1: Write failing tests** demonstrating that a missing country part, duplicate Source-1 ID, wrong Source-1 order, duplicate candidate ID, and prediction outside candidates each raise an error rather than emitting a file.
- [ ] **Step 2: Run focused tests** and confirm expected failures against current merge behavior.
- [ ] **Step 3: Implement an independent streaming checker** that reads both final files together, requires the exact Source-1 test-file order/coverage, validates IDs/uniqueness/subsets, and never fills absent data.
- [ ] **Step 4: Make merger fail closed.** It must require all expected completed country/shard manifests and concatenate verified parts in source ordinal order without loading a full country map.
- [ ] **Step 5: Run focused tests, complete test suite, and the official validator on an existing known-valid smoke artifact only as a validator wiring check.**

### Task 4: Operational integration and reproducibility

**Files:**
- Modify: `Implementation/run_pipeline.py`
- Modify: `Implementation/README.md`
- Create or modify: `Implementation/notebooks/` portable runner notebook only after the CLI path has a completed manifest

- [ ] **Step 1: Add explicit stage command wiring for the streamed runner; retain old Stage 50 only as deprecated diagnostic code and never as final-submission path.**
- [ ] **Step 2: Ensure each long run writes run ID, dataset/code/assets/model/decoder fingerprints, feature schema, candidate counts, elapsed time, and peak RSS.**
- [ ] **Step 3: Update documentation to distinguish measured results from targets and to give resumable, two-machine commands with unique run IDs.**
- [ ] **Step 4: Perform a fresh static review of candidate/model/output provenance and verify validation measurements use only production-retrievable candidates.**

### Acceptance gates

- [ ] Production candidate rows are exactly the candidate pairs passed into model scoring, not raw retrieval unions.
- [ ] All final candidate and match rows are present exactly once, in official Source-1 order; every prediction is a unique candidate for its row.
- [ ] No complete country-sized S1 list, candidate dict, target lookup, or both source indices coexist in the streamed runner.
- [ ] Resume skips only shards whose signed manifest, artifact hashes, model schema, code/config, and input fingerprints match.
- [ ] A representative France profile and at least one remote India/US profile have recorded completion/RSS before a full inference launch.
- [ ] Official validator and independent validator pass on the exact same final files.
