# Competitive Entity Resolution Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-reproducible, low-recall pipeline with a portable, checkpoint-safe sparse entity-resolution system that can produce a fresh candidate and prediction submission within the remaining challenge window.

**Architecture:** A deterministic run context fingerprints every dependency. Country-separated multi-channel retrieval writes frozen candidate shards, which are scored by a feature-schema-bound LightGBM model and decoded by one shared implementation. Full-gallery held-out evaluation selects compact candidate budgets and decoder policy before resumable final inference.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn, rapidfuzz, LightGBM, joblib, pytest.

---

## Scope and order

Implement in this dependency order: reproducibility contracts, normalization/features, blocker, candidate audit/training validation, shared decoder, sharded inference, notebook/package, then full-run launch. Do not delete legacy outputs; new output belongs to a fresh run ID.

### Task 1: Establish an executable regression-test harness

**Files:**
- Create: `Implementation/tests/conftest.py`
- Create: `Implementation/tests/test_regression_contracts.py`
- Modify: `Implementation/requirements.txt`

- [ ] **Step 1: Write failing import and smoke-isolation tests**

```python
def test_numbered_pipeline_modules_are_importable():
    import importlib
    assert callable(importlib.import_module("10_prepare_data").main)

def test_smoke_model_path_is_not_production_path(tmp_path):
    from src.matcher import PairwiseMatcher
    smoke = PairwiseMatcher(model_path=tmp_path / "smoke.joblib")
    assert smoke.model_path.name == "smoke.joblib"
```

- [ ] **Step 2: Run the tests and record the expected initial failure**

Run: `python -m pytest Implementation/tests/test_regression_contracts.py -q`

Expected: failure because numbered imports and explicit safe smoke path are absent.

- [ ] **Step 3: Add test dependencies and a root-path fixture**

Use a pinned, tested version set and make tests add `Implementation` to `sys.path` without relying on the current shell directory.

- [ ] **Step 4: Run the test file again**

Run: `python -m pytest Implementation/tests/test_regression_contracts.py -q`

Expected: the file collects and reports only behavior failures, never an import-path failure.

### Task 2: Replace weak manifests with provenance-safe run artifacts

**Files:**
- Create: `Implementation/src/run_context.py`
- Modify: `Implementation/src/checkpoint.py`
- Modify: `Implementation/src/config.py`
- Create: `Implementation/tests/test_run_context.py`

- [ ] **Step 1: Write failing fingerprint and invalidation tests**

```python
def test_manifest_rejects_changed_config(tmp_path):
    from src.checkpoint import write_manifest, is_checkpoint_valid
    manifest = tmp_path / "stage.json"
    write_manifest(manifest, {"schema": 2, "config_hash": "old", "completed": True})
    assert not is_checkpoint_valid(manifest, expected={"schema": 2, "config_hash": "new"})

def test_manifest_rejects_missing_or_changed_artifact(tmp_path):
    from src.checkpoint import artifact_fingerprint
    artifact = tmp_path / "x.txt"; artifact.write_text("one")
    first = artifact_fingerprint(artifact)
    artifact.write_text("two")
    assert first != artifact_fingerprint(artifact)
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest Implementation/tests/test_run_context.py -q`

Expected: missing `expected` matching and fingerprint behavior.

- [ ] **Step 3: Implement canonical hashing and explicit roots**

Implement SHA-256 helpers for canonical JSON and files, a `RunContext` dataclass with `run_id`, `data_root`, `implementation_root`, `runs_root`, and a stage manifest contract. Resolve roots from CLI/environment first and `Path(__file__)` second. Keep legacy `checkpoints` read-only.

- [ ] **Step 4: Verify green**

Run: `python -m pytest Implementation/tests/test_run_context.py -q`

Expected: all provenance tests pass.

### Task 3: Make normalization and multilingual views testable

**Files:**
- Modify: `Implementation/src/normalization.py`
- Modify: `Implementation/src/transliteration.py`
- Create: `Implementation/tests/test_normalization_views.py`

- [ ] **Step 1: Write failing multilingual-view tests**

```python
def test_accent_folded_and_original_views_are_distinct():
    from src.normalization import text_views
    views = text_views("Café Étoile")
    assert views.normalized == "café étoile"
    assert views.folded == "cafe etoile"

def test_missing_address_has_no_number_agreement():
    from src.normalization import numeric_tokens
    assert numeric_tokens("") == frozenset()
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest Implementation/tests/test_normalization_views.py -q`

Expected: `text_views` does not exist.

- [ ] **Step 3: Implement immutable text views**

Use Unicode NFKC, casefolding, optional Latin accent folding, compact/core token views, and a deterministic transliteration fallback. Return typed immutable values; do not replace raw text.

- [ ] **Step 4: Verify green**

Run: `python -m pytest Implementation/tests/test_normalization_views.py -q`

Expected: all text-view tests pass.

### Task 4: Implement compact multi-channel country retrieval

**Files:**
- Modify: `Implementation/src/blocking.py`
- Modify: `Implementation/src/features.py`
- Create: `Implementation/tests/test_blocking.py`

- [ ] **Step 1: Write failing candidate-contract tests**

```python
def test_ambiguous_exact_bucket_is_ranked_not_input_order():
    index = build_index([target("S2_a", "Acme", "99 Road"), target("S2_b", "Acme", "1 Road")])
    candidates = index.retrieve(query("Acme", "1 Road"), budget=2)
    assert candidates[0].target_id == "S2_b"

def test_retrieval_preserves_source_quotas_and_actual_rank():
    result = retrieve_two_sources(query("Northwind", "10 Main"), s2, s3, per_source=2, total=4)
    assert {c.source for c in result} == {"S2", "S3"}
    assert all(c.rank >= 1 for c in result)
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest Implementation/tests/test_blocking.py -q`

Expected: current blocker either returns input-order candidates or fabricated ranks.

- [ ] **Step 3: Implement channels and deterministic union**

Create independent source/country indices for exact keys, rare tokens, char 3-5 gram name TF-IDF, word name TF-IDF, and non-empty address TF-IDF. Add folded/transliterated views. Store channel mask, channel scores, and actual ranks. Re-rank large exact buckets by name/address/postal/number agreement, then preserve per-source quotas before the total Pareto-selected cap.

- [ ] **Step 4: Verify green**

Run: `python -m pytest Implementation/tests/test_blocking.py -q`

Expected: deterministic retrieval and quota/rank tests pass.

### Task 5: Build a truthful full-gallery candidate audit and matcher dataset

**Files:**
- Modify: `Implementation/20_candidate_retrieval.py`
- Modify: `Implementation/30_feature_and_train.py`
- Modify: `Implementation/src/metrics.py`
- Create: `Implementation/tests/test_candidate_audit.py`

- [ ] **Step 1: Write failing audit tests**

```python
def test_oracle_reports_non_singleton_coverage():
    report = candidate_report({"q1": {"S2_a"}, "q2": set()}, {"q1": {"S2_a"}, "q2": {"S3_b"}})
    assert report.non_singleton_complete_recall == 0.0

def test_candidate_output_equals_scored_universe():
    scored = {("q", "S2_a"), ("q", "S3_b")}
    assert assert_candidate_universe(scored, scored) is None
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest Implementation/tests/test_candidate_audit.py -q`

Expected: missing report fields and universe assertion.

- [ ] **Step 3: Implement exact-production validation protocol**

Hold out S1 deterministically, filter each query to its whole country gallery, persist long-form candidate rows, and calculate pair recall, non-singleton complete recall, candidate quantiles, per-channel gain, and oracle macro F0.5 at multiple budgets. Mine negatives from ranked retrieved nonmatches and split model early stopping by S1 ID.

- [ ] **Step 4: Verify green**

Run: `python -m pytest Implementation/tests/test_candidate_audit.py -q`

Expected: audit metrics and candidate universe assertions pass.

### Task 6: Bind the matcher and decoder to one feature and prediction contract

**Files:**
- Modify: `Implementation/src/matcher.py`
- Modify: `Implementation/src/decoder.py`
- Modify: `Implementation/40_tune_decoder.py`
- Create: `Implementation/tests/test_matcher_decoder.py`

- [ ] **Step 1: Write failing model/decoder parity tests**

```python
def test_smoke_training_does_not_write_production_model(tmp_path):
    production = tmp_path / "production.joblib"
    smoke = PairwiseMatcher(model_path=tmp_path / "smoke.joblib")
    smoke.train(X_train, y_train)
    assert not production.exists()

def test_shared_decoder_matches_inference_policy():
    rows = [{"s1_id": "q", "target_id": "S2_a", "score": .9}]
    assert decode_rows(rows, threshold=.8, enforce_target_exclusivity=False) == {"q": ["S2_a"]}
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest Implementation/tests/test_matcher_decoder.py -q`

Expected: current smoke behavior and separate inference implementation violate the contract.

- [ ] **Step 3: Implement model schema and shared threshold selection**

Require an explicit model path, persist feature-schema/training hashes, and reject incompatible scoring. Tune each exclusivity policy independently over observed score breakpoints. Make margin logic active only when tested; otherwise remove it from configuration. Export one `decode_scored_candidates` function for stage 40 and inference.

- [ ] **Step 4: Verify green**

Run: `python -m pytest Implementation/tests/test_matcher_decoder.py -q`

Expected: model isolation and decoder parity pass.

### Task 7: Replace country-only inference with atomic S1 shards

**Files:**
- Modify: `Implementation/50_full_inference.py`
- Create: `Implementation/src/shards.py`
- Create: `Implementation/tests/test_shards.py`

- [ ] **Step 1: Write failing shard/merge tests**

```python
def test_merge_rejects_missing_s1(tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        merge_shards([write_shard(tmp_path, {"q1": []})], expected_s1_ids=["q1", "q2"])

def test_merge_rejects_overlapping_shards(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        merge_shards([write_shard(tmp_path / "a", {"q": []}), write_shard(tmp_path / "b", {"q": []})], ["q"])
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest Implementation/tests/test_shards.py -q`

Expected: shard helpers do not exist.

- [ ] **Step 3: Implement atomic shard lifecycle**

Discover countries from test S1, sort S1 IDs into fixed-size chunks, write temporary matching/candidate TSV plus JSON manifest, fsync/rename atomically, and resume only when dependency and S1-list hashes match. Merge must enforce exact disjoint coverage and stream official S1 order.

- [ ] **Step 4: Verify green**

Run: `python -m pytest Implementation/tests/test_shards.py -q`

Expected: stale, incomplete, and overlapping shards are rejected.

### Task 8: Repair the orchestration, notebook, and release package

**Files:**
- Modify: `Implementation/run_pipeline.py`
- Modify: `Implementation/generate_notebook.py`
- Modify: `Implementation/notebooks/Amazon_ML_Challenge_2026.ipynb`
- Modify: `Implementation/60_validate_submission.py`
- Modify: `Implementation/README.md`
- Modify: `Implementation/documentation.md`
- Create: `Implementation/tests/test_portability.py`

- [ ] **Step 1: Write failing orchestration/portability tests**

```python
def test_orchestrator_declares_stage_20_and_force():
    source = Path("Implementation/run_pipeline.py").read_text(encoding="utf-8")
    assert "20_candidate_retrieval" in source
    assert "force=" in source

def test_notebook_contains_no_invalid_numbered_import():
    notebook = json.loads(Path("Implementation/notebooks/Amazon_ML_Challenge_2026.ipynb").read_text())
    assert not any(re.search(r"from\\s+\\d", "".join(c.get("source", []))) for c in notebook["cells"])
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest Implementation/tests/test_portability.py -q`

Expected: Stage 20/force and notebook-import assertions fail.

- [ ] **Step 3: Implement portable execution**

Make `run_pipeline.py` call every stage in order and create a fresh run under `runs/<run_id>`. Use `importlib.import_module` for numbered modules in the notebook, explicit root variables, and locked requirements. Make release validation enable target-ID checks and retain its command/output in a manifest. Correct documentation using measured values only.

- [ ] **Step 4: Verify green**

Run: `python -m pytest Implementation/tests/test_portability.py -q`

Expected: orchestration and notebook syntax contracts pass.

### Task 9: Run layered verification and launch the first fresh full-data attempt

**Files:**
- Create: `Implementation/tests/test_end_to_end_synthetic.py`
- Create: `Implementation/scripts/release_audit.py`

- [ ] **Step 1: Write a synthetic multilingual end-to-end test**

```python
def test_synthetic_multilingual_pipeline_recovers_and_resumes(tmp_path):
    result = run_synthetic_pipeline(tmp_path)
    assert result.oracle_f05 == 1.0
    assert result.final_predictions_are_candidate_subset
    assert result.resume_reuses_only_matching_shards
```

- [ ] **Step 2: Verify red, then implement the smallest orchestration adapters required**

Run: `python -m pytest Implementation/tests/test_end_to_end_synthetic.py -q`

Expected: red until the shared pipeline adapters exist; green after implementation.

- [ ] **Step 3: Run the full test and static checks**

Run: `python -m pytest Implementation/tests -q; python -m compileall -q Implementation`

Expected: zero test failures and exit code 0.

- [ ] **Step 4: Launch a fresh run and measure before committing to full inference**

Run: `python Implementation/run_pipeline.py --run-id first_fresh --stage audit --data-root <dataset-root>`

Expected: manifests and full-gallery held-out Pareto report. Choose the budget and decoder only from this report, then launch sharded train/inference.

## Final release gate

- [ ] Run full release audit with target-ID validation.
- [ ] Verify output hashes and release manifest refer to the same run ID/model/decoder/candidate universe.
- [ ] Execute the generated notebook in a clean kernel.
- [ ] Package the exact required challenge directory tree and pinned environment.
