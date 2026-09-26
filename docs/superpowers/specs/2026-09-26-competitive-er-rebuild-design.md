# Competitive Entity Resolution Rebuild Design

## Objective

Build a portable, reproducible entity-resolution pipeline for the Amazon ML Challenge that produces a high-recall, compact `candidate_pairs.tsv` and a valid `matching_results.tsv`. The system must be logically correct independently of the supplied validator and must safely resume free-tier notebook runs.

## Non-negotiable invariants

1. Smoke-test artifacts are isolated from production artifacts and cannot overwrite them.
2. Every reusable artifact has a schema version and fingerprints for its code, configuration, input data, split, and upstream artifacts. A mismatch invalidates the artifact.
3. Candidate IDs emitted for an S1 are exactly the IDs scored for that S1. Final predictions are a subset of those candidates.
4. Validation holds out S1 entities but searches the complete relevant country/source gallery. It may sample queries, never arbitrary gallery distractors.
5. Candidate budgets are chosen from measured recall/candidate-count Pareto points, not an unexamined hard cap.
6. Inference writes deterministic, atomic shards; merging fails on stale, missing, overlapping, or incomplete shards.
7. Production inference and validation call the same feature, matcher, and decoder functions.
8. The notebook runs top-to-bottom in a clean Kaggle, Colab, or local environment from explicit roots and a pinned dependency set.

## Architecture

### 1. Run context and artifact contracts

Introduce a small run-context layer that resolves roots from explicit CLI/environment values, then derives the implementation root from `__file__`. It records a canonical JSON configuration and SHA-256 fingerprints for source files, input TSV metadata/content samples, split IDs, model, decoder, and generated artifacts.

All stages receive an explicit `RunContext` and `--force` option. `--force` creates a new run namespace instead of reusing mismatched files. Existing current outputs are preserved as legacy artifacts and are never silently trusted as production inputs.

### 2. Normalized feature views

Each record retains raw values and generates deterministic derived views: normalized text, accent-folded Latin text, compact/core names, transliterated names where supported, normalized address, address tokens, postal tokens, and numeric tokens. Empty values remain explicit rather than becoming positive evidence.

### 3. Country-separated multi-channel retrieval

Build independent S2 and S3 indices per normalized country. Retrieve candidates through unioned channels:

- exact normalized name/core-name/address/postal keys;
- rare-token inverted keys;
- character 3-5 gram TF-IDF name retrieval;
- word n-gram name retrieval;
- character/word address retrieval only when an address is present;
- transliterated and accent-folded name views.

Every channel stores its actual rank and score. Oversized exact buckets are secondary-reranked using field agreement; they are never sliced in file order. Source budgets remain independent through final union, and final compactness is selected using validation Pareto measurements.

### 4. Frozen candidate audits and matcher data

For held-out S1s, generate a long-form, hashable candidate artifact using the same full country galleries as inference. Report pair recall, non-singleton complete coverage, macro oracle F0.5, channel marginal gains, and candidates-per-S1 quantiles. Training queries use all available or stratified S1 entities; all positives are retained. Negatives are ranked retrieved nonmatches, stratified by source/country/ambiguity, and early-stopping splits are grouped by S1.

### 5. Pair matcher and shared decoder

The matcher receives only materialized candidates and uses lexical, character, token, transliteration, address, numeric, country, retrieval-rank, retrieval-score, and ambiguity features. The model checkpoint includes feature-schema and training-artifact hashes. A single decoder implementation performs thresholding, optional target exclusivity, and any margin logic. Each decoder policy is independently tuned on scored held-out candidates with a finer score-based threshold sweep.

### 6. Resumable inference and final audit

Inference partitions by dynamically discovered test countries and deterministic S1 shards. Each shard writes temporary output then atomically renames it with a manifest containing its complete S1 key hash, candidate/prediction counts, and all dependency fingerprints. Final merge requires exact disjoint coverage of test S1 IDs and matching hashes. It streams final files in official S1 order.

The release audit checks format, cardinality, target-ID existence, candidate/prediction subset relation, duplicate IDs, source prefixes, row-order coverage, output hashes, and documented package layout.

## Quality gates

1. Unit tests cover normalization, retrieval union/rank/quota behavior, artifact invalidation, decoder parity, target exclusivity, and atomic shard merge rejection.
2. Integration tests run a synthetic multilingual entity set through every stage, including resume and intentional stale-artifact rejection.
3. A clean environment executes the notebook and pipeline smoke run without using existing checkpoints.
4. A full-data run produces fresh provenance manifests before final output validation.
5. The release documentation states only measured metrics from the exact final run.

## Deliberate exclusions

No external identity data is used. No model exceeding the challenge size/license constraints is introduced. Neural embedding retrieval is deferred unless the sparse multi-view baseline shows a measured recall deficit after the required repair work.
