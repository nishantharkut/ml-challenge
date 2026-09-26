# Implementation-Specific Instructions

Read the repository-root `AGENTS.md` first.  This file adds code-level rules.

## Source-of-truth boundaries

- `src/normalization.py` owns text views and address parsing.
- `src/blocking.py` owns candidate generation and candidate metadata.
- `src/features.py` owns the ordered model feature schema.
- `src/matcher.py` owns model serialization; never use an implicit default path.
- `src/decoder.py` owns thresholding and deterministic exclusivity.
- `src/outputs.py` owns official TSV construction and validation.
- `runs/<run-id>/` owns artifacts.  No stage may read a model, decoder, or
  dictionary from a different run without explicitly fingerprinting it.

## Implementation constraints

- Keep Python imports valid for notebook execution (`importlib` for scripts
  whose filename begins with a number).
- Use atomic writes and completion manifests for every shard/artifact.
- Store only compact primitive data in shard intermediates; do not serialize
  unbounded nested Python candidate dictionaries for a country.
- Preserve raw Source-1 file order with an ordinal through every shard phase.
- Target IDs are empirically exclusive in the supplied train labels, but tie
  resolution still must be stable: probability descending, then S1 ID ascending.
- When adding a feature, add an explicit regression contract and invalidate
  model/decoder checkpoints by artifact fingerprint/schema change.

## Required reporting from long runs

Print and persist: run ID, source code hash, model feature names, candidate
counts, candidate recall/oracle metrics where labels exist, elapsed time, and
peak RSS.  A run that terminates without these artifacts is diagnostic only and
must not be used for submission.
