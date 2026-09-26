# Competition Operating Contract

This repository is a time-critical Amazon ML Challenge entity-resolution system.
The objective is a valid, reproducible submission with high entity-level macro
F0.5 **and** a small final candidate set.  A green unit-test suite is necessary
but never sufficient evidence of readiness.

## Non-negotiable rules

- Treat the official problem statement and `candidate_pairs.tsv` update as the
  governing contract.  For each Source-1 row, predictions must be a subset of
  the exact candidate row supplied in the final package.
- Do not use external business-identity data, identity APIs, or manual lookups.
- Do not hard-code country values or assume France has training labels.
- Preserve an immutable prior run.  Every material experiment uses a distinct
  `AMAZON_ML_RUN_ID` and never overwrites a previous model, decoder, or output.
- Do not launch country-scale work until a representative partition profile has
  recorded RSS, elapsed time, candidate count, and completion behaviour.
- Do not merge partial country output.  Confirm exact Source-1 ID coverage for
  every partition before final assembly.

## Evidence required before accepting a change

Every change to normalization, blocking, feature generation, scoring, decoding,
or output must have all applicable evidence below:

1. A deterministic local contract check (unit or focused regression test).
2. A leakage-safe, entity-disjoint validation measurement on a frozen S1 slice.
3. Candidate certificate: pair recall, complete-entity recall, oracle macro
   F0.5, mean/P95/max candidates, per source, and per country.
4. A comparison against the previous artifact on the **same** frozen slice.
5. A memory/runtime profile at a representative partition size.

Never describe a planned measurement, an old measurement, or a metric from a
different validation slice as a current result.

## Critical quality gates

The following gates block any claim that a run is competition-ready:

- Candidate oracle F0.5 must be measured on the same validation entities used
  to tune the matcher and decoder.  A classifier cannot exceed this ceiling.
- Training positives must be candidates obtainable by the production blocker;
  blocker-missed ground-truth links are retrieval diagnostics, not synthetic
  classifier examples.
- The inference feature schema must exactly match the serialized model feature
  schema.  New blocking metadata features require retraining and retuning.
- Retrieval, training, validation, and inference must preserve the same
  normalization/transliteration assets and their fingerprints.
- Final inference must be shard-streamed: process S2 and S3 independently,
  persist source-shard evidence atomically, merge/score per S1 shard, and avoid
  country-wide candidate dictionaries or all-target hit maps in RAM.
- `matching_results.tsv` and `candidate_pairs.tsv` must pass the official
  validator and a separate subset/coverage/order check.

## Parallel-machine protocol

- Assign independent partitions or experiments, never duplicate a long run.
- Remote (20 GiB RAM): India/US retrieval or model training after a RAM profile.
- Local (16 GiB RAM, RTX 4050): France partition profiling/inference, artifact
  validation, notebook/package assembly, and smaller ablations.
- Each worker writes a unique run ID and log.  Record command, code hash,
  dataset fingerprint, start/end time, peak RSS, and artifact paths.

## Final-submission checklist

Before packaging, require all of the following:

- frozen dependency file and portable notebook execute from a clean environment;
- code recreates the exact final candidate and matching files from checkpoints;
- candidate file contains only model-scored final candidates;
- every Source-1 ID occurs exactly once in both outputs, in deterministic order;
- all predicted IDs exist, are unique per row, and are candidates for that row;
- documentation contains measured—not aspirational—candidate and validation
  statistics; and
- final package layout matches the official student-resource README.
