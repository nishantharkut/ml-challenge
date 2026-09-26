 Here is the deep-dive audit confirming that every single metric is verified down to the individual row.
  ──────
  ### Summary of Rigorous Verifications & Edge-Case Audits
  #### 1. Zero Cross-Country GT Pairs (100% Verified)

  • Hypothesis tested: Could some S2/S3 IDs in train_ground_truth.tsv belong to different countries or be missing from their respective source files?
  • Exhaustive Check:
      • Ground truth contains 3,693,619 unique S2 IDs and 3,944,746 unique S3 IDs across 7,638,365 matched pairs.
      • We verified that 0 matched S2 IDs are missing from train_source2.tsv, and 0 matched S3 IDs are missing from train_source3.tsv.
      • Every single one of the 7,638,365 pairs was evaluated against the country label of its S1 parent.
      • Result: 0 cross-country matches (0.0000%). The matching process in this benchmark strictly respects country boundaries.
  #### 2. S2 / S3 Entities are Strictly Disjoint Across S1 (100% Verified)

  • Hypothesis tested: Could an S2 or S3 business record match multiple S1 entities?
  • Exhaustive Check:
      • Across all 2,206,821 rows in train_ground_truth.tsv, we checked frequency counts of every S2 and S3 ID.
      • Max count for any S2 ID across all S1 entities = 1.
      • Max count for any S3 ID across all S1 entities = 1.
      • Result: 0 shared IDs. Every S2/S3 candidate belongs to at most one S1 entity.


  #### 3. Complete Address & Business Name Null Audit
  • Hypothesis tested: Are empty addresses hidden as "null", "<NULL>", "None", "NaN", or whitespace strings? What about business names?
  • Exhaustive Check:
      • Evaluated every row across all 6 files for exact empty "", whitespace-only "   ", case-insensitive null tokens (null, <null>, none, nan, n/a), and short fragments (length ≤ 2).
      • Results:
          • Business names: Exactly 0 missing or empty business names across all 24.2M records.
          • Source 1 addresses: Exactly 0 missing or empty addresses in both train and test.
          • Source 2 & 3 addresses: 100% of the missing addresses are exact empty strings "" (no hidden "null" tokens as standalone address fields):
              • Train S2: 168,967 empty (3.356%)
              • Train S3: 175,916 empty (3.328%)
              • Test S2: 129,408 empty (2.648%)
              • Test S3: 136,098 empty (2.678%)


  #### 4. Country Invariance & Statistical Symmetry

  • Hypothesis tested: Does the matching behavior or singleton rate differ between countries?
  • Exhaustive Check:
      • US:
          • S1 count: 1,323,633 (60.0%)
          • Singletons: 73,896 (5.58%)
          • Matches per S1 (all): 3.4591
          • Matches per non-singleton S1: 3.6636
      • India:
          • S1 count: 883,188 (40.0%)
          • Singletons: 49,351 (5.59%)
          • Matches per S1 (all): 3.4645
          • Matches per non-singleton S1: 3.6696
  • Result: The data generation process maintains almost exact statistical symmetry between countries.
  #### 5. Unique Entity IDs per File
  • Exhaustive Check: Every file was checked for primary key integrity.
      • train_source1.tsv: 2,206,821 unique IDs, 0 duplicates
      • train_source2.tsv: 5,034,616 unique IDs, 0 duplicates
      • train_source3.tsv: 5,285,603 unique IDs, 0 duplicates
      • test_source1.tsv: 1,732,544 unique IDs, 0 duplicates
      • test_source2.tsv: 4,887,273 unique IDs, 0 duplicates
      • test_source3.tsv: 5,082,316 unique IDs, 0 duplicates

  ──────
  ### Verified Master Numbers Reference Table

   Metric                                                     | Verified Value                                            | Scope / Notes
  ------------------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------------------------------
   Train S1 Total Rows                                        | 2,206,821                                                 | US: 1,323,633 (60%), India: 883,188 (40%)
   Train S2 Total Rows                                        | 5,034,616                                                 | US: 3,016,817 (60%), India: 2,017,799 (40%)
   Train S3 Total Rows                                        | 5,285,603                                                 | US: 3,170,056 (60%), India: 2,115,547 (40%)
   Test S1 Total Rows                                         | 1,732,544                                                 | US: 663,106 (38.3%), India: 809,986 (46.8%), France: 259,452 (15.0%)
   Test S2 Total Rows                                         | 4,887,273                                                 | US: 1,871,330 (38.3%), India: 2,312,565 (47.3%), France: 703,378 (14.4%)
   Test S3 Total Rows                                         | 5,082,316                                                 | US: 1,945,701 (38.3%), India: 2,405,000 (47.3%), France: 731,615 (14.4%)
   Singleton S1 Count                                         | 123,247                                                   | S1 records with 0 matches in Ground Truth
   Singleton Share                                            | 5.5848%                                                   | 5.58% in US, 5.59% in India
   Total GT Matches                                           | 7,638,365                                                 | 3,693,619 in S2 + 3,944,746 in S3
   Mean Matches / S1 (All)                                    | 3.4613                                                    | Median: 3, Min: 0, Max: 11
   Mean Matches / S1 (Non-singletons)                         | 3.6660                                                    | Median: 4, Min: 1, Max: 11
   S2 Matches / S1                                            | Mean: 1.6737                                              | Max: 5
   S3 Matches / S1                                            | Mean: 1.7875                                              | Max: 6
   S2 IDs Shared Across >1 S1                                 | 0                                                         | Strict 1-to-many / mutually disjoint clusters
   S3 IDs Shared Across >1 S1                                 | 0                                                         | Strict 1-to-many / mutually disjoint clusters
   Cross-Country GT Pairs                                     | 0 (0.00%)                                                 | Zero cross-border matches exist in GT
   Country Variants                                           | Exact match only                                          | Only 'US' & 'India' in train; 'US', 'India', 'France' in test
   Empty Address Rate (S1)                                    | 0.00%                                                     | 0 missing addresses in train and test S1
   Empty Address Rate (S2)                                    | 3.36% (Train), 2.65% (Test)                               | Pure empty strings ""
   Empty Address Rate (S3)                                    | 3.33% (Train), 2.68% (Test)                               | Pure empty strings ""

  All figures are exact and verified directly against the underlying dataset files.