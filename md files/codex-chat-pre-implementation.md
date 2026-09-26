 hi, you are an 20-30+ years ai and ml expert, you are exceptional and written various tier 1 to tier 1.5 rsearch
  papers. you have to help me excel and win this challenge.

  this is the problem statement :
  file:///c:/N%20Drive/Amazon%20ML%20Challenge/6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf

  see this properly, also, in addition to this problem statement, there is one more important instruciton : Update:
  candidate_pairs.tsv is part of your final submission
  1. Blocking has to scale. Amazon resolves business entities across billions of records, so
  comparing every record with every other one is not an option. Your blocking /
  candidate-generation step must cut the search space to a small candidate set per
  Source 1 entity.
  2. Candidate generation counts toward the final ranking. We will review your
  candidate_pairs.tsv and the code that produces it when deciding final rankings,
  alongside your matching_results.tsv score. The approach that generates a smaller
  candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the
  public/private leaderboard

  i did some research on my own, and laid out some plan :
  file:///c:/N%20Drive/Amazon%20ML%20Challenge/md%20files/main%20files

  ---

  now, your task is to hhelp me win this challenge, think deeply. see everything required proeprly.

  the dataset is here :
  file:///c:/N%20Drive/Amazon%20ML%20Challenge/dataset/6ab10eb3b23ba_student_resource/student_resource

  please please please help me, you are on yolo mode, you can go deep into the internet with no restriction at all.
  think deeply...


• I’ll read the challenge rules, your research notes, and the dataset first. Then I’ll build and validate a candidate
  generation and matching pipeline, with particular attention to keeping the candidate set small. I’m using the PDF,
  brainstorming, and planning workflows because the rules and evaluation details will shape the approach.

■ Conversation interrupted - tell the model what to do differently. Something went wrong? Hit `/feedback` to report the
issue.


› no, dont start the coding yet, first we should make the plan strong, think deeply. implementation shouldnt be started
  currently.


• Understood. I’ll stay at the strategy stage: inspect the problem statement, your notes, and the data, then produce a
  concrete plan with experiments and decision points. I won’t edit or implement code yet.

• Explored
  └ Read SKILL.md (brainstorming skill), SKILL.md (using-superpowers skill), SKILL.md (writing-plans skill), SKILL.md
  (pdf
         skill)

• Ran git status --short
  └ fatal: not a git repository (or any of the parent directories): .git

• Ran rg --files 'md files/main files'
  └ md files/main files\master_plan_v5.2_patch.md
    md files/main files\master_plan_v5.1.md

• Ran rg --files 'dataset/6ab10eb3b23ba_student_resource/student_resource'
  └ dataset/6ab10eb3b23ba_student_resource/student_resource\Documentation_template.md
    dataset/6ab10eb3b23ba_student_resource/student_resource\README.md
    dataset/6ab10eb3b23ba_student_resource/student_resource\utils\validate_submission.py
    +7 lines (ctrl+t to view transcript)

• Ran Get-ChildItem -Force -Name AGENTS.md,.agents,CLAUDE.md,GEMINI.md -ErrorAction SilentlyContinue
  └ (no output)

• Ran Get-ChildItem -Force | Select-Object Name,Mode,Length
  └
    Name                                                    Mode   Length
    ----                                                    ----   ------
    +8 lines (ctrl+t to view transcript)

• Ran rg --files pipeline scripts output
  └ scripts\test_translit.py
    scripts\test_indic_translit.py
    scripts\mine_dictionary.py
    +6 lines (ctrl+t to view transcript)

• Explored
  └ Read master_plan_v5.2_patch.md, Documentation_template.md, master_plan_v5.1.md, README.md

• Ran Get-Command pdftotext,pdfinfo,pdftoppm -ErrorAction SilentlyContinue | Select-Object Name,Source
  └
    Name          Source
    ----          ------
    +5 lines (ctrl+t to view transcript)

• The materials confirm two distinct targets: macro F0.5 for matches, and a small, reproducible candidate_pairs.tsv
  containing exactly what the matcher scores. Your v5.1 plan already covers many important ideas; I’m checking its
  assumptions against the official rules and the actual data before recommending priorities.

• Ran Get-Content -Raw -Encoding UTF8 -LiteralPath 'md files/main files/master_plan_v5.2_patch.md'
  └ # Master Plan v5.2 — Precision Patch on v5.1
    ## FINAL PLANNING DOCUMENT. Implementation begins now.

    +67 lines (ctrl+t to view transcript)

• Ran rg -n '^#{1,4} |^> \[!|^\|.*\|' 'md files/main files/master_plan_v5.1.md'
  └ 1:# 🏆 Amazon ML Challenge 2026 — Master Plan v5.1
    2:## Experimental Blueprint: Evidence-Driven Entity Resolution
    3:### All 20 Audit Points Incorporated (v5 + ChatGPT Deep Audit)
    +287 lines (ctrl+t to view transcript)

• Ran Get-Content -Raw -Encoding UTF8 -LiteralPath 'scripts/eda_results.json'
  └ {
      "file_stats": {
        "train_s1": {
    +624 lines (ctrl+t to view transcript)

• Ran Get-Content -Raw -Encoding UTF8 -LiteralPath 'pipeline/phase0_infra.py'
  └ """
    Amazon ML Challenge 2026 - Entity Resolution Pipeline
    Phase 0: Competition metric + data loading + validation infrastructure
    +379 lines (ctrl+t to view transcript)

• Ran pdfinfo '6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf'
  └ Title:           Emails Comms: Amazon ML Challenge 2026
    Producer:        Skia/PDF m156 Google Docs Renderer
    Custom Metadata: no
    +13 lines (ctrl+t to view transcript)

• Ran pdftotext -layout '6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf' -
  └                  Business Entity Resolution Challenge
    In large-scale commercial platforms, business identity data arrives from multiple independent
    sources — each contributing partial, noisy fragments of information about the same real-world
    +302 lines (ctrl+t to view transcript)

• Ran rg -n '^## |^### ' 'md files/main files/master_plan_v5.1.md'
  └ 2:## Experimental Blueprint: Evidence-Driven Entity Resolution
    3:### All 20 Audit Points Incorporated (v5 + ChatGPT Deep Audit)
    22:## 0. Competition Specification [FACT-STMT]
    +63 lines (ctrl+t to view transcript)

• Ran Get-Content -Raw -Encoding UTF8 -LiteralPath 'md files/main files/master_plan_v5.2_patch.md'
  └ # Master Plan v5.2 — Precision Patch on v5.1
    ## FINAL PLANNING DOCUMENT. Implementation begins now.

    +67 lines (ctrl+t to view transcript)

• Ran Get-Content -Raw -Encoding UTF8 -LiteralPath 'scripts/check_deps.py'
  └ import sys
    print(f"Python: {sys.version}")
    for pkg in ['sklearn', 'pandas', 'numpy', 'lightgbm', 'scipy', 'rapidfuzz', 'polars', 'sparse_dot_topn', 'anyascii',
    +7 lines (ctrl+t to view transcript)

• Ran Get-Content -Raw -Encoding UTF8 -LiteralPath 'pipeline/e0_e2_blocking.py'
  └ """
    Amazon ML Challenge 2026 - E0-E2: Multi-Channel Blocking
    Experiment: exact + rare-token + char-TF-IDF blocking
    +351 lines (ctrl+t to view transcript)

• Ran Get-ChildItem -Force 'output' | Select-Object Name,Length,LastWriteTime
  └ (no output)

• Ran Get-ChildItem 'dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/
  │ train','dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/test' -File | Select-Object Name,Length
  └
    Name                      Length
    ----                      ------
    +9 lines (ctrl+t to view transcript)

• Ran Get-Content -Encoding UTF8 'scripts/eda_results.json' -TotalCount 100
  └ {
      "file_stats": {
        "train_s1": {
    +97 lines (ctrl+t to view transcript)

• Ran Get-Content -Encoding UTF8 'dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/train/
  │ train_source1.tsv' -TotalCount 5
  └ entity_id    business_name    business_address    country
    S1-925783039    Orelee's Barbershop    1795 Westchester Drive, High Point, NC    US
    S1-773889195    Prime Money    17560 Ellis Road, Tahlequah, OK    US
    +2 lines (ctrl+t to view transcript)

• Ran Get-Item 'scripts/eda_results.json' | Select-Object Length
  └
    Length
    ------
    +3 lines (ctrl+t to view transcript)

• Ran Get-Content -Encoding UTF8 'md files/main files/master_plan_v5.1.md' | Select-Object -Skip 490 -First 320
  └ ## 11. S1-Level Set Decoder [HYPOTHESIS — compare alternatives]

    The prediction for each S1 is a **SET**. Independent pair decisions are suboptimal.
    +111 lines (ctrl+t to view transcript)

• Ran Get-Content -Encoding UTF8 'dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/train/
  │ train_ground_truth.tsv' -TotalCount 5
  └ source1_entity_id    matched_entity_ids
    S1-965667    S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-860443364
    S1-55344266    S2-249013014,S2-197070651,S3-478195123,S3-384364074
    +2 lines (ctrl+t to view transcript)

• Browsing the web

• Searched the web for arxiv entity resolution blocking benchmark recall reduction ratio blocker evaluation 2024 2025, …


› the end goal is to win the challenge, this is very famous amazon ml challenge, we should be on top of the
  leaderboard, think deeply...


• I’ll optimize the plan for the strongest defensible final result: leaderboard score plus candidate efficiency and
  auditability. One important issue has emerged: validation must keep the held-out Source 2/3 records searchable while
  keeping their labels out of training. I’m going to make that split explicit in the revised strategy.

• What compute can you realistically use for final training and inference? Your notes list about 15 GB RAM and no
  confirmed GPU.

  - This laptop only (about 15 GB RAM, CPU)
  - I can use a cloud GPU or larger RAM machine
  - I can use both local and cloud resources

• Ran Get-Content -Encoding UTF8 'dataset/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv'
  │ -TotalCount 5
  └ entity_id    business_name    business_address    country
    S1-714132312    Zephay Labs Inc    2621 Cotten Road, Tyler, TX    US
    S1-106407869    Vision Partners Corp    IA, Iowa City, 1064 Newton Rd, Unit 11    US
    +2 lines (ctrl+t to view transcript)

• Ran $j=Get-Content -Raw -Encoding UTF8 'scripts/eda_results.json' | ConvertFrom-Json; $j.ground_truth_stats |
  │ ConvertTo-Json -Depth 5
  └ {
        "total_s1":  2206821,
        "singletons":  123247,
    +68 lines (ctrl+t to view transcript)

• Ran $j=Get-Content -Raw -Encoding UTF8 'scripts/eda_results.json' | ConvertFrom-Json; $j.PSObject.Properties.Name
  └ file_stats
    country_variants
    ground_truth_stats
    +3 lines (ctrl+t to view transcript)

• Ran Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM
  └ Get-CimInstance : Access denied
    At line:2 char:1
    + Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM
    +5 lines (ctrl+t to view transcript)

• Ran Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,NumberOfLogicalProcessors
  └ Get-CimInstance : Access denied
    At line:2 char:1
    + Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMem ...
    +5 lines (ctrl+t to view transcript)

• Browsing the web

• Searched the web for site:arxiv.org entity resolution blocking survey pair completeness reduction ratio multi-pass bl…

• Browsing the web

• Opened https://arxiv.org/abs/2202.12521

• The research supports a staged approach: efficient lexical blocking is a strong first baseline, and learned dense
  blocking should earn its place by recovering matches the baseline misses at the same candidate budget. I also found
  that the current E0–E2 script has a per-country evaluation step that scans the Source 1 list repeatedly; at this
  scale, that needs to be addressed before its runtime results can guide decisions.

• I’ve read the problem statement (/C:/N Drive/Amazon ML
  Challenge/6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf), the supplied README (/C:/N Drive/Amazon ML
  Challenge/dataset/6ab10eb3b23ba_student_resource/student_resource/README.md), your v5.1 plan (/C:/N Drive/Amazon ML
  Challenge/md files/main files/master_plan_v5.1.md) and v5.2 patch (/C:/N Drive/Amazon ML Challenge/md files/main
  files/master_plan_v5.2_patch.md), and the existing exploratory scripts. I have not changed or run the pipeline.

  Your central strategy is right: treat candidate generation as a first-class model component. The test set has about
  1.73 million Source 1 records and 9.97 million Source 2/3 records, so exhaustive comparison would mean roughly 17
  trillion pairs. The final candidate file must contain exactly the pairs passed to the matcher, and its size matters in
  the final review.

  I recommend this sequence:

  1. Establish an honest validation setup. Hold out Source 1 entities and their labels, but leave all Source 2/3 records
     in the searchable validation index. Those held-out records must be retrievable, while their labels must never enter
     dictionaries, model training, or threshold tuning. The v5.2 instruction to fit “retrieval indices on fold-train
     only” needs this distinction; interpreted literally, it would remove validation targets. Keep one final holdout
     untouched. Use smaller development splits for early experiments, then full-scale checks once the architecture is
     promising.

  2. Find the candidate-efficiency frontier before building a sophisticated matcher. Start with bounded exact-name,
     distinctive-token, and address blocks; add ranked character and word retrieval; then evaluate transliteration and
     source-specific quotas. For every configuration, report pair recall, all-matches-per-entity recall, oracle macro
     F0.5, candidate counts per Source 1 entity (mean and tail), runtime, and peak memory. Compare configurations at
     equal candidate counts. Choose the smallest set that preserves the score ceiling needed for a top solution; there
     is no defensible fixed K yet.

  3. Train a strong, economical matcher on the actual candidate distribution. A gradient-boosted model with name,
     address, rare-token, numeric-component, script, and retrieval-evidence features is the first serious baseline. Mine
     hard negatives from the blocker’s mistakes and retain all available positives. Tune the final set of matches per
     Source 1 entity against macro F0.5, including the costly singleton mistakes. Test source exclusivity only if
     validation improves: its presence in training labels does not make a greedy assignment automatically optimal.

  4. Spend advanced-model compute only on measured failures. If lexical blocking misses cross-script or heavily
     rewritten names, test a learned dense blocker at the same final candidate budget. If the blocker finds matches but
     the matcher confuses close businesses, test a neural reranker on those borderline pairs. Large-scale ER research
     supports comparing blocking methods at matched recall and full-pipeline cost; it does not establish that a dense
     model will win on this dataset. Blocking comparison, SC-Block, UniBlocker.

  5. Treat France as a separate deployment risk. It has no training labels, so random US/India validation cannot
     estimate French performance. Keep normalization and features open to unseen countries, inspect French records
     without labels, and use cross-country transfer experiments as stress tests—not as a claimed French score. No
     external business lookup or augmentation is permitted by the rules.

  Two changes to your current plan deserve priority. First, the existing E0–E2 script (/C:/N Drive/Amazon ML Challenge/
  pipeline/e0_e2_blocking.py) has an evaluation expression that repeatedly scans the country’s Source 1 list while
  iterating ground truth; it is impractical at this scale. Its repeated loading and preprocessing of multi-million-row
  files also needs a memory-conscious design before we trust full-run comparisons. Second, three nested folds for every
  experiment would consume substantial time on this hardware; reserve that level of validation for finalists, and use a
  fixed, clean development protocol to eliminate weak ideas quickly.

  I would judge each experiment by a single question: does it improve honest macro F0.5, or preserve that score with
  fewer candidates and acceptable runtime? That gives us a path toward both leaderboard performance and the final
  blocking review. A first target to investigate is the candidate-count versus oracle-score curve—not an arbitrary “10
  candidates” rule.

  The one planning constraint I still need is your compute access: are we limited to the roughly 15 GB RAM CPU machine
  in your notes, or can you use a larger RAM machine or GPU? That determines whether learned blocking is a practical
  later experiment.

  Worked for 3m 42s · 23:06


› actually, we were thinking to use sagemaker on aws or use kaggle or google colab. but we have to make sure the
  implemeatation is that it has checkpoints.
  we can do locally too on this device but it wouldnt be running properly as it has too much contstraints


• That changes the execution plan in a useful way. I’ll make every long stage resumable, with saved progress and outputs
  after each partition. I’ll compare SageMaker, Kaggle, and Colab against that requirement, then fold the compute choice
  into the strategy. I’m still staying at the planning stage.

• Ran Get-Content -Raw -Encoding UTF8 -LiteralPath 'C:
  │ \Users\nhnis\.agents\skills\aws-billing-and-cost-management\SKILL.md'
  └ ---
    name: aws-billing-and-cost-management
    description: >-
    +161 lines (ctrl+t to view transcript)

• I’m using the AWS cost workflow to check the SageMaker option alongside checkpoint reliability. I won’t estimate a
  bill without an instance type, region, and expected runtime.

• Browsing the web

• Searched the web for site:docs.aws.amazon.com sagemaker training checkpoint S3 checkpoint_s3_uri spot training manage…

• What is your approximate cloud-compute budget for this challenge? This will determine whether we plan SageMaker as the
  main runner or reserve it for final full-scale jobs.

  - Free tiers only
  - Up to about $100 USD
  - Several hundred USD or more

• Yes—let’s design for cloud execution. The key is to make the pipeline resumable, not just save model weights. Blocking
  and scoring millions of records may take longer than model training.

  My recommended setup is a portable pipeline that runs on small local samples for development and on SageMaker for
  full-scale jobs, if the budget allows. Kaggle is useful for bounded experiments, but its documented notebook runs must
  finish within 12 hours and saved outputs are limited to 20 GB. Colab’s runtime and hardware availability vary, so I
  would not make it the sole home of a final multi-stage run. Kaggle notebook documentation, Colab FAQ.

  The checkpoint design should be part of every stage:

   Stage                             Checkpoint to retain                       Resume behavior
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Preprocessing and indices         Country/source partitions,                 Reuse completed partitions
                                     normalization version, index files
  ────────────────────────────────  ─────────────────────────────────────────  ─────────────────────────────────────────
   Candidate retrieval               Completed Source 1 chunks, candidate       Run only missing chunks
                                     counts, checksums
  ────────────────────────────────  ─────────────────────────────────────────  ─────────────────────────────────────────
   Feature extraction and scoring    Completed pair chunks and model version    Score only missing chunks
  ────────────────────────────────  ─────────────────────────────────────────  ─────────────────────────────────────────
   Model training                    Weights, optimizer state where             Continue from the last valid snapshot
                                     relevant, configuration
  ────────────────────────────────  ─────────────────────────────────────────  ─────────────────────────────────────────
   Final assembly                    Validated partition manifests              Rebuild the two TSVs from completed
                                                                                chunks

  A chunk is “complete” only after its data is written and a manifest records its row count, checksum, input version,
  and configuration. On restart, the runner verifies that manifest before skipping work. This also makes the final
  candidate_pairs.tsv auditable: it can be traced to the exact candidate chunks scored by the matcher. SageMaker
  supports syncing training checkpoints to S3 and restoring them after an interruption; the chunk checkpoints for
  blocking and inference will still need to be designed into our pipeline. SageMaker checkpoint documentation, storage
  paths.

  I would keep CPU, memory-heavy retrieval separate from any optional GPU training. We should earn the GPU experiment
  through the candidate-recall analysis first. This avoids paying for a GPU while building indices or formatting TSVs.
  Before choosing SageMaker instances or estimating cost, I need your approximate cloud budget; I’ve sent that as a
  short question. No implementation has started.

  23:07

/status

╭─────────────────────────────────────────────────────────────────────────────────╮
│  >_ OpenAI Codex (v0.156.0)                                                     │
│                                                                                 │
│ Visit https://chatgpt.com/codex/settings/usage for up-to-date                   │
│ information on rate limits and credits                                          │
│                                                                                 │
│  Model:                GPT-6-Sol (reasoning medium, summaries auto)             │
│  Model provider:       openai                                                   │
│  Directory:            C:\N Drive\Amazon ML Challenge                           │
│  Permissions:          Workspace (Ask for approval)                             │
│  Agents.md:            <none>                                                   │
│  Account:              test12hu34@gmail.com (Plus)                              │
│  Thread name:          Solve Amazon entity matching                             │
│  Collaboration mode:   Default                                                  │
│  Session:              01a0d99c-e9eb-76d2-a708-31aec9b8aae8                     │
│                                                                                 │
│  Context window:       65% left (98.6K used / 258K)                             │
│  5h limit:             [███████████████████░] 96% left (resets 04:02 on 26 Sep) │
│  Weekly limit:         [████████████████████] 99% left (resets 23:02 on 2 Oct)  │
╰─────────────────────────────────────────────────────────────────────────────────╯

• Model changed to gpt-5.6-terra high


› also, we dont have much time, only 16-18 hours for the implemenation and results. think deeply... we are on free tier
  only, we havent bought anything at all. also, the iplmentation should be in python notebook so that it's portable and
  used and imported everywhere properly..

  alkso, you didnt tell me whether the plan that i made in the @main files folders was perfect or not. have u seen the
  problem statemtn + additonal instruction + dataset, etc.. everything required??


• I have seen the official problem statement, your added candidate-generation instruction, the supplied README/template/
  validator, the dataset layout and sizes, your v5.1/v5.2 plans, EDA output, and the existing Phase 0/E0–E2 scripts.

  Your plan is thoughtful, but it is not perfect for 16–18 hours on free compute. It is a strong research roadmap, not
  an execution plan for the time available.

/status

╭─────────────────────────────────────────────────────────────────────────────────╮
│  >_ OpenAI Codex (v0.156.0)                                                     │
│                                                                                 │
│ Visit https://chatgpt.com/codex/settings/usage for up-to-date                   │
│ information on rate limits and credits                                          │
│                                                                                 │
│  Model:                GPT-5.6-Terra (reasoning high, summaries auto)           │
│  Model provider:       openai                                                   │
│  Directory:            C:\N Drive\Amazon ML Challenge                           │
│  Permissions:          Workspace (Ask for approval)                             │
│  Agents.md:            <none>                                                   │
│  Account:              test12hu34@gmail.com (Plus)                              │
│  Thread name:          Solve Amazon entity matching                             │
│  Collaboration mode:   Default                                                  │
│  Session:              01a0d99c-e9eb-76d2-a708-31aec9b8aae8                     │
│                                                                                 │
│  Context window:       65% left (98.6K used / 258K)                             │
│  5h limit:             [███████████████████░] 96% left (resets 04:02 on 26 Sep) │
│  Weekly limit:         [████████████████████] 99% left (resets 23:02 on 2 Oct)  │
╰─────────────────────────────────────────────────────────────────────────────────╯

• I have seen the official problem statement, your added candidate-generation instruction, the supplied README/template/
  validator, the dataset layout and sizes, your v5.1/v5.2 plans, EDA output, and the existing Phase 0/E0–E2 scripts.

  Your plan is thoughtful, but it is not perfect for 16–18 hours on free compute. It is a strong research roadmap, not
  an execution plan for the time available.

  What is correct in your plan:

  - It treats candidate_pairs.tsv as a final-ranking artifact, not a side file.
  - It uses macro F0.5 correctly, including singleton behavior.
  - It recognizes country blocking, source-specific retrieval, transliteration, address noise, and the unseen France
    split.

  - It measures oracle F0.5, which is the right way to quantify the candidate generator’s ceiling.
  - It recognizes that the final predictions must be a subset of candidates.
  - It aims for reproducibility and auditability.

  What must change:

  1. E0–E15, nested validation, learned dense blocking, neural reranking, dictionary mining, global assignment, and six
     decoders cannot all fit in 16–18 hours. Trying them will leave us with incomplete results.

  2. The current validation wording needs correction. We must hold out Source 1 labels for validation, but keep all
     Source 2/3 records searchable. Otherwise validation true targets may be absent from the retrieval index.

  3. The existing E0 code says large exact blocks need secondary discrimination, but currently skips them. That loses
     true matches and does not satisfy the plan’s own safety rule.

  4. Full cross-validation is too expensive now. Use one fixed, stratified validation split with a final untouched
     holdout only if time remains.

  5. Dense retrieval and cross-encoders are contingency experiments only. Free Kaggle/Colab sessions can end, and they
     are poor places to depend on a long, fragile final pipeline. SageMaker is not viable on a true free-only budget for
     this workload.

  6. The final candidate set needs a strict candidate budget policy. “Union every block” can generate a candidate file
     too large for final review. Every channel must compete for a limited per-Source-1 budget.

  For the next 16–18 hours, the winning strategy should be narrower:

  Portable Python notebook
    ├─ deterministic preprocessing
    ├─ country partition
    ├─ source-separated lexical retrieval
    │    ├─ exact/normalized name and address evidence
    │    ├─ rare-token retrieval
    │    ├─ char TF-IDF name retrieval
    │    └─ char TF-IDF address retrieval
    ├─ bounded, ranked candidate fusion per S2 and S3
    ├─ LightGBM / sklearn matcher on candidate pairs
    ├─ Source-1-level precision-oriented decoding
    ├─ candidate_pairs.tsv + matching_results.tsv
    └─ validator + experiment report

  The practical order:

  1. Build one portable notebook as the entry point, with configuration at the top and cells that can resume from disk
     checkpoints. Keep helpers inside the notebook initially; only extract a .py utility file if notebook reuse becomes
     messy.

  2. Use Parquet or compressed partition files for checkpoints. Partition by country × source × chunk. Each completed
     chunk writes:
      - data artifact
      - JSON manifest: input fingerprint, config hash, row count, candidate count, elapsed time
      - success marker only after the artifact is safely written

  3. Build a fast baseline blocker with country hard blocking, exact normalized keys, rare name tokens, and source-
     separated char TF-IDF retrieval. Keep the top candidates independently for S2 and S3. This directly respects the
     fact that each Source 1 entity commonly has multiple matches from both sources.

  4. Measure candidate recall, complete-entity recall, oracle macro F0.5, mean candidates, P95/P99, and candidate count
     by S2/S3. Do not proceed until this report exists.

  5. Train one strong tabular matcher using engineered string features and hard negatives from retrieved nonmatches.
     LightGBM is ideal if available; otherwise HistGradientBoosting or XGBoost if the environment supports it.

  6. Tune a conservative threshold and a simple per-Source-1 prefix rule on the fixed validation split. Because only
     about 5.6% of train Source 1 records are singletons, the challenge is mostly selecting complete multi-match sets
     without admitting distractors.

  7. Run final test inference from checkpoints, write both TSVs, then run the supplied validator.

  Notebook portability means it should work in Kaggle, Colab, or local Python with only path configuration changed. I
  recommend Kaggle first because it has documented persistent notebook outputs, roughly 29 GB RAM, and a clean “Save &
  Run All” execution path. Its free notebook runs still have a 12-hour ceiling, so checkpoints remain mandatory. Kaggle
  notebooks documentation

  We should use Kaggle/Colab for the full run only after the notebook works on a small local sample. Free Colab runtime
  availability changes and is less predictable. SageMaker has excellent checkpoint-to-S3 support, but it requires paid
  AWS usage beyond the free tier for the compute needed here. SageMaker checkpoints

  So: your research plan is high quality, but for this deadline we should freeze the architecture to a checkpointed
  lexical blocker + tabular matcher, then add exactly one improvement only if its validation result is clear. That is
  the best route to a complete, reproducible, competitive submission in time.

  23:10


› so what is the plan forward? thjink deeply..


• The forward plan is to build one complete, resumable contender first, then improve only where validation proves a
  weakness. With 16–18 hours and free compute, the path is a lean lexical retrieval system plus a strong tabular matcher
  —not a broad research program.

  ## Chosen approach

  Use a notebook-first, module-backed pipeline:

  competition_runner.ipynb
          │
          ├─ configuration + environment paths
          ├─ experiment controls and checkpoint resume
          ├─ calls importable pipeline functions
          ├─ validation reports
          └─ final-output generation
                   │
                   ├─ candidate_pairs.tsv
                   └─ matching_results.tsv

  src/entity_resolution/
          ├─ io_checkpointing.py
          ├─ normalize.py
          ├─ blocking.py
          ├─ features.py
          ├─ matcher.py
          ├─ decoder.py
          └─ evaluate.py

  The notebook is the portable control center for Kaggle, Colab, and local runs. The small Python modules prevent code
  duplication, enable clean imports, and satisfy the submission requirement for runnable source under src/.

  Every stage resumes from artifacts rather than rerunning:

  raw TSV
    → normalized country/source partitions
    → retrieval indices
    → candidate chunks
    → scored candidate chunks
    → decoded prediction chunks
    → final TSV assembly + validator

  Each artifact has a manifest containing configuration hash, input fingerprints, completed chunk IDs, row counts,
  candidate counts, and elapsed time. A notebook restart will skip only artifacts whose manifest matches the current
  configuration.

  ## What we will build first

  1. Reliable evaluation and checkpoint framework
      - Exact macro F0.5 and oracle F0.5.
      - One fixed stratified Source 1 validation split.
      - All Source 2/3 training records remain available in the search index; validation labels are held out from
        matcher training and tuning.

      - Country partition is hard blocking because your EDA found zero cross-country training links. This must be re-
        certified in the notebook.

      - Output validator runs as a final mandatory cell.

  2. High-value candidate generator

     Source 2 and Source 3 are retrieved independently. This prevents one source from consuming the other’s candidate
     budget.

     Candidate channels, in this exact order:
      - Exact normalized business name.
      - Exact normalized address.
      - Rare, high-IDF name-token overlap.
      - Structured numeric address evidence: house, plot, unit, ZIP/PIN-like numbers.
      - Character TF-IDF retrieval on names for records with weak or ambiguous deterministic evidence.
      - Character TF-IDF retrieval on addresses only where name retrieval is insufficient.

     Each channel returns ranked candidates. We then choose the smallest source-specific budgets that preserve
     validation oracle F0.5. We will test a compact grid such as K2/K3 = 4, 6, 8, 10, then retain the Pareto-optimal
     setting. The final candidate_pairs.tsv is the post-budget set that the matcher actually scores.

  3. One serious matcher

     Use LightGBM if Kaggle provides it; otherwise use a portable sklearn fallback.

     Features:
      - Name: character overlap, token overlap, edit similarity, length ratio, legal-suffix agreement, rare-token
        evidence.

      - Address: token and character overlap, number agreement/conflict, postal/PIN agreement, missing-address flags.
      - Cross-field: country, source identity, script mismatch, name/address disagreement.
      - Retrieval: channel hits, best rank, score, reciprocal rank, source-specific candidate rank.

     Train on all positives available from the validation-safe training partition plus retrieval-derived hard negatives.
     Avoid random negatives, because they are too easy and teach little.

  4. Precision-oriented Source 1 decoder

     We will compare only three cheap options:
      - Global probability threshold.
      - Separate threshold by Source 2 versus Source 3.
      - Select a ranked prefix only while score and score-gap rules support it.

     Then test target exclusivity as a final lightweight post-processing rule, because training data indicates each
     Source 2/3 record maps to at most one Source 1 entity. It stays only if it improves held-out macro F0.5.

  ## Time budget

   Time            Deliverable                                         Decision gate
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Hour 0–1.5      Kaggle-ready notebook skeleton, configuration,      Notebook runs on a small sample
                   manifest/resume mechanism, metric, validator
  ──────────────  ──────────────────────────────────────────────────  ──────────────────────────────────────────────────
   Hour 1.5–4      EDA certificate and deterministic candidate         Candidate recall/oracle report exists
                   baseline
  ──────────────  ──────────────────────────────────────────────────  ──────────────────────────────────────────────────
   Hour 4–6.5      TF-IDF residual retrieval, source-separated         Choose candidate frontier
                   budgets
  ──────────────  ──────────────────────────────────────────────────  ──────────────────────────────────────────────────
   Hour 6.5–9.5    Feature generation and first LightGBM matcher       First valid macro F0.5
  ──────────────  ──────────────────────────────────────────────────  ──────────────────────────────────────────────────
   Hour 9.5–11     Threshold/prefix/exclusivity ablations              Freeze decoder
  ──────────────  ──────────────────────────────────────────────────  ──────────────────────────────────────────────────
   Hour 11–14      Full train and test run with checkpoints            Both output TSVs validate
  ──────────────  ──────────────────────────────────────────────────  ──────────────────────────────────────────────────
   Hour 14–18      Error analysis and only one targeted improvement    Submit stable result; iterate if evidence
                                                                       supports it

  ## What we deliberately defer

  - Dense learned blocking.
  - Cross-encoder reranking.
  - Full nested cross-validation.
  - Large transliteration-dictionary mining program.
  - Multiple model families and broad hyperparameter sweeps.

  These are reasonable research directions in your original plan, but each risks consuming the entire deadline without
  yielding a valid final submission.

  ## Compute choice

  Use Kaggle as the primary free environment. It has documented saved notebook outputs and enough RAM for carefully
  partitioned work, but its execution window means checkpoints are non-negotiable. Do small development runs locally if
  useful; use Colab only as backup. The notebook should never assume a fixed path or retain critical state only in RAM.

  ## Success criteria

  We do not call a candidate generator good based only on pair recall. It must report:

  - Macro oracle F0.5.
  - Pair recall overall and separately for Source 2/3.
  - Complete-entity recall.
  - Mean, median, P95, P99, and maximum candidates per Source 1.
  - Total candidate pairs and reduction ratio.
  - Runtime and checkpoint status.

  The final submission must meet all of these:

  - Every test Source 1 ID appears exactly once.
  - Predictions are subsets of final candidates.
  - Candidate file contains precisely what inference scored.
  - No external data or business lookup.
  - Fully reproducible from provided data.
  - Supplied validator reports PASS.

  If you approve this design, the next step is to turn it into a concise execution checklist and then start
  implementation with the checkpointing foundation.

  Worked for 1m 10s · 23:13


› is it accortding to my plan that was there in the main files?


• Yes—the forward plan is based on your v5.1/v5.2 master plan. It is a time-boxed execution version of it, not a
  replacement.

   Your master plan                                   Forward plan
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Country-safe, multi-view preprocessing             Keep it; validate country blocking and preserve original plus
                                                      normalized text
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   Deterministic pool plus ranked retrieval fusion    Keep it; exact/rare evidence plus ranked lexical retrieval
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   Source-aware retrieval and budgets                 Keep it; retrieve and cap Source 2 and Source 3 separately
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   Candidate recall + oracle macro F0.5               Keep it as the main blocking decision metric
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   LightGBM plus hard negatives                       Keep it as the primary matcher
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   Source 1-level set decoder                         Keep a compact threshold/prefix comparison
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   France open-set handling                           Keep country-agnostic code and no France-specific learning
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   Exact candidate file and output validation         Keep it as a mandatory final gate
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   E0–E15 broad ablation program                      Compress it to only the highest-value experiments
  ─────────────────────────────────────────────────  ───────────────────────────────────────────────────────────────────
   Nested 3-fold validation                           Replace with a fixed validation split for this deadline

  The deliberate deviations are only for time:

  - We will not complete the full E0–E15 sequence.
  - We will not use dense retrieval or neural rerankers unless the lexical candidate-recall report proves a major
    unrecovered failure mode.

  - We will use one validation split rather than your full nested 3-fold protocol. This is less statistically rigorous,
    but appropriate for producing a complete submission in 16–18 hours.

  - Checkpointing and the notebook-first architecture are additions to make v5.1 reproducible on free cloud sessions.

  One correction to my earlier wording: we should not broadly defer transliteration. Your own EDA found about 551,240
  cross-script ground-truth links, so it is too material to ignore. We should include the smallest safe version of v5.1
  §4.3:

  1. Establish E0–E3 lexical baseline.
  2. Mine or use only high-confidence fold-safe transliteration mappings.
  3. Measure its marginal candidate recall, oracle F0.5, and candidate-count cost.
  4. Keep it only if validation improves.

  That follows your evidence hierarchy: it is a hypothesis tested by an ablation, not a presumed winning component.

  So the proposed path honors the core principle of your plan: no component is mandatory until it earns its place
  through candidate efficiency and held-out macro F0.5. The only thing changing is the number of experiments we can
  afford before final output generation.

  23:14

• You ran codex plugin marketplace add alexgreensh/token-optimizer
  └ Added marketplace `alexgreensh-token-optimizer` from https://github.com/alexgreensh/token-optimizer.git.
    Installed marketplace root: C:\Users\nhnis\.codex\.tmp\marketplaces\alexgreensh-token-optimizer

  23:14

• Installed Token Optimizer plugin. No additional app authentication is required.

• Model changed to gpt-5.6-sol medium


› also, we are group of 4 people. so we should be implementation of the code such that parallely we can run it so that
    results come up faster and we get time to improve it., think deeply.. see how can we do it..

    we can setup multiple accoutns too if required, just we have to win the ml challenge no matter what, think deeply..
    we can do everything., but you alsi have to do deep research proeprly.


• I’ll redesign the execution around four team members and independent, mergeable experiment tracks. I’m using the
  parallel-work workflow now: one track will examine free-compute and account-policy constraints, one will design a
  checkpointed sharding scheme, and one will research the highest-value entity-resolution experiments for this deadline.
  No implementation yet.

• Explored
  └ Read SKILL.md (brainstorming skill), SKILL.md (dispatching-parallel-agents skill)

• Started `/root/free_compute_policy`

• Started `/root/parallel_pipeline_design`

• Started `/root/er_research_priority`

◦ Browsing the web

• Searched the web for site:kaggle.com/terms multiple accounts prohibited Kaggle account quota notebooks, site:research…

• A critical policy point is already clear: do not create duplicate Kaggle or Colab identities to multiply quota. Kaggle
  allows one active account per person and can disqualify users with multiple accounts; Colab explicitly forbids using
  multiple accounts to work around resource limits. Four real teammates may each use their own account for distinct
  assigned work, without credential sharing or quota evasion.

• Waiting for agents

• Finished waiting
  └ No agents completed yet

• Waiting for agents

• Finished waiting
  └ No agents completed yet

• Waiting for agents

• Completed `/root/free_compute_policy`

• Finished waiting
  └ No agents completed yet

• Waiting for agents

• Completed `/root/parallel_pipeline_design`

• Finished waiting
  └ No agents completed yet

• Waiting for agents

/status

╭─────────────────────────────────────────────────────────────────────────────────╮
│  >_ OpenAI Codex (v0.156.0)                                                     │
│                                                                                 │
│ Visit https://chatgpt.com/codex/settings/usage for up-to-date                   │
│ information on rate limits and credits                                          │
│                                                                                 │
│  Model:                GPT-5.6-Sol (reasoning medium, summaries auto)           │
│  Model provider:       openai                                                   │
│  Directory:            C:\N Drive\Amazon ML Challenge                           │
│  Permissions:          Workspace (Ask for approval)                             │
│  Agents.md:            <none>                                                   │
│  Account:              test12hu34@gmail.com (Plus)                              │
│  Thread name:          Solve Amazon entity matching                             │
│  Collaboration mode:   Default                                                  │
│  Session:              01a0d99c-e9eb-76d2-a708-31aec9b8aae8                     │
│                                                                                 │
│  Context window:       51% left (132K used / 258K)                              │
│  5h limit:             [████████████████░░░░] 81% left (resets 04:02 on 26 Sep) │
│  Weekly limit:         [███████████████████░] 97% left (resets 23:02 on 2 Oct)  │
╰─────────────────────────────────────────────────────────────────────────────────╯

• Finished waiting
  └ No agents completed yet

• The parallel design is converging on component ownership rather than four incompatible end-to-end pipelines. The key
  is a frozen artifact contract: retrieval can improve independently while the matcher and decoder develop against an
  early candidate snapshot, then all lanes converge at fixed freeze times.

• Waiting for agents

• Completed `/root/er_research_priority`

• Finished waiting
  └ No agents completed yet

• Yes, four people materially improve our chances—but only if we avoid building four incompatible pipelines. The
  strongest arrangement is four component owners during development, followed by distributed data shards during final
  execution.

  Creating duplicate accounts to bypass quotas is too risky: Kaggle can disqualify users operating multiple accounts,
  Colab explicitly prohibits multi-account quota evasion, and AWS restricts multiplying Free Tier benefits. Four genuine
  teammates can use one genuine account each for distinct assigned work, without credential sharing, provided all four
  belong to the official challenge team. Kaggle Terms, Colab FAQ, AWS Free Tier terms.

  ## Three possible team structures

  ### 1. Four complete competing pipelines

  Each member builds everything independently.

  - Good: diverse ideas.
  - Bad: duplicates preprocessing and training; integration happens too late.
  - Verdict: too risky in 18 hours.

  ### 2. Each person handles one quarter of the data

  Each member runs the complete pipeline on a separate shard.

  - Good: final computation is faster.
  - Bad: nobody can work on matcher or decoder until the pipeline is complete; different code versions can make shards
    incompatible.

  - Verdict: useful only after architecture freeze.

  ### 3. Component ownership, then distributed execution — recommended

  Each person develops one independent component against fixed artifact formats. After hour 8, the pipeline is frozen
  and everyone runs compatible data shards.

  This gives parallel experimentation early and parallel computation later.

  ## Team ownership

   Person                                  Primary responsibility                  Deliverables
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   A — Integration and reliability         Data preparation, normalization         Normalized partitions, manifests,
                                           contract, validation split,             orchestration notebook, final zip
                                           checkpoint system, merge, packaging
  ──────────────────────────────────────  ──────────────────────────────────────  ──────────────────────────────────────
   B — Candidate retrieval                 Exact/rare-token blocking, char/word    Retrieval pool, recall/oracle
                                           retrieval, address retrieval,           reports, channel ablations
                                           source-separated fusion
  ──────────────────────────────────────  ──────────────────────────────────────  ──────────────────────────────────────
   C — Cross-script and candidate          India transliteration, fold-safe        Cross-script rescue channel, final
   selection                               aliases, ambiguity reranking,           candidate selector
                                           source-specific K policy
  ──────────────────────────────────────  ──────────────────────────────────────  ──────────────────────────────────────
   D — Matcher and decoder                 Pair features, hard negatives,          Model, validation predictions, final
                                           LightGBM, thresholds, Source 1 set      scored pairs
                                           decoding

  Person A also owns the exact metric implementation and acts as release integrator. This prevents each experimental
  owner from evaluating their own component differently.

  ## Notebook design

  Use several small notebook entry points backed by common importable Python modules:

  notebooks/
    00_smoke_and_contract.ipynb
    10_prepare_data.ipynb
    20_candidate_retrieval.ipynb
    30_candidate_selection.ipynb
    40_matcher_and_decoder.ipynb
    50_final_inference.ipynb
    60_validate_and_package.ipynb

  src/entity_resolution/
    config.py
    checkpoint.py
    normalization.py
    blocking.py
    transliteration.py
    features.py
    matcher.py
    decoder.py
    metrics.py
    outputs.py

  The notebooks remain the user-facing implementation. The modules make the same logic importable in Kaggle, Colab,
  local Jupyter, and the final audited package.

  No notebook may depend on state created manually in another notebook. All communication happens through versioned
  artifacts.

  ## Two-stage candidate generation

  Your master plan already points in this direction, and the additional candidate-ranking instruction makes it
  essential:

  Large internal retrieval pool
          ↓
  cheap evidence-based candidate selector
          ↓
  frozen final candidate set
          ↓
  final LightGBM matcher
          ↓
  matching_results.tsv

  The internal retrieval pool is not submitted. The final post-budget candidate set is:

  candidate_pairs.tsv
  = exact pairs scored by the final matcher

  This gives us room to retrieve broadly for recall, then produce a much smaller auditable set before final matching.

  ### Retrieval channels

  Person B works on:

  - Exact normalized name.
  - Legal-suffix-stripped name.
  - Exact normalized address.
  - Rare name and address tokens.
  - Address-number and postal-code evidence.
  - Character TF-IDF name retrieval.
  - Character TF-IDF address retrieval.
  - Source 2 and Source 3 retrieval independently.

  Person C works on:

  - Original Unicode view.
  - Latin accent-folded view.
  - Indic-to-Latin transliteration.
  - High-confidence aliases mined only from permitted training labels.
  - Cross-script rescue for Source 1 entities missed by the ordinary channels.
  - Secondary ranking of oversized/common-name buckets.
  - Smallest validated K2/K3 candidate budget.

  Research supports starting with tuned sparse blocking and string similarity joins rather than assuming dense retrieval
  will win. Configuration and candidate budgets have a large effect on blocking performance. Large-scale blocking
  comparison. Indic transliteration must preserve both original and transliterated views because romanization is
  variable; ICU provides script transformation support, while Unicode NFKC provides consistent compatibility
  normalization. ICU transforms, Unicode normalization.

  ## Artifact contract

  Every candidate row uses the stable key:

  (source1_entity_id, candidate_entity_id)

  Candidate artifacts contain:

  source1_entity_id
  candidate_entity_id
  candidate_source
  country
  channel_mask
  exact_name_hit
  exact_address_hit
  rare_token_hit
  name_rank
  address_rank
  transliteration_rank
  name_score
  address_score
  fusion_score

  Every artifact directory also contains a manifest:

  schema_version
  run_id
  configuration_hash
  input_hashes
  producer
  completed_shards
  row_count
  pair_count
  content_hash
  library_versions
  elapsed_seconds

  A checkpoint is reused only when input and configuration hashes match.

  ## Sharding strategy

  Use a deterministic SHA-256 function over source1_entity_id, producing 64 logical shards. Never use Python’s built-in
  hash(), because it is not stable between processes.

  Large work is additionally partitioned by:

  split × country × target_source × s1_shard

  For example:

  test / India / S2 / shard_017
  test / India / S3 / shard_017
  test / France / S2 / shard_042

  This allows any teammate to run any missing shard. Nobody writes to the same artifact simultaneously.

  During final execution, the six primary test partitions are:

  - US → S2
  - US → S3
  - India → S2
  - India → S3
  - France → S2
  - France → S3

  Four workers start four partitions; the first two to finish take France or the remaining partitions. If a partition is
  too slow, its 64 Source 1 shards can be distributed further.

  ## Deterministic merging

  Duplicate candidate pairs from multiple channels are merged as follows:

  - Union the channel flags.
  - Preserve every per-channel score and rank.
  - Use minimum rank for best_rank.
  - Use maximum compatible similarity where appropriate.
  - Apply fusion and source-specific budgets once.
  - Break exact score ties by candidate ID.
  - Reject duplicate pair keys after the final merge.

  The frozen candidate artifact receives a content hash. The matcher must produce exactly one score for every frozen
  pair:

  set(scored_pair_keys) == set(frozen_candidate_keys)

  The final candidate TSV is generated from that frozen artifact, never by rerunning blocking. Predictions must pass:

  matched_pair_keys ⊆ frozen_candidate_keys

  ## Seventeen-hour schedule

   Time           Required milestone
  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   0:00–0:30      Freeze schemas, paths, split map, configuration format and shard function
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   0:30–2:00      A prepares smoke data/checkpoints; B publishes baseline candidates; C prepares cross-script views; D
                  builds matcher on smoke candidates
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   Hour 2         Baseline candidate artifact available to every lane
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   2:00–5:00      Independent blocking, transliteration, matcher and decoder experiments
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   Hour 5         First complete validation score and candidate-efficiency report
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   5:00–8:00      Improve only measured failure modes
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   Hour 8         Architecture freeze; no new model families afterward
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   8:00–11:00     Distributed generation of frozen train/test candidates
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   11:00–13:00    Final model training and sharded scoring
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   13:00–14:30    Decode, construct both TSV files, run relational audits
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   14:30–16:00    Fresh-session notebook reproduction and official validator
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   16:00–17:00    Documentation, requirements, README and final package
  ─────────────  ───────────────────────────────────────────────────────────────────────────────────────────────────────
   17:00–18:00    Recovery buffer for failed shards or packaging only

  ## Promotion gates

  A retrieval component is retained only if it increases oracle F0.5 or preserves the same ceiling with fewer
  candidates.

  For each blocker report:

  - Pair recall overall, by source, country, script and address missingness.
  - Complete-entity recall.
  - Oracle macro F0.5.
  - Candidate mean, median, P95, P99 and maximum.
  - Added pairs per unit of oracle improvement.
  - Runtime and peak memory.

  The initial blocking target is at least 99.5% pair recall and roughly 0.995 oracle macro F0.5, but those are
  diagnostic targets rather than promises. Among configurations within 0.001 oracle F0.5 of the best observed
  configuration, choose the smallest candidate set. A larger candidate set must demonstrate a meaningful ceiling
  improvement.

  Matcher additions need a clear macro F0.5 gain on the fixed validation split. Dense retrieval or a transformer becomes
  eligible only if:

  - the classical blocker has a measurable residual recall failure;
  - the failure is concentrated in a tractable slice;
  - a small experiment proves improvement;
  - projected full inference fits the remaining time.

  LightGBM remains the primary matcher because it handles millions of numeric similarity rows efficiently. LightGBM
  paper. Ditto and multilingual encoders are credible later experiments, but applying them to tens of millions of pairs
  is too risky under this deadline. Ditto.

  ## Free-compute allocation

  Use Kaggle CPU notebooks as the primary runners. Official documentation currently lists approximately four CPU cores,
  about 30 GB RAM, a 12-hour notebook-run limit, and up to 20 GB persisted notebook output. Break every stage into 6–9-
  hour maximum jobs. Kaggle notebooks.

  Use:

  - Each teammate’s genuine Kaggle account for their assigned component or shard.
  - Colab only for short GPU or independent experiments; its resources and lifetime are not guaranteed.
  - Local machines for smoke tests, merge, validation, and documentation.
  - SageMaker only if the AWS console confirms an applicable free allowance and the job has been measured to fit. Free
    notebook instances are memory constrained; setup time also works against this deadline. SageMaker AI pricing.

  Do not:

  - Create extra Kaggle or Colab identities.
  - Share account credentials.
  - Upload the challenge data publicly.
  - Use external business databases, geocoding services, entity lookup APIs, or internet-derived identity data.
  - Submit through multiple identities.

  This design preserves your v5.1/v5.2 methodology while making it executable by four people within the deadline. If you
  approve it, the next step is to write the frozen design document and the exact per-person implementation checklist
  before touching code.