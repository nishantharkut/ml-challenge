"""
Master Pipeline Runner for Amazon ML Challenge 2026.
Orchestrates end-to-end execution across all stages with checkpointing,
manifest verification, progress logging, and final validation.
"""
import os
import sys
import time
import argparse
import importlib

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import ENV, BASE_DIR, IMPL_DIR, DATASET_DIR, OUTPUT_DIR, CHECKPOINT_DIR

STAGE_NUMBERS = (10, 20, 30, 40, 50, 60)


def invalidate_stage_manifests(stages, checkpoint_dir=CHECKPOINT_DIR):
    """Remove completion markers so existing stage APIs rerun unchanged."""
    for stage in stages:
        manifest_path = os.path.join(checkpoint_dir, f"stage_{stage}_manifest.json")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
            print(f"  Invalidated checkpoint manifest: {manifest_path}")

def print_banner():
    print("=" * 75)
    print(" 🏆 AMAZON ML CHALLENGE 2026 — LARGE-SCALE ENTITY RESOLUTION PIPELINE 🏆")
    print(f" Environment: {ENV.upper()} | Base Directory: {IMPL_DIR}")
    print(f" Dataset Path: {DATASET_DIR}")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    parser.add_argument("--smoke", action="store_true", help="Run only fast smoke test (1000 entities)")
    parser.add_argument("--stage", type=int, default=0, help="Run from specific stage (10, 20, 30, 40, 50, 60)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if checkpoint exists")
    args = parser.parse_args()

    print_banner()
    t_start = time.time()

    if args.smoke:
        print("\n>>> RUNNING SMOKE TEST ONLY <<<")
        smoke = importlib.import_module("00_smoke_and_verify")
        smoke.run_smoke_test()
        return

    if args.force:
        stages_to_run = [stage for stage in STAGE_NUMBERS if args.stage <= stage]
        print(f"\n>>> FORCE ENABLED: INVALIDATING STAGES {stages_to_run} <<<")
        invalidate_stage_manifests(stages_to_run)

    # Stage 10: Data Preparation & Disjoint Split
    if args.stage <= 10:
        print("\n>>> STAGE 10: DATA PREPARATION & DISJOINT SPLIT <<<")
        s10 = importlib.import_module("10_prepare_data")
        s10.prepare_data()

    # Stage 20: Candidate Retrieval & Blocking Evaluation
    if args.stage <= 20:
        print("\n>>> STAGE 20: CANDIDATE RETRIEVAL & BLOCKING EVALUATION <<<")
        s20 = importlib.import_module("20_candidate_retrieval")
        s20.run_candidate_retrieval()

    # Stage 30: Feature Engineering & Matcher Training
    if args.stage <= 30:
        print("\n>>> STAGE 30: FEATURE ENGINEERING & MATCHER TRAINING <<<")
        s30 = importlib.import_module("30_feature_and_train")
        s30.train_matcher()

    # Stage 40: Decision Decoder Optimization
    if args.stage <= 40:
        print("\n>>> STAGE 40: DECISION DECODER OPTIMIZATION <<<")
        s40 = importlib.import_module("40_tune_decoder")
        s40.tune_decoder()

    # Stage 50: Full Test Inference & TSV Generation
    if args.stage <= 50:
        print("\n>>> STAGE 50: FULL TEST INFERENCE & TSV GENERATION <<<")
        s50 = importlib.import_module("50_full_inference")
        s50.run_test_inference()

    # Stage 60: Official Submission Verification
    if args.stage <= 60:
        print("\n>>> STAGE 60: OFFICIAL SUBMISSION VALIDATION <<<")
        s60 = importlib.import_module("60_validate_submission")
        s60.main()

    print("\n" + "=" * 75)
    print(f"🎉 PIPELINE COMPLETE! Total Elapsed Time: {time.time() - t_start:.2f}s")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
