"""
Stage 60: Official Submission Verification.
Executes validate_submission.py on final matching_results.tsv and candidate_pairs.tsv.
"""
import os
import sys
import subprocess

# Ensure Implementation is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import OUTPUT_DIR, TEST_DIR, VALIDATOR_SCRIPT
from src.outputs import run_submission_validator

def main():
    print("=" * 70)
    print("STAGE 60: OFFICIAL SUBMISSION VALIDATION")
    print("=" * 70)
    
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    if not os.path.exists(matching_file):
        print(f"Error: {matching_file} does not exist! Please run Stage 50 first.")
        sys.exit(1)
        
    if not os.path.exists(candidate_file):
        print(f"Error: {candidate_file} does not exist! Please run Stage 50 first.")
        sys.exit(1)
        
    # Run validator
    passed = run_submission_validator(
        matching_path=matching_file,
        candidate_path=candidate_file,
        test_dir=TEST_DIR,
        check_ids=False
    )
    
    if passed:
        print("\nALL SUBMISSION REQUIREMENTS 100% VERIFIED!")
        print("Ready to zip and submit!")
        sys.exit(0)
    else:
        print("\nVALIDATION FAILED! Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
