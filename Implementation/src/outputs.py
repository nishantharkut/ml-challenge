"""
Submission File Formatting, Export, and Validation Execution Module.
Produces 100% compliant candidate_pairs.tsv and matching_results.tsv.
"""
import os
import subprocess
import sys
from .config import OUTPUT_DIR, VALIDATOR_SCRIPT, TEST_DIR

def export_candidate_pairs(candidates_by_s1, all_s1_ids, filepath=None):
    """
    Export candidate_pairs.tsv strictly matching official competition format:
    source1_entity_id\tcandidate_entity_ids
    """
    out_path = filepath or os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    print(f"Exporting candidate_pairs.tsv to {out_path} ({len(all_s1_ids):,} S1 rows)...")
    
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates_by_s1.get(s1_id, set())
            if isinstance(cands, dict):
                cand_list = sorted(cands.keys())
            else:
                cand_list = sorted(cands)
            cand_str = ",".join(cand_list)
            f.write(f"{s1_id}\t{cand_str}\n")
            
    print(f"  Successfully wrote {out_path}")
    return out_path

def export_matching_results(predictions_by_s1, all_s1_ids, filepath=None):
    """
    Export matching_results.tsv strictly matching official competition format:
    source1_entity_id\tmatched_entity_ids
    """
    out_path = filepath or os.path.join(OUTPUT_DIR, "matching_results.tsv")
    print(f"Exporting matching_results.tsv to {out_path} ({len(all_s1_ids):,} S1 rows)...")
    
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = predictions_by_s1.get(s1_id, set())
            match_list = sorted(matches)
            match_str = ",".join(match_list)
            f.write(f"{s1_id}\t{match_str}\n")
            
    print(f"  Successfully wrote {out_path}")
    return out_path

def run_submission_validator(matching_path=None, candidate_path=None, test_dir=None, check_ids=False):
    """
    Run official validate_submission.py tool on generated files.
    """
    matching_file = matching_path or os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = candidate_path or os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    test_folder = test_dir or TEST_DIR
    
    if not os.path.exists(VALIDATOR_SCRIPT):
        print(f"  Warning: Validator script not found at {VALIDATOR_SCRIPT}. Skipping external check.")
        return False
        
    cmd = [
        sys.executable,
        VALIDATOR_SCRIPT,
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", test_folder
    ]
    if check_ids:
        cmd.append("--check-ids")
        
    print(f"Running submission validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.stdout:
        safe_out = result.stdout.encode('ascii', errors='replace').decode('ascii')
        print(safe_out)
    if result.stderr:
        safe_err = result.stderr.encode('ascii', errors='replace').decode('ascii')
        print("Validator STDERR:", safe_err)
        
    if result.returncode == 0:
        print("  [PASSED] SUBMISSION VALIDATION PASSED!")
        return True
    else:
        print(f"  [FAILED] SUBMISSION VALIDATION FAILED with code {result.returncode}!")
        return False
