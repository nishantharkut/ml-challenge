"""
Exact Competition Metric, Candidate Recall, and Oracle Ceiling Evaluator.
Macro F0.5 with exact handling of singletons, one-to-many matches, and candidate sets.
"""
import numpy as np
from collections import defaultdict

def single_entity_f05(true_set, pred_set, beta=0.5):
    """
    Compute F0.5 for a single S1 entity.
    Special rules:
    - true is empty and pred is empty: 1.0 (correctly identified singleton)
    - true is empty and pred is non-empty: 0.0 (false merge on singleton)
    - true is non-empty and pred is empty: 0.0 (missed all matches)
    """
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0
    
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    # Direct formula: 5 * tp / (5 * tp + 4 * fp + fn) when beta = 0.5
    f05 = 5.0 * tp / (5.0 * tp + 4.0 * fp + fn)
    return f05

def competition_macro_f05(predictions, ground_truth, beta=0.5):
    """
    Calculate Macro F0.5 across all S1 entities in ground_truth.
    predictions: dict {s1_id: set of predicted ids}
    ground_truth: dict {s1_id: set of true ids}
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(single_entity_f05(true_set, pred_set, beta=beta))
    return float(np.mean(scores)) if scores else 0.0

def oracle_macro_f05(candidates, ground_truth, beta=0.5):
    """
    Calculate the Oracle Macro F0.5 ceiling for a candidate generator.
    The oracle selects the perfect intersection (GT & candidates).
    """
    oracle_preds = {}
    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        oracle_preds[s1_id] = true_set & cand_set
    return competition_macro_f05(oracle_preds, ground_truth, beta=beta)

def evaluate_blocking_quality(candidates, ground_truth):
    """
    Generate comprehensive diagnostic certificate for candidate generation.
    Returns dictionary with pair recall, complete entity recall, oracle ceiling, and counts.
    """
    total_pairs = 0
    found_pairs = 0
    s2_total = 0
    s2_found = 0
    s3_total = 0
    s3_found = 0
    
    complete_entities = 0
    total_entities = len(ground_truth)
    
    cand_counts = []
    
    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        cand_counts.append(len(cand_set))
        
        if len(true_set) == 0:
            # Singleton: candidate set contains all 0 true matches trivially
            complete_entities += 1
            continue
            
        found = true_set & cand_set
        total_pairs += len(true_set)
        found_pairs += len(found)
        
        if found == true_set:
            complete_entities += 1
            
        for tid in true_set:
            if tid.startswith("S2"):
                s2_total += 1
                if tid in cand_set:
                    s2_found += 1
            elif tid.startswith("S3"):
                s3_total += 1
                if tid in cand_set:
                    s3_found += 1
                    
    cand_counts.sort()
    n = len(cand_counts)
    
    report = {
        "total_s1": total_entities,
        "pair_recall": found_pairs / total_pairs if total_pairs > 0 else 1.0,
        "s2_pair_recall": s2_found / s2_total if s2_total > 0 else 1.0,
        "s3_pair_recall": s3_found / s3_total if s3_total > 0 else 1.0,
        "complete_entity_recall": complete_entities / total_entities if total_entities > 0 else 1.0,
        "oracle_macro_f05": oracle_macro_f05(candidates, ground_truth),
        "mean_candidates_per_s1": float(np.mean(cand_counts)) if cand_counts else 0.0,
        "median_candidates": cand_counts[n // 2] if n > 0 else 0,
        "p95_candidates": cand_counts[int(n * 0.95)] if n > 0 else 0,
        "p99_candidates": cand_counts[int(n * 0.99)] if n > 0 else 0,
        "max_candidates": cand_counts[-1] if n > 0 else 0,
        "total_candidate_pairs": sum(cand_counts)
    }
    return report
