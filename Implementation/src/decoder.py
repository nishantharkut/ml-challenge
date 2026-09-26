"""
S1-Level Precision-Oriented Set Decoder and Exclusivity Resolver.
Selects optimal prediction sets per S1 to maximize Macro F0.5.
"""
from collections import defaultdict
from .metrics import competition_macro_f05
from .config import DEFAULT_PROB_THRESHOLD, DEFAULT_MARGIN_THRESHOLD

class SetDecoder:
    def __init__(self, prob_threshold=DEFAULT_PROB_THRESHOLD,
                 margin_threshold=DEFAULT_MARGIN_THRESHOLD,
                 use_query_exclusivity=True):
        self.prob_threshold = prob_threshold
        self.margin_threshold = margin_threshold
        self.use_query_exclusivity = use_query_exclusivity
        
    def decode(self, scored_candidates, s1_all_ids=None):
        """
        Convert scored candidate dictionary into final matching predictions.
        scored_candidates: dict {s1_id: [(cand_id, prob), ...]}
        s1_all_ids: optional list/set of all S1 IDs to ensure every S1 is present.
        Returns: dict {s1_id: set of matched_entity_ids}
        """
        # Step 1: Initial filtering by probability threshold
        raw_matches = defaultdict(list)
        all_ids = sorted(set(s1_all_ids) if s1_all_ids is not None else set(scored_candidates.keys()))
        
        for s1_id in all_ids:
            cands = scored_candidates.get(s1_id, [])
            for cid, prob in cands:
                if prob >= self.prob_threshold:
                    raw_matches[s1_id].append((cid, prob))
                    
        # Step 2: Query Exclusivity (Each S2/S3 matches at most one S1)
        if self.use_query_exclusivity:
            # Map each target query ID to its best (s1_id, prob)
            best_s1_for_query = {}
            for s1_id, matches in raw_matches.items():
                for cid, prob in matches:
                    if (
                        cid not in best_s1_for_query
                        or prob > best_s1_for_query[cid][1]
                        or (
                            prob == best_s1_for_query[cid][1]
                            and s1_id < best_s1_for_query[cid][0]
                        )
                    ):
                        best_s1_for_query[cid] = (s1_id, prob)
                        
            # Keep only the pairs that won exclusivity
            exclusive_matches = {s1_id: set() for s1_id in all_ids}
            for s1_id in all_ids:
                matches = raw_matches.get(s1_id, [])
                for cid, prob in matches:
                    winner_s1, winner_prob = best_s1_for_query.get(cid, (None, 0.0))
                    if winner_s1 == s1_id:
                        exclusive_matches[s1_id].add(cid)
            return dict(exclusive_matches)
        else:
            final_matches = {}
            for s1_id in all_ids:
                final_matches[s1_id] = {cid for cid, _ in raw_matches.get(s1_id, [])}
            return final_matches

    def tune_thresholds(self, val_scored_candidates, val_ground_truth,
                        prob_range=(0.35, 0.65, 0.05)):
        """
        Grid search on validation set to find threshold that maximizes exact Macro F0.5.
        """
        best_score = -1.0
        best_t = self.prob_threshold
        s1_ids = list(val_ground_truth.keys())
        
        print("  Tuning decision threshold on validation set...")
        t_values = [round(t, 2) for t in list(
            [prob_range[0] + i * prob_range[2] 
             for i in range(int((prob_range[1] - prob_range[0]) / prob_range[2]) + 1)]
        )]
        
        for t in t_values:
            self.prob_threshold = t
            preds = self.decode(val_scored_candidates, s1_all_ids=s1_ids)
            score = competition_macro_f05(preds, val_ground_truth)
            print(f"    Threshold {t:.2f} -> Macro F0.5: {score:.6f}")
            if score > best_score:
                best_score = score
                best_t = t
                
        self.prob_threshold = best_t
        print(f"  Selected optimal threshold: {best_t:.2f} (Macro F0.5: {best_score:.6f})")
        return best_t, best_score


def tune_decoder_policies(scored_candidates, ground_truth, thresholds=None):
    """Tune exclusivity-on and exclusivity-off policies independently."""
    if thresholds is None:
        thresholds = [round(0.10 + step * 0.005, 3) for step in range(171)]
    threshold_values = sorted({float(value) for value in thresholds})
    if not threshold_values:
        raise ValueError("thresholds must contain at least one value")

    all_s1_ids = sorted(ground_truth)
    results = {}
    for use_exclusivity in (False, True):
        best = {"threshold": threshold_values[0], "score": -1.0}
        decoder = SetDecoder(
            prob_threshold=threshold_values[0],
            use_query_exclusivity=use_exclusivity,
        )
        for threshold in threshold_values:
            decoder.prob_threshold = threshold
            predictions = decoder.decode(scored_candidates, s1_all_ids=all_s1_ids)
            score = competition_macro_f05(predictions, ground_truth)
            if score > best["score"]:
                best = {"threshold": threshold, "score": score}
        results[use_exclusivity] = best
    return results
