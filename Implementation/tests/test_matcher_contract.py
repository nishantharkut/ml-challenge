from __future__ import annotations

import numpy as np
import pytest

from src.matcher import PairwiseMatcher


class _ProbabilityModel:
    def predict_proba(self, matrix):
        return np.column_stack((np.zeros(len(matrix)), np.ones(len(matrix))))


def test_matcher_refuses_feature_width_drift_before_model_inference(tmp_path):
    matcher = PairwiseMatcher(tmp_path / "matcher.joblib")
    matcher.model = _ProbabilityModel()
    matcher.feature_names = ["first", "second"]

    with pytest.raises(ValueError, match="feature width"):
        matcher.predict_proba(np.zeros((1, 3), dtype=np.float32))
