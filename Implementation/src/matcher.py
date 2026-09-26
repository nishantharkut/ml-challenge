"""
Pairwise Matcher Training and Inference Module.
Trains LightGBM classifier with structured hard negatives from retrieval candidates.
"""
import os
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from .config import LGBM_PARAMS, MAX_NEGATIVES_PER_POSITIVE, RANDOM_SEED

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

def smoke_model_path(smoke_dir):
    """Return the isolated model path used by smoke runs."""
    return Path(smoke_dir) / "smoke_lgbm_matcher.joblib"


class PairwiseMatcher:
    def __init__(self, model_path):
        if model_path is None:
            raise TypeError("model_path is required; production paths must be explicit")
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = []
        
    def train(self, X, y, X_val=None, y_val=None, feature_names=None):
        """Train LightGBM binary classifier with early stopping if validation data provided."""
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        
        if HAS_LGBM:
            print("  Training LightGBM Classifier...")
            self.model = lgb.LGBMClassifier(**LGBM_PARAMS)
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
            else:
                self.model.fit(X, y)
        else:
            print("  LightGBM not found. Using sklearn HistGradientBoostingClassifier fallback...")
            self.model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=45,
                random_state=RANDOM_SEED
            )
            self.model.fit(X, y)
            
        self.save()
        return self.model
        
    def predict_proba(self, X):
        """Predict match probability for feature matrix X."""
        if self.model is None:
            self.load()
        matrix = np.asarray(X)
        if matrix.ndim != 2:
            raise ValueError(f"feature matrix must be two-dimensional, got shape {matrix.shape}")
        if not self.feature_names:
            raise ValueError("model checkpoint has no ordered feature schema")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                "feature width does not match model schema: "
                f"matrix has {matrix.shape[1]}, model requires {len(self.feature_names)}"
            )
        if HAS_LGBM:
            return self.model.predict_proba(matrix)[:, 1]
        else:
            return self.model.predict_proba(matrix)[:, 1]
            
    def save(self):
        """Save model artifact."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, self.model_path)
        
    def load(self):
        """Load model artifact."""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.feature_names = data.get('feature_names', [])
            return True
        return False
        
    def get_feature_importances(self):
        """Return dict of feature importances."""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}
