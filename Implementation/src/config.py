"""
Configuration and Environment Settings for Amazon ML Challenge 2026.
Supports local Windows, Kaggle, and Google Colab environments automatically.
"""
import os
import sys
from pathlib import Path

from .run_context import RunContext

def detect_environment():
    """Detect whether running locally, on Kaggle, or on Google Colab."""
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    elif "google.colab" in sys.modules or os.path.exists("/content"):
        return "colab"
    else:
        return "local"

ENV = detect_environment()
RUN_CONTEXT = RunContext.from_environment(
    implementation_root=Path(__file__).resolve().parents[1]
)

BASE_DIR = str(RUN_CONTEXT.implementation_root.parent)
IMPL_DIR = str(RUN_CONTEXT.implementation_root)
DATASET_DIR = str(RUN_CONTEXT.data_root)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
CHECKPOINT_DIR = str(RUN_CONTEXT.checkpoint_dir)
OUTPUT_DIR = str(RUN_CONTEXT.output_dir)
SCRIPTS_DIR = os.path.join(IMPL_DIR, "scripts")

# Official validator path
VALIDATOR_SCRIPT = os.path.join(
    os.path.dirname(DATASET_DIR), "utils", "validate_submission.py"
) if os.path.exists(os.path.join(os.path.dirname(DATASET_DIR), "utils", "validate_submission.py")) else ""

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters & Constants
RANDOM_SEED = 42
BETA = 0.5  # Macro F0.5 evaluation

# Blocking Configuration
EXACT_NAME_BUCKET_LIMIT = 100
EXACT_ADDR_BUCKET_LIMIT = 100
RARE_TOKEN_IDF_PERCENTILE = 80
TFIDF_NAME_NGRAMS = (3, 4)
TFIDF_NAME_MAX_FEATURES = 100_000
TFIDF_NAME_TOP_K = 15
TFIDF_NAME_THRESHOLD = 0.12

TFIDF_ADDR_NGRAMS = (3, 4)
TFIDF_ADDR_MAX_FEATURES = 80_000
TFIDF_ADDR_TOP_K = 10
TFIDF_ADDR_THRESHOLD = 0.12

# Candidate Budgets (Source-Separated)
MAX_CANDIDATES_PER_SOURCE = 15
TOTAL_CANDIDATE_BUDGET = 25

# Matcher Configuration
MAX_NEGATIVES_PER_POSITIVE = 8
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 45,
    'max_depth': 8,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 600,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1
}

# Decision Decoder Defaults
DEFAULT_PROB_THRESHOLD = 0.45
DEFAULT_MARGIN_THRESHOLD = 0.03
USE_QUERY_EXCLUSIVITY = True
