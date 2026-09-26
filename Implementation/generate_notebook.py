"""
Notebook Generator: Creates a clean, portable Jupyter Notebook for Kaggle / Colab / Local execution.
"""
import json
import os

def create_notebook():
    notebook_path = r"C:\N Drive\Amazon ML Challenge\Implementation\notebooks\Amazon_ML_Challenge_2026.ipynb"
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏆 Amazon ML Challenge 2026 — Entity Resolution\n",
                "### End-to-End Scalable, Portable Pipeline (Kaggle / Colab / Local)\n",
                "\n",
                "This notebook implements the complete Entity Resolution system according to **Master Plan v5.1/v5.2**:\n",
                "- **Deterministic & Multi-View Normalization** (NFKC, Indic vowel mark protection, legal suffix parsing)\n",
                "- **Source-Separated Multi-Channel Blocking** (Exact Name/Address, Rare Token IDF, Char TF-IDF)\n",
                "- **Deterministic Feature Extraction** (RapidFuzz string distances, address agreement/conflict, retrieval agreement)\n",
                "- **LightGBM Pairwise Matcher** trained with structured hard negatives\n",
                "- **Precision-Oriented S1 Set Decoder** with query exclusivity\n",
                "- **Compliant TSV Generation** (`candidate_pairs.tsv` and `matching_results.tsv`)\n",
                "- **Official Submission Validation**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Environment Setup and Dependency Check\n",
                "import sys, os\n",
                "!pip install -q rapidfuzz sparse_dot_topn lightgbm polars anyascii\n",
                "print(f\"Python Version: {sys.version}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Import Pipeline Source Modules\n",
                "import os, sys\n",
                "# Add Implementation directory to sys.path\n",
                "for path in [\"/kaggle/working\", \"/content/Implementation\", \".\", \"..\"]:\n",
                "    if os.path.exists(path) and path not in sys.path:\n",
                "        sys.path.insert(0, os.path.abspath(path))\n",
                "\n",
                "from src.config import ENV, DATASET_DIR, TRAIN_DIR, TEST_DIR, OUTPUT_DIR, CHECKPOINT_DIR\n",
                "print(f\"Detected Environment: {ENV.upper()}\")\n",
                "print(f\"Dataset Directory:    {DATASET_DIR}\")\n",
                "print(f\"Output Directory:     {OUTPUT_DIR}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Fast Smoke Test (1,000 entities)\n",
                "import importlib\n",
                "smoke = importlib.import_module(\"00_smoke_and_verify\")\n",
                "smoke.run_smoke_test(sample_size=1000)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Stage 10 — Data Preparation & Entity-Disjoint Split\n",
                "stage10 = importlib.import_module(\"10_prepare_data\")\n",
                "stage10.prepare_data(val_sample_size=50000)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Stage 20 - Candidate Retrieval & Blocking Evaluation\n",
                "stage20 = importlib.import_module(\"20_candidate_retrieval\")\n",
                "stage20.run_candidate_retrieval()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Stage 30 — Feature Extraction & LightGBM Matcher Training\n",
                "stage30 = importlib.import_module(\"30_feature_and_train\")\n",
                "stage30.train_matcher(train_entity_limit=40000)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Stage 40 — Decision Decoder Optimization & Threshold Tuning\n",
                "stage40 = importlib.import_module(\"40_tune_decoder\")\n",
                "stage40.tune_decoder(val_eval_limit=15000)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8: Stage 50 — Full Test Inference & Output Generation\n",
                "stage50 = importlib.import_module(\"50_full_inference\")\n",
                "stage50.run_test_inference()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 9: Stage 60 — Official Submission Validation Check\n",
                "stage60 = importlib.import_module(\"60_validate_submission\")\n",
                "stage60.main()"
            ]
        }
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    print(f"Jupyter Notebook successfully created at {notebook_path}")

if __name__ == "__main__":
    create_notebook()
