"""Regression tests for failures that must never return."""
from __future__ import annotations

from pathlib import Path
import importlib
import ast

import pytest


def test_matcher_requires_an_explicit_model_path() -> None:
    """No call site may silently select the production checkpoint."""
    from src.matcher import PairwiseMatcher

    with pytest.raises(TypeError):
        PairwiseMatcher()


def test_smoke_model_path_is_inside_requested_sandbox(tmp_path: Path) -> None:
    """Smoke artifacts must be physically unable to alias release artifacts."""
    from src.matcher import smoke_model_path

    path = smoke_model_path(tmp_path)
    assert path.parent == tmp_path
    assert path.name == "smoke_lgbm_matcher.joblib"


def test_smoke_matcher_is_constructed_with_isolated_path(tmp_path: Path) -> None:
    smoke_module = importlib.import_module("00_smoke_and_verify")

    matcher = smoke_module.build_smoke_matcher(tmp_path)

    assert matcher.model_path == tmp_path / "smoke_lgbm_matcher.joblib"


@pytest.mark.parametrize(
    "script_name",
    ["30_feature_and_train.py", "40_tune_decoder.py", "50_full_inference.py"],
)
def test_pipeline_scripts_never_construct_implicit_matcher(script_name: str) -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((implementation_root / script_name).read_text(encoding="utf-8"))
    implicit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PairwiseMatcher"
        and not node.args
        and not node.keywords
    ]
    assert not implicit_calls
