import ast
import json
import re
import unittest
from pathlib import Path


IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
RUN_PIPELINE = IMPLEMENTATION_DIR / "run_pipeline.py"
NOTEBOOK_GENERATOR = IMPLEMENTATION_DIR / "generate_notebook.py"
NOTEBOOK = IMPLEMENTATION_DIR / "notebooks" / "Amazon_ML_Challenge_2026.ipynb"


def _python_from_notebook_cell(source):
    """Replace IPython shell escapes so ordinary Python syntax can be checked."""
    lines = []
    for line in source.splitlines():
        lines.append("pass  # IPython shell escape" if line.lstrip().startswith("!") else line)
    return "\n".join(lines)


class PipelinePortabilityTests(unittest.TestCase):
    def test_pipeline_orchestrates_stage_20(self):
        source = RUN_PIPELINE.read_text(encoding="utf-8")

        self.assertIn('import_module("20_candidate_retrieval")', source)
        self.assertLess(source.index('import_module("10_prepare_data")'),
                        source.index('import_module("20_candidate_retrieval")'))
        self.assertLess(source.index('import_module("20_candidate_retrieval")'),
                        source.index('import_module("30_feature_and_train")'))

    def test_force_flag_is_consumed_by_orchestration(self):
        source = RUN_PIPELINE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        referenced_attributes = {
            (node.value.id, node.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        }

        self.assertIn(("args", "force"), referenced_attributes)
        self.assertIn("invalidate_stage_manifests", source)

    def test_checked_in_notebook_code_cells_have_valid_python_import_syntax(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            with self.subTest(cell=index):
                compile(_python_from_notebook_cell(source), f"notebook-cell-{index}", "exec")

    def test_notebook_generator_does_not_emit_numbered_direct_imports(self):
        source = NOTEBOOK_GENERATOR.read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"from\s+\d\d_[A-Za-z0-9_]+\s+import", source))
        self.assertIn("importlib.import_module", source)

    def test_notebook_orchestrates_stage_20(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        notebook_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        generator_source = NOTEBOOK_GENERATOR.read_text(encoding="utf-8")

        self.assertIn('import_module("20_candidate_retrieval")', notebook_source)
        self.assertIn('import_module(\\\"20_candidate_retrieval\\\")', generator_source)


if __name__ == "__main__":
    unittest.main()
