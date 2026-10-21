"""Prevent accidentally reintroducing the legacy V1 preprocessing or raw dataset API."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CleanupLayoutTest(unittest.TestCase):
    def test_only_canonical_preprocessing(self):
        for legacy in (
            "src/preprocessing.py",
            "src/dataset.py",
            "src/pipeline.py",
            "src/segmentation_model.py",
        ):
            self.assertFalse((ROOT / legacy).exists(), legacy)
        self.assertTrue((ROOT / "src/preprocessing_v2.py").exists())

    def test_no_deleted_module_imports(self):
        forbidden = {
            "src.preprocessing", "src.dataset", "src.pipeline",
            "src.segmentation_model", "dataset", "dinov2_encoder",
        }
        for root_name in ("src", "scripts"):
            for file in (ROOT / root_name).rglob("*.py"):
                code = ast.parse(file.read_text(), filename=str(file))
                for node in ast.walk(code):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, forbidden, str(file))
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotIn(alias.name, forbidden, str(file))

    def test_central_root_config(self):
        from src.config import DATASET_ROOT
        self.assertEqual(DATASET_ROOT.name, "nffa_sem_100")


if __name__ == "__main__":
    unittest.main()
