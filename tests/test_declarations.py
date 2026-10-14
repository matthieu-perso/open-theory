"""The lightweight source index must recognize dotted namespace declarations."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_lean_decls", ROOT / "scripts/check_lean_decls.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DeclarationTests(unittest.TestCase):
    def test_dotted_namespace_theorem_is_indexed(self):
        self.assertIn("OpenTheory.Ridge.training", checker.declared_names())
