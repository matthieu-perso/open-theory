"""Regression tests for blueprint metadata and non-mutating freshness checks."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_blueprint", ROOT / "scripts/export_blueprint.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class BlueprintTests(unittest.TestCase):
    def test_export_uses_repository_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            result = exporter.parse_blueprint(Path(directory) / "blueprint.json")
            self.assertEqual(result["toolchain"], "leanprover/lean4:v4.34.0")

    def test_freshness_rejects_changed_node_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blueprint.json"
            result = exporter.parse_blueprint(path)
            result["nodes"][0]["title"] = "stale title"
            path.write_text(json.dumps(result))
            before = path.read_bytes()
            self.assertFalse(exporter.check_blueprint(path))
            self.assertEqual(path.read_bytes(), before)

    def test_freshness_ignores_only_generation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blueprint.json"
            result = exporter.parse_blueprint(path)
            result["generatedAt"] = "old timestamp"
            path.write_text(json.dumps(result))
            self.assertTrue(exporter.check_blueprint(path))


if __name__ == "__main__":
    unittest.main()
