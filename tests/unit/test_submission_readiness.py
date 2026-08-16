"""初赛提交完整性检查脚本测试。"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_FILE = Path(__file__).resolve().parents[2] / "scripts" / "verify_submission_readiness.py"
SPEC = importlib.util.spec_from_file_location("verify_submission_readiness", SCRIPT_FILE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SubmissionReadinessTests(unittest.TestCase):
    def test_reports_ready_when_all_public_contract_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative_path in MODULE.REQUIRED_FILES.values():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("example\n", encoding="utf-8")
            for relative_path in MODULE.JSON_EXAMPLES:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"status": "example"}), encoding="utf-8")
            request_path = root / "examples/create_experiment_request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "title": "C1 示例实验",
                        "user_request": "验证四阶段磨损分类。",
                        "dataset_id": "phm2010",
                        "cutter_ids": ["C1"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = MODULE.verify_submission_readiness(root)

            self.assertEqual(result["status"], "ready")
            self.assertTrue(all(result["checks"].values()))
            self.assertEqual(result["invalid_examples"], [])

    def test_reports_not_ready_for_invalid_json_example(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative_path in MODULE.REQUIRED_FILES.values():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("example\n", encoding="utf-8")
            for relative_path in MODULE.JSON_EXAMPLES:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            broken_path = root / MODULE.JSON_EXAMPLES[0]
            broken_path.write_text("{broken", encoding="utf-8")

            result = MODULE.verify_submission_readiness(root)

            self.assertEqual(result["status"], "not_ready")
            self.assertIn(MODULE.JSON_EXAMPLES[0], result["invalid_examples"])


if __name__ == "__main__":
    unittest.main()
