"""初赛白名单提交包测试。"""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from toolwear_agent.delivery.submission import SubmissionSecurityError, build_submission_package


class SubmissionPackageTests(unittest.TestCase):
    def test_builds_allowlisted_zip_without_runtime_or_secret_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            (root / "toolwear_agent").mkdir(parents=True)
            (root / "tests").mkdir()
            (root / ".env").write_text("LLM_API_KEY=real-secret-value", encoding="utf-8")
            (root / "toolwear_agent" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "toolwear_agent" / "cache.pyc").write_bytes(b"binary")
            (root / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            output = Path(temp_dir) / "submission.zip"

            result = build_submission_package(root, output)

            self.assertEqual(result.file_count, 3)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            self.assertIn("submission_manifest.json", names)
            self.assertIn("toolwear_agent/app.py", names)
            self.assertNotIn(".env", names)
            self.assertNotIn("toolwear_agent/cache.pyc", names)

    def test_rejects_secret_inside_an_allowlisted_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            (root / "toolwear_agent").mkdir(parents=True)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "toolwear_agent" / "bad.py").write_text(
                'TOKEN = "' + "sk-" + 'abcdefghijklmnopqrstuvwxyz123456"\n',
                encoding="utf-8",
            )

            with self.assertRaises(SubmissionSecurityError):
                build_submission_package(root, Path(temp_dir) / "unsafe.zip")

    def test_allows_short_placeholder_credentials_in_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            (root / "tests").mkdir(parents=True)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "tests" / "test_config.py").write_text(
                'llm_api_key="x"\ntool_api_token="local-test-token"\n',
                encoding="utf-8",
            )

            result = build_submission_package(root, Path(temp_dir) / "safe.zip")

            self.assertEqual(result.file_count, 2)

    def test_allows_empty_env_secret_before_another_setting(self) -> None:
        """空密钥不能跨行吞掉下一项配置并被误判为非空。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / ".env.example").write_text(
                "LLM_API_KEY=\nLLM_MODEL=qwen-placeholder\nTOOL_API_TOKEN=\n",
                encoding="utf-8",
            )

            result = build_submission_package(root, Path(temp_dir) / "safe.zip")

            self.assertEqual(result.file_count, 2)

    def test_excludes_presentation_inspection_sidecar(self) -> None:
        """PPT 诊断 sidecar 和过期截图不属于参赛材料，不能进入提交包。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            (root / "docs" / "submission").mkdir(parents=True)
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            sidecar = root / "docs" / "submission" / "deck.pptx.inspect.ndjson"
            sidecar.write_text('{"diagnostic": true}\n', encoding="utf-8")
            legacy_screenshot = root / "docs" / "submission" / "frontend-evaluation-desktop.png"
            legacy_screenshot.write_bytes(b"legacy")
            output = Path(temp_dir) / "submission.zip"

            build_submission_package(root, output)

            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            self.assertNotIn("docs/submission/deck.pptx.inspect.ndjson", names)
            self.assertNotIn("docs/submission/frontend-evaluation-desktop.png", names)

    def test_refuses_to_overwrite_existing_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            output = Path(temp_dir) / "submission.zip"
            output.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                build_submission_package(root, output)
