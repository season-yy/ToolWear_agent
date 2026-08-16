"""集中配置对象的单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from toolwear_agent.core.settings import Settings, load_settings, repository_root


class CoreSettingsTest(unittest.TestCase):
    """验证配置来源、派生路径和敏感字段保护。"""

    def test_repository_root_is_computed_from_source_tree(self) -> None:
        """仓库根目录应由源码位置计算，不依赖当前工作目录。"""

        expected = Path(__file__).resolve().parents[2]

        self.assertEqual(repository_root(), expected)

    def test_explicit_env_file_drives_all_derived_paths(self) -> None:
        """只配置 AI_INFRA_ROOT 时，其余运行目录应从该根目录派生。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            env_file = temp_root / "test.env"
            infra_root = temp_root / "infra"
            env_file.write_text(
                "\n".join(
                    [
                        f"AI_INFRA_ROOT={infra_root.as_posix()}",
                        "LLM_API_KEY=unit-test-secret",
                        "RANDOM_SEED=17",
                        "VB_STAGE_THRESHOLDS_UM=80,120,150",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_settings(env_file)

        self.assertEqual(settings.ai_infra_root, infra_root)
        self.assertEqual(settings.experiment_root, infra_root / "experiments" / "runs")
        self.assertEqual(settings.state_db_path, infra_root / "state" / "toolwear.db")
        self.assertEqual(settings.vb_stage_thresholds_um, (80.0, 120.0, 150.0))
        self.assertEqual(settings.random_seed, 17)
        self.assertNotIn("unit-test-secret", repr(settings))
        self.assertNotIn("llm_api_key", settings.model_dump())

    def test_environment_variable_overrides_dotenv_value(self) -> None:
        """系统环境变量的优先级必须高于 `.env`。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / "test.env"
            env_file.write_text("RANDOM_SEED=3", encoding="utf-8")

            with patch.dict(os.environ, {"RANDOM_SEED": "29"}, clear=False):
                settings = load_settings(env_file)

        self.assertEqual(settings.random_seed, 29)

    def test_stage_thresholds_must_be_three_increasing_values(self) -> None:
        """四阶段分类需要且只允许三个严格递增边界。"""

        with self.assertRaises(ValidationError):
            Settings(vb_stage_thresholds_um="90,130")

        with self.assertRaises(ValidationError):
            Settings(vb_stage_thresholds_um="90,90,160")

    def test_tool_api_token_can_be_loaded_from_default_secret_file(self) -> None:
        """服务和 Worker 应通过 Git 外 Secret 文件共享 API Token。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            infra_root = Path(temp_dir)
            secret_file = infra_root / "secrets" / "toolwear_api_token"
            secret_file.parent.mkdir(parents=True)
            secret_file.write_text("tool-api-secret\n", encoding="utf-8")

            settings = Settings(ai_infra_root=infra_root)

        self.assertEqual(settings.tool_api_token, "tool-api-secret")
        self.assertEqual(settings.tool_api_token_file, secret_file)
        self.assertNotIn("tool-api-secret", repr(settings))
        self.assertNotIn("tool_api_token", settings.model_dump())


if __name__ == "__main__":
    unittest.main()
