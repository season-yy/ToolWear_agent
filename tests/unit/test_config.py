"""配置读取的单元测试。

测试目标：
- 能从指定 `.env` 文件读取配置。
- 能把布尔值、整数、阈值转换成正确类型。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.common.config import load_settings


class LoadSettingsTest(unittest.TestCase):
    """验证 `load_settings()` 的基础行为。"""

    def test_load_settings_from_env_file(self) -> None:
        """给定一个临时 `.env` 文件，应能读取并转换核心配置。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=qwen",
                        "TRAIN_DEVICE=cpu",
                        "RANDOM_SEED=7",
                        "ENABLE_VB_REGRESSION=true",
                        "VB_STAGE_THRESHOLDS_UM=80,120,150",
                        "FASTAPI_PORT=19000",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_settings(env_file)

        self.assertEqual(settings.llm_provider, "qwen")
        self.assertEqual(settings.train_device, "cpu")
        self.assertEqual(settings.random_seed, 7)
        self.assertTrue(settings.enable_vb_regression)
        self.assertEqual(settings.vb_stage_thresholds_um, (80.0, 120.0, 150.0))
        self.assertEqual(settings.fastapi_port, 19000)


if __name__ == "__main__":
    unittest.main()
