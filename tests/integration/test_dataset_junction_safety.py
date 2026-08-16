"""验证 PHM2010 Junction 的真实路径安全边界。"""

from __future__ import annotations

import unittest

from toolwear_agent.core.errors import PathBoundaryError
from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.settings import load_settings


class DatasetJunctionSafetyTest(unittest.TestCase):
    """使用本机 C1 Junction 做只读集成验证。"""

    def test_junction_target_requires_explicit_manifest_root(self) -> None:
        """Junction 目标必须由 Dataset manifest 显式加入允许根目录。"""

        settings = load_settings()
        resolver = PathResolver(settings)
        cutter_link = settings.phm2010_raw_root / "c1"
        signal_file = cutter_link / "c_1_001.csv"
        if not signal_file.is_file():
            self.skipTest("当前机器未配置 PHM2010 C1 Junction。")

        resolved_cutter_root = cutter_link.resolve(strict=True)

        # 只信任表面 raw root 时，解析后的 Junction 目标已经越界，必须拒绝。
        with self.assertRaises(PathBoundaryError):
            resolver.assert_dataset_read_path(signal_file)

        # Dataset Registry 校验 manifest 后，可把真实 cutter 根加入本次只读允许列表。
        accepted = resolver.assert_dataset_read_path(
            signal_file,
            additional_roots=[resolved_cutter_root],
        )
        self.assertEqual(accepted, signal_file.resolve(strict=True))


if __name__ == "__main__":
    unittest.main()
