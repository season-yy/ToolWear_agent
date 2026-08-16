"""使用本机只读 PHM2010 数据验证通用 Adapter。"""

from __future__ import annotations

import unittest

from toolwear_agent.core.settings import load_settings
from toolwear_agent.data.adapters.phm2010 import PHM2010Adapter


class PHM2010AdapterIntegrationTest(unittest.TestCase):
    """验证真实 C1-C6 目录、标签和轻量体检结果。"""

    def test_real_dataset_inventory_is_complete(self) -> None:
        """真实数据应包含六把刀，且标签角色和 315 刀次符合 PHM2010。"""

        root = load_settings().phm2010_raw_root
        if not root.is_dir():
            self.skipTest("当前机器未配置 PHM2010 原始数据目录。")

        inspection = PHM2010Adapter().inspect(root)
        manifest = inspection.manifest

        self.assertTrue(inspection.validation.valid, inspection.validation.issues)
        self.assertEqual(set(manifest.available_cutter_ids), {"C1", "C2", "C3", "C4", "C5", "C6"})
        self.assertEqual(set(manifest.labeled_cutter_ids), {"C1", "C4", "C6"})
        self.assertEqual(set(manifest.unlabeled_cutter_ids), {"C2", "C3", "C5"})
        self.assertEqual(len(manifest.manifest_hash or ""), 64)

        for cutter in manifest.cutters.values():
            self.assertEqual(cutter.signal_file_count, 315)
            self.assertEqual(cutter.detected_channel_count, 7)
            self.assertTrue(cutter.sampled_signal_lengths)
            self.assertTrue(all(length > 0 for length in cutter.sampled_signal_lengths))
            self.assertIsNotNone(cutter.resolved_path)

        for cutter_id in manifest.labeled_cutter_ids:
            self.assertEqual(manifest.cutters[cutter_id].wear_row_count, 315)


if __name__ == "__main__":
    unittest.main()
