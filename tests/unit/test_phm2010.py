"""PHM2010 数据登记与体检的单元测试。

这些测试只使用临时目录中的小型假数据，不读取真实 PHM2010 数据集。
这样测试速度快，也不会依赖本机数据是否存在。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.training.phm2010 import (
    build_cutter_inventory,
    load_wear_labels,
    parse_signal_filename,
)


class Phm2010InventoryTest(unittest.TestCase):
    """验证 PHM2010 C1 数据登记核心逻辑。"""

    def test_parse_signal_filename_reads_cut_index(self) -> None:
        """给定 PHM2010 信号文件名，应能解析出刀次编号。"""

        cut_index = parse_signal_filename("c_1_023.csv", cutter="c1")

        self.assertEqual(cut_index, 23)

    def test_load_wear_labels_reads_three_flutes(self) -> None:
        """给定磨损标签 CSV，应能读取三刀刃 VB 数据。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            wear_file = Path(temp_dir) / "c1_wear.csv"
            wear_file.write_text(
                "\n".join(
                    [
                        "cut,flute_1,flute_2,flute_3",
                        "1,32.0,48.0,37.0",
                        "2,35.0,50.0,38.0",
                    ]
                ),
                encoding="utf-8",
            )

            labels = load_wear_labels(wear_file)

        self.assertEqual(labels[1].flute_1, 32.0)
        self.assertEqual(labels[1].flute_2, 48.0)
        self.assertEqual(labels[2].flute_3, 38.0)

    def test_build_cutter_inventory_matches_signal_and_label(self) -> None:
        """给定信号文件和标签文件，应能生成带标签匹配状态的数据清单。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            cutter_dir = Path(temp_dir) / "c1"
            cutter_dir.mkdir()
            (cutter_dir / "c_1_001.csv").write_text("1,2,3,4,5,6,7\n", encoding="utf-8")
            (cutter_dir / "c_1_002.csv").write_text("1,2,3,4,5,6,7\n", encoding="utf-8")
            (cutter_dir / "c1_wear.csv").write_text(
                "\n".join(
                    [
                        "cut,flute_1,flute_2,flute_3",
                        "1,32.0,48.0,37.0",
                        "2,35.0,50.0,38.0",
                    ]
                ),
                encoding="utf-8",
            )

            inventory = build_cutter_inventory(cutter_dir, cutter="c1")

        self.assertEqual(inventory.cutter, "c1")
        self.assertEqual(inventory.signal_file_count, 2)
        self.assertEqual(inventory.label_count, 2)
        self.assertEqual(inventory.records[0].cut, 1)
        self.assertTrue(inventory.records[0].has_label)
        self.assertEqual(inventory.records[0].channel_count_sample, 7)


if __name__ == "__main__":
    unittest.main()
