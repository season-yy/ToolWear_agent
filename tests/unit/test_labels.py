"""磨损标签生成的单元测试。

第 2 步的核心是把三刀刃 VB 转换成：
- 一个聚合后的磨损值，当前默认使用最大值 `max`
- 一个离散磨损阶段，当前默认按 90/130/160 um 分成四类
"""

from __future__ import annotations

import unittest

from toolwear_agent.training.labels import (
    DEFAULT_STAGE_NAMES,
    assign_wear_stage,
    build_label_records,
    compute_vb_value,
    summarize_stage_distribution,
)
from toolwear_agent.training.phm2010 import WearLabel


class WearLabelGenerationTest(unittest.TestCase):
    """验证 VB 聚合和四阶段标签生成。"""

    def test_compute_vb_value_uses_max_by_default(self) -> None:
        """给定三刀刃 VB，默认应取最大值作为本刀次磨损值。"""

        label = WearLabel(cut=1, flute_1=32.0, flute_2=48.0, flute_3=37.0)

        vb_value = compute_vb_value(label, aggregation="max")

        self.assertEqual(vb_value, 48.0)

    def test_assign_wear_stage_uses_three_thresholds(self) -> None:
        """给定 90/130/160 um 阈值，应生成四个磨损阶段。"""

        thresholds = (90.0, 130.0, 160.0)

        self.assertEqual(assign_wear_stage(80.0, thresholds).stage_name, DEFAULT_STAGE_NAMES[0])
        self.assertEqual(assign_wear_stage(90.0, thresholds).stage_name, DEFAULT_STAGE_NAMES[1])
        self.assertEqual(assign_wear_stage(140.0, thresholds).stage_name, DEFAULT_STAGE_NAMES[2])
        self.assertEqual(assign_wear_stage(170.0, thresholds).stage_name, DEFAULT_STAGE_NAMES[3])

    def test_build_label_records_keeps_cut_order(self) -> None:
        """生成标签记录时，应按刀次编号升序输出。"""

        labels = {
            2: WearLabel(cut=2, flute_1=100.0, flute_2=120.0, flute_3=110.0),
            1: WearLabel(cut=1, flute_1=30.0, flute_2=40.0, flute_3=50.0),
        }

        records = build_label_records(labels, aggregation="max", thresholds=(90.0, 130.0, 160.0))

        self.assertEqual([record.cut for record in records], [1, 2])
        self.assertEqual(records[0].vb_value, 50.0)
        self.assertEqual(records[1].stage_name, DEFAULT_STAGE_NAMES[1])

    def test_summarize_stage_distribution_counts_each_stage(self) -> None:
        """阶段分布统计应包含四个阶段，即使某阶段数量为 0。"""

        labels = {
            1: WearLabel(cut=1, flute_1=30.0, flute_2=40.0, flute_3=50.0),
            2: WearLabel(cut=2, flute_1=100.0, flute_2=120.0, flute_3=110.0),
        }
        records = build_label_records(labels, aggregation="max", thresholds=(90.0, 130.0, 160.0))

        distribution = summarize_stage_distribution(records)

        self.assertEqual(distribution[DEFAULT_STAGE_NAMES[0]], 1)
        self.assertEqual(distribution[DEFAULT_STAGE_NAMES[1]], 1)
        self.assertEqual(distribution[DEFAULT_STAGE_NAMES[2]], 0)
        self.assertEqual(distribution[DEFAULT_STAGE_NAMES[3]], 0)


if __name__ == "__main__":
    unittest.main()
