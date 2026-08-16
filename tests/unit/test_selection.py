"""候选方案确认逻辑的单元测试。

第 4 步的前端页面只是入口，真正需要长期稳定的是：
读取候选方案 -> 按 plan_id 选择 -> 保存确认结果。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.training.candidates import build_default_candidate_set, write_candidate_json
from toolwear_agent.training.selection import (
    SelectedPlan,
    find_candidate_plan,
    load_candidate_set,
    render_selected_plan_report,
    select_candidate_plan,
)


class CandidateSelectionTest(unittest.TestCase):
    """验证候选方案确认逻辑。"""

    def test_find_candidate_plan_by_plan_id(self) -> None:
        """给定 plan_id，应能找到对应候选方案。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")

        plan = find_candidate_plan(candidate_set, "multichannel_window_1d_cnn")

        self.assertEqual(plan.plan_id, "multichannel_window_1d_cnn")

    def test_find_candidate_plan_rejects_unknown_id(self) -> None:
        """给定不存在的 plan_id，应明确报错。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")

        with self.assertRaises(ValueError):
            find_candidate_plan(candidate_set, "not_exists")

    def test_select_candidate_plan_builds_result(self) -> None:
        """确认方案后，应生成结构完整的选择结果。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")

        selected = select_candidate_plan(
            candidate_set,
            plan_id="statistical_features_random_forest",
            confirmed_by="local_user",
        )

        self.assertIsInstance(selected, SelectedPlan)
        self.assertEqual(selected.selected_plan.plan_id, "statistical_features_random_forest")
        self.assertEqual(selected.confirmed_by, "local_user")
        self.assertEqual(selected.dataset_id, "phm2010")

    def test_load_candidate_set_from_json(self) -> None:
        """候选方案 JSON 应能被重新读取，供前端页面使用。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "candidate_plans.json"
            write_candidate_json(candidate_set, json_file)

            loaded = load_candidate_set(json_file)

        self.assertEqual(loaded.dataset_id, "phm2010")
        self.assertEqual(len(loaded.plans), 3)

    def test_render_selected_plan_report_contains_risk_and_cost(self) -> None:
        """确认报告必须包含成本和风险，方便用户复盘选择依据。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")
        selected = select_candidate_plan(candidate_set, "statistical_features_random_forest")

        report = render_selected_plan_report(selected)

        self.assertIn("预计成本", report)
        self.assertIn("风险", report)
        self.assertIn("statistical_features_random_forest", report)


if __name__ == "__main__":
    unittest.main()
