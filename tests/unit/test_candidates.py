"""候选算法方案生成的单元测试。

第 3 步不训练模型，只生成可执行的候选方案。
这些测试用于保证候选方案不是空泛描述，而是后续训练代码能读取的结构化配置。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.training.candidates import (
    CandidatePlan,
    build_default_candidate_set,
    validate_candidate_plan,
    write_candidate_json,
)


class CandidatePlanGenerationTest(unittest.TestCase):
    """验证 P0 候选算法方案。"""

    def test_default_candidate_set_has_three_plans(self) -> None:
        """默认候选集应生成 3 个方案，供后续页面展示和用户确认。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")

        self.assertEqual(len(candidate_set.plans), 3)

    def test_each_plan_has_required_fields(self) -> None:
        """每个方案都必须包含后续训练和展示所需的关键字段。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")

        for plan in candidate_set.plans:
            validate_candidate_plan(plan)
            self.assertTrue(plan.plan_id)
            self.assertTrue(plan.display_name)
            self.assertTrue(plan.model_family)
            self.assertTrue(plan.input_channels)
            self.assertTrue(plan.preprocess_steps)
            self.assertTrue(plan.risks)
            self.assertTrue(plan.suitable_for_p0)

    def test_recommended_order_is_unique(self) -> None:
        """候选方案的推荐顺序不能重复，方便页面按顺序展示。"""

        candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="c1")
        orders = [plan.recommended_order for plan in candidate_set.plans]

        self.assertEqual(sorted(orders), [1, 2, 3])

    def test_validate_candidate_plan_rejects_empty_plan(self) -> None:
        """如果方案缺少关键字段，应主动报错，而不是生成不可训练配置。"""

        bad_plan = CandidatePlan(
            plan_id="bad",
            display_name="",
            summary="",
            model_family="",
            input_channels=[],
            preprocess_steps=[],
            feature_strategy="",
            model_structure="",
            training_strategy="",
            expected_cost="",
            advantages=[],
            risks=[],
            recommended_reason="",
            user_confirm_params=[],
            suitable_for_p0=False,
            recommended_order=1,
        )

        with self.assertRaises(ValueError):
            validate_candidate_plan(bad_plan)

    def test_candidate_json_contains_canonical_pipeline_specs(self) -> None:
        """固定候选落盘时必须附带和 LLM 相同结构的 PipelineSpec。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "candidates.json"
            candidate_set = build_default_candidate_set(dataset_id="phm2010", cutter="C1")
            write_candidate_json(candidate_set, output_file)
            payload = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["pipeline_specs"]), 3)
        self.assertEqual(
            payload["pipeline_specs"][0]["pipeline_id"],
            candidate_set.plans[0].plan_id,
        )
        self.assertEqual(len(payload["registry_validations"]), 3)
        self.assertTrue(all(item["valid"] for item in payload["registry_validations"]))


if __name__ == "__main__":
    unittest.main()
