"""旧固定候选和 LLM 候选统一转换为 PipelineSpec 的测试。"""

from __future__ import annotations

import unittest

from toolwear_agent.agentteams.llm_candidates import LlmCandidatePlan
from toolwear_agent.schemas import PipelineSpec
from toolwear_agent.schemas.converters import candidate_plan_to_pipeline, llm_candidate_plan_to_pipeline
from toolwear_agent.training.candidates import build_default_candidate_set


class PipelineConverterTest(unittest.TestCase):
    """验证两类候选不再形成两套页面选项。"""

    def test_fixed_and_llm_random_forest_share_the_same_module_chain(self) -> None:
        """同一个方案无论来源为何，最终执行模块链必须相同。"""

        fixed_plan = build_default_candidate_set("phm2010", "C1").plans[0]
        llm_plan = LlmCandidatePlan(
            plan_id="statistical_features_random_forest",
            display_name="统计特征 + RandomForest",
            module_pipeline=["窗口统计特征", "RandomForestClassifier"],
            reason="LLM 推荐的低成本基线。",
            risk="跨刀具泛化待验证。",
            expected_cost="低",
            trainable_now=True,
            training_backend="sklearn_random_forest",
        )

        fixed_pipeline = candidate_plan_to_pipeline(fixed_plan)
        llm_pipeline = llm_candidate_plan_to_pipeline(llm_plan)

        self.assertIsInstance(fixed_pipeline, PipelineSpec)
        self.assertIsInstance(llm_pipeline, PipelineSpec)
        self.assertEqual(fixed_pipeline.pipeline_id, llm_pipeline.pipeline_id)
        self.assertEqual(fixed_pipeline.module_ids, llm_pipeline.module_ids)
        self.assertEqual(fixed_pipeline.source.value, "fixed")
        self.assertEqual(llm_pipeline.source.value, "llm")

    def test_all_default_fixed_candidates_convert_to_pipeline_spec(self) -> None:
        """三个已有固定候选都必须能进入统一 Schema。"""

        candidate_set = build_default_candidate_set("phm2010", "C1")

        pipelines = [candidate_plan_to_pipeline(plan) for plan in candidate_set.plans]

        self.assertEqual(len(pipelines), 3)
        self.assertTrue(all(isinstance(item, PipelineSpec) for item in pipelines))
        self.assertEqual(len({item.pipeline_id for item in pipelines}), 3)


if __name__ == "__main__":
    unittest.main()
