from __future__ import annotations

import unittest
from pathlib import Path

from toolwear_agent.agentteams.trace import build_p0_trace, render_trace_markdown


class TraceTests(unittest.TestCase):
    def test_build_p0_trace_contains_ten_ordered_steps(self) -> None:
        """P0 Trace 至少要覆盖初赛闭环的 10 个步骤。"""

        trace = build_p0_trace(
            ai_infra_root=Path("D:/AI_infra"),
            run_dir=Path("D:/AI_infra/experiments/runs/run_001"),
            run_id="run_001",
        )

        self.assertEqual(len(trace.steps), 10)
        self.assertEqual([step.step_no for step in trace.steps], list(range(1, 11)))
        self.assertEqual(trace.steps[0].agent_name, "DataStewardAgent")
        self.assertEqual(trace.steps[-1].agent_name, "ReportMemoryCuratorAgent")

    def test_build_p0_trace_uses_six_core_agents_only(self) -> None:
        """Trace 可以有 10 个流程步骤，但只能归属到 6 个核心 Agent。"""

        trace = build_p0_trace(
            ai_infra_root=Path("D:/AI_infra"),
            run_dir=Path("D:/AI_infra/experiments/runs/run_001"),
            run_id="run_001",
        )

        step_agent_names = {step.agent_name for step in trace.steps}
        self.assertEqual(len(trace.core_agents), 6)
        self.assertLessEqual(step_agent_names, set(trace.core_agents))
        self.assertEqual(
            step_agent_names,
            {
                "ExperimentManagerAgent",
                "DataStewardAgent",
                "AlgorithmArchitectAgent",
                "CodeTrainingEngineerAgent",
                "EvaluationGovernorAgent",
                "ReportMemoryCuratorAgent",
            },
        )

    def test_each_trace_step_has_agent_identity_and_evidence(self) -> None:
        """每个步骤都要说清 Agent 身份、输入、输出和状态。"""

        trace = build_p0_trace(
            ai_infra_root=Path("D:/AI_infra"),
            run_dir=Path("D:/AI_infra/experiments/runs/run_001"),
            run_id="run_001",
        )

        for step in trace.steps:
            self.assertTrue(step.agent_name)
            self.assertTrue(step.agent_role)
            self.assertTrue(step.status)
            self.assertTrue(step.output_files)

    def test_render_trace_markdown_explains_agentteams_mapping(self) -> None:
        """Markdown 需要面向比赛展示 AgentTeams 协作映射。"""

        trace = build_p0_trace(
            ai_infra_root=Path("D:/AI_infra"),
            run_dir=Path("D:/AI_infra/experiments/runs/run_001"),
            run_id="run_001",
        )
        markdown = render_trace_markdown(trace)

        self.assertIn("AgentTeams 协作记录", markdown)
        self.assertIn("角色编排", markdown)
        self.assertIn("任务拆解", markdown)
        self.assertIn("状态追踪", markdown)
        self.assertIn("六个核心 Agent", markdown)
        self.assertIn("ReportMemoryCuratorAgent", markdown)


if __name__ == "__main__":
    unittest.main()
