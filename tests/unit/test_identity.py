from __future__ import annotations

import unittest

from toolwear_agent.agentteams.identity import (
    build_core_agent_identities,
    build_p0_skill_manifest,
    render_agent_identity_markdown,
    render_skill_manifest_markdown,
)


class IdentityTests(unittest.TestCase):
    def test_core_agent_identities_are_fixed_to_six_agents(self) -> None:
        """初赛口径固定为六个核心 Agent。"""

        identities = build_core_agent_identities()

        self.assertEqual(len(identities), 6)
        self.assertEqual(
            [identity.agent_name for identity in identities],
            [
                "ExperimentManagerAgent",
                "DataStewardAgent",
                "AlgorithmArchitectAgent",
                "CodeTrainingEngineerAgent",
                "EvaluationGovernorAgent",
                "ReportMemoryCuratorAgent",
            ],
        )

    def test_skill_manifest_maps_every_skill_to_core_agent(self) -> None:
        """每个 Skill 都必须归属到六个核心 Agent 之一。"""

        core_agents = {identity.agent_name for identity in build_core_agent_identities()}
        skills = build_p0_skill_manifest()

        self.assertGreaterEqual(len(skills), 10)
        self.assertLessEqual({skill.owner_agent for skill in skills}, core_agents)

    def test_markdown_reports_include_boundaries_and_safety(self) -> None:
        """Identity 和 Skill 报告必须包含边界与安全说明。"""

        identity_markdown = render_agent_identity_markdown(build_core_agent_identities())
        skill_markdown = render_skill_manifest_markdown(build_p0_skill_manifest())

        self.assertIn("边界", identity_markdown)
        self.assertIn("ExperimentManagerAgent", identity_markdown)
        self.assertIn("Skill Manifest", skill_markdown)
        self.assertIn("安全说明", skill_markdown)


if __name__ == "__main__":
    unittest.main()
