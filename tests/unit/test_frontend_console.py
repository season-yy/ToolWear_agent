"""前端应用外壳的系统状态展示测试。"""

from __future__ import annotations

import unittest

from toolwear_agent.frontend.console import _agentteams_summary_lines
from toolwear_agent.frontend.ui_components import status_html


class FrontendConsoleTests(unittest.TestCase):
    def test_verified_agentteams_summary_uses_health_evidence(self) -> None:
        health = {
            "components": {
                "agentteams": {
                    "status": "verified",
                    "team": "ToolWear_agent",
                    "phase": "Active",
                    "leader_ready": True,
                    "ready_workers": 5,
                    "total_workers": 5,
                    "worker_count": 6,
                    "runtime": "copaw",
                    "models": ["qwen3.7-flash-2026-07-15"],
                    "verification_id": "agt-e2e-final",
                }
            }
        }

        lines = _agentteams_summary_lines(health)

        self.assertIn("Team：ToolWear_agent · Active", lines)
        self.assertIn("角色：6 · Leader Ready · Workers 5/5", lines)
        self.assertIn("模型：qwen3.7-flash-2026-07-15", lines)

    def test_pending_agentteams_summary_does_not_claim_success(self) -> None:
        lines = _agentteams_summary_lines(
            {"components": {"agentteams": {"status": "pending_verification"}}}
        )

        self.assertEqual(lines, ("尚无通过校验的 AgentTeams 部署证据。",))

    def test_verified_status_uses_success_style(self) -> None:
        self.assertIn("tw-status ok", status_html("已验证", "verified"))


if __name__ == "__main__":
    unittest.main()
