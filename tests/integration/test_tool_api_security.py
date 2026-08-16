"""Tool API 鉴权与 AgentTeams Skill 调用审计测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from toolwear_agent.backend.main import create_app
from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import DatasetRef, ExperimentState, LabelPolicy, SplitSpec


class ToolApiSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            ai_infra_root=Path(self.temp_dir.name),
            tool_api_token="local-tool-token",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _state() -> ExperimentState:
        return ExperimentState(
            experiment_id="exp-agentteams-audit",
            trace_id="trace-agentteams-audit",
            title="AgentTeams 审计测试",
            objective="验证 Skill 调用可鉴权并关联实验 Trace。",
            dataset_ref=DatasetRef(dataset_id="phm2010", cutter_ids=("C1",)),
            label_policy=LabelPolicy(),
            split_spec=SplitSpec(),
        )

    def test_configured_token_protects_api_but_keeps_health_public(self) -> None:
        with TestClient(create_app(settings=self.settings)) as client:
            health = client.get("/api/v1/health")
            missing = client.get("/api/v1/experiments")
            wrong = client.get(
                "/api/v1/experiments",
                headers={"Authorization": "Bearer wrong-token"},
            )
            accepted = client.get(
                "/api/v1/experiments",
                headers={"Authorization": "Bearer local-tool-token"},
            )

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["components"]["api"]["auth_required"])
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()["error"]["error_code"], "AUTHENTICATION_REQUIRED")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(accepted.status_code, 200)

    def test_valid_skill_headers_create_redacted_audit_event(self) -> None:
        headers = {
            "Authorization": "Bearer local-tool-token",
            "X-ToolWear-AgentTeams-Skill": "toolwear-data-profile",
            "X-ToolWear-AgentTeams-Agent": "DataStewardAgent",
            "X-ToolWear-Correlation-Id": "matrix-event-audit-001",
        }
        with TestClient(create_app(settings=self.settings)) as client:
            client.app.state.container.repository.create_experiment(
                self._state(),
                actor="human",
                reason="创建 AgentTeams 审计测试实验。",
            )
            response = client.get(
                "/api/v1/experiments/exp-agentteams-audit",
                headers=headers,
            )

        audit_file = self.settings.log_root / "agentteams" / "skill_invocations.jsonl"
        event = json.loads(audit_file.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(event["skill_name"], "toolwear-data-profile")
        self.assertEqual(event["agent_name"], "DataStewardAgent")
        self.assertEqual(event["experiment_id"], "exp-agentteams-audit")
        self.assertEqual(event["trace_id"], "trace-agentteams-audit")
        self.assertEqual(event["correlation_id"], "matrix-event-audit-001")
        self.assertNotIn("local-tool-token", audit_file.read_text(encoding="utf-8"))

    def test_skill_owner_mismatch_is_rejected_before_business_route(self) -> None:
        with TestClient(create_app(settings=self.settings)) as client:
            response = client.get(
                "/api/v1/experiments",
                headers={
                    "Authorization": "Bearer local-tool-token",
                    "X-ToolWear-AgentTeams-Skill": "toolwear-data-profile",
                    "X-ToolWear-AgentTeams-Agent": "AlgorithmArchitectAgent",
                    "X-ToolWear-Correlation-Id": "matrix-event-invalid-owner",
                },
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["error_code"], "SKILL_PERMISSION_DENIED")


if __name__ == "__main__":
    unittest.main()
