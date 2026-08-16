"""AgentTeams Worker 可执行 Skill 客户端的安全契约测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.agentteams.worker_skill_client import (
    SKILL_ROUTES,
    SkillClientError,
    build_http_request,
    read_api_token,
)


class WorkerSkillClientTests(unittest.TestCase):
    def test_read_operation_uses_fixed_route_and_audit_headers(self) -> None:
        request = build_http_request(
            skill_name="toolwear-data-profile",
            invocation={
                "operation": "inspect",
                "experiment_id": "exp-agentteams-smoke",
                "correlation_id": "matrix-event-001",
            },
            base_url="http://host.docker.internal:18100",
            token="local-tool-token",
        )

        self.assertEqual(request.method, "GET")
        self.assertEqual(
            request.url,
            "http://host.docker.internal:18100/api/v1/experiments/exp-agentteams-smoke",
        )
        self.assertIsNone(request.body)
        self.assertEqual(request.headers["Authorization"], "Bearer local-tool-token")
        self.assertEqual(
            request.headers["X-ToolWear-AgentTeams-Agent"],
            "DataStewardAgent",
        )
        self.assertEqual(
            request.headers["X-ToolWear-AgentTeams-Skill"],
            "toolwear-data-profile",
        )

    def test_write_operation_requires_confirmation_and_idempotency_key(self) -> None:
        invocation = {
            "operation": "execute",
            "experiment_id": "exp-agentteams-smoke",
            "correlation_id": "matrix-event-002",
            "payload": {"rationale": "由 AgentTeams 数据治理 Worker 请求。"},
        }

        with self.assertRaisesRegex(SkillClientError, "confirm_write"):
            build_http_request(
                skill_name="toolwear-data-profile",
                invocation=invocation,
                base_url="http://host.docker.internal:18100",
                token="",
            )

        invocation["confirm_write"] = True
        with self.assertRaisesRegex(SkillClientError, "idempotency_key"):
            build_http_request(
                skill_name="toolwear-data-profile",
                invocation=invocation,
                base_url="http://host.docker.internal:18100",
                token="",
            )

    def test_all_p0_skills_have_read_and_execute_routes(self) -> None:
        self.assertEqual(len(SKILL_ROUTES), 10)
        for skill_name, routes in SKILL_ROUTES.items():
            self.assertIn("inspect", routes, skill_name)
            self.assertIn("execute", routes, skill_name)

    def test_api_token_is_read_from_secret_file_without_returning_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            token_file = Path(temp_dir) / "tool_api_token"
            token_file.write_text("  local-secret-token\n", encoding="utf-8")

            token = read_api_token(env={}, default_file=token_file)

        self.assertEqual(token, "local-secret-token")


if __name__ == "__main__":
    unittest.main()
