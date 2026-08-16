"""状态驱动 Streamlit 使用的 FastAPI 客户端测试。"""

from __future__ import annotations

import json
import unittest

import httpx

from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient


class FrontendApiClientTest(unittest.TestCase):
    """验证页面参数、幂等键和稳定错误不会在客户端丢失。"""

    def test_create_experiment_sends_json_and_idempotency_key(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                201,
                json={"experiment_id": "exp-ui-1", "state": "DRAFT"},
            )

        client = ToolWearApiClient(
            "http://127.0.0.1:18100",
            transport=httpx.MockTransport(handler),
        )
        result = client.create_experiment(
            {"title": "界面实验", "window_length": 4096},
            idempotency_key="ui-create-1",
        )

        self.assertEqual(result["experiment_id"], "exp-ui-1")
        self.assertEqual(captured["path"], "/api/v1/experiments")
        self.assertEqual(captured["headers"]["idempotency-key"], "ui-create-1")
        self.assertEqual(captured["body"]["window_length"], 4096)

    def test_recommendation_and_training_use_experiment_scoped_paths(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json={"ok": True})

        client = ToolWearApiClient(
            "http://127.0.0.1:18100",
            transport=httpx.MockTransport(handler),
        )
        client.generate_recommendations(
            "exp-ui-2",
            user_request="生成候选",
            force_refresh=False,
            idempotency_key="recommend-1",
        )
        client.start_mini_run("exp-ui-2", idempotency_key="run-1")

        self.assertEqual(
            paths,
            [
                "/api/v1/experiments/exp-ui-2/recommendations",
                "/api/v1/experiments/exp-ui-2/runs/mini",
            ],
        )

    def test_evaluation_retry_is_explicit_and_experiment_scoped(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"operation": "evaluate"})

        client = ToolWearApiClient(
            "http://127.0.0.1:18100",
            transport=httpx.MockTransport(handler),
        )
        client.evaluate(
            "exp-ui-diagnosis",
            rationale="只重试 LLM。",
            force_refresh=True,
            idempotency_key="retry-diagnosis-1",
        )

        self.assertEqual(
            captured["path"],
            "/api/v1/experiments/exp-ui-diagnosis/evaluate",
        )
        self.assertEqual(captured["body"]["force_refresh"], True)

    def test_agent_identity_and_history_use_stable_paths(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json=[])

        client = ToolWearApiClient(
            "http://127.0.0.1:18100",
            transport=httpx.MockTransport(handler),
        )
        client.agent_definitions()
        client.agent_runs("exp-agent-ui")

        self.assertEqual(
            paths,
            [
                "/api/v1/agents",
                "/api/v1/experiments/exp-agent-ui/agent-runs",
            ],
        )

    def test_artifact_bytes_uses_bearer_auth_and_returns_binary_content(self) -> None:
        """前端显示受控图表时必须通过 API 鉴权读取，不能直接访问本机路径。"""

        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.raw_path.decode("ascii")
            captured["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, content=b"fake-png", headers={"content-type": "image/png"})

        client = ToolWearApiClient(
            "http://127.0.0.1:18100",
            token="local-test-token",
            transport=httpx.MockTransport(handler),
        )

        content = client.artifact_bytes("evidence/tsne 1")

        self.assertEqual(content, b"fake-png")
        self.assertEqual(captured["path"], "/api/v1/artifacts/evidence%2Ftsne%201/content")
        self.assertEqual(captured["authorization"], "Bearer local-test-token")

    def test_stable_api_error_is_exposed_to_page(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(
                409,
                json={
                    "error": {
                        "error_code": "INVALID_WORKFLOW_STATE",
                        "message": "当前状态不能训练。",
                        "trace_id": "trace-ui-1",
                        "context": {},
                    }
                },
            )

        client = ToolWearApiClient(
            "http://127.0.0.1:18100",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ToolApiError) as raised:
            client.start_mini_run("exp-ui-3", idempotency_key="run-invalid")

        self.assertEqual(raised.exception.error_code, "INVALID_WORKFLOW_STATE")
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.trace_id, "trace-ui-1")
        self.assertIn("当前状态不能训练", str(raised.exception))

    def test_network_error_has_recoverable_client_code(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = ToolWearApiClient(
            "http://127.0.0.1:18100",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ToolApiError) as raised:
            client.health()

        self.assertEqual(raised.exception.error_code, "API_UNREACHABLE")
        self.assertIsNone(raised.exception.status_code)

    def test_real_local_request_does_not_depend_on_system_proxy(self) -> None:
        """客户端构造必须显式绕过环境代理，避免本机 API 被转发成 502。"""

        client = ToolWearApiClient("http://127.0.0.1:9", timeout_seconds=0.05)

        with self.assertRaises(ToolApiError) as raised:
            client.health()

        self.assertEqual(raised.exception.error_code, "API_UNREACHABLE")


if __name__ == "__main__":
    unittest.main()
