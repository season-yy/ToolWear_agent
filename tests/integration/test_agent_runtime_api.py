"""六 Agent Runtime 的 FastAPI 契约与 SQLite 恢复测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from toolwear_agent.backend.main import create_app
from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import DatasetRef, ExperimentState, LabelPolicy, SplitSpec
from toolwear_agent.services.llm_chat import ChatCompletion


class _AgentApiChatClient:
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        json_mode: bool,
    ) -> ChatCompletion:
        del messages, temperature, json_mode
        payload = {
            "summary": "数据证据完整，当前没有泄漏阻断。",
            "next_actions": ["交给 AlgorithmArchitectAgent 生成候选"],
            "data_status": "pass",
            "findings": [
                {
                    "severity": "info",
                    "title": "cut 级切分通过",
                    "detail": "泄漏审计问题数为零。",
                    "evidence_ids": ["evidence-profile"],
                }
            ],
            "recommended_actions": ["保持当前 split lock"],
            "blocker": False,
        }
        return ChatCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider="qwen",
            model="qwen-test",
            latency_ms=20,
            total_tokens=120,
        )


class AgentRuntimeApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            ai_infra_root=Path(self.temp_dir.name),
            llm_api_key="x",
            llm_model="qwen-test",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _state() -> ExperimentState:
        return ExperimentState(
            experiment_id="exp-agent-api",
            trace_id="trace-agent-api",
            title="Agent API 测试",
            objective="验证六 Agent API。",
            dataset_ref=DatasetRef(dataset_id="phm2010", cutter_ids=("C1",)),
            label_policy=LabelPolicy(),
            split_spec=SplitSpec(),
        )

    @staticmethod
    def _request_body() -> dict[str, object]:
        return {
            "task_type": "interpret-data-profile",
            "objective": "解释真实数据体检结果。",
            "requested_skills": ["DataProfileSkill"],
            "evidence_ids": ["evidence-profile"],
            "input_payload": {
                "dataset_manifest": {"dataset_id": "phm2010"},
                "profile_summary": {"cut_count": 315},
                "label_policy": {"aggregation": "max"},
                "split_summary": {"group_by": "cut_id"},
                "leakage_summary": {"valid": True, "problem_count": 0},
            },
        }

    def test_identity_invoke_and_history_are_recoverable(self) -> None:
        app = create_app(
            settings=self.settings,
            agent_chat_client=_AgentApiChatClient(),
        )
        with TestClient(app) as client:
            client.app.state.container.repository.create_experiment(
                self._state(),
                actor="human",
                reason="创建 Agent API 测试实验。",
            )
            identities = client.get("/api/v1/agents")
            invoked = client.post(
                "/api/v1/experiments/exp-agent-api/agents/DataStewardAgent/invoke",
                json=self._request_body(),
                headers={"Idempotency-Key": "agent-api-data-steward"},
            )
            replayed = client.post(
                "/api/v1/experiments/exp-agent-api/agents/DataStewardAgent/invoke",
                json=self._request_body(),
                headers={"Idempotency-Key": "agent-api-data-steward"},
            )
            history = client.get("/api/v1/experiments/exp-agent-api/agent-runs")

        self.assertEqual(identities.status_code, 200)
        self.assertEqual(len(identities.json()), 6)
        self.assertEqual(invoked.status_code, 200)
        self.assertEqual(replayed.json(), invoked.json())
        self.assertEqual(invoked.json()["result"]["llm_call"]["status"], "success")
        self.assertEqual(len(history.json()), 1)
        self.assertEqual(history.json()[0]["task"]["assigned_to"], "DataStewardAgent")

    def test_unknown_agent_and_unauthorized_skill_do_not_create_history(self) -> None:
        app = create_app(
            settings=self.settings,
            agent_chat_client=_AgentApiChatClient(),
        )
        with TestClient(app) as client:
            client.app.state.container.repository.create_experiment(
                self._state(),
                actor="human",
                reason="创建 Agent API 测试实验。",
            )
            unknown = client.post(
                "/api/v1/experiments/exp-agent-api/agents/SeventhJudgeAgent/invoke",
                json=self._request_body(),
            )
            body = self._request_body()
            body["requested_skills"] = ["MiniTrainSkill"]
            unauthorized = client.post(
                "/api/v1/experiments/exp-agent-api/agents/DataStewardAgent/invoke",
                json=body,
            )
            history = client.get("/api/v1/experiments/exp-agent-api/agent-runs")

        self.assertEqual(unknown.status_code, 422)
        self.assertIn("固定六 Agent", unknown.json()["error"]["message"])
        self.assertEqual(unauthorized.status_code, 422)
        self.assertEqual(history.json(), [])


if __name__ == "__main__":
    unittest.main()
