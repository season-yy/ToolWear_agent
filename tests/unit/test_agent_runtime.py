"""六 Agent 统一运行时的权限、结构化输出和失败证据测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.agents.catalog import CORE_AGENT_NAMES, get_agent_definition
from toolwear_agent.agents.runtime import AgentPermissionError, AgentRuntimeService
from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import DatasetRef, ExperimentState, LabelPolicy, SplitSpec
from toolwear_agent.schemas.agent_runtime import (
    AgentInvocationRequest,
    ArchitectCandidate,
    EvaluationGovernorOutput,
)
from toolwear_agent.services.llm_chat import ChatCompletion
from toolwear_agent.state import SQLiteExperimentRepository


class _StubChatClient:
    """返回固定内容，并保留收到的提示词供边界断言。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict[str, str]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        json_mode: bool,
    ) -> ChatCompletion:
        self.messages = messages
        return ChatCompletion(
            content=self.content,
            provider="qwen",
            model="qwen-test",
            latency_ms=12,
            prompt_tokens=100,
            completion_tokens=80,
            total_tokens=180,
        )


def _experiment_state() -> ExperimentState:
    return ExperimentState(
        experiment_id="exp-agent-runtime",
        trace_id="trace-agent-runtime",
        title="Agent Runtime 测试",
        objective="验证六个 Agent 的真实结构化调用。",
        dataset_ref=DatasetRef(dataset_id="phm2010", cutter_ids=("C1",)),
        label_policy=LabelPolicy(),
        split_spec=SplitSpec(),
    )


def _data_steward_request(*, requested_skills: tuple[str, ...] = ()) -> AgentInvocationRequest:
    return AgentInvocationRequest(
        task_type="interpret-data-profile",
        objective="解释 C1 数据体检结果并判断是否存在阻断项。",
        requested_skills=requested_skills,
        evidence_ids=("evidence-profile",),
        input_payload={
            "dataset_manifest": {"dataset_id": "phm2010", "cutter_ids": ["C1"]},
            "profile_summary": {"cut_count": 315, "nan_count": 0},
            "label_policy": {"aggregation": "max", "thresholds_um": [90, 130, 160]},
            "split_summary": {"group_by": "cut_id", "leakage_count": 0},
            "leakage_summary": {"valid": True, "problem_count": 0},
        },
    )


class AgentRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.settings = Settings(
            ai_infra_root=self.root,
            llm_api_key="x",
            llm_model="qwen-test",
        )
        self.repository = SQLiteExperimentRepository(self.settings.state_db_path)
        self.repository.initialize()
        self.repository.create_experiment(
            _experiment_state(),
            actor="human",
            reason="创建 Agent Runtime 测试实验。",
            idempotency_key="create-agent-runtime-test",
        )

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    def test_catalog_is_fixed_to_six_agents_with_distinct_contracts(self) -> None:
        self.assertEqual(
            CORE_AGENT_NAMES,
            (
                "ExperimentManagerAgent",
                "DataStewardAgent",
                "AlgorithmArchitectAgent",
                "CodeTrainingEngineerAgent",
                "EvaluationGovernorAgent",
                "ReportMemoryCuratorAgent",
            ),
        )
        definitions = [get_agent_definition(name) for name in CORE_AGENT_NAMES]
        self.assertEqual(len({item.input_model for item in definitions}), 6)
        self.assertEqual(len({item.output_model for item in definitions}), 6)
        self.assertTrue(all(item.system_prompt for item in definitions))
        self.assertTrue(all(item.allowed_skills for item in definitions))
        self.assertTrue(
            all("ASCII" in item.system_prompt for item in definitions),
            "六个角色都必须明确约束结构化 ID 的字符集。",
        )

    def test_architect_cannot_mark_unregistered_candidate_as_trainable(self) -> None:
        with self.assertRaises(ValueError):
            ArchitectCandidate(
                rank=1,
                pipeline_id="unregistered_pipeline",
                reason="仅用于验证 Registry 边界。",
                risk="该方案没有在 Registry 中注册。",
                expected_cost="low",
                registry_compatible=False,
            )

    def test_evaluation_recommendation_always_requires_human_approval(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationGovernorOutput(
                summary="验证集存在可优化项。",
                next_actions=("调整类别权重",),
                diagnosis_categories=("imbalance",),
                confidence=0.8,
                recommended_action="adjust_parameters",
                evidence_ids=("evidence-metrics",),
                requires_human_approval=False,
            )

    def test_architect_output_outside_request_registry_is_persisted_as_failed(self) -> None:
        response = {
            "summary": "给出两个候选。",
            "next_actions": ["等待用户选择"],
            "candidates": [
                {
                    "rank": 1,
                    "pipeline_id": "statistical_features_random_forest",
                    "reason": "低成本基线。",
                    "risk": "时序表达有限。",
                    "expected_cost": "low",
                    "registry_compatible": True,
                },
                {
                    "rank": 2,
                    "pipeline_id": "invented_transformer",
                    "reason": "模型临时编造的方案。",
                    "risk": "当前代码无法训练。",
                    "expected_cost": "high",
                    "registry_compatible": True,
                },
            ],
            "experimental_extensions": [],
            "requires_human_choice": True,
        }
        runtime = AgentRuntimeService(
            self.settings,
            self.repository,
            chat_client=_StubChatClient(json.dumps(response, ensure_ascii=False)),
        )
        request = AgentInvocationRequest(
            task_type="recommend-pipelines",
            objective="只从本次 Registry 中推荐候选。",
            requested_skills=("PipelineRecommendSkill",),
            input_payload={
                "user_goal": "完成四阶段分类。",
                "data_summary": {"cutter_id": "C1"},
                "resource_summary": {"cuda_available": True},
                "registry_summary": {
                    "available_pipeline_ids": [
                        "statistical_features_random_forest",
                        "statistical_features_extra_trees",
                    ]
                },
                "memory_context": [],
            },
        )

        invocation = runtime.invoke(
            "exp-agent-runtime",
            "AlgorithmArchitectAgent",
            request,
            idempotency_key="invoke-architect-outside-registry",
        )

        self.assertEqual(invocation.result.status.value, "failed")
        self.assertEqual(
            invocation.result.error_code,
            "AGENT_OUTPUT_POLICY_VIOLATION",
        )

    def test_valid_structured_call_persists_task_result_and_trace_evidence(self) -> None:
        response = {
            "summary": "C1 数据完整，当前没有阻断项，但仍需保留 cut 级隔离。",
            "next_actions": ["继续生成候选方案"],
            "data_status": "warning",
            "findings": [
                {
                    "severity": "warning",
                    "title": "仅完成 C1 内部验证",
                    "detail": "当前证据不能代表跨刀具泛化。",
                    "evidence_ids": ["evidence-profile"],
                }
            ],
            "recommended_actions": ["保持 cut_id 分组切分"],
            "blocker": False,
        }
        client = _StubChatClient(json.dumps(response, ensure_ascii=False))
        runtime = AgentRuntimeService(
            self.settings,
            self.repository,
            chat_client=client,
        )

        invocation = runtime.invoke(
            "exp-agent-runtime",
            "DataStewardAgent",
            _data_steward_request(requested_skills=("DataProfileSkill",)),
            idempotency_key="invoke-data-steward",
        )

        self.assertEqual(invocation.result.status.value, "success")
        self.assertEqual(invocation.result.agent_name, "DataStewardAgent")
        self.assertEqual(invocation.result.trace_id, "trace-agent-runtime")
        self.assertEqual(invocation.result.llm_call.status, "success")
        self.assertEqual(invocation.result.output_schema, "DataStewardOutput")
        self.assertEqual(invocation.task.requested_skills, ("DataProfileSkill",))
        self.assertTrue(invocation.result.evidence[0].uri.endswith("agent_call.json"))
        self.assertTrue(Path(invocation.result.evidence[0].uri).is_file())
        self.assertEqual(
            self.repository.get_agent_result(invocation.task.task_id),
            invocation.result,
        )
        self.assertEqual(
            self.repository.list_agent_tasks("exp-agent-runtime"),
            (invocation.task,),
        )
        self.assertIn("不能修改原始数据", client.messages[0]["content"])

    def test_unauthorized_skill_is_rejected_before_task_is_saved(self) -> None:
        runtime = AgentRuntimeService(
            self.settings,
            self.repository,
            chat_client=_StubChatClient("{}"),
        )

        with self.assertRaises(AgentPermissionError):
            runtime.invoke(
                "exp-agent-runtime",
                "DataStewardAgent",
                _data_steward_request(requested_skills=("MiniTrainSkill",)),
                idempotency_key="invoke-unauthorized-skill",
            )

        self.assertEqual(self.repository.list_agent_tasks("exp-agent-runtime"), ())

    def test_invalid_llm_output_is_persisted_as_failed_result(self) -> None:
        runtime = AgentRuntimeService(
            self.settings,
            self.repository,
            chat_client=_StubChatClient("not-json"),
        )

        invocation = runtime.invoke(
            "exp-agent-runtime",
            "DataStewardAgent",
            _data_steward_request(),
            idempotency_key="invoke-invalid-output",
        )

        self.assertEqual(invocation.result.status.value, "failed")
        self.assertEqual(invocation.result.error_code, "AGENT_OUTPUT_INVALID")
        self.assertEqual(invocation.result.llm_call.status, "failed")
        self.assertTrue(invocation.result.evidence)
        restored = self.repository.get_agent_result(invocation.task.task_id)
        self.assertEqual(restored.error_code, "AGENT_OUTPUT_INVALID")


if __name__ == "__main__":
    unittest.main()
