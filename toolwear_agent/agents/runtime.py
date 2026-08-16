"""不依赖 AgentTeams 的六 Agent 结构化 LLM 运行时。"""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from toolwear_agent.agents.artifacts import persist_agent_result
from toolwear_agent.agents.catalog import get_agent_definition
from toolwear_agent.agents.llm_execution import execute_agent_llm, reject_agent_outcome
from toolwear_agent.agents.output_policy import (
    AgentOutputPolicyError,
    validate_agent_output_policy,
)
from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import AgentTask
from toolwear_agent.schemas.agent import ActorName, AgentResultStatus
from toolwear_agent.schemas.agent_runtime import (
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentOutputBase,
    AgentRunRecord,
)
from toolwear_agent.services.llm_chat import ChatClient, OpenAICompatibleChatClient
from toolwear_agent.state import EntityNotFoundError, SQLiteExperimentRepository


class AgentPermissionError(ValueError):
    """任务请求了当前角色没有权限使用的 Skill。"""


class AgentRuntimeService:
    """校验角色边界，执行一次 LLM，并保存 Task、Result 和 Trace。"""

    def __init__(
        self,
        settings: Settings,
        repository: SQLiteExperimentRepository,
        *,
        chat_client: ChatClient | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.chat_client = chat_client or OpenAICompatibleChatClient(settings)
        self.path_resolver = PathResolver(settings)

    @staticmethod
    def _task_id(
        experiment_id: str,
        agent_name: str,
        request: AgentInvocationRequest,
        idempotency_key: str | None,
    ) -> str:
        if idempotency_key is None:
            return f"agent-task-{uuid4().hex[:24]}"
        payload = json.dumps(request.model_dump(mode="json"), sort_keys=True)
        digest = hashlib.sha256(
            f"{experiment_id}:{agent_name}:{idempotency_key}:{payload}".encode("utf-8")
        ).hexdigest()[:24]
        return f"agent-task-{digest}"

    @staticmethod
    def _result_status(output: AgentOutputBase) -> AgentResultStatus:
        payload = output.model_dump(mode="python")
        needs_human = any(
            payload.get(key) is True
            for key in ("blocker", "requires_human_approval", "requires_human_choice")
        )
        return AgentResultStatus.NEEDS_HUMAN if needs_human else AgentResultStatus.SUCCESS

    def invoke(
        self,
        experiment_id: str,
        agent_name: str,
        request: AgentInvocationRequest,
        *,
        idempotency_key: str | None,
        requested_by: ActorName = "human",
    ) -> AgentInvocationResponse:
        definition = get_agent_definition(agent_name)
        unauthorized = set(request.requested_skills) - set(definition.allowed_skills)
        if unauthorized:
            raise AgentPermissionError(
                f"{definition.agent_name} 无权使用 Skill：{sorted(unauthorized)}"
            )
        validated_input = definition.input_model.model_validate(request.input_payload)
        state = self.repository.get_experiment(experiment_id)
        task_id = self._task_id(experiment_id, agent_name, request, idempotency_key)
        stored_idempotency_key = (
            "agent-key-"
            + hashlib.sha256(idempotency_key.encode()).hexdigest()[:16]
            if idempotency_key
            else None
        )
        task = AgentTask(
            task_id=task_id,
            experiment_id=experiment_id,
            revision=state.revision,
            assigned_to=definition.agent_name,
            requested_by=requested_by,
            trace_id=state.trace_id,
            task_type=request.task_type,
            objective=request.objective,
            input_schema=definition.input_model.__name__,
            output_schema=definition.output_model.__name__,
            input_payload=validated_input.model_dump(mode="json"),
            evidence_ids=request.evidence_ids,
            allowed_skills=definition.allowed_skills,
            requested_skills=request.requested_skills,
            idempotency_key=stored_idempotency_key,
        )
        try:
            task = self.repository.get_agent_task(task_id)
        except EntityNotFoundError:
            task = self.repository.save_agent_task(
                task,
                idempotency_key=(f"{idempotency_key}:task" if idempotency_key else None),
            )
        try:
            existing = self.repository.get_agent_result(task.task_id)
            return AgentInvocationResponse(task=task, result=existing)
        except EntityNotFoundError:
            pass

        outcome = execute_agent_llm(
            self.settings,
            self.chat_client,
            definition,
            task,
        )
        if outcome.output is not None:
            try:
                validate_agent_output_policy(
                    definition,
                    validated_input,
                    outcome.output,
                )
            except AgentOutputPolicyError as exc:
                outcome = reject_agent_outcome(
                    outcome,
                    error_code="AGENT_OUTPUT_POLICY_VIOLATION",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
        if outcome.output is None:
            result = persist_agent_result(
                self.path_resolver,
                self.repository,
                task,
                status=AgentResultStatus.FAILED,
                summary=f"{definition.agent_name} 调用失败，已阻断后续自动推进。",
                output_payload={},
                next_actions=("检查调用证据后重试同一角色任务。",),
                audit=outcome.audit,
                response_content=outcome.response_content,
                error_code=outcome.error_code,
                error_message=outcome.error_message,
                idempotency_key=idempotency_key,
            )
            return AgentInvocationResponse(task=task, result=result)
        output = outcome.output
        result = persist_agent_result(
            self.path_resolver,
            self.repository,
            task,
            status=self._result_status(output),
            summary=output.summary,
            output_payload=output.model_dump(mode="json"),
            next_actions=output.next_actions,
            audit=outcome.audit,
            response_content=outcome.response_content,
            error_code=None,
            error_message="",
            idempotency_key=idempotency_key,
        )
        return AgentInvocationResponse(task=task, result=result)

    def list_runs(self, experiment_id: str) -> tuple[AgentRunRecord, ...]:
        records: list[AgentRunRecord] = []
        for task in self.repository.list_agent_tasks(experiment_id):
            try:
                result = self.repository.get_agent_result(task.task_id)
            except EntityNotFoundError:
                result = None
            records.append(AgentRunRecord(task=task, result=result))
        return tuple(records)


__all__ = ["AgentPermissionError", "AgentRuntimeService"]
