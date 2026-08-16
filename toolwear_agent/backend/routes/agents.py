"""六 Agent Identity、单次调用和历史恢复路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header

from toolwear_agent.agents.catalog import list_agent_definitions
from toolwear_agent.agents.runtime import AgentRuntimeService
from toolwear_agent.backend.dependencies import get_agent_runtime
from toolwear_agent.schemas.agent_runtime import (
    AgentDefinitionView,
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentRunRecord,
)


router = APIRouter(prefix="/api/v1", tags=["agents"])
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]


@router.get("/agents", response_model=list[AgentDefinitionView])
def list_agents() -> tuple[AgentDefinitionView, ...]:
    """返回固定六 Agent 的公开 Identity 与权限，不包含完整 system prompt。"""

    return tuple(item.public_view() for item in list_agent_definitions())


@router.post(
    "/experiments/{experiment_id}/agents/{agent_name}/invoke",
    response_model=AgentInvocationResponse,
)
def invoke_agent(
    experiment_id: str,
    agent_name: str,
    body: AgentInvocationRequest,
    idempotency_key: IdempotencyKey = None,
    runtime: AgentRuntimeService = Depends(get_agent_runtime),
) -> AgentInvocationResponse:
    """以 human 身份调用角色；AgentTeams 后续由内部适配层传递身份。"""

    return runtime.invoke(
        experiment_id,
        agent_name,
        body,
        idempotency_key=idempotency_key,
        requested_by="human",
    )


@router.get(
    "/experiments/{experiment_id}/agent-runs",
    response_model=list[AgentRunRecord],
)
def list_agent_runs(
    experiment_id: str,
    runtime: AgentRuntimeService = Depends(get_agent_runtime),
) -> tuple[AgentRunRecord, ...]:
    """按创建顺序恢复当前实验的 AgentTask 与 AgentResult。"""

    return runtime.list_runs(experiment_id)
