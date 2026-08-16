"""六个真实 Agent 之间的任务、结果和轻量 Memory Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, TypeAlias

from pydantic import Field, JsonValue, model_validator

from toolwear_agent.schemas.base import (
    EntityId,
    NonEmptyText,
    SchemaModel,
    Sha256Hex,
    utc_now,
)
from toolwear_agent.schemas.evidence import EvidenceRef


AgentName: TypeAlias = Literal[
    "ExperimentManagerAgent",
    "DataStewardAgent",
    "AlgorithmArchitectAgent",
    "CodeTrainingEngineerAgent",
    "EvaluationGovernorAgent",
    "ReportMemoryCuratorAgent",
]

ActorName: TypeAlias = AgentName | Literal["human", "system"]


class AgentTask(SchemaModel):
    """Manager/Leader 分派给一个明确 Agent 的结构化任务。"""

    task_id: EntityId
    experiment_id: EntityId
    revision: int = Field(ge=1)
    assigned_to: AgentName
    requested_by: ActorName
    trace_id: EntityId | None = None
    task_type: EntityId
    objective: NonEmptyText
    input_schema: EntityId = "AgentInput"
    output_schema: EntityId = "AgentOutput"
    input_payload: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_ids: tuple[EntityId, ...] = ()
    allowed_skills: tuple[EntityId, ...] = ()
    requested_skills: tuple[EntityId, ...] = ()
    idempotency_key: EntityId | None = None
    created_at: datetime = Field(default_factory=utc_now)


class AgentResultStatus(str, Enum):
    """Agent 调用的稳定结果状态。"""

    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"


class AgentLlmCallAudit(SchemaModel):
    """一次 Agent LLM 调用的最小审计信息，不保存密钥或完整 Prompt。"""

    runtime: Literal["toolwear-local", "agentteams"] = "toolwear-local"
    provider: str = ""
    model: str = ""
    status: Literal["success", "failed"]
    latency_ms: int = Field(ge=0)
    prompt_template_version: EntityId
    prompt_sha256: Sha256Hex
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    error_code: EntityId | None = None
    error_message: str = ""

    @model_validator(mode="after")
    def _failed_call_has_error_code(self) -> "AgentLlmCallAudit":
        if self.status == "failed" and not self.error_code:
            raise ValueError("failed Agent LLM 调用必须包含 error_code。")
        return self


class AgentResult(SchemaModel):
    """Agent 的结构化输出、证据和失败信息。"""

    task_id: EntityId
    agent_name: AgentName
    trace_id: EntityId | None = None
    status: AgentResultStatus
    summary: NonEmptyText
    output_schema: EntityId = "AgentOutput"
    output_payload: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()
    next_actions: tuple[str, ...] = ()
    llm_call: AgentLlmCallAudit | None = None
    error_code: EntityId | None = None
    error_message: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _failure_has_machine_readable_error(self) -> "AgentResult":
        if self.status is AgentResultStatus.FAILED and not self.error_code:
            raise ValueError("failed AgentResult 必须包含 error_code。")
        return self


class MemoryCase(SchemaModel):
    """SQLite FTS5 可检索的一条实验经验。"""

    memory_id: EntityId
    dataset_id: EntityId
    task_type: EntityId
    problem: NonEmptyText
    intervention: NonEmptyText
    outcome: NonEmptyText
    summary: NonEmptyText
    tags: tuple[EntityId, ...] = ()
    evidence_ids: tuple[EntityId, ...] = ()
    created_by: ActorName = "system"
    created_at: datetime = Field(default_factory=utc_now)
