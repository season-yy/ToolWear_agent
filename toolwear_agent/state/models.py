"""SQLite 状态层专用的事件、运行记录和领域错误。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, JsonValue, model_validator

from toolwear_agent.core.errors import ToolWearError
from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, utc_now
from toolwear_agent.schemas.experiment import ExperimentStatus


class StateRepositoryError(ToolWearError):
    """状态仓库可预期错误的基类。"""

    error_code = "STATE_REPOSITORY_ERROR"


class EntityNotFoundError(StateRepositoryError):
    """请求的实验、修订、审批或运行不存在。"""

    error_code = "STATE_ENTITY_NOT_FOUND"


class StateConflictError(StateRepositoryError):
    """写入与当前持久化状态冲突。"""

    error_code = "STATE_CONFLICT"


class InvalidStateTransitionError(StateRepositoryError):
    """状态转换不在 master spec 允许的边集合中。"""

    error_code = "INVALID_STATE_TRANSITION"


class IdempotencyConflictError(StateRepositoryError):
    """同一个幂等键被用于不同写请求。"""

    error_code = "IDEMPOTENCY_KEY_CONFLICT"


class RevisionLockedError(StateRepositoryError):
    """训练或评估期间禁止切换当前修订。"""

    error_code = "REVISION_LOCKED"


class StateTransitionEvent(SchemaModel):
    """每次状态变化对应的一条不可覆盖审计记录。"""

    event_id: EntityId
    sequence: int = Field(ge=1)
    experiment_id: EntityId
    revision: int = Field(ge=1)
    before_state: ExperimentStatus | None = None
    after_state: ExperimentStatus
    actor: NonEmptyText
    reason: NonEmptyText
    trace_id: EntityId
    evidence_ids: tuple[EntityId, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


class RunStatus(str, Enum):
    """一次训练或评估运行的持久化生命周期。"""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunRecord(SchemaModel):
    """数据库只保存运行摘要；大模型、图表和日志仍保存为文件。"""

    run_id: EntityId
    experiment_id: EntityId
    revision: int = Field(ge=1)
    pipeline_id: EntityId
    run_kind: EntityId
    status: RunStatus = RunStatus.QUEUED
    progress: float = Field(default=0.0, ge=0, le=1)
    progress_message: str = "等待训练 Worker。"
    current_epoch: int = Field(default=0, ge=0)
    total_epochs: int = Field(default=0, ge=0)
    cancel_requested: bool = False
    budget_accounted: bool = False
    consumed_epochs: int = Field(default=0, ge=0)
    result_summary: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_uri: str | None = None
    error_code: EntityId | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _timestamps_match_status(self) -> "RunRecord":
        if self.status is RunStatus.RUNNING and self.started_at is None:
            raise ValueError("running 运行必须包含 started_at。")
        if self.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
            if self.completed_at is None:
                raise ValueError("已结束运行必须包含 completed_at。")
        if self.status is RunStatus.FAILED and not self.error_code:
            raise ValueError("failed 运行必须包含 error_code。")
        return self
