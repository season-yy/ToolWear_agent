"""ExperimentState、不可变修订、审批和决策 Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from toolwear_agent.schemas.agent import AgentName
from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex, utc_now
from toolwear_agent.schemas.dataset import DatasetRef, LabelPolicy, SplitSpec
from toolwear_agent.schemas.pipeline import PipelineSpec, RunConfig


class ExperimentStatus(str, Enum):
    """master spec 规定的完整实验状态机。"""

    DRAFT = "DRAFT"
    DATA_VALIDATING = "DATA_VALIDATING"
    BLOCKED_DATA = "BLOCKED_DATA"
    WAITING_PLAN_SELECTION = "WAITING_PLAN_SELECTION"
    PIPELINE_VALIDATING = "PIPELINE_VALIDATING"
    CODE_PREPARING = "CODE_PREPARING"
    MINI_TRAINING = "MINI_TRAINING"
    EVALUATING = "EVALUATING"
    DECIDING = "DECIDING"
    WAITING_FULL_APPROVAL = "WAITING_FULL_APPROVAL"
    FULL_TRAINING = "FULL_TRAINING"
    EVALUATING_FULL = "EVALUATING_FULL"
    WAITING_USER_REVIEW = "WAITING_USER_REVIEW"
    COMPLETED_MINI = "COMPLETED_MINI"
    COMPLETED_FULL = "COMPLETED_FULL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExperimentBudget(SchemaModel):
    """限制自动迭代次数和训练成本，避免实验无限循环。"""

    max_mini_runs: int = Field(default=3, ge=1, le=20)
    completed_mini_runs: int = Field(default=0, ge=0)
    max_full_runs: int = Field(default=1, ge=1, le=5)
    completed_full_runs: int = Field(default=0, ge=0)
    max_total_epochs: int = Field(default=100, ge=1, le=10_000)
    consumed_epochs: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _usage_does_not_exceed_budget(self) -> "ExperimentBudget":
        if self.completed_mini_runs > self.max_mini_runs:
            raise ValueError("小样本运行次数不能超过预算上限。")
        if self.completed_full_runs > self.max_full_runs:
            raise ValueError("完整训练次数不能超过预算上限。")
        if self.consumed_epochs > self.max_total_epochs:
            raise ValueError("已消耗 epoch 不能超过预算上限。")
        return self


class ExperimentPreferences(SchemaModel):
    """用户在创建实验时给出的真实数据和训练偏好。"""

    input_channels: tuple[EntityId, ...] = ()
    window_length: int = Field(default=4096, ge=256, le=131_072)
    overlap: float = Field(default=0.5, ge=0, lt=0.95)
    sample_fraction: float = Field(default=0.2, gt=0, le=1)
    max_windows_per_cut: int = Field(default=32, ge=1, le=1024)
    mode: Literal["quick", "balanced"] = "quick"

    @field_validator("input_channels")
    @classmethod
    def _channels_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("input_channels 不能重复。")
        return value


class ExperimentState(SchemaModel):
    """SQLite 中当前实验状态的类型化快照。"""

    experiment_id: EntityId
    trace_id: EntityId = Field(default_factory=lambda: f"trace-{uuid4().hex}")
    title: NonEmptyText
    objective: NonEmptyText = "完成刀具磨损四阶段分类实验。"
    dataset_ref: DatasetRef
    label_policy: LabelPolicy
    split_spec: SplitSpec
    preferences: ExperimentPreferences = Field(default_factory=ExperimentPreferences)
    state: ExperimentStatus = ExperimentStatus.DRAFT
    revision: int = Field(default=1, ge=1)
    selected_pipeline_ref: EntityId | None = None
    latest_recommendation_id: EntityId | None = None
    current_agent: AgentName | None = None
    pending_approval: EntityId | None = None
    budget: ExperimentBudget = Field(default_factory=ExperimentBudget)
    best_run_id: EntityId | None = None
    last_event_sequence: int = Field(default=0, ge=0)
    error_code: EntityId | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def status(self) -> ExperimentStatus:
        """兼容迁移期旧调用；新代码统一使用 `state`。"""

        return self.state

    @property
    def current_revision(self) -> int:
        """兼容迁移期旧调用；新代码统一使用 `revision`。"""

        return self.revision

    @property
    def selected_pipeline_id(self) -> EntityId | None:
        """兼容迁移期旧调用；新代码统一使用 `selected_pipeline_ref`。"""

        return self.selected_pipeline_ref

    @property
    def active_run_id(self) -> EntityId | None:
        """旧字段不再作为状态真相，保留只读映射到 best run。"""

        return self.best_run_id


class ExperimentRevision(SchemaModel):
    """用户或 Agent 确认后形成的不可变实验修订。"""

    experiment_id: EntityId
    revision: int = Field(ge=1)
    pipeline: PipelineSpec
    run_config: RunConfig
    created_by: NonEmptyText
    change_reason: NonEmptyText
    parent_revision: int | None = Field(default=None, ge=1)
    content_hash: Sha256Hex | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _nested_ids_match(self) -> "ExperimentRevision":
        if self.run_config.experiment_id != self.experiment_id:
            raise ValueError("RunConfig.experiment_id 与 Revision 不一致。")
        if self.run_config.revision != self.revision:
            raise ValueError("RunConfig.revision 与 Revision 不一致。")
        if self.run_config.pipeline_id != self.pipeline.pipeline_id:
            raise ValueError("RunConfig.pipeline_id 与 PipelineSpec 不一致。")
        if self.parent_revision is not None and self.parent_revision >= self.revision:
            raise ValueError("parent_revision 必须小于当前 revision。")
        return self


class ApprovalAction(str, Enum):
    """需要人工确认的高影响动作。"""

    APPROVE_PIPELINE = "approve_pipeline"
    APPROVE_MINI_TRAIN = "approve_mini_train"
    APPROVE_FULL_TRAIN = "approve_full_train"
    APPROVE_FINAL_EVALUATION = "approve_final_evaluation"
    APPROVE_ROLLBACK = "approve_rollback"


class ApprovalStatus(str, Enum):
    """审批生命周期。"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalRecord(SchemaModel):
    """将“Agent 建议”与“真实执行”分开的人工审批记录。"""

    approval_id: EntityId
    experiment_id: EntityId
    revision: int = Field(ge=1)
    action: ApprovalAction
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_by: NonEmptyText
    decided_by: str | None = None
    rationale: str = ""
    request_hash: Sha256Hex | None = None
    requested_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def _decision_fields_match_status(self) -> "ApprovalRecord":
        is_pending = self.status is ApprovalStatus.PENDING
        if is_pending and (self.decided_by is not None or self.decided_at is not None):
            raise ValueError("pending 审批不能包含决定人或决定时间。")
        if not is_pending and (not self.decided_by or self.decided_at is None):
            raise ValueError("已结束审批必须包含 decided_by 和 decided_at。")
        return self


class DecisionAction(str, Enum):
    """EvaluationGovernor 可提出的下一步动作。"""

    ADJUST_PARAMETERS = "adjust_parameters"
    CHANGE_PIPELINE = "change_pipeline"
    STOP = "stop"
    APPROVE_FULL_TRAIN = "approve_full_train"
    FINALIZE = "finalize"


class DecisionRecord(SchemaModel):
    """只基于 validation 形成的调参、换方案或停止建议。"""

    decision_id: EntityId
    experiment_id: EntityId
    run_id: EntityId
    action: DecisionAction
    basis_split: Literal["validation"] = "validation"
    reason: NonEmptyText
    decided_by: NonEmptyText
    recommended_changes: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
