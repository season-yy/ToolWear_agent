"""FastAPI Tool API 的请求、响应和稳定错误契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, field_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel
from toolwear_agent.schemas.evaluation import ValidationResult
from toolwear_agent.schemas.evidence import EvidenceRef
from toolwear_agent.schemas.experiment import ExperimentState


class ApiErrorDetail(SchemaModel):
    """前端和 Skill 可以稳定处理的错误详情。"""

    error_code: EntityId
    message: NonEmptyText
    trace_id: EntityId | None = None
    context: dict[str, JsonValue] = Field(default_factory=dict)


class ApiErrorResponse(SchemaModel):
    """所有业务错误共用的外层结构。"""

    error: ApiErrorDetail


class CreateExperimentRequest(SchemaModel):
    """创建实验时一次性保存页面真实参数。"""

    experiment_id: EntityId | None = None
    title: NonEmptyText
    user_request: NonEmptyText
    dataset_id: EntityId
    cutter_ids: tuple[EntityId, ...] = Field(min_length=1)
    input_channels: tuple[EntityId, ...] = ()
    vb_aggregation: str = "max"
    vb_thresholds_um: tuple[float, float, float] = (90.0, 130.0, 160.0)
    enable_vb_regression: bool = False
    specified_flute: int | None = Field(default=None, ge=1, le=3)
    train_ratio: float = Field(default=0.6, gt=0, lt=1)
    validation_ratio: float = Field(default=0.2, gt=0, lt=1)
    test_ratio: float = Field(default=0.2, gt=0, lt=1)
    random_seed: int = Field(default=42, ge=0)
    window_length: int = Field(default=4096, ge=256, le=131_072)
    overlap: float = Field(default=0.5, ge=0, lt=0.95)
    sample_fraction: float = Field(default=0.2, gt=0, le=1)
    max_windows_per_cut: int = Field(default=32, ge=1, le=1024)
    mode: Literal["quick", "balanced"] = "quick"

    @field_validator("cutter_ids", "input_channels")
    @classmethod
    def _values_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("同一列表中的 ID 不能重复。")
        return value


class RecommendationRequest(SchemaModel):
    """请求 LLM 或规则引擎生成兼容候选。"""

    user_request: NonEmptyText
    force_refresh: bool = False


class PipelineApprovalRequest(SchemaModel):
    """用户选择一个候选，并锁定进入训练的参数。"""

    pipeline_id: EntityId
    rationale: NonEmptyText = "用户在实验台确认该方案。"
    input_channels: tuple[EntityId, ...] | None = None
    module_parameters: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    batch_size: int = Field(default=64, ge=1, le=4096)
    epochs: int = Field(default=2, ge=1, le=10_000)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    device: str = "auto"
    num_workers: int = Field(default=0, ge=0, le=32)
    max_samples: int | None = Field(default=None, ge=4)


class ActionRequest(SchemaModel):
    """只需要说明理由的状态动作。"""

    rationale: NonEmptyText = "由用户在实验台触发。"


class EvaluationRequest(SchemaModel):
    """生成评估诊断；可只重试 LLM，不重复训练或覆盖旧证据。"""

    rationale: NonEmptyText = "仅依据 train/validation 事实生成诊断。"
    force_refresh: bool = False


class RunStartRequest(SchemaModel):
    """启动小样本训练；未覆盖字段沿用已审批 revision。"""

    run_id: EntityId | None = None
    max_samples: int | None = Field(default=None, ge=4)
    rationale: NonEmptyText = "用户批准启动小样本训练。"


class DecisionRequest(SchemaModel):
    """用户可采用规则建议，也可明确停止或更换模块。"""

    action: Literal["auto", "stop", "change_pipeline", "approve_full"] = "auto"
    rationale: NonEmptyText = "根据 validation 评估决定下一步。"


class ExperimentActionResponse(SchemaModel):
    """数据准备、评估、决策和报告共用的动作响应。"""

    operation: EntityId
    summary: NonEmptyText
    state: ExperimentState
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()


class PipelineValidationResponse(SchemaModel):
    """Pipeline Registry 校验和转换后的实验状态。"""

    state: ExperimentState
    validation: ValidationResult
