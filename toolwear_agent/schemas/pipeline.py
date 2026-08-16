"""候选方案、模块链和单次运行配置的统一 Schema。"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex


class ModuleKind(str, Enum):
    """Module Registry 中的稳定模块类别。"""

    WINDOWING = "windowing"
    PREPROCESS = "preprocess"
    FEATURE = "feature"
    FUSION = "fusion"
    MODEL = "model"
    LOSS = "loss"
    TRAINER = "trainer"


class ModuleSpec(SchemaModel):
    """Pipeline 中一个已注册模块及其参数。"""

    module_id: EntityId
    kind: ModuleKind
    order: int = Field(ge=0)
    enabled: bool = True
    module_version: str = "1"
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ExperimentalExtensionSpec(SchemaModel):
    """尚未进入本地 Registry 的研究能力提案。"""

    extension_id: EntityId
    display_name: NonEmptyText
    kind: ModuleKind
    rationale: NonEmptyText
    proposed_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    implemented: Literal[False] = False


class PipelineSource(str, Enum):
    """候选方案来源，用于证据和页面展示。"""

    FIXED = "fixed"
    LLM = "llm"
    USER = "user"
    MEMORY = "memory"


class ExpectedCost(str, Enum):
    """用于粗粒度预算提示的成本级别。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PipelineSpec(SchemaModel):
    """固定候选、LLM 候选和用户编辑共享的唯一执行方案。"""

    pipeline_id: EntityId
    display_name: NonEmptyText
    task_type: str = "four_stage_classification"
    source: PipelineSource
    input_channels: tuple[EntityId, ...] = Field(min_length=1)
    modules: tuple[ModuleSpec, ...] = Field(min_length=1)
    rationale: NonEmptyText
    risks: tuple[NonEmptyText, ...] = Field(min_length=1)
    expected_cost: ExpectedCost
    trainable: bool = True
    recommended_rank: int | None = Field(default=None, ge=1)
    compatibility_tags: tuple[EntityId, ...] = ()
    experimental_extensions: tuple[ExperimentalExtensionSpec, ...] = ()

    @field_validator("input_channels", "compatibility_tags")
    @classmethod
    def _values_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("列表项不能重复。")
        return value

    @model_validator(mode="after")
    def _validate_module_chain(self) -> "PipelineSpec":
        enabled = [module for module in self.modules if module.enabled]
        module_ids = [module.module_id for module in enabled]
        orders = [module.order for module in enabled]
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("启用的 module_id 不能重复。")
        if len(orders) != len(set(orders)) or orders != sorted(orders):
            raise ValueError("启用模块的 order 必须唯一且按升序提供。")

        kind_counts = {kind: sum(module.kind is kind for module in enabled) for kind in ModuleKind}
        if kind_counts[ModuleKind.WINDOWING] != 1:
            raise ValueError("Pipeline 必须且只能有一个 windowing 模块。")
        if kind_counts[ModuleKind.MODEL] != 1:
            raise ValueError("Pipeline 必须且只能有一个 model 模块。")
        if kind_counts[ModuleKind.TRAINER] != 1:
            raise ValueError("Pipeline 必须且只能有一个 trainer 模块。")
        extension_ids = [extension.extension_id for extension in self.experimental_extensions]
        if len(extension_ids) != len(set(extension_ids)):
            raise ValueError("experimental_extensions 的 extension_id 不能重复。")
        if self.trainable and self.experimental_extensions:
            raise ValueError("包含 experimental_extensions 的 Pipeline 不能标记为可训练。")
        return self

    @property
    def module_ids(self) -> tuple[str, ...]:
        """按执行顺序返回启用模块 ID，供页面和 Registry 使用。"""

        return tuple(module.module_id for module in self.modules if module.enabled)


class RunKind(str, Enum):
    """Run 的成本与测试集访问级别。"""

    SMOKE = "smoke"
    MINI_TRAIN = "mini_train"
    FULL_TRAIN = "full_train"
    FINAL_EVALUATION = "final_evaluation"


class RunConfig(SchemaModel):
    """一次可复现训练或最终评估的参数快照。"""

    run_id: EntityId
    experiment_id: EntityId
    revision: int = Field(ge=1)
    pipeline_id: EntityId
    run_kind: RunKind = RunKind.MINI_TRAIN
    split_hash: Sha256Hex | None = None
    sample_fraction: float = Field(default=0.2, gt=0, le=1)
    max_samples: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=64, ge=1, le=4096)
    epochs: int = Field(default=5, ge=1, le=10_000)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    device: str = "auto"
    random_seed: int = Field(default=42, ge=0)
    num_workers: int = Field(default=0, ge=0, le=64)
    evaluate_test: bool = False

    @field_validator("device")
    @classmethod
    def _device_is_supported(cls, value: str) -> str:
        """只接受稳定设备别名或带非负编号的 CUDA 设备。"""

        normalized = value.strip().lower()
        if normalized in {"auto", "cpu", "cuda"}:
            return normalized
        if normalized.startswith("cuda:") and normalized[5:].isdigit():
            return normalized
        raise ValueError("device 只能是 auto、cpu、cuda 或 cuda:N。")

    @model_validator(mode="after")
    def _isolate_test_evaluation(self) -> "RunConfig":
        if self.evaluate_test and self.run_kind is not RunKind.FINAL_EVALUATION:
            raise ValueError("只有 final_evaluation Run 可以读取 test 指标。")
        if self.run_kind is RunKind.FINAL_EVALUATION and not self.evaluate_test:
            raise ValueError("final_evaluation 必须显式设置 evaluate_test=true。")
        return self
