"""Module Registry 与 Trainer Registry 共用的强类型定义。"""

from __future__ import annotations

import math
from enum import Enum
from typing import TypeAlias

from pydantic import Field, model_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex
from toolwear_agent.schemas.pipeline import ModuleKind


ParameterScalar: TypeAlias = str | int | float | bool


class FeatureType(str, Enum):
    """模块链中可被静态检查的数据表示。"""

    RAW_SIGNAL = "raw_signal"
    WINDOWED_SIGNAL = "windowed_signal"
    TABULAR_FEATURES = "tabular_features"
    RAW_1D = "raw_1d"
    LOGITS = "logits"


class TrainerBackend(str, Enum):
    """P0 支持的训练后端。"""

    SKLEARN = "sklearn"
    PYTORCH = "pytorch"


class ResourceClass(str, Enum):
    """面向用户展示的粗粒度资源成本。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ParameterType(str, Enum):
    """Registry 可静态校验的参数类型。"""

    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"


class ParameterRule(SchemaModel):
    """一个模块或训练器参数的白名单和范围约束。"""

    value_type: ParameterType
    description: NonEmptyText
    default: ParameterScalar | None = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[ParameterScalar, ...] = ()

    @model_validator(mode="after")
    def _range_and_choices_are_consistent(self) -> "ParameterRule":
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("参数 minimum 不能大于 maximum。")
        if len(self.choices) != len(set(self.choices)):
            raise ValueError("参数 choices 不能重复。")
        if self.required and self.default is not None:
            raise ValueError("required 参数不应同时声明 default。")
        if self.default is None:
            return self

        value = self.default
        type_valid = {
            ParameterType.BOOLEAN: isinstance(value, bool),
            ParameterType.INTEGER: isinstance(value, int) and not isinstance(value, bool),
            ParameterType.NUMBER: (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ),
            ParameterType.STRING: isinstance(value, str),
        }[self.value_type]
        if not type_valid:
            raise ValueError(f"参数 default 不符合 {self.value_type.value} 类型。")
        if self.choices and value not in self.choices:
            raise ValueError("参数 default 不在 choices 中。")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.minimum is not None and float(value) < self.minimum:
                raise ValueError("参数 default 小于 minimum。")
            if self.maximum is not None and float(value) > self.maximum:
                raise ValueError("参数 default 大于 maximum。")
        return self


class InputPresetDefinition(SchemaModel):
    """页面可直接选择的一组传感器通道。"""

    preset_id: EntityId
    display_name: NonEmptyText
    channel_ids: tuple[EntityId, ...] = Field(min_length=1)
    description: NonEmptyText

    @model_validator(mode="after")
    def _channels_are_unique(self) -> "InputPresetDefinition":
        if len(self.channel_ids) != len(set(self.channel_ids)):
            raise ValueError("Input preset 的 channel_ids 不能重复。")
        return self


class ModuleDefinition(SchemaModel):
    """一个可由 PipelineSpec 引用的模块能力。"""

    module_id: EntityId
    display_name: NonEmptyText
    kind: ModuleKind
    supported_tasks: tuple[EntityId, ...] = ("four_stage_classification",)
    required_feature_type: FeatureType
    output_feature_type: FeatureType
    min_channels: int = Field(default=1, ge=1, le=128)
    max_channels: int = Field(default=7, ge=1, le=128)
    supported_channel_counts: tuple[int, ...] = ()
    parameters_schema: dict[str, ParameterRule] = Field(default_factory=dict)
    trainer_backend: TrainerBackend | None = None
    resource_class: ResourceClass
    implemented: bool
    experimental: bool = False

    @model_validator(mode="after")
    def _definition_is_consistent(self) -> "ModuleDefinition":
        if self.max_channels < self.min_channels:
            raise ValueError("max_channels 不能小于 min_channels。")
        if len(self.supported_channel_counts) != len(set(self.supported_channel_counts)):
            raise ValueError("supported_channel_counts 不能重复。")
        if any(
            channel_count < self.min_channels or channel_count > self.max_channels
            for channel_count in self.supported_channel_counts
        ):
            raise ValueError("supported_channel_counts 必须位于 min/max 范围内。")
        if self.kind is ModuleKind.TRAINER:
            raise ValueError("trainer 必须登记到 Trainer Registry。")
        if self.kind in {ModuleKind.MODEL, ModuleKind.LOSS} and self.trainer_backend is None:
            raise ValueError("model/loss 模块必须声明 trainer_backend。")
        return self


class TrainerDefinition(SchemaModel):
    """一个训练后端及其支持的模型、损失和资源边界。"""

    trainer_id: EntityId
    display_name: NonEmptyText
    backend: TrainerBackend
    supported_tasks: tuple[EntityId, ...] = ("four_stage_classification",)
    supported_model_ids: tuple[EntityId, ...] = Field(min_length=1)
    supported_loss_ids: tuple[EntityId, ...] = ()
    parameters_schema: dict[str, ParameterRule] = Field(default_factory=dict)
    resource_class: ResourceClass
    requires_cuda: bool = False
    implemented: bool
    experimental: bool = False

    @model_validator(mode="after")
    def _identifiers_are_unique(self) -> "TrainerDefinition":
        if len(self.supported_model_ids) != len(set(self.supported_model_ids)):
            raise ValueError("supported_model_ids 不能重复。")
        if len(self.supported_loss_ids) != len(set(self.supported_loss_ids)):
            raise ValueError("supported_loss_ids 不能重复。")
        return self


class RegistryCatalog(SchemaModel):
    """可直接作为 FastAPI 响应和页面选项来源的能力目录。"""

    input_presets: tuple[InputPresetDefinition, ...] = Field(min_length=1)
    modules: tuple[ModuleDefinition, ...] = Field(min_length=1)
    trainers: tuple[TrainerDefinition, ...] = Field(min_length=1)
    catalog_hash: Sha256Hex | None = None

    @model_validator(mode="after")
    def _catalog_ids_are_unique(self) -> "RegistryCatalog":
        groups = (
            ("preset_id", [item.preset_id for item in self.input_presets]),
            ("module_id", [item.module_id for item in self.modules]),
            ("trainer_id", [item.trainer_id for item in self.trainers]),
        )
        for field_name, identifiers in groups:
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"Catalog 中的 {field_name} 不能重复。")

        modules = {module.module_id: module for module in self.modules}
        for trainer in self.trainers:
            for model_id in trainer.supported_model_ids:
                model = modules.get(model_id)
                if model is None or model.kind is not ModuleKind.MODEL:
                    raise ValueError(f"训练器 {trainer.trainer_id} 引用了未知模型 {model_id}。")
                if model.trainer_backend is not trainer.backend:
                    raise ValueError(f"模型 {model_id} 与训练器 {trainer.trainer_id} 后端不一致。")
            for loss_id in trainer.supported_loss_ids:
                loss = modules.get(loss_id)
                if loss is None or loss.kind is not ModuleKind.LOSS:
                    raise ValueError(f"训练器 {trainer.trainer_id} 引用了未知损失 {loss_id}。")
                if loss.trainer_backend is not trainer.backend:
                    raise ValueError(f"损失 {loss_id} 与训练器 {trainer.trainer_id} 后端不一致。")
        return self
