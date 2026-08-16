"""PipelineSpec 对 Module/Trainer Registry 的静态兼容性校验。"""

from __future__ import annotations

import math
from collections.abc import Mapping

from toolwear_agent.registry.module_registry import PHM2010_CHANNEL_IDS, ModuleRegistry
from toolwear_agent.registry.trainer_registry import TrainerRegistry
from toolwear_agent.schemas import (
    FeatureType,
    ModuleDefinition,
    ParameterRule,
    PipelineSpec,
    TrainerDefinition,
    ValidationIssue,
    ValidationResult,
)


def _issue(
    code: str,
    message: str,
    field_path: str,
    *,
    severity: str = "error",
    remediation: str | None = None,
) -> ValidationIssue:
    """构造字段完整且便于 API 映射的校验问题。"""

    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        field_path=field_path,
        remediation=remediation,
    )


def _matches_parameter_type(value: object, rule: ParameterRule) -> bool:
    """严格区分 bool、整数、数值和字符串，避免 Python 隐式类型混淆。"""

    if rule.value_type.value == "boolean":
        return isinstance(value, bool)
    if rule.value_type.value == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if rule.value_type.value == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if rule.value_type.value == "string":
        return isinstance(value, str)
    return False


def _validate_parameters(
    values: Mapping[str, object],
    rules: Mapping[str, ParameterRule],
    *,
    field_path: str,
) -> list[ValidationIssue]:
    """校验参数白名单、类型、范围和可选值。"""

    issues: list[ValidationIssue] = []
    for name in sorted(set(values) - set(rules)):
        issues.append(
            _issue(
                "PARAMETER_UNKNOWN",
                f"参数 {name} 未在 Registry 中声明。",
                f"{field_path}.parameters.{name}",
                remediation="从 Registry 参数 Schema 中选择合法参数。",
            )
        )

    for name, rule in rules.items():
        parameter_path = f"{field_path}.parameters.{name}"
        if name not in values:
            if rule.required:
                issues.append(_issue("PARAMETER_REQUIRED", f"缺少必填参数 {name}。", parameter_path))
            continue

        value = values[name]
        if not _matches_parameter_type(value, rule):
            issues.append(
                _issue(
                    "PARAMETER_TYPE_MISMATCH",
                    f"参数 {name} 必须是 {rule.value_type.value}。",
                    parameter_path,
                )
            )
            continue
        if rule.choices and value not in rule.choices:
            issues.append(
                _issue(
                    "PARAMETER_NOT_ALLOWED",
                    f"参数 {name} 的值不在允许列表 {list(rule.choices)} 中。",
                    parameter_path,
                )
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if rule.minimum is not None and float(value) < rule.minimum:
                issues.append(
                    _issue(
                        "PARAMETER_OUT_OF_RANGE",
                        f"参数 {name} 不能小于 {rule.minimum}。",
                        parameter_path,
                    )
                )
            if rule.maximum is not None and float(value) > rule.maximum:
                issues.append(
                    _issue(
                        "PARAMETER_OUT_OF_RANGE",
                        f"参数 {name} 不能大于 {rule.maximum}。",
                        parameter_path,
                    )
                )
    return issues


def _implementation_issue(
    *,
    capability_id: str,
    capability_type: str,
    field_path: str,
    pipeline_trainable: bool,
) -> ValidationIssue:
    """未实现能力在预览方案中告警，在可训练方案中报错。"""

    return _issue(
        "MODULE_NOT_IMPLEMENTED",
        f"{capability_type} {capability_id} 已登记但当前尚未实现。",
        field_path,
        severity="error" if pipeline_trainable else "warning",
        remediation="保留为非训练候选，或先完成实现并更新 Registry。",
    )


def _validate_module_definition(
    pipeline: PipelineSpec,
    module_index: int,
    definition: ModuleDefinition,
    module_kind: object,
    module_parameters: Mapping[str, object],
) -> list[ValidationIssue]:
    """校验单个非 trainer 模块的静态属性。"""

    field_path = f"modules.{module_index}"
    issues: list[ValidationIssue] = []
    if definition.kind is not module_kind:
        issues.append(
            _issue(
                "MODULE_KIND_MISMATCH",
                f"模块 {definition.module_id} 的类别应为 {definition.kind.value}。",
                f"{field_path}.kind",
            )
        )
    if pipeline.task_type not in definition.supported_tasks:
        issues.append(
            _issue(
                "TASK_NOT_SUPPORTED",
                f"模块 {definition.module_id} 不支持任务 {pipeline.task_type}。",
                field_path,
            )
        )
    channel_count = len(pipeline.input_channels)
    channel_count_invalid = not definition.min_channels <= channel_count <= definition.max_channels
    if definition.supported_channel_counts:
        channel_count_invalid = channel_count not in definition.supported_channel_counts
    if channel_count_invalid:
        supported_text = (
            f"{list(definition.supported_channel_counts)}"
            if definition.supported_channel_counts
            else f"{definition.min_channels}-{definition.max_channels}"
        )
        issues.append(
            _issue(
                "CHANNEL_COUNT_NOT_SUPPORTED",
                (
                    f"模块 {definition.module_id} 支持 {supported_text} 个通道，"
                    f"当前为 {channel_count}。"
                ),
                field_path,
            )
        )
    if not definition.implemented:
        issues.append(
            _implementation_issue(
                capability_id=definition.module_id,
                capability_type="模块",
                field_path=field_path,
                pipeline_trainable=pipeline.trainable,
            )
        )
    issues.extend(
        _validate_parameters(module_parameters, definition.parameters_schema, field_path=field_path)
    )
    return issues


def _validate_trainer_definition(
    pipeline: PipelineSpec,
    trainer_index: int,
    trainer: TrainerDefinition,
    trainer_parameters: Mapping[str, object],
) -> list[ValidationIssue]:
    """校验训练器实现状态、任务和参数。"""

    field_path = f"modules.{trainer_index}"
    issues: list[ValidationIssue] = []
    if pipeline.task_type not in trainer.supported_tasks:
        issues.append(
            _issue(
                "TASK_NOT_SUPPORTED",
                f"训练器 {trainer.trainer_id} 不支持任务 {pipeline.task_type}。",
                field_path,
            )
        )
    if not trainer.implemented:
        issues.append(
            _implementation_issue(
                capability_id=trainer.trainer_id,
                capability_type="训练器",
                field_path=field_path,
                pipeline_trainable=pipeline.trainable,
            )
        )
    issues.extend(_validate_parameters(trainer_parameters, trainer.parameters_schema, field_path=field_path))
    return issues


def validate_pipeline_against_registries(
    pipeline: PipelineSpec,
    module_registry: ModuleRegistry,
    trainer_registry: TrainerRegistry,
) -> ValidationResult:
    """在执行前校验模块存在性、特征链、后端和参数兼容性。"""

    issues: list[ValidationIssue] = []
    unknown_channels = sorted(set(pipeline.input_channels) - set(PHM2010_CHANNEL_IDS))
    if unknown_channels:
        issues.append(
            _issue(
                "INPUT_CHANNEL_NOT_REGISTERED",
                f"Pipeline 包含未登记通道: {unknown_channels}。",
                "input_channels",
            )
        )

    current_feature_type = FeatureType.RAW_SIGNAL
    model_definition: ModuleDefinition | None = None
    loss_definitions: list[ModuleDefinition] = []
    trainer_definition: TrainerDefinition | None = None
    trainer_index: int | None = None

    for index, module in enumerate(pipeline.modules):
        if not module.enabled:
            continue
        if module.kind.value == "trainer":
            trainer_index = index
            try:
                trainer_definition = trainer_registry.get(module.module_id)
            except KeyError:
                issues.append(
                    _issue(
                        "TRAINER_NOT_REGISTERED",
                        f"训练器 {module.module_id} 不在 Trainer Registry 中。",
                        f"modules.{index}.module_id",
                    )
                )
                continue
            issues.extend(
                _validate_trainer_definition(pipeline, index, trainer_definition, module.parameters)
            )
            continue

        try:
            definition = module_registry.get(module.module_id)
        except KeyError:
            issues.append(
                _issue(
                    "MODULE_NOT_REGISTERED",
                    f"模块 {module.module_id} 不在 Module Registry 中。",
                    f"modules.{index}.module_id",
                    remediation="使用已登记模块，或输出 experimental_extension。",
                )
            )
            continue

        issues.extend(
            _validate_module_definition(pipeline, index, definition, module.kind, module.parameters)
        )
        if definition.required_feature_type is not current_feature_type:
            issues.append(
                _issue(
                    "FEATURE_TYPE_MISMATCH",
                    (
                        f"模块 {definition.module_id} 需要 {definition.required_feature_type.value}，"
                        f"上一步输出为 {current_feature_type.value}。"
                    ),
                    f"modules.{index}",
                )
            )
        current_feature_type = definition.output_feature_type
        if definition.kind.value == "model":
            model_definition = definition
        elif definition.kind.value == "loss":
            loss_definitions.append(definition)

    if trainer_definition is not None and model_definition is not None:
        if model_definition.trainer_backend is not trainer_definition.backend:
            issues.append(
                _issue(
                    "TRAINER_BACKEND_MISMATCH",
                    (
                        f"模型 {model_definition.module_id} 需要 {model_definition.trainer_backend.value}，"
                        f"当前训练器为 {trainer_definition.backend.value}。"
                    ),
                    f"modules.{trainer_index}",
                )
            )
        if model_definition.module_id not in trainer_definition.supported_model_ids:
            issues.append(
                _issue(
                    "MODEL_NOT_SUPPORTED_BY_TRAINER",
                    f"训练器 {trainer_definition.trainer_id} 不支持模型 {model_definition.module_id}。",
                    f"modules.{trainer_index}",
                )
            )

        if trainer_definition.backend.value == "pytorch" and len(loss_definitions) != 1:
            issues.append(
                _issue(
                    "LOSS_COUNT_INVALID",
                    "PyTorch 分类 Pipeline 必须且只能配置一个损失模块。",
                    "modules",
                )
            )
        if trainer_definition.backend.value == "sklearn" and loss_definitions:
            issues.append(
                _issue(
                    "LOSS_NOT_SUPPORTED",
                    "sklearn 分类器由模型内部处理目标函数，不应配置独立 loss。",
                    "modules",
                )
            )
        for loss in loss_definitions:
            if loss.trainer_backend is not trainer_definition.backend:
                issues.append(
                    _issue(
                        "LOSS_BACKEND_MISMATCH",
                        f"损失 {loss.module_id} 与训练器后端不一致。",
                        "modules",
                    )
                )
            if loss.module_id not in trainer_definition.supported_loss_ids:
                issues.append(
                    _issue(
                        "LOSS_NOT_SUPPORTED_BY_TRAINER",
                        f"训练器 {trainer_definition.trainer_id} 不支持损失 {loss.module_id}。",
                        "modules",
                    )
                )

    for index, extension in enumerate(pipeline.experimental_extensions):
        issues.append(
            _issue(
                "EXPERIMENTAL_EXTENSION",
                f"扩展 {extension.extension_id} 仅为研究提案，当前不可执行。",
                f"experimental_extensions.{index}",
                severity="warning",
            )
        )

    has_error = any(issue.severity.value == "error" for issue in issues)
    return ValidationResult(
        valid=not has_error,
        scope="pipeline.registry",
        issues=tuple(issues),
    )


def validate_pipeline_with_default_registries(pipeline: PipelineSpec) -> ValidationResult:
    """使用项目默认 Registry 校验候选，供页面与 LLM 边界复用。"""

    from toolwear_agent.registry.module_registry import build_default_module_registry
    from toolwear_agent.registry.trainer_registry import build_default_trainer_registry

    return validate_pipeline_against_registries(
        pipeline,
        build_default_module_registry(),
        build_default_trainer_registry(),
    )
