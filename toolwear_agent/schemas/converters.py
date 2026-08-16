"""把迁移期旧候选对象转换为唯一 PipelineSpec。"""

from __future__ import annotations

from toolwear_agent.agentteams.llm_candidates import LlmCandidatePlan
from toolwear_agent.schemas.pipeline import (
    ExpectedCost,
    ExperimentalExtensionSpec,
    ModuleSpec,
    PipelineSource,
    PipelineSpec,
)
from toolwear_agent.training.candidates import CandidatePlan


def _module_chain(plan_id: str) -> tuple[ModuleSpec, ...]:
    """返回已知候选的规范模块链；未知 LLM 方案不能直接执行。"""

    chains: dict[str, tuple[ModuleSpec, ...]] = {
        "statistical_features_random_forest": (
            ModuleSpec(module_id="sliding_window", kind="windowing", order=10),
            ModuleSpec(module_id="statistical_features", kind="feature", order=20),
            ModuleSpec(module_id="random_forest", kind="model", order=30),
            ModuleSpec(module_id="sklearn", kind="trainer", order=40),
        ),
        "statistical_features_extra_trees": (
            ModuleSpec(module_id="sliding_window", kind="windowing", order=10),
            ModuleSpec(module_id="statistical_features", kind="feature", order=20),
            ModuleSpec(module_id="extra_trees", kind="model", order=30),
            ModuleSpec(module_id="sklearn", kind="trainer", order=40),
        ),
        "multichannel_window_1d_cnn": (
            ModuleSpec(module_id="sliding_window", kind="windowing", order=10),
            ModuleSpec(module_id="zscore", kind="preprocess", order=20),
            ModuleSpec(module_id="raw_1d", kind="feature", order=30),
            ModuleSpec(module_id="cnn_1d", kind="model", order=40),
            ModuleSpec(module_id="cross_entropy", kind="loss", order=50),
            ModuleSpec(module_id="pytorch", kind="trainer", order=60),
        ),
        "light_multibranch_cnn_attention": (
            ModuleSpec(module_id="sliding_window", kind="windowing", order=10),
            ModuleSpec(module_id="zscore", kind="preprocess", order=20),
            ModuleSpec(module_id="raw_1d", kind="feature", order=30),
            ModuleSpec(module_id="early_concat", kind="fusion", order=40),
            ModuleSpec(module_id="cnn_1d", kind="model", order=50),
            ModuleSpec(module_id="cross_entropy", kind="loss", order=60),
            ModuleSpec(module_id="pytorch", kind="trainer", order=70),
        ),
    }
    try:
        return chains[plan_id]
    except KeyError as exc:
        raise ValueError(f"候选方案尚未映射到本地 Module Registry：{plan_id}") from exc


def _experimental_extensions(
    plan_id: str,
    proposed_extension: dict[str, object] | None = None,
) -> tuple[ExperimentalExtensionSpec, ...]:
    """把尚未实现的研究结构放入显式扩展区，而不是伪装成 Registry 模块。"""

    extensions: list[ExperimentalExtensionSpec] = []
    if plan_id == "light_multibranch_cnn_attention":
        extensions.append(
            ExperimentalExtensionSpec(
                extension_id="multibranch_channel_attention",
                display_name="多传感器分支与通道注意力",
                kind="fusion",
                rationale="体现力、振动、声发射分支建模设想，待基础 1D-CNN 跑通后实现。",
            )
        )
    if proposed_extension is not None:
        extensions.append(ExperimentalExtensionSpec.model_validate(proposed_extension))
    return tuple(extensions)


def _expected_cost(raw_cost: str) -> ExpectedCost:
    """把旧中文说明归一化为稳定成本枚举。"""

    lowered = raw_cost.lower()
    if "高" in raw_cost or "high" in lowered:
        return ExpectedCost.HIGH
    if "中" in raw_cost or "medium" in lowered:
        return ExpectedCost.MEDIUM
    return ExpectedCost.LOW


def candidate_plan_to_pipeline(plan: CandidatePlan) -> PipelineSpec:
    """把规则化 CandidatePlan 转为统一 PipelineSpec。"""

    return PipelineSpec(
        pipeline_id=plan.plan_id,
        display_name=plan.display_name,
        source=PipelineSource.FIXED,
        input_channels=tuple(plan.input_channels),
        modules=_module_chain(plan.plan_id),
        rationale=plan.recommended_reason,
        risks=tuple(plan.risks),
        expected_cost=_expected_cost(plan.expected_cost),
        trainable=plan.plan_id in {
            "statistical_features_random_forest",
            "statistical_features_extra_trees",
            "multichannel_window_1d_cnn",
        },
        recommended_rank=plan.recommended_order,
        compatibility_tags=(plan.model_family,),
        experimental_extensions=_experimental_extensions(plan.plan_id),
    )


def llm_candidate_plan_to_pipeline(plan: LlmCandidatePlan) -> PipelineSpec:
    """把通过旧白名单校验的 LLM 候选转为统一 PipelineSpec。"""

    return PipelineSpec(
        pipeline_id=plan.plan_id,
        display_name=plan.display_name,
        source=PipelineSource.LLM,
        input_channels=(
            "force_x",
            "force_y",
            "force_z",
            "vibration_x",
            "vibration_y",
            "vibration_z",
            "acoustic_emission_rms",
        ),
        modules=_module_chain(plan.plan_id),
        rationale=plan.reason,
        risks=(plan.risk,),
        expected_cost=_expected_cost(plan.expected_cost),
        trainable=plan.trainable_now,
        compatibility_tags=(plan.training_backend,),
        experimental_extensions=_experimental_extensions(
            plan.plan_id,
            plan.experimental_extension,
        ),
    )
