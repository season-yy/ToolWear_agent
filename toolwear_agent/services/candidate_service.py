"""把 LLM 候选统一转换为 Registry 校验后的 PipelineSpec。"""

from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from toolwear_agent.agentteams.llm_candidates import generate_llm_candidate_set
from toolwear_agent.core.settings import Settings
from toolwear_agent.registry import (
    build_default_registry_catalog,
    validate_pipeline_with_default_registries,
)
from toolwear_agent.schemas import CandidateRecommendationSet, ExperimentState, PipelineSpec
from toolwear_agent.schemas.converters import llm_candidate_plan_to_pipeline


class CandidateProvider(Protocol):
    """允许 API 测试或后续 Agent runtime 注入候选来源。"""

    def recommend(
        self,
        state: ExperimentState,
        user_request: str,
    ) -> CandidateRecommendationSet:
        """返回 2-3 个已通过 Registry 的候选。"""


class DefaultCandidateProvider:
    """默认调用千问；失败时保留可审计的规则 fallback。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def recommend(
        self,
        state: ExperimentState,
        user_request: str,
    ) -> CandidateRecommendationSet:
        cutter_id = state.dataset_ref.cutter_ids[0]
        generated = generate_llm_candidate_set(
            self.settings,
            user_request,
            dataset_id=state.dataset_ref.dataset_id,
            cutter=cutter_id,
        )
        pipelines: list[PipelineSpec] = []
        for plan in generated.plans:
            pipeline = llm_candidate_plan_to_pipeline(plan)
            if state.preferences.input_channels:
                payload = pipeline.model_dump(mode="python")
                payload["input_channels"] = state.preferences.input_channels
                pipeline = PipelineSpec.model_validate(payload)
            validation = validate_pipeline_with_default_registries(pipeline)
            if validation.valid and pipeline.pipeline_id not in {
                item.pipeline_id for item in pipelines
            }:
                pipelines.append(pipeline)
        if len(pipelines) < 2:
            raise ValueError("候选生成后不足两个可兼容 Pipeline，请调整输入通道或模块要求。")
        catalog = build_default_registry_catalog()
        if catalog.catalog_hash is None:  # pragma: no cover - builder 已附加 hash
            raise ValueError("Registry Catalog 缺少 hash。")
        return CandidateRecommendationSet(
            recommendation_id=f"recommendation-{uuid4().hex}",
            experiment_id=state.experiment_id,
            revision=state.revision,
            dataset_ref=state.dataset_ref,
            user_request=user_request,
            provider=generated.provider,
            model=generated.model,
            used_fallback=generated.used_fallback,
            fallback_reason=generated.fallback_reason,
            registry_catalog_hash=catalog.catalog_hash,
            pipelines=tuple(pipelines[:3]),
        )
