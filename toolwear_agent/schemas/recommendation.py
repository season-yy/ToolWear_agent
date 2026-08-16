"""候选方案在 LLM、页面、审批和数据库之间的统一契约。"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex, utc_now
from toolwear_agent.schemas.dataset import DatasetRef
from toolwear_agent.schemas.pipeline import PipelineSpec


class CandidateRecommendationSet(SchemaModel):
    """一次可恢复候选生成的结构化结果。"""

    recommendation_id: EntityId
    experiment_id: EntityId
    revision: int = Field(ge=1)
    dataset_ref: DatasetRef
    user_request: NonEmptyText
    provider: NonEmptyText
    model: str = ""
    used_fallback: bool = False
    fallback_reason: str = ""
    registry_catalog_hash: Sha256Hex
    pipelines: tuple[PipelineSpec, ...] = Field(min_length=2, max_length=3)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _pipeline_ids_are_unique(self) -> "CandidateRecommendationSet":
        pipeline_ids = [pipeline.pipeline_id for pipeline in self.pipelines]
        if len(pipeline_ids) != len(set(pipeline_ids)):
            raise ValueError("候选 Pipeline ID 不能重复。")
        return self
