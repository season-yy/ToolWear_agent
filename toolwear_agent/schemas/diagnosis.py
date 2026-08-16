"""EvaluationGovernor 使用的 validation 事实与结构化诊断契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex, utc_now


class ConfusionObservation(SchemaModel):
    """混淆矩阵中一个真实类别到预测类别的错误方向。"""

    actual_label: NonEmptyText
    predicted_label: NonEmptyText
    count: int = Field(ge=1)
    row_rate: float = Field(ge=0, le=1)


class EvaluationFacts(SchemaModel):
    """由确定性代码生成、且不允许携带 final test 的诊断事实。"""

    facts_id: EntityId
    experiment_id: EntityId
    run_id: EntityId
    pipeline_id: EntityId
    basis_split: Literal["validation"] = "validation"
    final_test_used: Literal[False] = False
    train_macro_f1: float | None = Field(default=None, ge=0, le=1)
    validation_macro_f1: float = Field(ge=0, le=1)
    validation_balanced_accuracy: float = Field(ge=0, le=1)
    generalization_gap_macro_f1: float | None = Field(default=None, ge=-1, le=1)
    train_loss: float | None = Field(default=None, ge=0)
    validation_loss: float | None = Field(default=None, ge=0)
    weakest_class: NonEmptyText
    weakest_class_f1: float = Field(ge=0, le=1)
    weakest_class_recall: float = Field(ge=0, le=1)
    class_support: dict[str, int]
    support_imbalance_ratio: float = Field(ge=1)
    top_confusions: tuple[ConfusionObservation, ...] = Field(default=(), max_length=5)
    training_trend: Literal["not_available", "improving", "stable", "degrading", "unstable"]
    epoch_count: int = Field(ge=0)
    module_ids: tuple[EntityId, ...] = ()
    completed_mini_runs: int = Field(ge=0)
    max_mini_runs: int = Field(ge=1)
    source_evidence_ids: tuple[EntityId, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _budget_and_support_are_consistent(self) -> "EvaluationFacts":
        if self.completed_mini_runs > self.max_mini_runs:
            raise ValueError("completed_mini_runs 不能超过 max_mini_runs。")
        if not self.class_support or any(value < 1 for value in self.class_support.values()):
            raise ValueError("class_support 必须包含每个阶段的正整数样本量。")
        return self


class DiagnosisFinding(SchemaModel):
    """诊断中的一条可定位发现。"""

    finding_id: EntityId
    severity: Literal["info", "warning", "critical"]
    category: EntityId
    title: NonEmptyText
    detail: NonEmptyText
    evidence: NonEmptyText


class DiagnosisRecommendation(SchemaModel):
    """供用户建立下一 revision 的建议，不直接修改训练配置。"""

    recommendation_id: EntityId
    action_type: Literal[
        "adjust_parameter",
        "change_pipeline",
        "inspect_data",
        "approve_full",
        "stop",
    ]
    target: NonEmptyText
    suggestion: NonEmptyText
    rationale: NonEmptyText
    priority: Literal["low", "medium", "high"]
    requires_human_approval: Literal[True] = True


class DiagnosisAdvice(SchemaModel):
    """LLM 只能填充的建议部分，事实字段由系统单独提供。"""

    overall_conclusion: NonEmptyText
    risk_level: Literal["low", "medium", "high"]
    findings: tuple[DiagnosisFinding, ...] = Field(min_length=1, max_length=8)
    recommendations: tuple[DiagnosisRecommendation, ...] = Field(min_length=1, max_length=6)
    recommended_action: Literal[
        "approve_full",
        "adjust_parameters",
        "change_pipeline",
        "stop",
    ]


class LlmCallAudit(SchemaModel):
    """不含密钥和完整 Prompt 的 LLM 调用审计摘要。"""

    provider: str = ""
    model: str = ""
    status: Literal["success", "fallback"]
    used_fallback: bool
    fallback_reason: str = ""
    latency_ms: int = Field(ge=0)
    prompt_template_version: EntityId
    prompt_sha256: Sha256Hex
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _status_matches_fallback(self) -> "LlmCallAudit":
        if (self.status == "fallback") != self.used_fallback:
            raise ValueError("LLM status 与 used_fallback 不一致。")
        if self.used_fallback and not self.fallback_reason:
            raise ValueError("fallback 调用必须记录原因。")
        return self


class EvaluationDiagnosis(SchemaModel):
    """事实、LLM 建议和调用审计组成的完整诊断证据。"""

    diagnosis_id: EntityId
    facts: EvaluationFacts
    advice: DiagnosisAdvice
    llm_call: LlmCallAudit
    created_at: datetime = Field(default_factory=utc_now)
