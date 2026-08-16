"""训练指标、最终评估和通用校验结果 Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, JsonValue, model_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, utc_now


class MetricSplit(str, Enum):
    """指标所属数据分区。"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class MetricBundle(SchemaModel):
    """一个 split 的核心分类指标。"""

    split: MetricSplit
    sample_count: int = Field(ge=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    loss: float | None = Field(default=None, ge=0)
    per_class: dict[str, JsonValue] = Field(default_factory=dict)
    confusion_matrix: tuple[tuple[int, ...], ...] = ()


class EvaluationReport(SchemaModel):
    """不会把 test 指标混入候选选择的评估快照。"""

    evaluation_id: EntityId
    experiment_id: EntityId
    run_id: EntityId
    pipeline_id: EntityId
    metrics: tuple[MetricBundle, ...] = Field(min_length=1)
    class_labels: tuple[NonEmptyText, ...] = Field(min_length=2)
    evidence_ids: tuple[EntityId, ...] = ()
    final_test: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _test_metrics_are_final_only(self) -> "EvaluationReport":
        splits = [bundle.split for bundle in self.metrics]
        if len(splits) != len(set(splits)):
            raise ValueError("同一报告不能重复同一个 split 的指标。")
        has_test = MetricSplit.TEST in splits
        if has_test and not self.final_test:
            raise ValueError("包含 test 指标的报告必须设置 final_test=true。")
        if self.final_test and not has_test:
            raise ValueError("final_test=true 时必须包含 test 指标。")
        return self


class ValidationSeverity(str, Enum):
    """校验问题级别。"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(SchemaModel):
    """一个可定位、可机器处理的校验问题。"""

    code: EntityId
    severity: ValidationSeverity
    message: NonEmptyText
    field_path: str | None = None
    remediation: str | None = None


class ValidationResult(SchemaModel):
    """Dataset、Pipeline、Run 或 Skill 的统一校验结果。"""

    valid: bool
    scope: EntityId
    issues: tuple[ValidationIssue, ...] = ()
    validated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _valid_flag_matches_issues(self) -> "ValidationResult":
        has_error = any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)
        if self.valid and has_error:
            raise ValueError("存在 error 问题时 valid 不能为 true。")
        if not self.valid and not has_error:
            raise ValueError("valid=false 时至少需要一个 error 问题。")
        return self
