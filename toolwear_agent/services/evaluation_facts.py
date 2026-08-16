"""从真实 train/validation 指标提取不可由 LLM 改写的诊断事实。"""

from __future__ import annotations

from collections.abc import Sequence

from toolwear_agent.schemas import EpochLoss, MetricBundle
from toolwear_agent.schemas.diagnosis import ConfusionObservation, EvaluationFacts


def _metric_value(values: object, key: str) -> float:
    if not isinstance(values, dict):
        return 0.0
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _support_value(values: object) -> int:
    if not isinstance(values, dict):
        return 1
    value = values.get("support", 1)
    return max(1, int(value)) if isinstance(value, (int, float)) else 1


def _top_confusions(
    validation: MetricBundle,
    class_labels: Sequence[str],
) -> tuple[ConfusionObservation, ...]:
    observations: list[ConfusionObservation] = []
    for actual_index, row in enumerate(validation.confusion_matrix):
        if actual_index >= len(class_labels):
            break
        row_total = sum(row)
        if row_total <= 0:
            continue
        for predicted_index, count in enumerate(row):
            if predicted_index >= len(class_labels) or predicted_index == actual_index or count <= 0:
                continue
            observations.append(
                ConfusionObservation(
                    actual_label=class_labels[actual_index],
                    predicted_label=class_labels[predicted_index],
                    count=count,
                    row_rate=count / row_total,
                )
            )
    observations.sort(key=lambda item: (item.count, item.row_rate), reverse=True)
    return tuple(observations[:5])


def _training_trend(epoch_history: Sequence[EpochLoss]) -> str:
    if len(epoch_history) < 2:
        return "not_available"
    losses = [item.validation_loss for item in epoch_history]
    net_change = losses[-1] - losses[0]
    if len(losses) >= 3:
        directions = [right - left for left, right in zip(losses, losses[1:])]
        has_up = any(value > 0.03 for value in directions)
        has_down = any(value < -0.03 for value in directions)
        if has_up and has_down and max(losses) - min(losses) > 0.15:
            return "unstable"
    if net_change <= -0.05:
        return "improving"
    if net_change >= 0.05:
        return "degrading"
    return "stable"


def build_evaluation_facts(
    *,
    experiment_id: str,
    run_id: str,
    pipeline_id: str,
    train: MetricBundle | None,
    validation: MetricBundle,
    class_labels: Sequence[str],
    epoch_history: Sequence[EpochLoss],
    module_ids: Sequence[str],
    completed_mini_runs: int,
    max_mini_runs: int,
    source_evidence_ids: Sequence[str],
) -> EvaluationFacts:
    """构建只含 train/validation 的事实快照。"""

    if validation.split.value != "validation":
        raise ValueError("诊断事实必须使用 validation MetricBundle。")
    if train is not None and train.split.value != "train":
        raise ValueError("train 参数只能是 train MetricBundle。")
    class_support = {
        label: _support_value(validation.per_class.get(label)) for label in class_labels
    }
    class_scores = {
        label: _metric_value(validation.per_class.get(label), "f1-score")
        for label in class_labels
    }
    weakest_class = min(class_scores, key=class_scores.get)
    train_macro = train.macro_f1 if train is not None else None
    return EvaluationFacts(
        facts_id=f"{run_id}-validation-facts",
        experiment_id=experiment_id,
        run_id=run_id,
        pipeline_id=pipeline_id,
        train_macro_f1=train_macro,
        validation_macro_f1=validation.macro_f1,
        validation_balanced_accuracy=validation.balanced_accuracy,
        generalization_gap_macro_f1=(
            None if train_macro is None else train_macro - validation.macro_f1
        ),
        train_loss=train.loss if train is not None else None,
        validation_loss=validation.loss,
        weakest_class=weakest_class,
        weakest_class_f1=class_scores[weakest_class],
        weakest_class_recall=_metric_value(
            validation.per_class.get(weakest_class),
            "recall",
        ),
        class_support=class_support,
        support_imbalance_ratio=max(class_support.values()) / min(class_support.values()),
        top_confusions=_top_confusions(validation, class_labels),
        training_trend=_training_trend(epoch_history),
        epoch_count=len(epoch_history),
        module_ids=tuple(module_ids),
        completed_mini_runs=completed_mini_runs,
        max_mini_runs=max_mini_runs,
        source_evidence_ids=tuple(source_evidence_ids),
    )
