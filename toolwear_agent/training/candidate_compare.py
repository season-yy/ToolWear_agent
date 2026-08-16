"""多候选训练对比。

本模块复用窗口特征表，对当前已实现的候选模型做小样本训练对比。
第 13 步先支持 RandomForest 和 ExtraTrees，后续再接 CNN。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from toolwear_agent.agentteams.diagnosis import load_json
from toolwear_agent.agentteams.reporting import find_latest_decided_run
from toolwear_agent.common.config import Settings
from toolwear_agent.data.splitting import normalize_split_name
from toolwear_agent.training.mini_train import _evaluate_classifier


SUPPORTED_COMPARE_PLAN_IDS = (
    "statistical_features_random_forest",
    "statistical_features_extra_trees",
)


@dataclass(frozen=True)
class CandidateCompareMetric:
    """单个候选模型的对比指标。"""

    plan_id: str
    display_name: str
    classifier_name: str
    train_count: int
    validation_macro_f1: float
    validation_balanced_accuracy: float
    recommendation: str


@dataclass(frozen=True)
class CandidateCompareResult:
    """候选对比结果。"""

    compare_run_id: str
    source_run_dir: str
    metrics: list[CandidateCompareMetric]
    best_plan_id: str
    report_file: str
    result_json: str
    log_file: str


def _now_shanghai_compact() -> str:
    """生成适合作为目录名的上海时间戳。"""

    shanghai_timezone = timezone(timedelta(hours=8))
    return datetime.now(shanghai_timezone).strftime("%Y%m%d_%H%M%S")


def load_feature_table(feature_table_file: Path) -> tuple[list[list[float]], list[int], list[str]]:
    """读取窗口特征表。"""

    x_values: list[list[float]] = []
    y_values: list[int] = []
    splits: list[str] = []
    with feature_table_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames or []
        feature_names = [name for name in fieldnames if name.endswith(("_mean", "_std", "_min", "_max", "_ptp", "_rms"))]
        for row in reader:
            x_values.append([float(row[name]) for name in feature_names])
            y_values.append(int(row["stage_id"]))
            splits.append(normalize_split_name(str(row["split"])))
    return x_values, y_values, splits


def _split_indices(splits: list[str], split_name: str) -> list[int]:
    """返回指定 split 的行索引。"""

    return [index for index, value in enumerate(splits) if value == split_name]


def build_classifier(plan_id: str, random_seed: int) -> tuple[str, object]:
    """根据候选 ID 构建分类器。"""

    if plan_id == "statistical_features_random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return "RandomForestClassifier", RandomForestClassifier(
            n_estimators=160,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        )
    if plan_id == "statistical_features_extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return "ExtraTreesClassifier", ExtraTreesClassifier(
            n_estimators=180,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        )
    raise ValueError(f"不支持的候选训练方案: {plan_id}")


def train_compare_candidate(
    plan_id: str,
    x_values: list[list[float]],
    y_values: list[int],
    splits: list[str],
    random_seed: int,
) -> CandidateCompareMetric:
    """训练并评估一个候选模型。"""

    classifier_name, classifier = build_classifier(plan_id, random_seed)
    train_indices = _split_indices(splits, "train")
    val_indices = _split_indices(splits, "validation")
    classifier.fit([x_values[index] for index in train_indices], [y_values[index] for index in train_indices])
    validation = _evaluate_classifier(classifier, [x_values[index] for index in val_indices], [y_values[index] for index in val_indices])
    display_name = "统计特征 + RandomForest" if plan_id.endswith("random_forest") else "统计特征 + ExtraTrees"
    recommendation = "可作为当前稳定候选" if validation["macro_f1"] >= 0.9 else "暂不建议进入完整训练"
    return CandidateCompareMetric(
        plan_id=plan_id,
        display_name=display_name,
        classifier_name=classifier_name,
        train_count=len(train_indices),
        validation_macro_f1=float(validation["macro_f1"]),
        validation_balanced_accuracy=float(validation["balanced_accuracy"]),
        recommendation=recommendation,
    )


def select_best_candidate(metrics: list[CandidateCompareMetric]) -> CandidateCompareMetric:
    """只按 validation 指标选择候选，禁止 test 参与模型结构决策。"""

    if not metrics:
        raise ValueError("候选指标不能为空。")
    return max(
        metrics,
        key=lambda item: (item.validation_macro_f1, item.validation_balanced_accuracy, item.plan_id),
    )


def render_compare_report(result: CandidateCompareResult) -> str:
    """渲染候选对比 Markdown。"""

    lines = [
        "# PHM2010 C1 多候选训练对比报告",
        "",
        f"- 对比运行编号：`{result.compare_run_id}`",
        f"- 来源运行目录：`{result.source_run_dir}`",
        f"- 推荐候选：`{result.best_plan_id}`",
        "",
        "## 1. 指标对比",
        "",
        "| 候选 | 分类器 | Validation Macro-F1 | Validation Balanced Acc | 建议 |",
        "|---|---|---:|---:|---|",
    ]
    for metric in result.metrics:
        lines.append(
            f"| {metric.display_name} | {metric.classifier_name} | {metric.validation_macro_f1:.4f} | "
            f"{metric.validation_balanced_accuracy:.4f} | {metric.recommendation} |"
        )
    lines.extend(
        [
            "",
            "## 2. Agent 解释",
            "",
            "当前对比使用同一份窗口特征表和 cut 级别 split，因此结果可横向比较。",
            "候选排序只读取 validation 指标；test 在方案冻结前不会进入本报告，也不会参与推荐。",
            "如果多个树模型指标都接近满分，说明 C1 内部统计特征已经很强；下一步重点不应继续刷 C1 分数，而应做跨刀具验证或 CNN 对照。",
            "",
        ]
    )
    return "\n".join(lines)


def run_c1_candidate_compare(settings: Settings, plan_ids: tuple[str, ...] = SUPPORTED_COMPARE_PLAN_IDS) -> CandidateCompareResult:
    """执行 C1 多候选训练对比。"""

    source_run_dir = find_latest_decided_run(settings.experiment_root)
    feature_table_file = source_run_dir / "feature_table.csv"
    x_values, y_values, splits = load_feature_table(feature_table_file)
    metrics = [train_compare_candidate(plan_id, x_values, y_values, splits, settings.random_seed) for plan_id in plan_ids]
    best = select_best_candidate(metrics)

    compare_run_id = f"phm2010_c1_candidate_compare_{_now_shanghai_compact()}"
    compare_dir = settings.experiment_root / compare_run_id
    result_json = compare_dir / "candidate_compare_result.json"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_candidate_compare_report.md"
    log_file = settings.log_root / "phm2010_c1_candidate_compare.log"
    result = CandidateCompareResult(
        compare_run_id=compare_run_id,
        source_run_dir=str(source_run_dir),
        metrics=metrics,
        best_plan_id=best.plan_id,
        report_file=str(report_file),
        result_json=str(result_json),
        log_file=str(log_file),
    )
    compare_dir.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_compare_report(result), encoding="utf-8")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 多候选训练对比日志",
                f"compare_run_id: {compare_run_id}",
                f"source_run_dir: {source_run_dir}",
                f"plan_ids: {', '.join(plan_ids)}",
                f"best_plan_id: {best.plan_id}",
                f"result_json: {result_json}",
                f"report_file: {report_file}",
            ]
        ),
        encoding="utf-8",
    )
    return result
