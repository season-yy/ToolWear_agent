"""训练结果可视化与证据报告。

本模块对应 P0 第 6 步：把训练产生的 metrics.json 和 feature_table.csv
整理成图表、表格和 Markdown 报告，方便比赛展示和后续 Agent 诊断。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from toolwear_agent.common.config import Settings
from toolwear_agent.data.splitting import normalize_split_name


STAGE_ID_TO_NAME = {
    0: "初期磨损",
    1: "正常磨损",
    2: "剧烈磨损",
    3: "失效磨损",
}


@dataclass(frozen=True)
class VisualReportResult:
    """第 6 步可视化报告产物索引。"""

    run_id: str
    run_dir: str
    figure_dir: str
    report_file: str
    metrics_summary_file: str
    validation_report_csv: str
    validation_confusion_matrix_png: str
    stage_distribution_png: str
    split_distribution_png: str
    tsne_png: str
    loss_curve_note_file: str
    final_test_status: str


def find_latest_window_run(experiment_root: Path) -> Path:
    """查找最新的 C1 窗口小范围训练目录。"""

    runs = [
        item
        for item in experiment_root.glob("phm2010_c1_window_mini_train_*")
        if item.is_dir() and (item / "metrics.json").exists() and (item / "feature_table.csv").exists()
    ]
    if not runs:
        raise FileNotFoundError(f"未找到窗口训练运行目录: {experiment_root}")
    return max(runs, key=lambda item: item.stat().st_mtime)


def load_metrics(metrics_file: Path) -> dict[str, object]:
    """读取训练指标 JSON。"""

    return json.loads(metrics_file.read_text(encoding="utf-8"))


def classification_report_to_rows(report: dict[str, object]) -> list[dict[str, object]]:
    """把 sklearn 的 classification_report 字典转换成 CSV 行。"""

    rows: list[dict[str, object]] = []
    for raw_label, values in report.items():
        if raw_label == "accuracy":
            rows.append(
                {
                    "label": "accuracy",
                    "stage_name": "整体准确率",
                    "precision": "",
                    "recall": "",
                    "f1_score": values,
                    "support": "",
                }
            )
            continue
        if not isinstance(values, dict):
            continue
        stage_name = STAGE_ID_TO_NAME.get(int(raw_label), raw_label) if raw_label.isdigit() else raw_label
        rows.append(
            {
                "label": raw_label,
                "stage_name": stage_name,
                "precision": values.get("precision", ""),
                "recall": values.get("recall", ""),
                "f1_score": values.get("f1-score", ""),
                "support": values.get("support", ""),
            }
        )
    return rows


def write_rows_csv(rows: list[dict[str, object]], output_file: Path) -> Path:
    """写出结构化 CSV 表格。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "stage_name", "precision", "recall", "f1_score", "support"]
    with output_file.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_file


def _prepare_matplotlib() -> None:
    """配置 matplotlib 后端和中文字体。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(matrix: list[list[int]], title: str, output_file: Path) -> Path:
    """绘制混淆矩阵图。"""

    _prepare_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(matrix)
    labels = [STAGE_ID_TO_NAME[index] for index in range(data.shape[0])]
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=160)
    image = ax.imshow(data, cmap="Blues")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("预测阶段")
    ax.set_ylabel("真实阶段")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    threshold = data.max() / 2 if data.size else 0
    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            value = data[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#1f2937",
                fontsize=10,
            )
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_bar_chart(counts: dict[str, int], title: str, output_file: Path) -> Path:
    """绘制分布柱状图。"""

    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    bars = ax.bar(labels, values, color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"][: len(labels)])
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("样本数")
    ax.tick_params(axis="x", labelrotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _read_feature_table(feature_table_file: Path) -> tuple[list[list[float]], list[int], list[str], list[str]]:
    """读取特征表中的数值特征、阶段标签、split 和窗口 ID。"""

    feature_values: list[list[float]] = []
    stage_ids: list[int] = []
    splits: list[str] = []
    window_ids: list[str] = []
    with feature_table_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames or []
        feature_names = [name for name in fieldnames if name.endswith(("_mean", "_std", "_min", "_max", "_ptp", "_rms"))]
        for row in reader:
            feature_values.append([float(row[name]) for name in feature_names])
            stage_ids.append(int(row["stage_id"]))
            splits.append(normalize_split_name(row["split"]))
            window_ids.append(row["window_id"])
    return feature_values, stage_ids, splits, window_ids


def plot_tsne(feature_table_file: Path, output_file: Path, random_seed: int = 42) -> Path:
    """绘制 t-SNE 特征降维图。

    sklearn 官方 TSNE 文档：https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """

    _prepare_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    feature_values, stage_ids, splits, _window_ids = _read_feature_table(feature_table_file)
    x_values = StandardScaler().fit_transform(np.array(feature_values))
    # 当前小范围样本 2021 条，t-SNE 可直接运行。perplexity 取 30 是常见稳健默认值。
    coordinates = TSNE(n_components=2, perplexity=30, random_state=random_seed, init="pca", learning_rate="auto").fit_transform(
        x_values
    )

    colors = {0: "#2563eb", 1: "#16a34a", 2: "#f59e0b", 3: "#dc2626"}
    markers = {"train": "o", "validation": "^"}
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    for split in ["train", "validation"]:
        for stage_id in sorted(set(stage_ids)):
            indices = [index for index, value in enumerate(stage_ids) if value == stage_id and splits[index] == split]
            if not indices:
                continue
            ax.scatter(
                coordinates[indices, 0],
                coordinates[indices, 1],
                s=14,
                alpha=0.72,
                marker=markers.get(split, "o"),
                color=colors.get(stage_id, "#6b7280"),
                label=f"{split}-{STAGE_ID_TO_NAME.get(stage_id, stage_id)}",
            )
    ax.set_title("t-SNE 特征分布图", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=7, ncols=2)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    return output_file


def write_metrics_summary(metrics: dict[str, object], output_file: Path) -> Path:
    """保存适合前端和报告读取的指标摘要。"""

    validation = metrics["validation"]
    summary = {
        "run_id": metrics["run_id"],
        "full_window_count": metrics["full_window_count"],
        "sample_count": metrics["sample_count"],
        "sample_fraction": metrics["sample_fraction"],
        "train_count": metrics["train_count"],
        "validation_count": validation["count"],
        "validation_macro_f1": validation["macro_f1"],
        "validation_balanced_accuracy": validation["balanced_accuracy"],
        "validation_accuracy": validation["accuracy"],
        "final_test_status": metrics.get("final_test_status", "not_run_pipeline_not_frozen"),
        "stage_distribution": metrics["stage_distribution"],
        "split_distribution": metrics["split_distribution"],
        "note": "RandomForest 不产生 epoch loss 曲线，loss 曲线将在 CNN 等深度学习模型训练时输出。",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def write_loss_curve_note(output_file: Path) -> Path:
    """写出 loss 曲线说明，避免伪造不存在的训练曲线。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(
            [
                "# Loss 曲线说明",
                "",
                "当前第 5 步使用的是 RandomForestClassifier。",
                "该模型不是按 epoch 迭代优化的神经网络，因此不会产生训练 loss / 验证 loss 曲线。",
                "",
                "为了保证证据真实，本步骤不伪造 loss 曲线。",
                "后续接入 1D CNN、多分支 CNN 或迁移学习模型后，会输出真实的 epoch loss 曲线。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_file


def _markdown_image(path: Path) -> str:
    """把图片路径转换成 Markdown 可读文本。"""

    return f"`{path}`"


def write_visual_markdown_report(result: VisualReportResult, metrics: dict[str, object], output_file: Path) -> Path:
    """生成第 6 步 Markdown 可视化报告。"""

    validation = metrics["validation"]
    lines = [
        "# PHM2010 C1 窗口训练可视化报告",
        "",
        "## 1. 报告结论",
        "",
        f"- 运行编号：`{result.run_id}`",
        f"- 全量窗口样本数：{metrics['full_window_count']}",
        f"- 本次小范围训练样本数：{metrics['sample_count']}",
        f"- 验证集 Macro-F1：{validation['macro_f1']:.4f}",
        f"- 验证集 Balanced Accuracy：{validation['balanced_accuracy']:.4f}",
        "- 最终测试集：未执行（当前仍处于候选训练阶段）",
        "",
        "## 2. 图表产物",
        "",
        f"- 验证集混淆矩阵：{_markdown_image(Path(result.validation_confusion_matrix_png))}",
        f"- 阶段分布图：{_markdown_image(Path(result.stage_distribution_png))}",
        f"- split 分布图：{_markdown_image(Path(result.split_distribution_png))}",
        f"- t-SNE 图：{_markdown_image(Path(result.tsne_png))}",
        "",
        "## 3. 表格产物",
        "",
        f"- 验证集分类报告：`{result.validation_report_csv}`",
        f"- 指标摘要：`{result.metrics_summary_file}`",
        "",
        "## 4. Loss 曲线说明",
        "",
        "当前方案是 RandomForest，不产生 epoch loss 曲线。本步骤保存了说明文件，后续 CNN 训练时输出真实 loss 曲线。",
        "",
        f"- Loss 曲线说明：`{result.loss_curve_note_file}`",
        "",
        "## 5. 风险说明",
        "",
        "当前结果来自 PHM2010 C1 内部 cut 级别划分，可以作为 P0 初赛闭环证据。",
        "候选判断只使用 validation；test 会在方案和参数冻结后通过独立最终评估执行。",
        "当前结果也不等价于跨刀具、跨工况泛化能力。后续 C1/C4/C6 跨刀具验证仍然是更关键的检验。",
        "",
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def run_c1_visual_report(settings: Settings, run_dir: Path | None = None) -> VisualReportResult:
    """执行 PHM2010 C1 第 6 步可视化报告生成。"""

    selected_run_dir = run_dir or find_latest_window_run(settings.experiment_root)
    metrics_file = selected_run_dir / "metrics.json"
    feature_table_file = selected_run_dir / "feature_table.csv"
    metrics = load_metrics(metrics_file)
    run_id = str(metrics["run_id"])
    figure_dir = selected_run_dir / "figures"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_visual_report.md"
    metrics_summary_file = selected_run_dir / "metrics_summary.json"
    validation_report_csv = selected_run_dir / "classification_report_validation.csv"
    validation_confusion_matrix_png = figure_dir / "confusion_matrix_validation.png"
    stage_distribution_png = figure_dir / "stage_distribution.png"
    split_distribution_png = figure_dir / "split_distribution.png"
    tsne_png = figure_dir / "tsne_feature_distribution.png"
    loss_curve_note_file = selected_run_dir / "loss_curve_note.md"

    write_rows_csv(classification_report_to_rows(metrics["validation"]["classification_report"]), validation_report_csv)
    write_metrics_summary(metrics, metrics_summary_file)
    plot_confusion_matrix(metrics["validation"]["confusion_matrix"], "验证集混淆矩阵", validation_confusion_matrix_png)
    plot_bar_chart(metrics["stage_distribution"], "小范围训练阶段分布", stage_distribution_png)
    plot_bar_chart(metrics["split_distribution"], "小范围训练 split 分布", split_distribution_png)
    plot_tsne(feature_table_file, tsne_png, random_seed=42)
    write_loss_curve_note(loss_curve_note_file)

    result = VisualReportResult(
        run_id=run_id,
        run_dir=str(selected_run_dir),
        figure_dir=str(figure_dir),
        report_file=str(report_file),
        metrics_summary_file=str(metrics_summary_file),
        validation_report_csv=str(validation_report_csv),
        validation_confusion_matrix_png=str(validation_confusion_matrix_png),
        stage_distribution_png=str(stage_distribution_png),
        split_distribution_png=str(split_distribution_png),
        tsne_png=str(tsne_png),
        loss_curve_note_file=str(loss_curve_note_file),
        final_test_status=str(metrics.get("final_test_status", "not_run_pipeline_not_frozen")),
    )
    write_visual_markdown_report(result, metrics, report_file)
    (selected_run_dir / "visual_report_manifest.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
