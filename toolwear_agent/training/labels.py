"""磨损标签生成工具。

本模块对应初赛 P0 的第 2 步：
把 PHM2010 的三刀刃 VB 磨损值转换成模型训练可用的标签表。

当前默认策略：
- VB 聚合方式：`max`，也就是三刀刃取最大磨损值
- 四阶段阈值：90/130/160 um
- 阶段名称：初期磨损、正常磨损、剧烈磨损、失效磨损

注意：
这些阈值是 Demo 默认值，不代表所有刀具磨损任务的通用标准。
后续会继续做成用户可配置项，并在页面中展示给用户确认。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from toolwear_agent.training.phm2010 import WearLabel, load_wear_labels


DEFAULT_STAGE_NAMES = ("初期磨损", "正常磨损", "剧烈磨损", "失效磨损")


@dataclass(frozen=True)
class WearStage:
    """单个 VB 数值对应的离散阶段。"""

    stage_id: int
    stage_name: str


@dataclass(frozen=True)
class WearLabelRecord:
    """单个刀次的最终标签记录。

    这条记录会被后续特征提取、小样本训练和报告模块复用。
    """

    cut: int
    flute_1: float
    flute_2: float
    flute_3: float
    vb_aggregation: str
    vb_value: float
    stage_id: int
    stage_name: str


@dataclass(frozen=True)
class LabelDataset:
    """某把刀具的标签生成结果。"""

    dataset_id: str
    cutter: str
    wear_file: str
    aggregation: str
    thresholds_um: tuple[float, float, float]
    stage_names: tuple[str, str, str, str]
    record_count: int
    stage_distribution: dict[str, int]
    records: list[WearLabelRecord]


def compute_vb_value(
    label: WearLabel,
    aggregation: str = "max",
    *,
    specified_flute: int | None = None,
) -> float:
    """根据指定策略聚合三刀刃 VB。

    当前支持：
    - `max`：取三刀刃最大值，作为本 Demo 默认策略
    - `mean`：取三刀刃平均值，后续可在页面中开放给用户选择
    """

    values = (label.flute_1, label.flute_2, label.flute_3)
    if aggregation == "max":
        return max(values)
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "specified_flute":
        if specified_flute not in {1, 2, 3}:
            raise ValueError("specified_flute 聚合必须指定 1-3 号刀刃。")
        return values[specified_flute - 1]
    raise ValueError(f"不支持的 VB 聚合方式: {aggregation}")


def validate_thresholds(thresholds: tuple[float, ...]) -> tuple[float, float, float]:
    """校验四阶段分类所需的三个阈值。

    三个阈值必须严格递增，否则阶段边界会含糊。
    """

    if len(thresholds) != 3:
        raise ValueError("四阶段标签生成必须提供 3 个阈值")
    first, second, third = thresholds
    if not first < second < third:
        raise ValueError("四阶段阈值必须严格递增")
    return first, second, third


def assign_wear_stage(
    vb_value: float,
    thresholds: tuple[float, ...],
    stage_names: tuple[str, str, str, str] = DEFAULT_STAGE_NAMES,
) -> WearStage:
    """把连续 VB 值映射为四阶段磨损标签。

    边界规则采用左闭右开：
    - vb < 90：初期磨损
    - 90 <= vb < 130：正常磨损
    - 130 <= vb < 160：剧烈磨损
    - vb >= 160：失效磨损
    """

    first, second, third = validate_thresholds(thresholds)
    if len(stage_names) != 4:
        raise ValueError("四阶段标签必须提供 4 个阶段名称")

    if vb_value < first:
        return WearStage(stage_id=0, stage_name=stage_names[0])
    if vb_value < second:
        return WearStage(stage_id=1, stage_name=stage_names[1])
    if vb_value < third:
        return WearStage(stage_id=2, stage_name=stage_names[2])
    return WearStage(stage_id=3, stage_name=stage_names[3])


def build_label_records(
    labels: dict[int, WearLabel],
    aggregation: str,
    thresholds: tuple[float, ...],
    stage_names: tuple[str, str, str, str] = DEFAULT_STAGE_NAMES,
    specified_flute: int | None = None,
) -> list[WearLabelRecord]:
    """把原始磨损标签转换成训练可用的标签记录。"""

    records: list[WearLabelRecord] = []
    for cut in sorted(labels):
        label = labels[cut]
        vb_value = compute_vb_value(
            label,
            aggregation=aggregation,
            specified_flute=specified_flute,
        )
        stage = assign_wear_stage(vb_value, thresholds=thresholds, stage_names=stage_names)
        records.append(
            WearLabelRecord(
                cut=cut,
                flute_1=label.flute_1,
                flute_2=label.flute_2,
                flute_3=label.flute_3,
                vb_aggregation=aggregation,
                vb_value=vb_value,
                stage_id=stage.stage_id,
                stage_name=stage.stage_name,
            )
        )
    return records


def summarize_stage_distribution(
    records: list[WearLabelRecord],
    stage_names: tuple[str, str, str, str] = DEFAULT_STAGE_NAMES,
) -> dict[str, int]:
    """统计四阶段标签数量。

    即使某个阶段数量为 0，也会保留在结果中，方便报告和前端展示。
    """

    distribution = {stage_name: 0 for stage_name in stage_names}
    for record in records:
        distribution[record.stage_name] = distribution.get(record.stage_name, 0) + 1
    return distribution


def build_label_dataset(
    wear_file: Path,
    cutter: str,
    aggregation: str,
    thresholds: tuple[float, ...],
    stage_names: tuple[str, str, str, str] = DEFAULT_STAGE_NAMES,
    specified_flute: int | None = None,
    dataset_id: str = "phm2010",
) -> LabelDataset:
    """从磨损标签文件构建完整标签数据集。"""

    validated_thresholds = validate_thresholds(thresholds)
    raw_labels = load_wear_labels(wear_file)
    records = build_label_records(
        raw_labels,
        aggregation=aggregation,
        thresholds=validated_thresholds,
        stage_names=stage_names,
        specified_flute=specified_flute,
    )
    return LabelDataset(
        dataset_id=dataset_id,
        cutter=cutter,
        wear_file=str(wear_file),
        aggregation=aggregation,
        thresholds_um=validated_thresholds,
        stage_names=stage_names,
        record_count=len(records),
        stage_distribution=summarize_stage_distribution(records, stage_names),
        records=records,
    )


def write_label_json(label_dataset: LabelDataset, output_file: Path) -> Path:
    """写出 JSON 标签文件。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(asdict(label_dataset), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def write_label_csv(label_dataset: LabelDataset, output_file: Path) -> Path:
    """写出 CSV 标签文件。

    CSV 更方便后续用 pandas、Excel 或其他工具快速查看。
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "cut",
                "flute_1",
                "flute_2",
                "flute_3",
                "vb_aggregation",
                "vb_value",
                "stage_id",
                "stage_name",
            ],
        )
        writer.writeheader()
        for record in label_dataset.records:
            writer.writerow(asdict(record))
    return output_file


def render_label_report(label_dataset: LabelDataset) -> str:
    """生成标签报告 Markdown。"""

    lines = [
        f"# PHM2010 {label_dataset.cutter.upper()} 四阶段标签生成报告",
        "",
        "## 1. 标签生成策略",
        "",
        f"- VB 聚合方式：`{label_dataset.aggregation}`",
        f"- 阈值：{label_dataset.thresholds_um[0]}/{label_dataset.thresholds_um[1]}/{label_dataset.thresholds_um[2]} um",
        "- 阶段规则：左闭右开，最后一段包含所有大于等于最后阈值的样本",
        "",
        "## 2. 阶段定义",
        "",
        "| stage_id | stage_name | VB 范围 |",
        "|---:|---|---|",
        f"| 0 | {label_dataset.stage_names[0]} | VB < {label_dataset.thresholds_um[0]} |",
        f"| 1 | {label_dataset.stage_names[1]} | {label_dataset.thresholds_um[0]} <= VB < {label_dataset.thresholds_um[1]} |",
        f"| 2 | {label_dataset.stage_names[2]} | {label_dataset.thresholds_um[1]} <= VB < {label_dataset.thresholds_um[2]} |",
        f"| 3 | {label_dataset.stage_names[3]} | VB >= {label_dataset.thresholds_um[2]} |",
        "",
        "## 3. 标签分布",
        "",
        "| 阶段 | 数量 |",
        "|---|---:|",
    ]

    for stage_name in label_dataset.stage_names:
        lines.append(f"| {stage_name} | {label_dataset.stage_distribution.get(stage_name, 0)} |")

    lines.extend(
        [
            "",
            "## 4. 前 10 条标签样例",
            "",
            "| 刀次 | flute_1 | flute_2 | flute_3 | VB max | stage_id | stage_name |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for record in label_dataset.records[:10]:
        lines.append(
            f"| {record.cut} | {record.flute_1:.3f} | {record.flute_2:.3f} | "
            f"{record.flute_3:.3f} | {record.vb_value:.3f} | "
            f"{record.stage_id} | {record.stage_name} |"
        )

    lines.extend(
        [
            "",
            "## 5. 对下一步的意义",
            "",
            "本报告将连续 VB 磨损值转换成四阶段分类标签。下一步可以基于这些标签，"
            "生成 2-3 个候选算法方案，并让用户确认其中一个方案进入小样本训练。",
            "",
        ]
    )

    return "\n".join(lines)


def write_label_report(label_dataset: LabelDataset, output_file: Path) -> Path:
    """写出 Markdown 标签生成报告。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_label_report(label_dataset), encoding="utf-8")
    return output_file
