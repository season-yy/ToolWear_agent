"""PHM2010 滑窗样本索引构建。

这个模块负责把“一个刀次 CSV 文件”切成多个窗口样本索引。
注意：这里不把窗口另存成大量 CSV，而是保存 file_path + start_row + end_row。
训练时按索引回到原始 CSV 读取窗口，这样更省空间，也更容易追溯。
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from toolwear_agent.data.leakage import (
    assert_no_window_leakage,
    assert_windows_match_split_manifest,
    write_leakage_audit,
)
from toolwear_agent.data.sampling import build_training_sample
from toolwear_agent.data.splitting import (
    attach_split_hash,
    build_split_manifest as build_locked_split_manifest,
    create_or_verify_split_lock,
    normalize_split_name,
    write_split_manifest as write_split_manifest_json,
)
from toolwear_agent.schemas import SplitSpec


DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_MAX_WINDOWS_PER_CUT = 32
DEFAULT_TRAIN_RATIO = 0.60
DEFAULT_VAL_RATIO = 0.20
DEFAULT_TEST_RATIO = 0.20
DEFAULT_SPLIT_RANDOM_SEED = 42


@dataclass(frozen=True)
class CutLabel:
    """单个刀次的标签和原始文件信息。"""

    cut: int
    file_path: str
    row_count: int
    vb_value: float
    stage_id: int
    stage_name: str


@dataclass(frozen=True)
class WindowRecord:
    """单个窗口样本的索引记录。"""

    window_id: str
    cut: int
    file_path: str
    row_count: int
    start_row: int
    end_row: int
    window_size: int
    stride: int
    overlap_ratio: float
    vb_value: float
    stage_id: int
    stage_name: str
    split: str


@dataclass(frozen=True)
class WindowBuildResult:
    """窗口索引构建结果。"""

    dataset_id: str
    cutter: str
    window_size: int
    stride: int
    overlap_ratio: float
    max_windows_per_cut: int
    split_random_seed: int
    cut_count: int
    window_count: int
    split_distribution: dict[str, int]
    stage_distribution: dict[str, int]
    group_distribution: dict[str, int]
    split_manifest_file: str
    split_manifest_json_file: str
    split_hash: str
    split_lock_file: str
    leakage_audit_file: str
    window_manifest_file: str
    report_file: str
    log_file: str


def overlap_to_stride(window_size: int, overlap_ratio: float) -> int:
    """把重叠率转换成滑窗步长。

    例如窗口长度 4096、重叠率 0.5，则步长为 2048。
    """

    if window_size <= 0:
        raise ValueError("window_size 必须大于 0")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio 必须位于 [0, 1) 区间")
    stride = int(window_size * (1 - overlap_ratio))
    return max(stride, 1)


def count_csv_rows(signal_file: Path) -> int:
    """统计单个信号 CSV 的行数。"""

    with signal_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return sum(1 for _ in file_obj)


def load_cut_labels(label_file: Path, cutter_dir: Path, cutter: str) -> list[CutLabel]:
    """读取阶段标签，并补充每个刀次对应的信号文件路径和行数。"""

    cutter_number = cutter.lower().replace("c", "")
    records: list[CutLabel] = []
    with label_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            cut = int(row["cut"])
            signal_file = cutter_dir / f"c_{cutter_number}_{cut:03d}.csv"
            records.append(
                CutLabel(
                    cut=cut,
                    file_path=str(signal_file),
                    row_count=count_csv_rows(signal_file),
                    vb_value=float(row["vb_value"]),
                    stage_id=int(row["stage_id"]),
                    stage_name=str(row["stage_name"]),
                )
            )
    return records


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """按阶段内数量计算 train/val/test 的 cut 数。

    阶段样本较少时，仍尽量保证验证集和测试集至少有一个 cut。
    """

    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    test_count = total - train_count - val_count
    if total >= 3:
        if val_count == 0:
            val_count = 1
            train_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1
    return train_count, val_count, test_count


def assign_cut_splits(
    cut_labels: Iterable[CutLabel],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    random_seed: int = DEFAULT_SPLIT_RANDOM_SEED,
) -> dict[int, str]:
    """按磨损阶段分层划分 cut，避免同一个 cut 泄露到多个集合。

    这里先划分 cut，再生成窗口。这样同一个 CSV 切出的所有窗口只会属于同一个 split。
    """

    by_stage: dict[int, list[CutLabel]] = {}
    for label in cut_labels:
        by_stage.setdefault(label.stage_id, []).append(label)

    split_by_cut: dict[int, str] = {}
    for stage_id in sorted(by_stage):
        stage_records = sorted(by_stage[stage_id], key=lambda item: item.cut)
        stage_random = random.Random(random_seed + stage_id)
        stage_random.shuffle(stage_records)
        train_count, val_count, _test_count = _split_counts(len(stage_records), train_ratio, val_ratio)
        for index, label in enumerate(stage_records):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "validation"
            else:
                split = "test"
            split_by_cut[label.cut] = split
    return split_by_cut


def all_window_starts(row_count: int, window_size: int, stride: int) -> list[int]:
    """生成一个 CSV 文件内所有可用窗口起点。"""

    if row_count < window_size:
        return []
    return list(range(0, row_count - window_size + 1, stride))


def uniformly_pick_starts(starts: list[int], limit: int) -> list[int]:
    """从所有候选窗口中均匀抽取固定数量，覆盖整段加工过程。"""

    if limit <= 0:
        raise ValueError("limit 必须大于 0")
    if len(starts) <= limit:
        return starts
    if limit == 1:
        return [starts[0]]

    picked_indices = [round(index * (len(starts) - 1) / (limit - 1)) for index in range(limit)]
    return [starts[index] for index in picked_indices]


def build_window_records(
    cut_labels: list[CutLabel],
    split_by_cut: dict[int, str],
    cutter: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    max_windows_per_cut: int = DEFAULT_MAX_WINDOWS_PER_CUT,
) -> list[WindowRecord]:
    """为所有 cut 生成窗口索引记录。"""

    stride = overlap_to_stride(window_size, overlap_ratio)
    records: list[WindowRecord] = []
    for label in sorted(cut_labels, key=lambda item: item.cut):
        starts = all_window_starts(label.row_count, window_size, stride)
        picked_starts = uniformly_pick_starts(starts, max_windows_per_cut)
        for window_index, start_row in enumerate(picked_starts):
            records.append(
                WindowRecord(
                    window_id=f"{cutter.lower()}_{label.cut:03d}_w{window_index:03d}",
                    cut=label.cut,
                    file_path=label.file_path,
                    row_count=label.row_count,
                    start_row=start_row,
                    end_row=start_row + window_size,
                    window_size=window_size,
                    stride=stride,
                    overlap_ratio=overlap_ratio,
                    vb_value=label.vb_value,
                    stage_id=label.stage_id,
                    stage_name=label.stage_name,
                    split=normalize_split_name(split_by_cut[label.cut]),
                )
            )
    return records


def write_split_manifest(cut_labels: list[CutLabel], split_by_cut: dict[int, str], output_file: Path) -> Path:
    """保存 cut 级别的数据集划分。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["cut", "file_path", "row_count", "vb_value", "stage_id", "stage_name", "split"])
        for label in sorted(cut_labels, key=lambda item: item.cut):
            writer.writerow(
                [
                    label.cut,
                    label.file_path,
                    label.row_count,
                    label.vb_value,
                    label.stage_id,
                    label.stage_name,
                    split_by_cut[label.cut],
                ]
            )
    return output_file


def write_window_manifest(records: list[WindowRecord], output_file: Path) -> Path:
    """保存窗口级别的数据索引。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(asdict(records[0]).keys()) if records else []
    with output_file.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return output_file


def load_window_manifest(manifest_file: Path) -> list[WindowRecord]:
    """读取窗口 manifest。"""

    records: list[WindowRecord] = []
    with manifest_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            records.append(
                WindowRecord(
                    window_id=row["window_id"],
                    cut=int(row["cut"]),
                    file_path=row["file_path"],
                    row_count=int(row["row_count"]),
                    start_row=int(row["start_row"]),
                    end_row=int(row["end_row"]),
                    window_size=int(row["window_size"]),
                    stride=int(row["stride"]),
                    overlap_ratio=float(row["overlap_ratio"]),
                    vb_value=float(row["vb_value"]),
                    stage_id=int(row["stage_id"]),
                    stage_name=row["stage_name"],
                    split=normalize_split_name(row["split"]),
                )
            )
    return records


def _count_by(records: Iterable[WindowRecord], field_name: str) -> dict[str, int]:
    """按指定字段统计窗口数量。"""

    counts: dict[str, int] = {}
    for record in records:
        value = str(getattr(record, field_name))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_by_split_stage(records: Iterable[WindowRecord]) -> dict[str, int]:
    """统计 split + stage 的组合分布。"""

    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.split}:{record.stage_name}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def validate_no_cut_leakage(records: Iterable[WindowRecord]) -> None:
    """检查同一个 cut 是否只出现在一个 split 中。"""

    assert_no_window_leakage(records)


def write_window_report(result: WindowBuildResult, output_file: Path) -> Path:
    """写出窗口构建报告。"""

    lines = [
        "# PHM2010 C1 窗口样本构建报告",
        "",
        "## 1. 参数",
        "",
        f"- window_size：{result.window_size}",
        f"- stride：{result.stride}",
        f"- overlap_ratio：{result.overlap_ratio}",
        f"- max_windows_per_cut：{result.max_windows_per_cut}",
        f"- split_random_seed：{result.split_random_seed}",
        f"- split_hash：`{result.split_hash}`",
        "",
        "## 2. 样本规模",
        "",
        f"- cut 数量：{result.cut_count}",
        f"- 窗口样本数量：{result.window_count}",
        "",
        "## 3. split 分布",
        "",
    ]
    lines.extend(f"- {key}：{value}" for key, value in result.split_distribution.items())
    lines.extend(["", "## 4. 阶段分布", ""])
    lines.extend(f"- {key}：{value}" for key, value in result.stage_distribution.items())
    lines.extend(["", "## 5. split + 阶段分布", ""])
    lines.extend(f"- {key}：{value}" for key, value in result.group_distribution.items())
    lines.extend(
        [
            "",
            "## 6. 产物索引",
            "",
            f"- cut 划分文件：`{result.split_manifest_file}`",
            f"- cut 划分 JSON：`{result.split_manifest_json_file}`",
            f"- split lock：`{result.split_lock_file}`",
            f"- 泄漏审计：`{result.leakage_audit_file}`",
            f"- 窗口索引文件：`{result.window_manifest_file}`",
            f"- 日志文件：`{result.log_file}`",
            "",
        ]
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def build_c1_window_manifest(
    label_file: Path,
    cutter_dir: Path,
    output_root: Path,
    report_file: Path,
    log_file: Path,
    cutter: str = "c1",
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    max_windows_per_cut: int = DEFAULT_MAX_WINDOWS_PER_CUT,
    split_random_seed: int = DEFAULT_SPLIT_RANDOM_SEED,
    split_lock_file: Path | None = None,
    experiment_id: str = "phm2010_c1_p0",
    revision: int = 1,
) -> WindowBuildResult:
    """构建 PHM2010 C1 固定窗口索引、泄漏审计和修订级 split lock。"""

    split_manifest_file = output_root / "phm2010_c1_split_manifest.csv"
    split_manifest_json_file = output_root / "phm2010_c1_split_manifest.json"
    window_manifest_file = output_root / "phm2010_c1_window_manifest.csv"
    leakage_audit_file = output_root / "phm2010_c1_leakage_audit.json"
    resolved_lock_file = split_lock_file or output_root / "phm2010_c1_split_lock.json"
    cut_labels = load_cut_labels(label_file, cutter_dir, cutter)
    split_spec = SplitSpec(
        train_ratio=DEFAULT_TRAIN_RATIO,
        validation_ratio=DEFAULT_VAL_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
        random_seed=split_random_seed,
    )
    split_by_cut = assign_cut_splits(
        cut_labels,
        train_ratio=split_spec.train_ratio,
        val_ratio=split_spec.validation_ratio,
        random_seed=split_spec.random_seed,
    )
    records = build_window_records(
        cut_labels=cut_labels,
        split_by_cut=split_by_cut,
        cutter=cutter,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        max_windows_per_cut=max_windows_per_cut,
    )
    leakage_audit = assert_no_window_leakage(records)
    split_manifest = attach_split_hash(
        build_locked_split_manifest(
            cut_labels=cut_labels,
            split_by_cut=split_by_cut,
            dataset_id="phm2010",
            cutter_id=cutter.lower(),
            split_spec=split_spec,
        )
    )
    if split_manifest.split_hash is None:  # pragma: no cover - attach 后的类型防御
        raise ValueError("split_hash 不能为空。")
    assert_windows_match_split_manifest(records, split_manifest)
    # 先验证已有 lock，再写 Manifest；冲突时不能覆盖仍被旧 lock 引用的文件。
    create_or_verify_split_lock(
        manifest=split_manifest,
        lock_file=resolved_lock_file,
        experiment_id=experiment_id,
        revision=revision,
        manifest_file=split_manifest_json_file,
    )
    write_split_manifest_json(split_manifest, split_manifest_json_file)
    write_leakage_audit(leakage_audit, leakage_audit_file)
    write_split_manifest(cut_labels, split_by_cut, split_manifest_file)
    write_window_manifest(records, window_manifest_file)

    result = WindowBuildResult(
        dataset_id="phm2010",
        cutter=cutter,
        window_size=window_size,
        stride=overlap_to_stride(window_size, overlap_ratio),
        overlap_ratio=overlap_ratio,
        max_windows_per_cut=max_windows_per_cut,
        split_random_seed=split_random_seed,
        cut_count=len(cut_labels),
        window_count=len(records),
        split_distribution=_count_by(records, "split"),
        stage_distribution=_count_by(records, "stage_name"),
        group_distribution=_count_by_split_stage(records),
        split_manifest_file=str(split_manifest_file),
        split_manifest_json_file=str(split_manifest_json_file),
        split_hash=split_manifest.split_hash,
        split_lock_file=str(resolved_lock_file),
        leakage_audit_file=str(leakage_audit_file),
        window_manifest_file=str(window_manifest_file),
        report_file=str(report_file),
        log_file=str(log_file),
    )
    write_window_report(result, report_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def stratified_fraction_sample(
    records: Iterable[WindowRecord],
    fraction: float,
    minimum_per_group: int = 1,
    random_seed: int = DEFAULT_SPLIT_RANDOM_SEED,
) -> list[WindowRecord]:
    """兼容旧入口：仅从 train 中执行阶段/cut/时间感知抽样。

    `minimum_per_group` 为旧参数，保留它只为避免调用方报错；新算法始终保证每个
    可用阶段至少一个窗口，并在预算允许时覆盖该阶段的全部 cut。
    """

    if minimum_per_group < 1:
        raise ValueError("minimum_per_group 必须大于等于 1。")
    materialized = list(records)
    cutter_id = materialized[0].window_id.split("_", 1)[0] if materialized else "c1"
    sample = build_training_sample(
        materialized,
        dataset_id="phm2010",
        cutter_id=cutter_id,
        split_hash="0" * 64,
        fraction=fraction,
        random_seed=random_seed,
    )
    return list(sample.records)
