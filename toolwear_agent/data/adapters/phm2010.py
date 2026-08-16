"""PHM2010 C1-C6 的通用只读数据集 Adapter。"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path

from toolwear_agent.data.adapters.base import DatasetAdapter
from toolwear_agent.data.manifest import attach_manifest_hash
from toolwear_agent.schemas import (
    CutterManifest,
    DatasetInspection,
    DatasetManifest,
    ValidationIssue,
    ValidationResult,
)


PHM2010_CUTTER_IDS = tuple(f"C{number}" for number in range(1, 7))
PHM2010_LABELED_CUTTERS = frozenset({"C1", "C4", "C6"})
PHM2010_CHANNELS = (
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "acoustic_emission_rms",
)


def _sha256_file(path: Path) -> str:
    """流式计算小型标签文件哈希，不把整个文件一次读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_hash(files: list[Path]) -> str | None:
    """根据完整文件名和大小生成轻量清单指纹。

    原始信号总体积很大，登记阶段不逐字节重算所有 CSV。文件名和大小足以发现
    缺失、重复或截断等常见变化，真正进入 Run 后仍会保存 split 和产物哈希。
    """

    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.name}\0{path.stat().st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def _signal_files(cutter_dir: Path, cutter_number: int) -> list[Path]:
    """返回严格匹配当前刀具编号并按 cut 排序的信号文件。"""

    pattern = re.compile(rf"^c_{cutter_number}_(\d{{3}})\.csv$", re.IGNORECASE)
    matched: list[tuple[int, Path]] = []
    for path in cutter_dir.glob(f"c_{cutter_number}_*.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            matched.append((int(match.group(1)), path))
    return [path for _, path in sorted(matched)]


def _representative_files(files: list[Path]) -> list[Path]:
    """选择前、中、后三个文件做采样长度和通道检查。"""

    if not files:
        return []
    indexes = sorted({0, len(files) // 2, len(files) - 1})
    return [files[index] for index in indexes]


def _inspect_signal(path: Path) -> tuple[int | None, int, bool]:
    """扫描一个代表文件，返回通道数、样本长度和列数是否始终一致。"""

    channel_count: int | None = None
    row_count = 0
    consistent = True
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            row_count += 1
            if channel_count is None:
                channel_count = len(row)
            elif len(row) != channel_count:
                consistent = False
    return channel_count, row_count, consistent


def _inspect_wear(path: Path) -> tuple[int, bool, bool, set[int]]:
    """返回 wear 行数、表头状态、数值状态和 cut 编号集合。"""

    required = {"cut", "flute_1", "flute_2", "flute_3"}
    cut_ids: set[int] = set()
    values_valid = True
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        header_valid = required.issubset(set(reader.fieldnames or ()))
        row_count = 0
        for row in reader:
            row_count += 1
            try:
                cut_ids.add(int(row.get("cut", "")))
            except (TypeError, ValueError):
                header_valid = False
            try:
                values = [float(row.get(column, "")) for column in ("flute_1", "flute_2", "flute_3")]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    values_valid = False
            except (TypeError, ValueError):
                values_valid = False
    return row_count, header_valid, values_valid, cut_ids


class PHM2010Adapter(DatasetAdapter):
    """自动发现 PHM2010 六把刀并生成统一 DatasetManifest。"""

    adapter_id = "phm2010"

    def __init__(self, *, expected_signal_count: int = 315, expected_channel_count: int = 7) -> None:
        if expected_signal_count < 1 or expected_channel_count < 1:
            raise ValueError("预期文件数和通道数都必须为正整数。")
        self.expected_signal_count = expected_signal_count
        self.expected_channel_count = expected_channel_count

    def inspect(self, root: Path) -> DatasetInspection:
        """只读发现 C1-C6，并执行不加载完整数据矩阵的轻量体检。"""

        logical_root = Path(root).expanduser().absolute()
        if not logical_root.is_dir():
            raise FileNotFoundError(f"PHM2010 数据集根目录不存在: {logical_root}")

        issues: list[ValidationIssue] = []
        cutters: dict[str, CutterManifest] = {}

        for cutter_id in PHM2010_CUTTER_IDS:
            cutter_number = int(cutter_id[1:])
            relative_path = cutter_id.lower()
            logical_cutter_path = logical_root / relative_path
            available = logical_cutter_path.is_dir()
            labeled = cutter_id in PHM2010_LABELED_CUTTERS
            wear_name = f"{relative_path}_wear.csv" if labeled else None

            if not available:
                issues.append(
                    ValidationIssue(
                        code="CUTTER_NOT_AVAILABLE",
                        severity="warning",
                        field_path=f"cutters.{cutter_id}",
                        message=f"当前机器未检测到 {cutter_id} 目录，页面不会把它列为可选数据。",
                    )
                )
                cutters[cutter_id] = CutterManifest(
                    cutter_id=cutter_id,
                    relative_path=relative_path,
                    labeled=labeled,
                    signal_glob=f"c_{cutter_number}_*.csv",
                    wear_file=wear_name,
                    available=False,
                )
                continue

            resolved_path = logical_cutter_path.resolve(strict=True)
            signal_files = _signal_files(logical_cutter_path, cutter_number)
            signal_cut_ids = {int(path.stem.rsplit("_", maxsplit=1)[1]) for path in signal_files}
            expected_cut_ids = set(range(1, self.expected_signal_count + 1))
            representative = _representative_files(signal_files)
            channel_counts: list[int] = []
            sample_lengths: list[int] = []

            for sample_file in representative:
                channel_count, sample_length, consistent = _inspect_signal(sample_file)
                if channel_count is not None:
                    channel_counts.append(channel_count)
                sample_lengths.append(sample_length)
                if not consistent:
                    issues.append(
                        ValidationIssue(
                            code="INCONSISTENT_SIGNAL_COLUMNS",
                            severity="error",
                            field_path=f"cutters.{cutter_id}.{sample_file.name}",
                            message=f"{sample_file.name} 内部存在列数不一致的信号行。",
                        )
                    )

            detected_channel_count = channel_counts[0] if channel_counts else None
            if len(signal_files) != self.expected_signal_count:
                issues.append(
                    ValidationIssue(
                        code="SIGNAL_COUNT_MISMATCH",
                        severity="error",
                        field_path=f"cutters.{cutter_id}.signal_file_count",
                        message=(
                            f"{cutter_id} 检测到 {len(signal_files)} 个信号文件，"
                            f"预期 {self.expected_signal_count} 个。"
                        ),
                    )
                )
            if signal_cut_ids != expected_cut_ids:
                missing_count = len(expected_cut_ids - signal_cut_ids)
                unexpected_count = len(signal_cut_ids - expected_cut_ids)
                issues.append(
                    ValidationIssue(
                        code="SIGNAL_CUT_SET_MISMATCH",
                        severity="error",
                        field_path=f"cutters.{cutter_id}.signal_files",
                        message=(
                            f"{cutter_id} 的 cut 编号集合不完整：缺失 {missing_count} 个，"
                            f"超出预期 {unexpected_count} 个。"
                        ),
                    )
                )
            if not channel_counts or any(count != self.expected_channel_count for count in channel_counts):
                issues.append(
                    ValidationIssue(
                        code="CHANNEL_COUNT_MISMATCH",
                        severity="error",
                        field_path=f"cutters.{cutter_id}.detected_channel_count",
                        message=f"{cutter_id} 代表文件未稳定检测到 {self.expected_channel_count} 个通道。",
                    )
                )
            if not sample_lengths or any(length < 1 for length in sample_lengths):
                issues.append(
                    ValidationIssue(
                        code="EMPTY_SIGNAL_FILE",
                        severity="error",
                        field_path=f"cutters.{cutter_id}.sampled_signal_lengths",
                        message=f"{cutter_id} 的代表信号文件为空或无法读取。",
                    )
                )

            wear_row_count: int | None = None
            wear_sha256: str | None = None
            if labeled:
                wear_path = logical_cutter_path / str(wear_name)
                if not wear_path.is_file():
                    issues.append(
                        ValidationIssue(
                            code="WEAR_FILE_MISSING",
                            severity="error",
                            field_path=f"cutters.{cutter_id}.wear_file",
                            message=f"{cutter_id} 是有标签刀具，但缺少 {wear_name}。",
                        )
                    )
                else:
                    wear_row_count, header_valid, values_valid, wear_cut_ids = _inspect_wear(wear_path)
                    wear_sha256 = _sha256_file(wear_path)
                    if not header_valid:
                        issues.append(
                            ValidationIssue(
                                code="WEAR_HEADER_INVALID",
                                severity="error",
                                field_path=f"cutters.{cutter_id}.wear_file",
                                message=f"{wear_name} 缺少 cut 和三刃 VB 必要列。",
                            )
                        )
                    if not values_valid:
                        issues.append(
                            ValidationIssue(
                                code="WEAR_VALUES_INVALID",
                                severity="error",
                                field_path=f"cutters.{cutter_id}.wear_file",
                                message=f"{wear_name} 包含缺失、非有限或负数 VB 值。",
                            )
                        )
                    if wear_row_count != self.expected_signal_count:
                        issues.append(
                            ValidationIssue(
                                code="WEAR_COUNT_MISMATCH",
                                severity="error",
                                field_path=f"cutters.{cutter_id}.wear_row_count",
                                message=(
                                    f"{cutter_id} 检测到 {wear_row_count} 行磨损标签，"
                                    f"预期 {self.expected_signal_count} 行。"
                                ),
                            )
                        )
                    if wear_cut_ids != signal_cut_ids or wear_cut_ids != expected_cut_ids:
                        issues.append(
                            ValidationIssue(
                                code="WEAR_CUT_SET_MISMATCH",
                                severity="error",
                                field_path=f"cutters.{cutter_id}.wear_file",
                                message=f"{cutter_id} 的 wear cut 编号与信号 cut 编号不能一一对应。",
                            )
                        )

            cutters[cutter_id] = CutterManifest(
                cutter_id=cutter_id,
                relative_path=relative_path,
                labeled=labeled,
                signal_glob=f"c_{cutter_number}_*.csv",
                wear_file=wear_name,
                available=True,
                resolved_path=resolved_path,
                signal_file_count=len(signal_files),
                wear_row_count=wear_row_count,
                detected_channel_count=detected_channel_count,
                sampled_signal_lengths=tuple(sample_lengths),
                inventory_hash=_inventory_hash(signal_files),
                wear_sha256=wear_sha256,
            )

        manifest = DatasetManifest(
            dataset_id="phm2010",
            display_name="PHM 2010",
            adapter=self.adapter_id,
            root=logical_root,
            channels=PHM2010_CHANNELS,
            cutters=cutters,
            sampling_rate_hz=50_000.0,
            description="PHM 2010 铣削刀具磨损数据集；C1/C4/C6 有 VB 标签。",
        )
        manifest = attach_manifest_hash(manifest)
        has_error = any(issue.severity.value == "error" for issue in issues)
        validation = ValidationResult(
            valid=not has_error,
            scope="dataset.phm2010",
            issues=tuple(issues),
        )
        return DatasetInspection(manifest=manifest, validation=validation)
