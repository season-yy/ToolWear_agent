"""PHM2010 数据集登记工具。

本模块负责把 PHM2010 的原始文件整理成“机器能读、人也能查”的清单。
当前第 1 步只做 C1 数据登记与体检，不生成磨损阶段标签，也不训练模型。

PHM2010 C1 目录约定：
- 信号文件：`c_1_001.csv`、`c_1_002.csv` ... `c_1_315.csv`
- 磨损标签：`c1_wear.csv`
- 信号文件无表头，默认 7 个通道
- 标签文件含表头：`cut,flute_1,flute_2,flute_3`
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


EXPECTED_CHANNEL_COUNT = 7
EXPECTED_CUTS_PER_CUTTER = 315


@dataclass(frozen=True)
class WearLabel:
    """单个刀次对应的三刀刃磨损值。"""

    cut: int
    flute_1: float
    flute_2: float
    flute_3: float


@dataclass(frozen=True)
class SignalRecord:
    """单个信号 CSV 的登记信息。

    这里不保存原始信号数据，只保存文件路径、刀次编号和体检摘要。
    这样清单文件会很小，也不会把大数据重复写入运行目录。
    """

    cut: int
    file_path: str
    file_name: str
    file_size_bytes: int
    has_label: bool
    channel_count_sample: int | None
    sampled_rows: int
    missing_value_count_sample: int
    read_error: str | None


@dataclass(frozen=True)
class CutterInventory:
    """某一把刀具的数据登记结果。"""

    dataset_id: str
    cutter: str
    cutter_dir: str
    wear_file: str
    signal_file_count: int
    label_count: int
    expected_signal_file_count: int
    expected_channel_count: int
    missing_signal_cuts: list[int]
    missing_label_cuts: list[int]
    unexpected_signal_cuts: list[int]
    records: list[SignalRecord]


def parse_signal_filename(file_name: str, cutter: str) -> int:
    """从 PHM2010 信号文件名中解析刀次编号。

    示例：`c_1_023.csv` 会解析为 `23`。
    参数 `cutter` 使用 `c1`、`c4` 这样的写法。
    """

    cutter_number = cutter.lower().replace("c", "")
    pattern = rf"^c_{re.escape(cutter_number)}_(\d{{3}})\.csv$"
    match = re.match(pattern, file_name)
    if not match:
        raise ValueError(f"无法从文件名解析刀次编号: {file_name}")
    return int(match.group(1))


def signal_file_pattern(cutter: str) -> str:
    """返回某把刀具的信号文件通配模式。"""

    cutter_number = cutter.lower().replace("c", "")
    return f"c_{cutter_number}_*.csv"


def wear_file_name(cutter: str) -> str:
    """返回某把刀具的磨损标签文件名。"""

    return f"{cutter.lower()}_wear.csv"


def load_wear_labels(wear_file: Path) -> dict[int, WearLabel]:
    """读取磨损标签文件。

    返回值以刀次编号为 key，方便后续和信号文件做匹配。
    """

    labels: dict[int, WearLabel] = {}
    with wear_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required_columns = {"cut", "flute_1", "flute_2", "flute_3"}
        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"磨损标签缺少必要列: {missing}")

        for row in reader:
            cut = int(row["cut"])
            labels[cut] = WearLabel(
                cut=cut,
                flute_1=float(row["flute_1"]),
                flute_2=float(row["flute_2"]),
                flute_3=float(row["flute_3"]),
            )
    return labels


def inspect_signal_file(signal_file: Path, sample_rows: int = 100) -> tuple[int | None, int, int, str | None]:
    """抽样检查单个信号 CSV。

    返回：
    - 抽样检测到的列数
    - 实际抽样行数
    - 抽样范围内的空值数量
    - 读取错误信息，正常时为 None

    注意：这里默认只抽样前 100 行，不扫描完整文件。
    这样第 1 步体检速度更快，也足够发现列数不对、空文件、明显坏文件等问题。
    """

    detected_channel_count: int | None = None
    sampled_row_count = 0
    missing_value_count = 0

    try:
        with signal_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.reader(file_obj)
            for row in reader:
                if sampled_row_count >= sample_rows:
                    break
                sampled_row_count += 1
                if detected_channel_count is None:
                    detected_channel_count = len(row)
                missing_value_count += sum(1 for item in row if item.strip() == "")
    except Exception as exc:  # noqa: BLE001 - 体检报告需要记录具体异常文本
        return detected_channel_count, sampled_row_count, missing_value_count, str(exc)

    return detected_channel_count, sampled_row_count, missing_value_count, None


def sorted_signal_files(cutter_dir: Path, cutter: str) -> list[Path]:
    """按刀次编号排序返回信号文件列表。"""

    files = list(cutter_dir.glob(signal_file_pattern(cutter)))
    return sorted(files, key=lambda file_path: parse_signal_filename(file_path.name, cutter))


def expected_cut_range() -> set[int]:
    """返回 PHM2010 单把刀具预期刀次集合。"""

    return set(range(1, EXPECTED_CUTS_PER_CUTTER + 1))


def build_cutter_inventory(cutter_dir: Path, cutter: str = "c1") -> CutterInventory:
    """生成某一把刀具的数据登记清单。

    这个函数是第 1 步的核心：它把信号文件和磨损标签按刀次编号对齐，
    并对每个信号文件做轻量体检。
    """

    if not cutter_dir.exists():
        raise FileNotFoundError(f"刀具目录不存在: {cutter_dir}")

    wear_file = cutter_dir / wear_file_name(cutter)
    if not wear_file.exists():
        raise FileNotFoundError(f"磨损标签文件不存在: {wear_file}")

    labels = load_wear_labels(wear_file)
    records: list[SignalRecord] = []
    signal_files = sorted_signal_files(cutter_dir, cutter)
    signal_cuts: set[int] = set()

    for signal_file in signal_files:
        cut = parse_signal_filename(signal_file.name, cutter)
        signal_cuts.add(cut)
        channel_count, sampled_rows, missing_count, read_error = inspect_signal_file(signal_file)
        records.append(
            SignalRecord(
                cut=cut,
                file_path=str(signal_file),
                file_name=signal_file.name,
                file_size_bytes=signal_file.stat().st_size,
                has_label=cut in labels,
                channel_count_sample=channel_count,
                sampled_rows=sampled_rows,
                missing_value_count_sample=missing_count,
                read_error=read_error,
            )
        )

    expected_cuts = expected_cut_range()
    label_cuts = set(labels)

    return CutterInventory(
        dataset_id="phm2010",
        cutter=cutter,
        cutter_dir=str(cutter_dir),
        wear_file=str(wear_file),
        signal_file_count=len(signal_files),
        label_count=len(labels),
        expected_signal_file_count=EXPECTED_CUTS_PER_CUTTER,
        expected_channel_count=EXPECTED_CHANNEL_COUNT,
        missing_signal_cuts=sorted(expected_cuts - signal_cuts),
        missing_label_cuts=sorted(signal_cuts - label_cuts),
        unexpected_signal_cuts=sorted(signal_cuts - expected_cuts),
        records=records,
    )


def inventory_to_dict(inventory: CutterInventory) -> dict[str, object]:
    """把数据清单转换为可写入 JSON 的字典。"""

    return asdict(inventory)


def write_inventory_json(inventory: CutterInventory, output_file: Path) -> Path:
    """把数据登记清单写入 JSON 文件。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(inventory_to_dict(inventory), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def count_records_with_errors(records: Iterable[SignalRecord]) -> int:
    """统计存在读取错误的信号文件数量。"""

    return sum(1 for record in records if record.read_error)
