"""PHM2010 数据体检报告生成。

本模块把 `CutterInventory` 转换成 Markdown 报告。
报告面向人阅读，用来支持初赛展示和后续 Agent 诊断。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from toolwear_agent.training.phm2010 import (
    EXPECTED_CHANNEL_COUNT,
    CutterInventory,
    SignalRecord,
    count_records_with_errors,
)


@dataclass(frozen=True)
class ProfileSummary:
    """数据体检摘要。"""

    channel_error_count: int
    empty_sample_count: int
    read_error_count: int
    missing_value_file_count: int
    min_file_size_bytes: int
    max_file_size_bytes: int


def summarize_inventory(inventory: CutterInventory) -> ProfileSummary:
    """从登记清单中汇总体检结果。"""

    records = inventory.records
    file_sizes = [record.file_size_bytes for record in records]

    return ProfileSummary(
        channel_error_count=sum(
            1 for record in records if record.channel_count_sample != EXPECTED_CHANNEL_COUNT
        ),
        empty_sample_count=sum(1 for record in records if record.sampled_rows == 0),
        read_error_count=count_records_with_errors(records),
        missing_value_file_count=sum(1 for record in records if record.missing_value_count_sample > 0),
        min_file_size_bytes=min(file_sizes) if file_sizes else 0,
        max_file_size_bytes=max(file_sizes) if file_sizes else 0,
    )


def _format_cut_list(cuts: list[int], limit: int = 20) -> str:
    """把刀次编号列表格式化成适合报告展示的短文本。"""

    if not cuts:
        return "无"
    shown = ", ".join(str(cut) for cut in cuts[:limit])
    if len(cuts) > limit:
        return f"{shown} ... 共 {len(cuts)} 个"
    return shown


def _records_with_channel_errors(records: list[SignalRecord]) -> list[SignalRecord]:
    """筛选列数异常的信号文件。"""

    return [record for record in records if record.channel_count_sample != EXPECTED_CHANNEL_COUNT]


def render_profile_markdown(inventory: CutterInventory) -> str:
    """生成 C1 数据体检 Markdown 报告。"""

    summary = summarize_inventory(inventory)
    channel_error_records = _records_with_channel_errors(inventory.records)
    first_records = inventory.records[:5]

    lines = [
        f"# PHM2010 {inventory.cutter.upper()} 数据登记与体检报告",
        "",
        "## 1. 体检结论",
        "",
        f"- 信号 CSV 数量：{inventory.signal_file_count}",
        f"- 预期 CSV 数量：{inventory.expected_signal_file_count}",
        f"- 磨损标签数量：{inventory.label_count}",
        f"- 预期通道数量：{inventory.expected_channel_count}",
        f"- 列数异常文件数量：{summary.channel_error_count}",
        f"- 抽样空文件数量：{summary.empty_sample_count}",
        f"- 读取错误文件数量：{summary.read_error_count}",
        f"- 抽样发现空值的文件数量：{summary.missing_value_file_count}",
        "",
        "## 2. 数据路径",
        "",
        f"- 刀具目录：`{inventory.cutter_dir}`",
        f"- 磨损标签文件：`{inventory.wear_file}`",
        "",
        "## 3. 刀次完整性",
        "",
        f"- 缺失信号刀次：{_format_cut_list(inventory.missing_signal_cuts)}",
        f"- 缺失标签刀次：{_format_cut_list(inventory.missing_label_cuts)}",
        f"- 超出预期范围的信号刀次：{_format_cut_list(inventory.unexpected_signal_cuts)}",
        "",
        "## 4. 文件大小范围",
        "",
        f"- 最小文件大小：{summary.min_file_size_bytes} bytes",
        f"- 最大文件大小：{summary.max_file_size_bytes} bytes",
        "",
        "## 5. 前 5 个信号文件样例",
        "",
        "| 刀次 | 文件名 | 文件大小 bytes | 抽样列数 | 抽样行数 | 是否有标签 |",
        "|---:|---|---:|---:|---:|---|",
    ]

    for record in first_records:
        lines.append(
            "| "
            f"{record.cut} | {record.file_name} | {record.file_size_bytes} | "
            f"{record.channel_count_sample} | {record.sampled_rows} | "
            f"{'是' if record.has_label else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 6. 异常文件摘要",
            "",
        ]
    )

    if channel_error_records:
        lines.append("| 刀次 | 文件名 | 抽样列数 | 读取错误 |")
        lines.append("|---:|---|---:|---|")
        for record in channel_error_records[:20]:
            lines.append(
                f"| {record.cut} | {record.file_name} | "
                f"{record.channel_count_sample} | {record.read_error or '无'} |"
            )
    else:
        lines.append("未发现抽样列数异常文件。")

    lines.extend(
        [
            "",
            "## 7. 对下一步的意义",
            "",
            "本报告确认 C1 数据是否可以进入标签生成步骤。下一步会基于磨损标签中的三刀刃 VB，"
            "默认取最大值，并按 90/130/160 um 生成四阶段磨损标签。",
            "",
        ]
    )

    return "\n".join(lines)


def write_profile_report(inventory: CutterInventory, output_file: Path) -> Path:
    """写出 Markdown 数据体检报告。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_profile_markdown(inventory), encoding="utf-8")
    return output_file
