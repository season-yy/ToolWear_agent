"""信号统计特征提取工具。

这一层只负责把一刀对应的 7 通道原始信号 CSV 转换成一行机器学习特征。
P0 阶段先做简单、稳定、容易解释的统计特征，避免一开始就把训练流程复杂化。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_CHANNEL_NAMES = (
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "acoustic_emission_rms",
)

STATISTIC_NAMES = ("mean", "std", "min", "max", "ptp", "rms")


@dataclass(frozen=True)
class SignalFeatureRow:
    """单个刀次的统计特征结果。

    `features` 按 `feature_names` 的顺序保存数值，后续训练时可以直接组成二维矩阵。
    """

    cut: int
    feature_names: list[str]
    features: list[float]
    sampled_rows: int


def build_feature_names(channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES) -> list[str]:
    """生成统计特征列名。

    例如 `force_x_mean` 表示 force_x 通道在抽样窗口内的均值。
    """

    return [f"{channel}_{statistic}" for channel in channel_names for statistic in STATISTIC_NAMES]


def _empty_channel_accumulators(channel_count: int) -> list[dict[str, float]]:
    """为每个通道创建统计量累加器。

    这里用流式累加，而不是把完整 CSV 读进内存，主要是为了后续处理大文件更稳。
    """

    return [
        {
            "count": 0.0,
            "sum": 0.0,
            "sum_square": 0.0,
            "min": math.inf,
            "max": -math.inf,
        }
        for _ in range(channel_count)
    ]


def _update_accumulators(accumulators: list[dict[str, float]], values: list[float]) -> None:
    """把一行信号值累加到各通道统计量中。"""

    for index, value in enumerate(values):
        stats = accumulators[index]
        stats["count"] += 1.0
        stats["sum"] += value
        stats["sum_square"] += value * value
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)


def _finalize_features(accumulators: list[dict[str, float]]) -> list[float]:
    """把累加器转换成最终特征向量。"""

    features: list[float] = []
    for stats in accumulators:
        count = stats["count"]
        if count <= 0:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        mean_value = stats["sum"] / count
        variance = max(stats["sum_square"] / count - mean_value * mean_value, 0.0)
        std_value = math.sqrt(variance)
        min_value = stats["min"]
        max_value = stats["max"]
        peak_to_peak = max_value - min_value
        rms_value = math.sqrt(stats["sum_square"] / count)
        features.extend([mean_value, std_value, min_value, max_value, peak_to_peak, rms_value])
    return features


def extract_signal_statistics(
    signal_file: Path,
    cut: int,
    channel_names: Sequence[str] = DEFAULT_CHANNEL_NAMES,
    max_rows: int = 2000,
    start_row: int = 0,
) -> SignalFeatureRow:
    """从单个 PHM2010 信号 CSV 中提取统计特征。

    `max_rows` 控制最多读取多少行，`start_row` 控制从第几行开始读取。
    对滑窗样本来说，一个窗口就是从 `start_row` 开始、长度为 `max_rows` 的片段。
    """

    if max_rows <= 0:
        raise ValueError("max_rows 必须大于 0")
    if start_row < 0:
        raise ValueError("start_row 不能小于 0")
    if not signal_file.exists():
        raise FileNotFoundError(f"信号文件不存在: {signal_file}")

    channel_count = len(channel_names)
    accumulators = _empty_channel_accumulators(channel_count)
    sampled_rows = 0

    with signal_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row_index, raw_row in enumerate(reader):
            if row_index < start_row:
                continue
            if sampled_rows >= max_rows:
                break
            if len(raw_row) != channel_count:
                raise ValueError(
                    f"信号文件列数不符合预期: {signal_file}, "
                    f"期望 {channel_count} 列, 实际 {len(raw_row)} 列"
                )
            values = [float(item) for item in raw_row]
            _update_accumulators(accumulators, values)
            sampled_rows += 1

    if sampled_rows == 0:
        raise ValueError(f"信号文件为空或未读取到有效行: {signal_file}")

    return SignalFeatureRow(
        cut=cut,
        feature_names=build_feature_names(channel_names),
        features=_finalize_features(accumulators),
        sampled_rows=sampled_rows,
    )
