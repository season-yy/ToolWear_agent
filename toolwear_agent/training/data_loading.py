"""训练前的数据证据校验、窗口选择和原始信号读取。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.errors import PathBoundaryError
from toolwear_agent.data.registry import DatasetRegistry
from toolwear_agent.data.leakage import assert_no_window_leakage, assert_windows_match_split_manifest
from toolwear_agent.data.sampling import load_sample_manifest
from toolwear_agent.data.splitting import assert_manifest_matches_lock, load_split_lock, load_split_manifest
from toolwear_agent.schemas import RunConfig, TrainingDataRef, TrainingSampleManifest
from toolwear_agent.training.features import DEFAULT_CHANNEL_NAMES, STATISTIC_NAMES
from toolwear_agent.training.windows import WindowRecord, load_window_manifest


@dataclass(frozen=True)
class PreparedTrainingData:
    """通过全部数据正确性检查后，可交给训练后端的窗口记录。"""

    train_records: tuple[WindowRecord, ...]
    validation_records: tuple[WindowRecord, ...]
    class_labels: tuple[str, ...]
    sample_manifest: TrainingSampleManifest


@dataclass(frozen=True)
class RawWindowBatch:
    """内存中的时序窗口、标签和追溯 ID。"""

    values: np.ndarray
    labels: np.ndarray
    window_ids: tuple[str, ...]


def _normalized_path(value: str | Path) -> str:
    """统一 Windows 路径分隔符和大小写，供 Manifest 对照。"""

    return str(value).replace("\\", "/").casefold()


def _stable_record_key(record: WindowRecord) -> tuple[int, int, int, str]:
    """定义不依赖 CSV 原始排列的稳定顺序。"""

    return record.stage_id, record.cut, record.start_row, record.window_id


def _allocate_stage_quotas(records: list[WindowRecord], limit: int) -> dict[int, int]:
    """按原阶段比例分配上限，并保证每个阶段至少保留一个窗口。"""

    by_stage: dict[int, int] = {}
    for record in records:
        by_stage[record.stage_id] = by_stage.get(record.stage_id, 0) + 1
    if limit < len(by_stage):
        raise ValueError("max_samples 小于阶段数量，无法保证四阶段均有样本。")

    total = len(records)
    quotas = {
        stage_id: min(count, max(1, int(limit * count / total)))
        for stage_id, count in by_stage.items()
    }
    # 先按小数余量和阶段 ID 稳定补齐，再从超额阶段稳定回收。
    remainders = sorted(
        by_stage,
        key=lambda stage_id: (-(limit * by_stage[stage_id] / total - quotas[stage_id]), stage_id),
    )
    while sum(quotas.values()) < limit:
        for stage_id in remainders:
            if quotas[stage_id] < by_stage[stage_id]:
                quotas[stage_id] += 1
                break
        else:  # pragma: no cover - limit 已受总样本数约束
            break
    while sum(quotas.values()) > limit:
        for stage_id in reversed(remainders):
            if quotas[stage_id] > 1:
                quotas[stage_id] -= 1
                break
        else:  # pragma: no cover - limit >= 阶段数时不会发生
            break
    return quotas


def stratified_record_limit(
    records: Iterable[WindowRecord],
    limit: int | None,
    *,
    random_seed: int,
) -> tuple[WindowRecord, ...]:
    """在需要 smoke 限流时保持阶段比例并可复现地选取窗口。"""

    materialized = sorted(records, key=_stable_record_key)
    if limit is None or limit >= len(materialized):
        return tuple(materialized)
    if limit < 1:
        raise ValueError("max_samples 必须大于 0。")

    quotas = _allocate_stage_quotas(materialized, limit)
    selected: list[WindowRecord] = []
    for stage_id in sorted(quotas):
        stage_records = [record for record in materialized if record.stage_id == stage_id]
        random.Random(f"{random_seed}:{stage_id}").shuffle(stage_records)
        selected.extend(stage_records[: quotas[stage_id]])
    return tuple(sorted(selected, key=_stable_record_key))


def _match_sample_windows(
    all_windows: list[WindowRecord],
    sample_manifest: TrainingSampleManifest,
) -> tuple[WindowRecord, ...]:
    """把小样本引用还原为原窗口，并拒绝路径、标签或区间被修改。"""

    windows_by_id = {record.window_id: record for record in all_windows}
    matched: list[WindowRecord] = []
    for reference in sample_manifest.windows:
        record = windows_by_id.get(reference.window_id)
        if record is None:
            raise ValueError(f"小样本窗口不在窗口 Manifest 中: {reference.window_id}")
        mismatched = (
            record.split != "train"
            or record.cut != reference.cut_id
            or record.stage_id != reference.stage_id
            or record.stage_name != reference.stage_name
            or record.start_row != reference.start_row
            or record.end_row != reference.end_row
            or _normalized_path(record.file_path) != _normalized_path(reference.source_file)
        )
        if mismatched:
            raise ValueError(f"小样本窗口与窗口 Manifest 不一致: {reference.window_id}")
        matched.append(record)
    return tuple(matched)


def prepare_training_data(
    *,
    data_ref: TrainingDataRef,
    run_config: RunConfig,
    path_resolver: PathResolver,
) -> PreparedTrainingData:
    """校验锁、hash、泄漏审计和小样本后，返回 train/validation 记录。

    该函数故意不返回 test 记录。候选训练、诊断和调参调用方从类型和数据两侧
    都无法接触 test；最终评估将在独立的 final_evaluation 入口实现。
    """

    referenced_files = (
        data_ref.window_manifest_file,
        data_ref.training_sample_manifest_file,
        data_ref.split_manifest_file,
        data_ref.split_lock_file,
        data_ref.leakage_audit_file,
    )
    for referenced_file in referenced_files:
        path_resolver.assert_within(referenced_file, (path_resolver.settings.ai_infra_root,))
    missing = [str(path) for path in referenced_files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("训练数据证据文件不存在: " + ", ".join(missing))

    split_manifest = load_split_manifest(data_ref.split_manifest_file)
    split_lock = load_split_lock(data_ref.split_lock_file)
    assert_manifest_matches_lock(split_manifest, split_lock)
    if split_manifest.dataset_id != data_ref.dataset_id or split_manifest.cutter_id != data_ref.cutter_id:
        raise ValueError("TrainingDataRef 与 SplitManifest 数据集身份不一致。")
    if run_config.split_hash != split_manifest.split_hash:
        raise ValueError("RunConfig.split_hash 与锁定 SplitManifest 不一致。")
    if split_lock.experiment_id != run_config.experiment_id or split_lock.revision != run_config.revision:
        raise ValueError("RunConfig 与 split lock 的 experiment/revision 不一致。")

    leakage_payload = json.loads(data_ref.leakage_audit_file.read_text(encoding="utf-8"))
    if leakage_payload.get("valid") is not True:
        raise ValueError("已有泄漏审计未通过，训练被阻断。")

    all_windows = load_window_manifest(data_ref.window_manifest_file)
    assert_no_window_leakage(all_windows)
    assert_windows_match_split_manifest(all_windows, split_manifest)
    sample_manifest = load_sample_manifest(data_ref.training_sample_manifest_file)
    if sample_manifest.dataset_id != data_ref.dataset_id or sample_manifest.cutter_id != data_ref.cutter_id:
        raise ValueError("TrainingSampleManifest 与 TrainingDataRef 身份不一致。")
    if sample_manifest.source_split_hash != split_manifest.split_hash:
        raise ValueError("TrainingSampleManifest 没有引用当前锁定 split_hash。")
    if abs(sample_manifest.fraction - run_config.sample_fraction) > 1e-9:
        raise ValueError("RunConfig.sample_fraction 与小样本 Manifest 不一致。")

    train_records = _match_sample_windows(all_windows, sample_manifest)
    validation_records = tuple(record for record in all_windows if record.split == "validation")
    if not validation_records:
        raise ValueError("窗口 Manifest 中没有 validation 样本。")

    train_records = stratified_record_limit(
        train_records,
        run_config.max_samples,
        random_seed=run_config.random_seed,
    )
    validation_limit = run_config.max_samples if run_config.run_kind.value == "smoke" else None
    validation_records = stratified_record_limit(
        validation_records,
        validation_limit,
        random_seed=run_config.random_seed + 1,
    )
    # 普通目录必须位于逻辑 raw root；Junction 解析越界时，只接受 Dataset Registry
    # 已体检并记录的当前 cutter 真实目录，绝不把整个磁盘加入白名单。
    source_files = {record.file_path for record in (*train_records, *validation_records)}
    try:
        for source_file in source_files:
            path_resolver.assert_dataset_read_path(source_file)
    except PathBoundaryError:
        registry = DatasetRegistry(path_resolver.settings.dataset_manifest)
        dataset_manifest = registry.get(data_ref.dataset_id)
        cutter_manifest = next(
            (
                cutter
                for cutter_id, cutter in dataset_manifest.cutters.items()
                if cutter_id.casefold() == data_ref.cutter_id.casefold()
            ),
            None,
        )
        if cutter_manifest is None or cutter_manifest.resolved_path is None:
            raise PathBoundaryError(
                f"Dataset Registry 未登记 cutter {data_ref.cutter_id} 的真实只读路径。"
            )
        for source_file in source_files:
            path_resolver.assert_dataset_read_path(
                source_file,
                additional_roots=(cutter_manifest.resolved_path,),
            )

    labels_by_id: dict[int, str] = {}
    for record in all_windows:
        existing = labels_by_id.setdefault(record.stage_id, record.stage_name)
        if existing != record.stage_name:
            raise ValueError(f"stage_id={record.stage_id} 对应了多个阶段名称。")
    if sorted(labels_by_id) != list(range(len(labels_by_id))):
        raise ValueError("阶段 ID 必须从 0 开始连续编号。")
    return PreparedTrainingData(
        train_records=train_records,
        validation_records=validation_records,
        class_labels=tuple(labels_by_id[index] for index in sorted(labels_by_id)),
        sample_manifest=sample_manifest,
    )


def _channel_indices(input_channels: tuple[str, ...]) -> tuple[int, ...]:
    """把 Registry 通道 ID 转为 PHM2010 CSV 的固定列索引。"""

    unknown = set(input_channels) - set(DEFAULT_CHANNEL_NAMES)
    if unknown:
        raise ValueError(f"存在未知输入通道: {sorted(unknown)}")
    return tuple(DEFAULT_CHANNEL_NAMES.index(channel) for channel in input_channels)


def load_raw_window_batch(
    records: Iterable[WindowRecord],
    input_channels: tuple[str, ...],
) -> RawWindowBatch:
    """按源文件分组读取 CSV，一次读取后切出该文件的全部目标窗口。

    相比逐窗口重复跳过几十万行，这种方式既保留原始数据只读，又显著降低真实
    C1 smoke 的 I/O 时间。最终只保留选中的窗口数组，不长期缓存整刀信号。
    """

    materialized = list(records)
    if not materialized:
        raise ValueError("没有可加载的窗口记录。")
    indices = _channel_indices(input_channels)
    records_by_file: dict[str, list[WindowRecord]] = {}
    source_path_by_key: dict[str, Path] = {}
    for record in materialized:
        key = _normalized_path(record.file_path)
        records_by_file.setdefault(key, []).append(record)
        source_path_by_key[key] = Path(record.file_path)

    values_by_id: dict[str, np.ndarray] = {}
    for key in sorted(records_by_file):
        source_file = source_path_by_key[key]
        frame = pd.read_csv(source_file, header=None, usecols=list(indices), dtype=np.float32)
        # pandas 不承诺按 usecols 参数顺序返回列，因此显式恢复 Pipeline 通道顺序。
        frame = frame.loc[:, list(indices)]
        source_values = frame.to_numpy(dtype=np.float32, copy=False)
        if source_values.ndim == 1:
            source_values = source_values.reshape(-1, 1)
        for record in records_by_file[key]:
            window = source_values[record.start_row : record.end_row]
            if window.shape != (record.window_size, len(indices)):
                raise ValueError(
                    f"窗口 {record.window_id} 实际形状 {window.shape}，"
                    f"期望 {(record.window_size, len(indices))}。"
                )
            values_by_id[record.window_id] = np.ascontiguousarray(window.T, dtype=np.float32)

    window_lengths = {record.window_size for record in materialized}
    if len(window_lengths) != 1:
        raise ValueError("同一次训练的窗口长度必须一致。")
    return RawWindowBatch(
        values=np.stack([values_by_id[record.window_id] for record in materialized]),
        labels=np.asarray([record.stage_id for record in materialized], dtype=np.int64),
        window_ids=tuple(record.window_id for record in materialized),
    )


def statistical_feature_matrix(
    raw_windows: np.ndarray,
    channel_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """把 ``[N, C, L]`` 原始窗口转换为六类时域统计特征。"""

    if raw_windows.ndim != 3:
        raise ValueError("统计特征输入必须是 [samples, channels, length]。")
    means = raw_windows.mean(axis=2)
    standard_deviations = raw_windows.std(axis=2)
    minima = raw_windows.min(axis=2)
    maxima = raw_windows.max(axis=2)
    peak_to_peak = np.ptp(raw_windows, axis=2)
    rms = np.sqrt(np.mean(np.square(raw_windows), axis=2))
    # 先按通道，再按统计量排列，与旧版特征表的解释顺序保持一致。
    stacked = np.stack((means, standard_deviations, minima, maxima, peak_to_peak, rms), axis=2)
    resolved_channel_names = channel_names or tuple(
        f"channel_{channel_index}" for channel_index in range(raw_windows.shape[1])
    )
    if len(resolved_channel_names) != raw_windows.shape[1]:
        raise ValueError("channel_names 数量必须与输入通道数一致。")
    feature_names = tuple(
        f"{channel_name}_{statistic}"
        for channel_name in resolved_channel_names
        for statistic in STATISTIC_NAMES
    )
    return stacked.reshape(raw_windows.shape[0], -1).astype(np.float32), feature_names


def fit_channel_normalizer(raw_train: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """只用训练窗口拟合逐通道均值和标准差。"""

    if raw_train.ndim != 3:
        raise ValueError("Z-score 输入必须是 [samples, channels, length]。")
    means = raw_train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    standard_deviations = raw_train.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    standard_deviations = np.maximum(standard_deviations, np.float32(epsilon))
    return means, standard_deviations


def apply_channel_normalizer(
    raw_windows: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> np.ndarray:
    """应用训练集统计量，不在 validation 上重新拟合。"""

    return ((raw_windows - means[None, :, None]) / standard_deviations[None, :, None]).astype(np.float32)
