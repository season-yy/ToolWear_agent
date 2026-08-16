"""训练集内、按阶段与刀次感知的可复现小样本抽取。"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Protocol, TypeVar

from toolwear_agent.data.splitting import normalize_split_name
from toolwear_agent.schemas import SampledWindowRef, TrainingSampleManifest


class SampleWindowLike(Protocol):
    """小样本抽取所需的最小窗口字段。"""

    window_id: str
    cut: int
    file_path: str
    start_row: int
    end_row: int
    stage_id: int
    stage_name: str
    split: str


WindowT = TypeVar("WindowT", bound=SampleWindowLike)


@dataclass(frozen=True)
class TrainingSample(Generic[WindowT]):
    """抽中的原窗口对象及其可落盘 Manifest。"""

    records: tuple[WindowT, ...]
    manifest: TrainingSampleManifest


def _source_key(record: SampleWindowLike) -> tuple[str, int]:
    """使用源文件和 cut 共同标识一个刀次，兼容未来多刀具数据。"""

    normalized_path = str(record.file_path).replace("\\", "/").casefold()
    return normalized_path, record.cut


def _stable_record_sort_key(record: SampleWindowLike) -> tuple[int, int, str, int, int, str]:
    """定义与输入顺序无关的稳定窗口排序。"""

    return (
        record.stage_id,
        record.cut,
        str(record.file_path).casefold(),
        record.start_row,
        record.end_row,
        record.window_id,
    )


def _temporal_pick(records: list[WindowT], limit: int) -> list[WindowT]:
    """在一个刀次内部均匀取点，优先覆盖加工开始与结束。"""

    ordered = sorted(records, key=lambda item: (item.start_row, item.end_row, item.window_id))
    if limit >= len(ordered):
        return ordered
    if limit == 1:
        return [ordered[(len(ordered) - 1) // 2]]

    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    # round 在极端小数组上理论上可能产生重复；这里做稳定补齐，保证数量严格等于 limit。
    unique_indices = list(dict.fromkeys(indices))
    if len(unique_indices) < limit:
        unique_indices.extend(index for index in range(len(ordered)) if index not in unique_indices)
    return [ordered[index] for index in sorted(unique_indices[:limit])]


def _allocate_cut_quotas(
    records_by_cut: dict[tuple[str, int], list[WindowT]],
    target_count: int,
    *,
    random_seed: int,
    stage_id: int,
) -> dict[tuple[str, int], int]:
    """先最大化 cut 覆盖，再把剩余名额均匀分配到各 cut。"""

    cut_keys = sorted(records_by_cut)
    shuffled_keys = list(cut_keys)
    random.Random(f"{random_seed}:{stage_id}").shuffle(shuffled_keys)
    quotas = {key: 0 for key in cut_keys}

    initial_keys = shuffled_keys if target_count >= len(cut_keys) else shuffled_keys[:target_count]
    for key in initial_keys:
        quotas[key] = 1

    remaining = target_count - len(initial_keys)
    while remaining > 0:
        allocated_this_round = False
        for key in shuffled_keys:
            if quotas[key] >= len(records_by_cut[key]):
                continue
            quotas[key] += 1
            remaining -= 1
            allocated_this_round = True
            if remaining == 0:
                break
        if not allocated_this_round:
            raise ValueError("小样本目标数量超过可用训练窗口数量。")
    return quotas


def calculate_sample_hash(manifest: TrainingSampleManifest) -> str:
    """计算不包含生成时间和 sample_hash 自身的稳定 SHA-256。"""

    payload = manifest.model_dump(mode="json", exclude={"sample_hash", "created_at"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def attach_sample_hash(manifest: TrainingSampleManifest) -> TrainingSampleManifest:
    """返回带正确哈希的小样本 Manifest，并拒绝内容漂移。"""

    expected = calculate_sample_hash(manifest)
    if manifest.sample_hash is not None and manifest.sample_hash != expected:
        raise ValueError("TrainingSampleManifest 内容与 sample_hash 不一致。")
    payload = manifest.model_dump(mode="python")
    payload["sample_hash"] = expected
    return TrainingSampleManifest.model_validate(payload)


def build_training_sample(
    records: Iterable[WindowT],
    *,
    dataset_id: str,
    cutter_id: str,
    split_hash: str,
    fraction: float,
    random_seed: int,
) -> TrainingSample[WindowT]:
    """仅从 train 中按阶段抽取固定比例，并覆盖尽量多的 cut 与加工时段。"""

    if not 0 < fraction <= 1:
        raise ValueError("fraction 必须位于 (0, 1] 区间。")
    if random_seed < 0:
        raise ValueError("random_seed 不能小于 0。")

    train_records = [record for record in records if normalize_split_name(record.split) == "train"]
    if not train_records:
        raise ValueError("没有可用于小样本抽取的 train 窗口。")

    records_by_stage: dict[int, list[WindowT]] = {}
    for record in train_records:
        records_by_stage.setdefault(record.stage_id, []).append(record)

    selected: list[WindowT] = []
    for stage_id in sorted(records_by_stage):
        stage_records = records_by_stage[stage_id]
        target_count = min(len(stage_records), max(1, math.ceil(len(stage_records) * fraction)))
        records_by_cut: dict[tuple[str, int], list[WindowT]] = {}
        for record in stage_records:
            records_by_cut.setdefault(_source_key(record), []).append(record)
        quotas = _allocate_cut_quotas(
            records_by_cut,
            target_count,
            random_seed=random_seed,
            stage_id=stage_id,
        )
        for cut_key in sorted(records_by_cut):
            if quotas[cut_key] > 0:
                selected.extend(_temporal_pick(records_by_cut[cut_key], quotas[cut_key]))

    selected.sort(key=_stable_record_sort_key)
    stage_distribution: dict[str, int] = {}
    cut_distribution: dict[str, int] = {}
    window_refs: list[SampledWindowRef] = []
    for record in selected:
        stage_key = str(record.stage_id)
        cut_key = f"{cutter_id}:{record.cut}"
        stage_distribution[stage_key] = stage_distribution.get(stage_key, 0) + 1
        cut_distribution[cut_key] = cut_distribution.get(cut_key, 0) + 1
        window_refs.append(
            SampledWindowRef(
                window_id=record.window_id,
                cutter_id=cutter_id,
                cut_id=record.cut,
                source_file=str(record.file_path).replace("\\", "/"),
                stage_id=record.stage_id,
                stage_name=record.stage_name,
                start_row=record.start_row,
                end_row=record.end_row,
            )
        )

    fraction_token = f"{fraction:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    manifest = TrainingSampleManifest(
        sample_id=f"{dataset_id}_{cutter_id}_train_s{random_seed}_f{fraction_token}",
        dataset_id=dataset_id,
        cutter_id=cutter_id,
        source_split_hash=split_hash,
        fraction=fraction,
        random_seed=random_seed,
        full_train_count=len(train_records),
        selected_count=len(selected),
        stage_distribution=stage_distribution,
        cut_distribution=cut_distribution,
        windows=tuple(window_refs),
    )
    return TrainingSample(records=tuple(selected), manifest=attach_sample_hash(manifest))


def write_sample_manifest(manifest: TrainingSampleManifest, output_file: Path) -> Path:
    """原子写出训练小样本清单。"""

    verified = attach_sample_hash(manifest)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(verified.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return output_file


def load_sample_manifest(input_file: Path) -> TrainingSampleManifest:
    """加载小样本清单并重新计算哈希。"""

    manifest = TrainingSampleManifest.model_validate_json(Path(input_file).read_text(encoding="utf-8"))
    return attach_sample_hash(manifest)
