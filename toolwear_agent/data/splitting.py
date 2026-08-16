"""Cut 级数据切分的稳定哈希、持久化和修订锁。

本模块只管理“哪一个刀次属于哪个 split”，不负责生成窗口或训练模型。
把切分单独锁定后，即使某次模型效果不好，也不能在同一实验修订中重划 test。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Protocol

from toolwear_agent.core.errors import ToolWearError
from toolwear_agent.schemas import SplitAssignment, SplitLock, SplitManifest, SplitSpec


class CutLabelLike(Protocol):
    """构建 split manifest 所需的最小刀次标签接口。"""

    cut: int
    file_path: str
    row_count: int
    vb_value: float
    stage_id: int
    stage_name: str


class SplitLockConflictError(ToolWearError):
    """同一实验修订尝试使用不同 split 时抛出。"""

    error_code = "SPLIT_LOCK_CONFLICT"


def normalize_split_name(value: str) -> str:
    """把旧版 `val` 统一为正式名称 `validation`。"""

    normalized = value.strip().lower()
    aliases = {"val": "validation", "valid": "validation"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"train", "validation", "test"}:
        raise ValueError(f"未知 split 名称: {value}")
    return normalized


def build_split_manifest(
    *,
    cut_labels: Iterable[CutLabelLike],
    split_by_cut: dict[int, str],
    dataset_id: str,
    cutter_id: str,
    split_spec: SplitSpec,
) -> SplitManifest:
    """从刀次标签和归属映射构建稳定排序的 SplitManifest。"""

    assignments: list[SplitAssignment] = []
    for label in sorted(cut_labels, key=lambda item: item.cut):
        if label.cut not in split_by_cut:
            raise ValueError(f"刀次 {label.cut} 缺少 split 归属。")
        assignments.append(
            SplitAssignment(
                cutter_id=cutter_id,
                cut_id=label.cut,
                source_file=str(label.file_path).replace("\\", "/"),
                row_count=label.row_count,
                vb_value=label.vb_value,
                stage_id=label.stage_id,
                stage_name=label.stage_name,
                split=normalize_split_name(split_by_cut[label.cut]),
            )
        )

    unknown_cuts = set(split_by_cut) - {item.cut_id for item in assignments}
    if unknown_cuts:
        raise ValueError(f"split 映射包含不存在的刀次: {sorted(unknown_cuts)}")
    return SplitManifest(
        dataset_id=dataset_id,
        cutter_id=cutter_id,
        split_spec=split_spec,
        assignments=tuple(assignments),
    )


def calculate_split_hash(manifest: SplitManifest) -> str:
    """计算不包含 split_hash 自身的稳定 SHA-256。"""

    payload = manifest.model_dump(mode="json", exclude={"split_hash"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def attach_split_hash(manifest: SplitManifest) -> SplitManifest:
    """返回带正确哈希的 Manifest，并拒绝错误的已有哈希。"""

    expected = calculate_split_hash(manifest)
    if manifest.split_hash is not None and manifest.split_hash != expected:
        raise ValueError("SplitManifest 内容与 split_hash 不一致。")
    payload = manifest.model_dump(mode="python")
    payload["split_hash"] = expected
    return SplitManifest.model_validate(payload)


def write_split_manifest(manifest: SplitManifest, output_file: Path) -> Path:
    """原子写出带哈希的 split manifest。"""

    verified = attach_split_hash(manifest)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(verified.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return output_file


def load_split_manifest(input_file: Path) -> SplitManifest:
    """加载并重新计算哈希，拒绝人工篡改或半写入文件。"""

    manifest = SplitManifest.model_validate_json(Path(input_file).read_text(encoding="utf-8"))
    return attach_split_hash(manifest)


def load_split_lock(input_file: Path) -> SplitLock:
    """读取实验修订的 split lock。"""

    return SplitLock.model_validate_json(Path(input_file).read_text(encoding="utf-8"))


def _assert_lock_identity(
    lock: SplitLock,
    *,
    manifest: SplitManifest,
    experiment_id: str,
    revision: int,
) -> None:
    """校验已有 lock 是否与当前实验修订和 manifest 完全一致。"""

    conflicts: list[str] = []
    if lock.experiment_id != experiment_id:
        conflicts.append(f"experiment_id={lock.experiment_id}")
    if lock.revision != revision:
        conflicts.append(f"revision={lock.revision}")
    if lock.dataset_id != manifest.dataset_id:
        conflicts.append(f"dataset_id={lock.dataset_id}")
    if lock.cutter_id != manifest.cutter_id:
        conflicts.append(f"cutter_id={lock.cutter_id}")
    if lock.split_hash != manifest.split_hash:
        conflicts.append(f"split_hash={lock.split_hash}")
    if conflicts:
        raise SplitLockConflictError(
            "同一实验修订的 split 已锁定，不能用新结果覆盖；已有值: " + ", ".join(conflicts)
        )


def create_or_verify_split_lock(
    *,
    manifest: SplitManifest,
    lock_file: Path,
    experiment_id: str,
    revision: int,
    manifest_file: Path,
) -> SplitLock:
    """首次创建 split lock；后续调用只能验证相同 split，不能替换。"""

    verified_manifest = attach_split_hash(manifest)
    if verified_manifest.split_hash is None:  # pragma: no cover - attach 后的类型防御
        raise ValueError("split_hash 不能为空。")

    lock_file = Path(lock_file)
    if lock_file.exists():
        existing = load_split_lock(lock_file)
        _assert_lock_identity(
            existing,
            manifest=verified_manifest,
            experiment_id=experiment_id,
            revision=revision,
        )
        return existing

    lock = SplitLock(
        experiment_id=experiment_id,
        revision=revision,
        dataset_id=verified_manifest.dataset_id,
        cutter_id=verified_manifest.cutter_id,
        split_hash=verified_manifest.split_hash,
        manifest_file=str(Path(manifest_file).absolute()),
    )
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = lock_file.with_suffix(lock_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(lock.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(lock_file)
    return lock


def assert_manifest_matches_lock(manifest: SplitManifest, lock: SplitLock) -> None:
    """在训练前再次确认 manifest 未脱离其修订锁。"""

    verified = attach_split_hash(manifest)
    _assert_lock_identity(
        lock,
        manifest=verified,
        experiment_id=lock.experiment_id,
        revision=lock.revision,
    )
