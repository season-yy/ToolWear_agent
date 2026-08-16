"""DatasetManifest 的稳定序列化与完整性哈希工具。"""

from __future__ import annotations

import hashlib
import json

from toolwear_agent.schemas import DatasetManifest


def calculate_manifest_hash(manifest: DatasetManifest) -> str:
    """计算不包含 `manifest_hash` 自身的稳定 SHA-256。"""

    payload = manifest.model_dump(
        mode="json",
        exclude={"manifest_hash"},
        exclude_computed_fields=True,
    )
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def attach_manifest_hash(manifest: DatasetManifest) -> DatasetManifest:
    """返回带正确 hash 的新 Manifest；已有错误 hash 会被拒绝。"""

    expected = calculate_manifest_hash(manifest)
    if manifest.manifest_hash is not None and manifest.manifest_hash != expected:
        raise ValueError("DatasetManifest 内容与 manifest_hash 不一致。")
    payload = manifest.model_dump(mode="python", exclude_computed_fields=True)
    payload["manifest_hash"] = expected
    return DatasetManifest.model_validate(payload)
