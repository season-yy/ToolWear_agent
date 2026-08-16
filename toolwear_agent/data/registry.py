"""强类型 Dataset Registry 的 YAML 持久化实现。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from toolwear_agent.data.manifest import attach_manifest_hash, calculate_manifest_hash
from toolwear_agent.schemas import DatasetManifest
from toolwear_agent.schemas.base import SchemaModel


class DatasetRegistryDocument(SchemaModel):
    """Registry YAML 文件的顶层结构。"""

    datasets: dict[str, DatasetManifest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _keys_match_dataset_ids(self) -> "DatasetRegistryDocument":
        for key, manifest in self.datasets.items():
            if key != manifest.dataset_id:
                raise ValueError(f"datasets 键 {key} 与 dataset_id {manifest.dataset_id} 不一致。")
        return self


class DatasetRegistry:
    """管理可供 API、Streamlit 和 Agent 共用的数据集清单。"""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path).expanduser().absolute()
        self._document = self._load() if self.registry_path.is_file() else DatasetRegistryDocument()

    def _load(self) -> DatasetRegistryDocument:
        """从 YAML 加载并验证所有 Manifest 的内容哈希。"""

        raw = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        document = DatasetRegistryDocument.model_validate(raw)
        for manifest in document.datasets.values():
            attach_manifest_hash(manifest)
        return document

    def _save(self) -> None:
        """先写同目录临时文件再原子替换，避免写到一半留下损坏 YAML。"""

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._document.model_dump(mode="json", exclude_computed_fields=True)
        content = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        temporary_path = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(self.registry_path)

    def register(self, manifest: DatasetManifest) -> DatasetManifest:
        """新增或更新一份经过 hash 校验的 Manifest。"""

        verified = attach_manifest_hash(manifest)
        datasets = dict(self._document.datasets)
        datasets[verified.dataset_id] = verified
        self._document = DatasetRegistryDocument(datasets=datasets)
        self._save()
        return verified

    def get(self, dataset_id: str) -> DatasetManifest:
        """按 ID 返回数据集；不存在时抛出语义明确的 KeyError。"""

        try:
            return self._document.datasets[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Dataset Registry 中不存在数据集: {dataset_id}") from exc

    def list(self) -> tuple[DatasetManifest, ...]:
        """按 dataset_id 稳定排序返回所有注册数据集。"""

        return tuple(self._document.datasets[key] for key in sorted(self._document.datasets))

    def allowed_resolved_roots(self, dataset_id: str) -> tuple[Path, ...]:
        """返回 manifest 已体检的真实刀具目录，供路径安全边界使用。"""

        manifest = self.get(dataset_id)
        return tuple(
            cutter.resolved_path
            for cutter in manifest.cutters.values()
            if cutter.available and cutter.resolved_path is not None
        )
