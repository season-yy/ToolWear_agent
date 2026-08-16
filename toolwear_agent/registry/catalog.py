"""Module/Trainer Registry 的可审计 Catalog 输出。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from toolwear_agent.registry.module_registry import build_default_module_registry
from toolwear_agent.registry.trainer_registry import build_default_trainer_registry
from toolwear_agent.schemas import RegistryCatalog


def calculate_catalog_hash(catalog: RegistryCatalog) -> str:
    """计算不包含 `catalog_hash` 自身的稳定 SHA-256。"""

    payload = catalog.model_dump(mode="json", exclude={"catalog_hash"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def attach_catalog_hash(catalog: RegistryCatalog) -> RegistryCatalog:
    """返回带正确 hash 的新 Catalog，并拒绝不一致的已有 hash。"""

    expected = calculate_catalog_hash(catalog)
    if catalog.catalog_hash is not None and catalog.catalog_hash != expected:
        raise ValueError("RegistryCatalog 内容与 catalog_hash 不一致。")
    payload = catalog.model_dump(mode="python")
    payload["catalog_hash"] = expected
    return RegistryCatalog.model_validate(payload)


def build_default_registry_catalog() -> RegistryCatalog:
    """汇总默认输入、模块和训练器定义。"""

    module_registry = build_default_module_registry()
    trainer_registry = build_default_trainer_registry()
    catalog = RegistryCatalog(
        input_presets=module_registry.list_input_presets(),
        modules=module_registry.list_modules(),
        trainers=trainer_registry.list_trainers(),
    )
    return attach_catalog_hash(catalog)


def write_registry_catalog(catalog: RegistryCatalog, output_file: Path) -> Path:
    """原子写出 Catalog JSON，避免留下半写入文件。"""

    verified = attach_catalog_hash(catalog)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(verified.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return output_file


def load_registry_catalog(input_file: Path) -> RegistryCatalog:
    """加载 Catalog，并重新计算 hash 拒绝内容漂移。"""

    catalog = RegistryCatalog.model_validate_json(Path(input_file).read_text(encoding="utf-8"))
    attach_catalog_hash(catalog)
    return catalog
