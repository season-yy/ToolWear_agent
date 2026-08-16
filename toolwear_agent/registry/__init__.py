"""模块、训练器及 Pipeline 兼容性注册入口。"""

from toolwear_agent.registry.module_registry import (
    PHM2010_CHANNEL_IDS,
    ModuleRegistry,
    build_default_module_registry,
)
from toolwear_agent.registry.catalog import (
    build_default_registry_catalog,
    load_registry_catalog,
    write_registry_catalog,
)
from toolwear_agent.registry.trainer_registry import TrainerRegistry, build_default_trainer_registry
from toolwear_agent.registry.validation import (
    validate_pipeline_against_registries,
    validate_pipeline_with_default_registries,
)

__all__ = [
    "ModuleRegistry",
    "PHM2010_CHANNEL_IDS",
    "TrainerRegistry",
    "build_default_module_registry",
    "build_default_registry_catalog",
    "build_default_trainer_registry",
    "load_registry_catalog",
    "validate_pipeline_against_registries",
    "validate_pipeline_with_default_registries",
    "write_registry_catalog",
]
