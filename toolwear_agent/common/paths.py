"""运行目录管理。

本模块只负责路径创建和路径检查，不负责读写业务数据。
这样做的好处是：后续训练脚本、报告生成、后端服务都能复用同一套目录规则。
"""

from __future__ import annotations

from pathlib import Path

from toolwear_agent.common.config import Settings
from toolwear_agent.core.paths import PathResolver


def runtime_dirs(settings: Settings) -> list[Path]:
    """返回项目运行时需要提前存在的目录列表。"""

    return list(PathResolver(settings).runtime_directories())


def ensure_runtime_dirs(settings: Settings) -> list[Path]:
    """确保运行目录存在。

    这里只创建明确配置过的目录，不扫描、不清理、不删除任何路径。
    """

    return list(PathResolver(settings).ensure_runtime_directories())
