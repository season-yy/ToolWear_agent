"""旧配置导入路径的兼容层。

新代码应从 :mod:`toolwear_agent.core.settings` 导入。本文件暂时保留，避免现有
C1 命令和测试在增量迁移期间一次性失效。
"""

from toolwear_agent.core.settings import Settings, find_env_file, load_settings

__all__ = ["Settings", "find_env_file", "load_settings"]
