"""日志初始化工具。

早期先使用 Python 标准库 logging，避免引入过多依赖。
后续如果需要结构化 JSON 日志，可以在这里统一替换，不影响业务代码。
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """获取统一格式的 logger。

    如果 logger 已经配置过 handler，就直接复用，避免重复打印日志。
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
