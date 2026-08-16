"""项目内可稳定映射到 API `error_code` 的错误类型。"""

from __future__ import annotations


class ToolWearError(Exception):
    """所有可预期业务错误的基类。"""

    error_code = "TOOLWEAR_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidIdentifierError(ToolWearError):
    """实体 ID 为空、过长或包含路径字符。"""

    error_code = "INVALID_IDENTIFIER"


class PathBoundaryError(ToolWearError):
    """解析后的路径逃出了允许访问的根目录。"""

    error_code = "PATH_OUTSIDE_ALLOWED_ROOT"
