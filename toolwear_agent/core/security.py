"""不依赖 Web 框架的输入与路径安全工具。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from toolwear_agent.core.errors import InvalidIdentifierError, PathBoundaryError


# ID 会直接参与目录构造，因此只允许短横线、下划线和点等非路径字符。
_ENTITY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_entity_id(value: str, *, field_name: str = "entity_id") -> str:
    """校验用于目录和数据库主键的实体 ID。

    这里不做自动替换。静默把非法字符换掉会导致两个不同输入落到同一目录，
    因此非法输入必须由调用方明确修正。
    """

    if not _ENTITY_ID_PATTERN.fullmatch(value):
        raise InvalidIdentifierError(
            f"{field_name} 必须以字母或数字开头，只能包含字母、数字、点、下划线和短横线，长度不超过 64。"
        )
    return value


def resolve_path(path: str | Path) -> Path:
    """返回规范化的绝对路径，并解析已有的 Junction/符号链接。"""

    return Path(path).expanduser().resolve(strict=False)


def ensure_path_within(path: str | Path, allowed_roots: Iterable[str | Path]) -> Path:
    """确认路径位于至少一个允许根目录内。

    候选路径和根目录都会先 `resolve`。因此在 Windows 上，Junction 的真实目标
    若不在显式允许列表中也会被拒绝，不能只凭表面路径绕过边界。
    """

    resolved_path = resolve_path(path)
    resolved_roots = [resolve_path(root) for root in allowed_roots]
    if not resolved_roots:
        raise PathBoundaryError("允许根目录列表为空，拒绝访问。")

    for root in resolved_roots:
        if resolved_path == root or root in resolved_path.parents:
            return resolved_path

    roots_text = ", ".join(str(root) for root in resolved_roots)
    raise PathBoundaryError(f"路径不在允许根目录内：{resolved_path}；允许范围：{roots_text}")
