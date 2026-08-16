"""所有 ToolWear API、状态和 Agent 消息共享的 Schema 基础类型。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, StringConstraints


SCHEMA_VERSION = "1.0"

EntityId: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        strip_whitespace=True,
    ),
]
NonEmptyText: TypeAlias = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Sha256Hex: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Fa-f0-9]{64}$", to_lower=True),
]


def utc_now() -> datetime:
    """返回带 UTC 时区的当前时间，避免本地时区混入状态库。"""

    return datetime.now(timezone.utc)


class SchemaModel(BaseModel):
    """默认禁止未知字段且不可原地修改的契约模型。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=False,
    )

    schema_version: Literal["1.0"] = SCHEMA_VERSION
