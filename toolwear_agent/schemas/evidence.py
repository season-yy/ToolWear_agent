"""文件、指标、日志和协作记录的证据引用 Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex, utc_now


class EvidenceKind(str, Enum):
    """P0 证据的稳定类别。"""

    CONFIG = "config"
    SPLIT = "split"
    METRICS = "metrics"
    MODEL = "model"
    FIGURE = "figure"
    LOG = "log"
    TRACE = "trace"
    REPORT = "report"
    CODE = "code"
    APPROVAL = "approval"


class EvidenceRef(SchemaModel):
    """可校验、可定位但不内嵌大文件内容的证据索引。"""

    evidence_id: EntityId
    experiment_id: EntityId
    run_id: EntityId | None = None
    kind: EvidenceKind
    uri: NonEmptyText
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    description: str = ""
    created_by: str = "system"
    created_at: datetime = Field(default_factory=utc_now)
