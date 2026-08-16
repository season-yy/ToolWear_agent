"""受路径白名单保护的证据内容读取路由。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

from toolwear_agent.backend.dependencies import get_workflow
from toolwear_agent.services.workflow import ExperimentWorkflowService


router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])


@router.get("/{evidence_id}/content")
def artifact_content(
    evidence_id: str,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> Response:
    """返回小型 JSON/文本证据或浏览器可显示的图片，不接受任意路径。"""

    evidence, path = workflow.artifact_content(evidence_id)
    media_type = evidence.media_type
    if media_type == "application/json":
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    if media_type.startswith("text/") or media_type in {
        "application/x-ndjson",
        "text/csv",
    }:
        return PlainTextResponse(path.read_text(encoding="utf-8"), media_type=media_type)
    return FileResponse(path, media_type=media_type, filename=path.name)
