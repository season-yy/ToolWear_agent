"""Tool API Bearer 鉴权与 AgentTeams Skill 调用审计。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from secrets import compare_digest
from threading import Lock
from time import perf_counter
from typing import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from toolwear_agent.agentteams.worker_skill_client import SKILL_OWNERS
from toolwear_agent.core.settings import Settings
from toolwear_agent.state import EntityNotFoundError


_PUBLIC_API_PATHS = {"/api/v1/health"}
_EXPERIMENT_PATH = re.compile(r"^/api/v1/experiments/([^/]+)")
_CORRELATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _error(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"error_code": error_code, "message": message, "trace_id": None, "context": {}}},
    )


class AgentTeamsAuditWriter:
    """以 JSONL 追加保存脱敏 Skill 调用事实。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def append(self, event: dict[str, object]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
        except OSError:
            # 审计写入失败不能掩盖原始 API 结果，服务日志会继续保留 HTTP 失败。
            return


def _bearer_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def _skill_identity(request: Request) -> tuple[str, str, str] | None:
    skill_name = request.headers.get("X-ToolWear-AgentTeams-Skill", "").strip()
    agent_name = request.headers.get("X-ToolWear-AgentTeams-Agent", "").strip()
    correlation_id = request.headers.get("X-ToolWear-Correlation-Id", "").strip()
    if not any((skill_name, agent_name, correlation_id)):
        return None
    if not all((skill_name, agent_name, correlation_id)):
        raise ValueError("AgentTeams Skill 调用必须同时提供 Skill、Agent 和 correlation ID。")
    owner = SKILL_OWNERS.get(skill_name)
    if owner is None or owner != agent_name:
        raise PermissionError("Skill 不属于声明的 Agent。")
    if not _CORRELATION_PATTERN.fullmatch(correlation_id):
        raise ValueError("correlation ID 格式不合法。")
    return skill_name, agent_name, correlation_id


def _experiment_and_trace(request: Request) -> tuple[str, str]:
    match = _EXPERIMENT_PATH.match(request.url.path)
    if match is None:
        return "", ""
    experiment_id = match.group(1)
    try:
        trace_id = request.app.state.container.repository.get_experiment(experiment_id).trace_id
    except (AttributeError, EntityNotFoundError):
        trace_id = ""
    return experiment_id, trace_id


def install_tool_api_security(app: FastAPI, settings: Settings) -> None:
    """安装一次鉴权与审计中间件；未配置 Token 时保持本机开发兼容。"""

    audit = AgentTeamsAuditWriter(settings.log_root / "agentteams" / "skill_invocations.jsonl")
    call_next_type = Callable[[Request], Awaitable[Response]]

    @app.middleware("http")
    async def tool_api_security(request: Request, call_next: call_next_type) -> Response:
        started = perf_counter()
        protected = request.url.path.startswith("/api/v1/") and request.url.path not in _PUBLIC_API_PATHS
        authenticated = False
        if protected and settings.tool_api_token:
            supplied = _bearer_token(request)
            authenticated = bool(supplied) and compare_digest(supplied, settings.tool_api_token)
            if not authenticated:
                return _error(401, "AUTHENTICATION_REQUIRED", "需要有效的 ToolWear API Bearer Token。")

        try:
            skill_identity = _skill_identity(request)
        except PermissionError as exc:
            return _error(403, "SKILL_PERMISSION_DENIED", str(exc))
        except ValueError as exc:
            return _error(422, "INVALID_SKILL_HEADERS", str(exc))

        response = await call_next(request)
        if skill_identity is not None:
            skill_name, agent_name, correlation_id = skill_identity
            experiment_id, trace_id = _experiment_and_trace(request)
            audit.append(
                {
                    "event_id": f"skill-call-{uuid4().hex[:24]}",
                    "event_type": "agentteams_skill_invocation",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "skill_name": skill_name,
                    "agent_name": agent_name,
                    "correlation_id": correlation_id,
                    "experiment_id": experiment_id,
                    "trace_id": trace_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                    "authentication": "bearer" if settings.tool_api_token else "disabled_local_mode",
                }
            )
            response.headers["X-ToolWear-Correlation-Id"] = correlation_id
        return response


__all__ = ["AgentTeamsAuditWriter", "install_tool_api_security"]
