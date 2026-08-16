"""AgentTeams Worker 内可独立执行的 ToolWear HTTP Skill 客户端。"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


RouteSpec = tuple[str, str, bool]
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ALLOWED_API_HOSTS = {"host.docker.internal", "127.0.0.1", "localhost", "toolwear-api"}

SKILL_OWNERS = {
    "toolwear-data-profile": "DataStewardAgent",
    "toolwear-stage-label": "DataStewardAgent",
    "toolwear-window-split": "DataStewardAgent",
    "toolwear-pipeline-recommend": "AlgorithmArchitectAgent",
    "toolwear-human-selection": "ExperimentManagerAgent",
    "toolwear-mini-train": "CodeTrainingEngineerAgent",
    "toolwear-visualization": "EvaluationGovernorAgent",
    "toolwear-diagnosis": "EvaluationGovernorAgent",
    "toolwear-decision": "EvaluationGovernorAgent",
    "toolwear-report-trace": "ReportMemoryCuratorAgent",
}

SKILL_ROUTES: dict[str, dict[str, RouteSpec]] = {
    "toolwear-data-profile": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/profile", True),
    },
    "toolwear-stage-label": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/artifacts", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/labels", True),
    },
    "toolwear-window-split": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/artifacts", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/split", True),
    },
    "toolwear-pipeline-recommend": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/recommendations", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/recommendations", True),
    },
    "toolwear-human-selection": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/recommendations", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/approve-pipeline", True),
    },
    "toolwear-mini-train": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/runs", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/runs/mini", True),
        "validate": ("POST", "/api/v1/experiments/{experiment_id}/validate", True),
        "run-status": ("GET", "/api/v1/experiments/{experiment_id}/runs/{run_id}", False),
        "run-logs": ("GET", "/api/v1/experiments/{experiment_id}/runs/{run_id}/logs", False),
    },
    "toolwear-visualization": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/runs", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/evaluate", True),
    },
    "toolwear-diagnosis": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/agent-runs", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/evaluate", True),
    },
    "toolwear-decision": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/events", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/decision", True),
    },
    "toolwear-report-trace": {
        "inspect": ("GET", "/api/v1/experiments/{experiment_id}/artifacts", False),
        "execute": ("POST", "/api/v1/experiments/{experiment_id}/report", True),
    },
}


class SkillClientError(ValueError):
    """Skill 输入违反白名单或安全约束。"""


@dataclass(frozen=True)
class PreparedSkillRequest:
    """完成边界校验、尚未发往网络的 HTTP 请求。"""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    skill_name: str
    owner_agent: str
    operation: str
    experiment_id: str
    correlation_id: str
    is_write: bool


def _validate_id(name: str, value: object, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if not text and not required:
        return ""
    if not _ID_PATTERN.fullmatch(text):
        raise SkillClientError(f"{name} 为空或包含不允许的字符。")
    return text


def _normalize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _ALLOWED_API_HOSTS:
        raise SkillClientError("TOOLWEAR_API_BASE_URL 不是允许的 ToolWear API 地址。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SkillClientError("TOOLWEAR_API_BASE_URL 不得包含凭据、查询参数或片段。")
    return base_url.rstrip("/")


def read_api_token(
    *,
    env: Mapping[str, str] | None = None,
    default_file: Path = Path("/run/secrets/toolwear_api_token"),
) -> str:
    """从环境变量或容器 Secret 文件读取 Tool API Token，不输出其内容。"""

    values = os.environ if env is None else env
    direct = values.get("TOOLWEAR_API_TOKEN", "").strip()
    if direct:
        return direct
    token_file = Path(values.get("TOOLWEAR_API_TOKEN_FILE", str(default_file)))
    if not token_file.is_file():
        return ""
    return token_file.read_text(encoding="utf-8").strip()


def build_http_request(
    *,
    skill_name: str,
    invocation: Mapping[str, object],
    base_url: str,
    token: str,
) -> PreparedSkillRequest:
    """把结构化 Skill 输入转换为固定白名单 HTTP 请求。"""

    if skill_name not in SKILL_ROUTES:
        raise SkillClientError(f"未知 Skill：{skill_name}")
    operation = str(invocation.get("operation", "")).strip()
    route = SKILL_ROUTES[skill_name].get(operation)
    if route is None:
        raise SkillClientError(f"{skill_name} 不允许 operation={operation!r}。")
    method, path_template, is_write = route
    experiment_id = _validate_id("experiment_id", invocation.get("experiment_id"))
    correlation_id = _validate_id("correlation_id", invocation.get("correlation_id"))
    run_id = _validate_id("run_id", invocation.get("run_id"), required="{run_id}" in path_template)
    if is_write and invocation.get("confirm_write") is not True:
        raise SkillClientError("写操作必须显式设置 confirm_write=true。")
    idempotency_key = _validate_id(
        "idempotency_key",
        invocation.get("idempotency_key"),
        required=is_write,
    )
    payload = invocation.get("payload", {})
    if not isinstance(payload, dict):
        raise SkillClientError("payload 必须是 JSON object。")
    if not is_write and payload:
        raise SkillClientError("只读操作不得携带 payload。")

    path = path_template.format(experiment_id=experiment_id, run_id=run_id)
    headers = {
        "Accept": "application/json",
        "X-ToolWear-AgentTeams-Skill": skill_name,
        "X-ToolWear-AgentTeams-Agent": SKILL_OWNERS[skill_name],
        "X-ToolWear-Correlation-Id": correlation_id,
    }
    body = None
    if is_write:
        headers["Content-Type"] = "application/json"
        headers["Idempotency-Key"] = idempotency_key
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return PreparedSkillRequest(
        method=method,
        url=_normalize_base_url(base_url) + path,
        headers=headers,
        body=body,
        skill_name=skill_name,
        owner_agent=SKILL_OWNERS[skill_name],
        operation=operation,
        experiment_id=experiment_id,
        correlation_id=correlation_id,
        is_write=is_write,
    )


def _extract_trace_id(value: object) -> str:
    if isinstance(value, dict):
        trace_id = value.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            return trace_id
        for nested in value.values():
            found = _extract_trace_id(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _extract_trace_id(nested)
            if found:
                return found
    return ""


def _extract_evidence(value: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "evidence_id" in value and "sha256" in value:
            found.append(value)
        for nested in value.values():
            found.extend(_extract_evidence(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_extract_evidence(nested))
    return found


Transport = Callable[[PreparedSkillRequest, float], tuple[int, object]]


def _urllib_transport(prepared: PreparedSkillRequest, timeout_seconds: float) -> tuple[int, object]:
    request = Request(
        prepared.url,
        data=prepared.body,
        headers=prepared.headers,
        method=prepared.method,
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
        return response.status, json.loads(raw) if raw else {}


def execute_request(
    prepared: PreparedSkillRequest,
    *,
    timeout_seconds: float = 30.0,
    retries: int = 2,
    transport: Transport = _urllib_transport,
) -> dict[str, object]:
    """执行请求；仅对瞬时网络错误和明确可重试状态进行有限重试。"""

    attempts = 0
    while True:
        attempts += 1
        try:
            status_code, data = transport(prepared, timeout_seconds)
            return {
                "ok": 200 <= status_code < 300,
                "skill_name": prepared.skill_name,
                "owner_agent": prepared.owner_agent,
                "operation": prepared.operation,
                "experiment_id": prepared.experiment_id,
                "correlation_id": prepared.correlation_id,
                "status_code": status_code,
                "attempts": attempts,
                "trace_id": _extract_trace_id(data),
                "evidence_refs": _extract_evidence(data),
                "data": data,
                "error": None,
            }
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                error_data: object = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                error_data = {"message": "Tool API 返回了非 JSON 错误。"}
            if exc.code in {408, 429, 502, 503, 504} and attempts <= retries:
                time.sleep(min(0.5 * attempts, 2.0))
                continue
            return _error_envelope(prepared, exc.code, attempts, error_data)
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempts <= retries:
                time.sleep(min(0.5 * attempts, 2.0))
                continue
            return _error_envelope(
                prepared,
                0,
                attempts,
                {"error_code": "TOOL_API_UNREACHABLE", "message": type(exc).__name__},
            )


def _error_envelope(
    prepared: PreparedSkillRequest,
    status_code: int,
    attempts: int,
    error: object,
) -> dict[str, object]:
    return {
        "ok": False,
        "skill_name": prepared.skill_name,
        "owner_agent": prepared.owner_agent,
        "operation": prepared.operation,
        "experiment_id": prepared.experiment_id,
        "correlation_id": prepared.correlation_id,
        "status_code": status_code,
        "attempts": attempts,
        "trace_id": "",
        "evidence_refs": [],
        "data": None,
        "error": error,
    }


def _parse_cli(argv: list[str] | None) -> tuple[str, dict[str, object]]:
    parser = argparse.ArgumentParser(description="受控调用 ToolWear FastAPI 的 AgentTeams Worker Skill。")
    parser.add_argument("--operation", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--correlation-id", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--payload-file", type=Path)
    parser.add_argument("--idempotency-key", default="")
    parser.add_argument("--confirm-write", action="store_true")
    parser.add_argument("--skill-name", default="")
    args = parser.parse_args(argv)
    skill_name = args.skill_name or Path(__file__).resolve().parents[1].name
    payload = {}
    if args.payload_file:
        payload = json.loads(args.payload_file.read_text(encoding="utf-8"))
    return skill_name, {
        "operation": args.operation,
        "experiment_id": args.experiment_id,
        "correlation_id": args.correlation_id,
        "run_id": args.run_id,
        "payload": payload,
        "idempotency_key": args.idempotency_key,
        "confirm_write": args.confirm_write,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口只向 stdout 输出单个 JSON，不输出 Token 或请求头。"""

    try:
        skill_name, invocation = _parse_cli(argv)
        prepared = build_http_request(
            skill_name=skill_name,
            invocation=invocation,
            base_url=os.environ.get(
                "TOOLWEAR_API_BASE_URL",
                "http://host.docker.internal:18100",
            ),
            token=read_api_token(),
        )
        result = execute_request(
            prepared,
            timeout_seconds=float(os.environ.get("TOOLWEAR_API_TIMEOUT_SECONDS", "30")),
            retries=int(os.environ.get("TOOLWEAR_API_RETRIES", "2")),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"ok": False, "error": {"error_code": "SKILL_INPUT_ERROR", "message": str(exc)}}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
