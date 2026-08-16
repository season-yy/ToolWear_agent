"""Streamlit 调用 ToolWear FastAPI 的唯一 HTTP 客户端。"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


JsonPayload = dict[str, Any]


class ToolApiError(RuntimeError):
    """把网络错误和后端稳定错误统一成页面可处理的异常。"""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int | None = None,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code
        self.trace_id = trace_id


class ToolWearApiClient:
    """封装页面所需 API；不读取运行目录，也不启动 CLI 子进程。"""

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0))
        self.transport = transport

    @staticmethod
    def _experiment_path(experiment_id: str) -> str:
        return f"/api/v1/experiments/{quote(experiment_id, safe='')}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: JsonPayload | None = None,
        idempotency_key: str | None = None,
        return_bytes: bool = False,
    ) -> Any:
        headers = {"Accept": "*/*" if return_bytes else "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
                headers=headers,
                # Tool API 固定运行在本机；禁止系统代理把 127.0.0.1 请求转发到外网。
                trust_env=False,
            ) as client:
                response = client.request(method, path, json=json_body)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ToolApiError(
                "API_UNREACHABLE",
                f"无法连接 ToolWear API：{exc}",
            ) from exc

        if response.is_error:
            error_code = "HTTP_ERROR"
            message = f"ToolWear API 返回 HTTP {response.status_code}。"
            trace_id: str | None = None
            try:
                payload = response.json()
                detail = payload.get("error", {}) if isinstance(payload, dict) else {}
                error_code = str(detail.get("error_code", error_code))
                message = str(detail.get("message", message))
                raw_trace_id = detail.get("trace_id")
                trace_id = str(raw_trace_id) if raw_trace_id else None
            except ValueError:
                pass
            raise ToolApiError(
                error_code,
                message,
                status_code=response.status_code,
                trace_id=trace_id,
            )
        if not response.content:
            return b"" if return_bytes else {}
        if return_bytes:
            return response.content
        try:
            return response.json()
        except ValueError as exc:
            raise ToolApiError(
                "INVALID_API_RESPONSE",
                "ToolWear API 返回了无法解析的 JSON。",
                status_code=response.status_code,
            ) from exc

    def health(self) -> JsonPayload:
        return self._request("GET", "/api/v1/health")

    def capabilities(self) -> JsonPayload:
        return self._request("GET", "/api/v1/capabilities")

    def agent_definitions(self) -> list[JsonPayload]:
        return self._request("GET", "/api/v1/agents")

    def datasets(self) -> list[JsonPayload]:
        return self._request("GET", "/api/v1/datasets")

    def list_experiments(self) -> list[JsonPayload]:
        return self._request("GET", "/api/v1/experiments")

    def create_experiment(
        self,
        payload: JsonPayload,
        *,
        idempotency_key: str,
    ) -> JsonPayload:
        return self._request(
            "POST",
            "/api/v1/experiments",
            json_body=payload,
            idempotency_key=idempotency_key,
        )

    def get_experiment(self, experiment_id: str) -> JsonPayload:
        return self._request("GET", self._experiment_path(experiment_id))

    def events(self, experiment_id: str) -> list[JsonPayload]:
        return self._request("GET", self._experiment_path(experiment_id) + "/events")

    def artifacts(self, experiment_id: str) -> list[JsonPayload]:
        return self._request("GET", self._experiment_path(experiment_id) + "/artifacts")

    def artifact_json(self, evidence_id: str) -> Any:
        return self._request(
            "GET",
            f"/api/v1/artifacts/{quote(evidence_id, safe='')}/content",
        )

    def artifact_bytes(self, evidence_id: str) -> bytes:
        """经 Bearer 鉴权读取图片等二进制证据。"""

        return self._request(
            "GET",
            f"/api/v1/artifacts/{quote(evidence_id, safe='')}/content",
            return_bytes=True,
        )

    def artifact_url(self, evidence_id: str) -> str:
        """返回受控证据下载地址；页面不会拼接或暴露任意文件路径。"""

        return self.base_url + f"/api/v1/artifacts/{quote(evidence_id, safe='')}/content"

    def latest_recommendations(self, experiment_id: str) -> JsonPayload:
        return self._request(
            "GET",
            self._experiment_path(experiment_id) + "/recommendations",
        )

    def runs(self, experiment_id: str) -> list[JsonPayload]:
        return self._request("GET", self._experiment_path(experiment_id) + "/runs")

    def agent_runs(self, experiment_id: str) -> list[JsonPayload]:
        return self._request(
            "GET",
            self._experiment_path(experiment_id) + "/agent-runs",
        )

    def invoke_agent(
        self,
        experiment_id: str,
        agent_name: str,
        payload: JsonPayload,
        *,
        idempotency_key: str,
    ) -> JsonPayload:
        path = self._experiment_path(experiment_id)
        return self._request(
            "POST",
            path + f"/agents/{quote(agent_name, safe='')}/invoke",
            json_body=payload,
            idempotency_key=idempotency_key,
        )

    def get_revision(self, experiment_id: str, revision: int) -> JsonPayload:
        return self._request(
            "GET",
            self._experiment_path(experiment_id) + f"/revisions/{revision}",
        )

    def get_run(self, experiment_id: str, run_id: str) -> JsonPayload:
        path = self._experiment_path(experiment_id)
        return self._request("GET", path + f"/runs/{quote(run_id, safe='')}")

    def run_logs(
        self,
        experiment_id: str,
        run_id: str,
        *,
        tail: int = 100,
    ) -> JsonPayload:
        path = self._experiment_path(experiment_id)
        return self._request(
            "GET",
            path + f"/runs/{quote(run_id, safe='')}/logs?tail={tail}",
        )

    def action(
        self,
        experiment_id: str,
        action: str,
        *,
        rationale: str,
        idempotency_key: str,
    ) -> JsonPayload:
        return self._request(
            "POST",
            self._experiment_path(experiment_id) + f"/{quote(action, safe='')}",
            json_body={"rationale": rationale},
            idempotency_key=idempotency_key,
        )

    def evaluate(
        self,
        experiment_id: str,
        *,
        rationale: str,
        force_refresh: bool,
        idempotency_key: str,
    ) -> JsonPayload:
        """生成诊断，或在保留旧证据的前提下只重试 LLM。"""

        return self._request(
            "POST",
            self._experiment_path(experiment_id) + "/evaluate",
            json_body={
                "rationale": rationale,
                "force_refresh": force_refresh,
            },
            idempotency_key=idempotency_key,
        )

    def generate_recommendations(
        self,
        experiment_id: str,
        *,
        user_request: str,
        force_refresh: bool,
        idempotency_key: str,
    ) -> JsonPayload:
        return self._request(
            "POST",
            self._experiment_path(experiment_id) + "/recommendations",
            json_body={
                "user_request": user_request,
                "force_refresh": force_refresh,
            },
            idempotency_key=idempotency_key,
        )

    def approve_pipeline(
        self,
        experiment_id: str,
        payload: JsonPayload,
        *,
        idempotency_key: str,
    ) -> JsonPayload:
        return self._request(
            "POST",
            self._experiment_path(experiment_id) + "/approve-pipeline",
            json_body=payload,
            idempotency_key=idempotency_key,
        )

    def start_mini_run(
        self,
        experiment_id: str,
        *,
        idempotency_key: str,
    ) -> JsonPayload:
        return self._request(
            "POST",
            self._experiment_path(experiment_id) + "/runs/mini",
            json_body={"rationale": "用户从实验台批准启动小样本训练。"},
            idempotency_key=idempotency_key,
        )

    def decide(
        self,
        experiment_id: str,
        *,
        action: str,
        rationale: str,
        idempotency_key: str,
    ) -> JsonPayload:
        return self._request(
            "POST",
            self._experiment_path(experiment_id) + "/decision",
            json_body={"action": action, "rationale": rationale},
            idempotency_key=idempotency_key,
        )
