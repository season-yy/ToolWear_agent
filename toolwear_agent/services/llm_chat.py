"""最小 OpenAI 兼容聊天客户端，只负责一次受控 JSON 调用。"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from toolwear_agent.core.settings import Settings


@dataclass(frozen=True)
class ChatCompletion:
    """LLM 正文及可观测元数据，不保存 API Key。"""

    content: str
    provider: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatClient(Protocol):
    """评估服务可注入的最小 LLM 边界。"""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        json_mode: bool,
    ) -> ChatCompletion:
        """返回一次聊天完成结果。"""


class OpenAICompatibleChatClient:
    """复用项目已有千问/OpenAI-compatible 配置。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _optional_token(usage: dict[str, object], key: str) -> int | None:
        value = usage.get(key)
        return int(value) if isinstance(value, int) and value >= 0 else None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        json_mode: bool = True,
    ) -> ChatCompletion:
        if not self.settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY 为空")
        if not self.settings.llm_model:
            raise RuntimeError("LLM_MODEL 为空")

        payload: dict[str, object] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1800,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.settings.llm_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(
            request,
            timeout=self.settings.llm_timeout_seconds,
        ) as response:
            raw = json.loads(response.read().decode("utf-8"))
        latency_ms = round((time.monotonic() - started) * 1000)

        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM 响应缺少 choices。")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM 响应缺少非空 content。")
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return ChatCompletion(
            content=content,
            provider=self.settings.llm_provider,
            model=self.settings.llm_model,
            latency_ms=latency_ms,
            prompt_tokens=self._optional_token(usage, "prompt_tokens"),
            completion_tokens=self._optional_token(usage, "completion_tokens"),
            total_tokens=self._optional_token(usage, "total_tokens"),
        )
