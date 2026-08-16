"""单次 Agent LLM 调用、输出校验和审计构造。"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

from toolwear_agent.agents.catalog import AgentDefinition
from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import AgentLlmCallAudit, AgentTask
from toolwear_agent.schemas.agent_runtime import AgentOutputBase
from toolwear_agent.services.diagnosis_prompt import extract_json_object
from toolwear_agent.services.llm_chat import ChatClient, ChatCompletion


@dataclass(frozen=True)
class AgentCallOutcome:
    """运行时持久化所需的 LLM 调用结果。"""

    output: AgentOutputBase | None
    audit: AgentLlmCallAudit
    response_content: str
    error_code: str | None = None
    error_message: str = ""


def _messages(
    definition: AgentDefinition,
    task: AgentTask,
) -> list[dict[str, str]]:
    output_schema = definition.output_model.model_json_schema()
    system = (
        definition.system_prompt
        + " 只输出一个满足下述 JSON Schema 的对象，不要输出 Markdown 或解释："
        + json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
    )
    user = json.dumps(
        {
            "task_type": task.task_type,
            "objective": task.objective,
            "input": task.input_payload,
            "evidence_ids": task.evidence_ids,
            "requested_skills": task.requested_skills,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _failed_outcome(
    settings: Settings,
    definition: AgentDefinition,
    prompt_sha256: str,
    started: float,
    completion: ChatCompletion | None,
    error_code: str,
    exc: Exception,
) -> AgentCallOutcome:
    message = f"{type(exc).__name__}: {str(exc)[:300]}"
    return AgentCallOutcome(
        output=None,
        audit=AgentLlmCallAudit(
            provider=completion.provider if completion else settings.llm_provider,
            model=completion.model if completion else settings.llm_model,
            status="failed",
            latency_ms=(
                completion.latency_ms
                if completion
                else round((time.monotonic() - started) * 1000)
            ),
            prompt_template_version=definition.prompt_template_version,
            prompt_sha256=prompt_sha256,
            prompt_tokens=completion.prompt_tokens if completion else None,
            completion_tokens=completion.completion_tokens if completion else None,
            total_tokens=completion.total_tokens if completion else None,
            error_code=error_code,
            error_message=message,
        ),
        response_content=completion.content if completion else "",
        error_code=error_code,
        error_message=message,
    )


def execute_agent_llm(
    settings: Settings,
    chat_client: ChatClient,
    definition: AgentDefinition,
    task: AgentTask,
) -> AgentCallOutcome:
    """调用模型一次；任何失败都返回结构化 outcome，不向上伪装成功。"""

    messages = _messages(definition, task)
    prompt_sha256 = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    started = time.monotonic()
    try:
        completion = chat_client.complete(messages, temperature=0.1, json_mode=True)
    except (RuntimeError, OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        return _failed_outcome(
            settings,
            definition,
            prompt_sha256,
            started,
            None,
            "AGENT_LLM_CALL_FAILED",
            exc,
        )
    try:
        output = cast(
            AgentOutputBase,
            definition.output_model.model_validate(extract_json_object(completion.content)),
        )
    except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _failed_outcome(
            settings,
            definition,
            prompt_sha256,
            started,
            completion,
            "AGENT_OUTPUT_INVALID",
            exc,
        )
    return AgentCallOutcome(
        output=output,
        audit=AgentLlmCallAudit(
            provider=completion.provider,
            model=completion.model,
            status="success",
            latency_ms=completion.latency_ms,
            prompt_template_version=definition.prompt_template_version,
            prompt_sha256=prompt_sha256,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
        ),
        response_content=completion.content,
    )


def reject_agent_outcome(
    outcome: AgentCallOutcome,
    *,
    error_code: str,
    error_message: str,
) -> AgentCallOutcome:
    """把 Schema 合法但违反业务策略的响应转成可审计失败。"""

    audit_payload = outcome.audit.model_dump(mode="python")
    audit_payload.update(
        status="failed",
        error_code=error_code,
        error_message=error_message[:300],
    )
    return AgentCallOutcome(
        output=None,
        audit=AgentLlmCallAudit.model_validate(audit_payload),
        response_content=outcome.response_content,
        error_code=error_code,
        error_message=error_message[:300],
    )


__all__ = ["AgentCallOutcome", "execute_agent_llm", "reject_agent_outcome"]
