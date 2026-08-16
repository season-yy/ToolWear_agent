"""结构化 LLM 诊断 Provider，以及可审计的确定性降级策略。"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Protocol

from pydantic import ValidationError

from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas.diagnosis import (
    DiagnosisAdvice,
    EvaluationDiagnosis,
    EvaluationFacts,
    LlmCallAudit,
)
from toolwear_agent.services.diagnosis_prompt import (
    PROMPT_TEMPLATE_VERSION,
    build_diagnosis_messages,
    extract_json_object,
    normalize_advice_payload,
)
from toolwear_agent.services.diagnosis_rules import build_rule_based_advice
from toolwear_agent.services.evaluation_facts import build_evaluation_facts
from toolwear_agent.services.llm_chat import ChatClient, ChatCompletion, OpenAICompatibleChatClient


class DiagnosisProvider(Protocol):
    """评估服务依赖的诊断边界。"""

    def diagnose(self, facts: EvaluationFacts) -> EvaluationDiagnosis:
        """返回经过 Schema 校验的诊断。"""


class DefaultDiagnosisProvider:
    """调用千问生成建议；任何外部失败都显式回退到规则诊断。"""

    def __init__(self, settings: Settings, *, chat_client: ChatClient | None = None) -> None:
        self.settings = settings
        self.chat_client = chat_client or OpenAICompatibleChatClient(settings)

    def diagnose(self, facts: EvaluationFacts) -> EvaluationDiagnosis:
        messages = build_diagnosis_messages(facts)
        prompt_payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        prompt_sha256 = hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest()
        started = time.monotonic()
        completion: ChatCompletion | None = None
        try:
            completion = self.chat_client.complete(
                messages,
                temperature=0.1,
                json_mode=True,
            )
            advice = DiagnosisAdvice.model_validate(
                normalize_advice_payload(extract_json_object(completion.content))
            )
            audit = LlmCallAudit(
                provider=completion.provider,
                model=completion.model,
                status="success",
                used_fallback=False,
                latency_ms=completion.latency_ms,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                prompt_sha256=prompt_sha256,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=completion.total_tokens,
            )
        except (
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            OSError,
            TimeoutError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            advice = build_rule_based_advice(facts)
            reason = f"{type(exc).__name__}: {str(exc)[:240]}"
            audit = LlmCallAudit(
                provider=completion.provider if completion else self.settings.llm_provider,
                model=completion.model if completion else self.settings.llm_model,
                status="fallback",
                used_fallback=True,
                fallback_reason=reason,
                latency_ms=(
                    completion.latency_ms
                    if completion is not None
                    else round((time.monotonic() - started) * 1000)
                ),
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                prompt_sha256=prompt_sha256,
                prompt_tokens=completion.prompt_tokens if completion else None,
                completion_tokens=completion.completion_tokens if completion else None,
                total_tokens=completion.total_tokens if completion else None,
            )
        digest = hashlib.sha256(
            (facts.facts_id + audit.prompt_sha256).encode("utf-8")
        ).hexdigest()[:24]
        return EvaluationDiagnosis(
            diagnosis_id=f"diagnosis-{digest}",
            facts=facts,
            advice=advice,
            llm_call=audit,
        )


__all__ = [
    "DefaultDiagnosisProvider",
    "DiagnosisProvider",
    "build_evaluation_facts",
    "build_rule_based_advice",
]
