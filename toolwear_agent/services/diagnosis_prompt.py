"""评估诊断提示词、LLM JSON 提取和保守归一化。"""

from __future__ import annotations

import json
import re

from toolwear_agent.schemas.diagnosis import EvaluationFacts


PROMPT_TEMPLATE_VERSION = "evaluation-governor-v1"


def build_diagnosis_messages(facts: EvaluationFacts) -> list[dict[str, str]]:
    """只把 train/validation 事实放入提示词，隔离 final test。"""

    system = (
        "你是 EvaluationGovernorAgent。只能依据给定的 train/validation 事实诊断刀具磨损分类，"
        "不得假设或索取 final test 结果，不得声称已完成跨刀具泛化。只输出 JSON 对象。"
        "findings 请给 3-5 条，recommendations 请给 2-4 条；建议只能供人审批，不能声称已经执行。"
    )
    user = (
        "请输出 overall_conclusion、risk_level(low/medium/high)、findings、recommendations、"
        "recommended_action(approve_full/adjust_parameters/change_pipeline/stop)。\n"
        "finding 字段：finding_id,severity,category,title,detail,evidence；"
        "severity 只能是 info、warning、critical。\n"
        "recommendation 字段：recommendation_id,action_type,target,suggestion,rationale,priority,"
        "requires_human_approval=true；action_type 只能是 adjust_parameter、change_pipeline、"
        "inspect_data、approve_full、stop；priority 只能是 low、medium、high。\n"
        "所有 ID 使用 ASCII 小写字母、数字和连字符。\n事实 JSON："
        + facts.model_dump_json(exclude={"created_at"})
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_object(content: str) -> dict[str, object]:
    """兼容纯 JSON 与包裹在说明文字中的单个 JSON 对象。"""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM 诊断必须是 JSON 对象。")
    return payload


def _safe_entity_id(value: object, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-._")
    if not normalized or not normalized[0].isalnum():
        normalized = fallback
    return normalized[:64]


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_advice_payload(payload: dict[str, object]) -> dict[str, object]:
    """只归一化常见枚举和 ID 别名，缺失语义字段仍交给 Schema 拒绝。"""

    normalized = dict(payload)
    normalized["risk_level"] = {
        "critical": "high",
        "moderate": "medium",
        "warning": "medium",
    }.get(str(payload.get("risk_level", "")).lower(), payload.get("risk_level"))
    normalized["recommended_action"] = {
        "approve_full_train": "approve_full",
        "full_train": "approve_full",
        "adjust_parameter": "adjust_parameters",
        "tune_parameters": "adjust_parameters",
        "change_model": "change_pipeline",
        "terminate": "stop",
    }.get(
        str(payload.get("recommended_action", "")).lower(),
        payload.get("recommended_action"),
    )

    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list):
        raise ValueError("LLM findings 必须是 JSON 数组。")
    findings: list[dict[str, object]] = []
    for index, raw in enumerate(raw_findings, start=1):
        if not isinstance(raw, dict):
            raise ValueError("LLM finding 必须是 JSON 对象。")
        item = dict(raw)
        item["finding_id"] = _safe_entity_id(
            item.get("finding_id"),
            f"finding-{index}",
        )
        item["category"] = _safe_entity_id(
            item.get("category"),
            f"category-{index}",
        )
        item["severity"] = {
            "low": "info",
            "medium": "warning",
            "moderate": "warning",
            "high": "critical",
        }.get(str(item.get("severity", "")).lower(), item.get("severity"))
        if "evidence" in item:
            item["evidence"] = _as_text(item["evidence"])
        findings.append(item)
    normalized["findings"] = findings

    raw_recommendations = payload.get("recommendations")
    if not isinstance(raw_recommendations, list):
        raise ValueError("LLM recommendations 必须是 JSON 数组。")
    recommendations: list[dict[str, object]] = []
    for index, raw in enumerate(raw_recommendations, start=1):
        if not isinstance(raw, dict):
            raise ValueError("LLM recommendation 必须是 JSON 对象。")
        item = dict(raw)
        item["recommendation_id"] = _safe_entity_id(
            item.get("recommendation_id"),
            f"recommendation-{index}",
        )
        item["action_type"] = {
            "adjust_parameters": "adjust_parameter",
            "parameter_adjustment": "adjust_parameter",
            "approve_full_train": "approve_full",
            "full_train": "approve_full",
            "inspect_dataset": "inspect_data",
            "terminate": "stop",
        }.get(str(item.get("action_type", "")).lower(), item.get("action_type"))
        item["priority"] = {
            "critical": "high",
            "urgent": "high",
            "moderate": "medium",
            "warning": "medium",
        }.get(str(item.get("priority", "")).lower(), item.get("priority"))
        item["requires_human_approval"] = True
        recommendations.append(item)
    normalized["recommendations"] = recommendations
    return normalized


__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "build_diagnosis_messages",
    "extract_json_object",
    "normalize_advice_payload",
]
