"""LLM 候选方案生成。

AlgorithmArchitectAgent 通过千问 OpenAI 兼容接口生成候选方案；系统随后做
Schema 和白名单校验。LLM 只负责建议和解释，不直接写代码、不执行训练。
"""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from toolwear_agent.common.config import Settings


TRAINABLE_PLAN_IDS = {
    "statistical_features_random_forest",
    "statistical_features_extra_trees",
    "multichannel_window_1d_cnn",
}


@dataclass(frozen=True)
class LlmCandidatePlan:
    """LLM 生成后的候选方案。"""

    plan_id: str
    display_name: str
    module_pipeline: list[str]
    reason: str
    risk: str
    expected_cost: str
    trainable_now: bool
    training_backend: str
    experimental_extension: dict[str, object] | None = None


@dataclass(frozen=True)
class LlmCandidateSet:
    """LLM 候选方案集合。"""

    dataset_id: str
    cutter: str
    user_request: str
    provider: str
    model: str
    used_fallback: bool
    fallback_reason: str
    plans: list[LlmCandidatePlan]


def _fallback_plans(
    user_request: str,
    reason: str,
    *,
    dataset_id: str = "phm2010",
    cutter: str = "c1",
) -> LlmCandidateSet:
    """当 LLM 不可用时，返回可审计的规则候选。"""

    plans = [
        LlmCandidatePlan(
            plan_id="statistical_features_random_forest",
            display_name="统计特征 + RandomForest",
            module_pipeline=["窗口统计特征", "RandomForestClassifier", "Macro-F1/Balanced Accuracy 评估"],
            reason="适合作为低成本、可解释、可快速复现的 P0 基线。",
            risk="C1 内部高分不代表跨刀具泛化，需要后续 C4/C6 验证。",
            expected_cost="低",
            trainable_now=True,
            training_backend="sklearn_random_forest",
        ),
        LlmCandidatePlan(
            plan_id="statistical_features_extra_trees",
            display_name="统计特征 + ExtraTrees",
            module_pipeline=["窗口统计特征", "ExtraTreesClassifier", "Macro-F1/Balanced Accuracy 评估"],
            reason="与 RandomForest 同属树模型，但随机性更强，可作为快速对照候选。",
            risk="如果特征已经过强，可能与 RandomForest 指标接近，无法证明深度模型价值。",
            expected_cost="低",
            trainable_now=True,
            training_backend="sklearn_extra_trees",
        ),
        LlmCandidatePlan(
            plan_id="multichannel_window_1d_cnn",
            display_name="多通道窗口 + 1D CNN",
            module_pipeline=["多通道窗口", "轻量 1D CNN", "真实 epoch loss 曲线"],
            reason="更贴近原始时序建模，后续可扩展到多分支融合和注意力。",
            risk="小样本下可能过拟合，需要结合真实 loss 和 validation 指标判断。",
            expected_cost="中",
            trainable_now=True,
            training_backend="pytorch",
        ),
    ]
    return LlmCandidateSet(dataset_id, cutter, user_request, "fallback", "", True, reason, plans)


def _extract_json_array(text: str) -> list[dict[str, object]]:
    """从模型返回文本中提取 JSON 数组。"""

    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("LLM 返回中未找到 JSON 数组")
    return json.loads(text[start : end + 1])


def normalize_plan_id(raw_plan_id: str, module_pipeline: list[str], training_backend: str) -> str:
    """把 LLM 可能生成的别名规范化到本地白名单 ID。"""

    text = " ".join([raw_plan_id, training_backend, *module_pipeline]).lower()
    alias_tokens = {
        token for token in re.split(r"[^a-z0-9]+", text) if token
    }
    if (
        "random_forest" in text
        or "random forest" in text
        or "randomforest" in text
        or "rf" in alias_tokens
        or "随机森林" in text
    ):
        return "statistical_features_random_forest"
    if (
        "extra_trees" in text
        or "extra trees" in text
        or "extratrees" in text
        or "et" in alias_tokens
        or "极端随机树" in text
    ):
        return "statistical_features_extra_trees"
    if (
        "1d_cnn" in text
        or "1d cnn" in text
        or "cnn_1d" in text
        or "1dcnn" in text
        or ({"1d", "cnn"} <= alias_tokens)
    ):
        return "multichannel_window_1d_cnn"
    return raw_plan_id


def _plan_from_dict(data: dict[str, object]) -> LlmCandidatePlan:
    """把 LLM 字典转换成候选方案，并套用白名单约束。"""

    from toolwear_agent.schemas import ExperimentalExtensionSpec

    module_pipeline = [str(item) for item in data["module_pipeline"]]
    training_backend = str(data["training_backend"])
    plan_id = normalize_plan_id(str(data["plan_id"]), module_pipeline, training_backend)
    raw_extension = data.get("experimental_extension")
    experimental_extension: dict[str, object] | None = None
    if raw_extension is not None:
        if not isinstance(raw_extension, dict):
            raise ValueError("experimental_extension 必须是 JSON 对象。")
        experimental_extension = ExperimentalExtensionSpec.model_validate(raw_extension).model_dump(
            mode="python"
        )
    return LlmCandidatePlan(
        plan_id=plan_id,
        display_name=str(data["display_name"]),
        module_pipeline=module_pipeline,
        reason=str(data["reason"]),
        risk=str(data["risk"]),
        expected_cost=str(data["expected_cost"]),
        trainable_now=(
            plan_id in TRAINABLE_PLAN_IDS
            and bool(data.get("trainable_now", False))
            and experimental_extension is None
        ),
        training_backend=training_backend if plan_id in TRAINABLE_PLAN_IDS else "not_implemented_yet",
        experimental_extension=experimental_extension,
    )


def validate_candidate_payload(raw_plans: list[dict[str, object]]) -> list[LlmCandidatePlan]:
    """校验 LLM 候选，并确认每项都能转换成 PipelineSpec。"""

    # 使用函数内导入，避免 schemas.converters 在导入本模块类型时产生循环。
    from toolwear_agent.registry import validate_pipeline_with_default_registries
    from toolwear_agent.schemas.converters import llm_candidate_plan_to_pipeline

    required_keys = {
        "plan_id",
        "display_name",
        "module_pipeline",
        "reason",
        "risk",
        "expected_cost",
        "trainable_now",
        "training_backend",
    }
    plans: list[LlmCandidatePlan] = []
    for raw_plan in raw_plans[:3]:
        missing = required_keys - set(raw_plan)
        if missing:
            raise ValueError(f"LLM 候选方案缺少字段: {sorted(missing)}")
        plan = _plan_from_dict(raw_plan)
        pipeline = llm_candidate_plan_to_pipeline(plan)
        validation = validate_pipeline_with_default_registries(pipeline)
        if not validation.valid:
            messages = [
                issue.message for issue in validation.issues if issue.severity.value == "error"
            ]
            raise ValueError("LLM 候选未通过 Module Registry：" + "；".join(messages))
        plans.append(plan)
    if not plans:
        raise ValueError("LLM 未生成有效候选方案")
    return plans


def build_architect_prompt(
    user_request: str,
    *,
    dataset_id: str = "phm2010",
    cutter: str = "c1",
) -> list[dict[str, str]]:
    """构造 AlgorithmArchitectAgent 的 LLM 消息。"""

    from toolwear_agent.registry import build_default_module_registry

    registry = build_default_module_registry()
    registered_ids = ", ".join(module.module_id for module in registry.list_modules())
    system_prompt = (
        f"你是 AlgorithmArchitectAgent，负责为 {dataset_id} {cutter} 刀具磨损四阶段分类生成候选算法方案。"
        "只能输出 JSON 数组，不要输出 Markdown。候选必须考虑小样本训练、cut 级别防泄漏和 12GB 显存。"
        "当前可训练白名单包括 statistical_features_random_forest、"
        "statistical_features_extra_trees 和 multichannel_window_1d_cnn；"
        "DANN、多分支注意力等未实现结构只能作为不可训练候选展示。"
        f"当前 Module Registry ID 为：{registered_ids}。"
        "不得把未登记能力写入 module_pipeline；如确需提出新能力，必须增加 experimental_extension 对象，"
        "其中包含 extension_id、display_name、kind、rationale，并将 trainable_now 设为 false。"
    )
    user_prompt = (
        f"用户需求：{user_request}\n"
        f"数据背景：{dataset_id} {cutter}，工业多通道时序，小样本比例默认 20%，"
        "标签为 VB max 按 90/130/160 um 生成四阶段。\n"
        "请生成 2-3 个候选，每个对象必须包含字段："
        "plan_id, display_name, module_pipeline, reason, risk, expected_cost, trainable_now, training_backend。"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def call_openai_compatible_chat(settings: Settings, messages: list[dict[str, str]]) -> str:
    """调用 OpenAI 兼容 Chat Completions 接口。"""

    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY 为空")
    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL 为空")

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=settings.llm_timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def generate_llm_candidate_set(
    settings: Settings,
    user_request: str,
    *,
    dataset_id: str = "phm2010",
    cutter: str = "c1",
) -> LlmCandidateSet:
    """生成 LLM 候选方案；失败时降级为规则候选。"""

    try:
        content = call_openai_compatible_chat(
            settings,
            build_architect_prompt(user_request, dataset_id=dataset_id, cutter=cutter),
        )
        raw = json.loads(content)
        raw_plans = raw["plans"] if isinstance(raw, dict) and "plans" in raw else _extract_json_array(content)
        plans = validate_candidate_payload(raw_plans)
        return LlmCandidateSet(
            dataset_id,
            cutter,
            user_request,
            settings.llm_provider,
            settings.llm_model,
            False,
            "",
            plans,
        )
    except (
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        urllib.error.URLError,
        TimeoutError,
        http.client.HTTPException,
    ) as exc:
        return _fallback_plans(
            user_request,
            str(exc),
            dataset_id=dataset_id,
            cutter=cutter,
        )


def write_llm_candidate_outputs(candidate_set: LlmCandidateSet, settings: Settings) -> tuple[Path, Path, Path]:
    """写出 LLM 候选 JSON、Markdown 和日志。"""

    from toolwear_agent.registry import validate_pipeline_with_default_registries
    from toolwear_agent.schemas.converters import llm_candidate_plan_to_pipeline

    output_token = f"{candidate_set.dataset_id}_{candidate_set.cutter}".lower()
    candidate_file = (
        settings.ai_infra_root
        / "experiments"
        / "candidates"
        / f"{output_token}_llm_candidate_plans.json"
    )
    report_file = settings.ai_infra_root / "reports" / f"{output_token}_llm_candidate_plans.md"
    log_file = settings.log_root / f"{output_token}_llm_candidate_plans.log"
    payload = asdict(candidate_set)
    pipelines = [llm_candidate_plan_to_pipeline(plan) for plan in candidate_set.plans]
    validations = [validate_pipeline_with_default_registries(pipeline) for pipeline in pipelines]
    invalid = [result for result in validations if not result.valid]
    if invalid:
        messages = [issue.message for result in invalid for issue in result.issues if issue.severity.value == "error"]
        raise ValueError("LLM 候选未通过 Module Registry：" + "；".join(messages))
    payload["pipeline_specs"] = [pipeline.model_dump(mode="json") for pipeline in pipelines]
    payload["registry_validations"] = [result.model_dump(mode="json") for result in validations]
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# PHM2010 C1 LLM 候选方案",
        "",
        f"- Provider：{candidate_set.provider}",
        f"- Model：{candidate_set.model or '未使用'}",
        f"- 是否 fallback：{candidate_set.used_fallback}",
        f"- fallback 原因：{candidate_set.fallback_reason or '无'}",
        "",
    ]
    for plan in candidate_set.plans:
        lines.extend(
            [
                f"## {plan.display_name}",
                "",
                f"- plan_id：`{plan.plan_id}`",
                f"- 当前可训练：{plan.trainable_now}",
                f"- 训练后端：`{plan.training_backend}`",
                f"- 模块链：{' -> '.join(plan.module_pipeline)}",
                f"- 推荐理由：{plan.reason}",
                f"- 风险：{plan.risk}",
                f"- 预计成本：{plan.expected_cost}",
                "",
            ]
        )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 LLM 候选方案生成日志",
                f"provider: {candidate_set.provider}",
                f"model: {candidate_set.model or '未使用'}",
                f"used_fallback: {candidate_set.used_fallback}",
                f"fallback_reason: {candidate_set.fallback_reason or '无'}",
                f"candidate_file: {candidate_file}",
                f"report_file: {report_file}",
            ]
        ),
        encoding="utf-8",
    )
    return candidate_file, report_file, log_file
