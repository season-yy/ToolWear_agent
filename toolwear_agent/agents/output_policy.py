"""补充 JSON Schema 无法表达的跨输入输出 Agent 策略。"""

from __future__ import annotations

from toolwear_agent.agents.catalog import AgentDefinition
from toolwear_agent.schemas.agent_runtime import (
    AgentOutputBase,
    AlgorithmArchitectInput,
    AlgorithmArchitectOutput,
)


class AgentOutputPolicyError(ValueError):
    """模型输出虽然结构合法，但违反当前任务的动态边界。"""


def _validate_architect_registry(
    agent_input: AlgorithmArchitectInput,
    output: AlgorithmArchitectOutput,
) -> None:
    raw_ids = agent_input.registry_summary.get("available_pipeline_ids")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise AgentOutputPolicyError("Registry 输入缺少 available_pipeline_ids。")
    available_ids = {
        item for item in raw_ids if isinstance(item, str) and item.strip()
    }
    candidate_ids = tuple(item.pipeline_id for item in output.candidates)
    unknown_ids = sorted(set(candidate_ids) - available_ids)
    if unknown_ids:
        raise AgentOutputPolicyError(
            f"候选不属于本次 Registry：{unknown_ids}"
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise AgentOutputPolicyError("候选 pipeline_id 不能重复。")
    expected_ranks = tuple(range(1, len(output.candidates) + 1))
    actual_ranks = tuple(item.rank for item in output.candidates)
    if actual_ranks != expected_ranks:
        raise AgentOutputPolicyError("候选 rank 必须从 1 开始连续递增。")


def validate_agent_output_policy(
    definition: AgentDefinition,
    validated_input: object,
    output: AgentOutputBase,
) -> None:
    """按角色校验需要同时读取任务输入和模型输出的规则。"""

    if definition.agent_name != "AlgorithmArchitectAgent":
        return
    if not isinstance(validated_input, AlgorithmArchitectInput) or not isinstance(
        output, AlgorithmArchitectOutput
    ):
        raise AgentOutputPolicyError("算法方案 Agent 的输入输出类型不匹配。")
    _validate_architect_registry(validated_input, output)


__all__ = ["AgentOutputPolicyError", "validate_agent_output_policy"]
