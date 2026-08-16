"""六个 Agent 各自的输入、输出以及运行时 API 契约。"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, JsonValue

from toolwear_agent.schemas.agent import AgentName, AgentResult, AgentTask
from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel
from toolwear_agent.schemas.diagnosis import EvaluationFacts


JsonMap: TypeAlias = dict[str, JsonValue]


class ExperimentManagerInput(SchemaModel):
    user_goal: NonEmptyText
    experiment_state: JsonMap
    worker_results: tuple[JsonMap, ...] = ()
    pending_approval: JsonMap = Field(default_factory=dict)


class DataStewardInput(SchemaModel):
    dataset_manifest: JsonMap
    profile_summary: JsonMap
    label_policy: JsonMap
    split_summary: JsonMap
    leakage_summary: JsonMap


class AlgorithmArchitectInput(SchemaModel):
    user_goal: NonEmptyText
    data_summary: JsonMap
    resource_summary: JsonMap
    registry_summary: JsonMap
    memory_context: tuple[JsonMap, ...] = ()


class CodeTrainingEngineerInput(SchemaModel):
    approved_pipeline: JsonMap
    split_summary: JsonMap
    training_budget: JsonMap
    runtime_summary: JsonMap


class EvaluationGovernorInput(SchemaModel):
    evaluation_facts: EvaluationFacts
    deterministic_checks: tuple[JsonMap, ...] = ()
    resource_summary: JsonMap = Field(default_factory=dict)


class ReportMemoryCuratorInput(SchemaModel):
    experiment_state: JsonMap
    metrics_summary: JsonMap
    decision: JsonMap
    evidence_index: tuple[JsonMap, ...]
    user_feedback: str = ""


class AgentOutputBase(SchemaModel):
    summary: NonEmptyText
    next_actions: tuple[NonEmptyText, ...] = ()


class DelegatedStep(SchemaModel):
    order: int = Field(ge=1, le=20)
    assigned_to: AgentName
    task_type: EntityId
    objective: NonEmptyText
    requires_human_approval: bool = False


class ExperimentManagerOutput(AgentOutputBase):
    plan_steps: tuple[DelegatedStep, ...] = Field(min_length=1, max_length=12)
    next_agent: AgentName | None = None
    requires_human_approval: bool
    approval_question: str = ""


class AgentFinding(SchemaModel):
    severity: Literal["info", "warning", "critical"]
    title: NonEmptyText
    detail: NonEmptyText
    evidence_ids: tuple[EntityId, ...] = ()


class DataStewardOutput(AgentOutputBase):
    data_status: Literal["pass", "warning", "blocker"]
    findings: tuple[AgentFinding, ...] = Field(min_length=1, max_length=10)
    recommended_actions: tuple[NonEmptyText, ...] = ()
    blocker: bool


class ArchitectCandidate(SchemaModel):
    rank: int = Field(ge=1, le=3)
    pipeline_id: EntityId
    reason: NonEmptyText
    risk: NonEmptyText
    expected_cost: Literal["low", "medium", "high"]
    registry_compatible: Literal[True] = True


class AlgorithmArchitectOutput(AgentOutputBase):
    candidates: tuple[ArchitectCandidate, ...] = Field(min_length=2, max_length=3)
    experimental_extensions: tuple[JsonMap, ...] = ()
    requires_human_choice: Literal[True] = True


class PreflightCheck(SchemaModel):
    check_name: EntityId
    status: Literal["pass", "warning", "blocker"]
    detail: NonEmptyText


class CodeTrainingEngineerOutput(AgentOutputBase):
    execution_steps: tuple[NonEmptyText, ...] = Field(min_length=1, max_length=12)
    preflight_checks: tuple[PreflightCheck, ...] = Field(min_length=1, max_length=12)
    selected_backend: NonEmptyText
    selected_device: NonEmptyText
    blocker: bool
    requires_human_approval: bool


class EvaluationGovernorOutput(AgentOutputBase):
    diagnosis_categories: tuple[EntityId, ...] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0, le=1)
    recommended_action: Literal[
        "approve_full",
        "adjust_parameters",
        "change_pipeline",
        "stop",
    ]
    evidence_ids: tuple[EntityId, ...] = ()
    requires_human_approval: Literal[True] = True


class ReportMemoryCuratorOutput(AgentOutputBase):
    limitations: tuple[NonEmptyText, ...] = Field(min_length=1, max_length=10)
    memory_problem: NonEmptyText
    memory_intervention: NonEmptyText
    memory_outcome: NonEmptyText
    memory_tags: tuple[EntityId, ...] = ()
    evidence_ids: tuple[EntityId, ...] = ()
    requires_publish_approval: Literal[True] = True


class AgentInvocationRequest(SchemaModel):
    task_type: EntityId
    objective: NonEmptyText
    input_payload: JsonMap
    evidence_ids: tuple[EntityId, ...] = ()
    requested_skills: tuple[EntityId, ...] = ()


class AgentInvocationResponse(SchemaModel):
    task: AgentTask
    result: AgentResult


class AgentRunRecord(SchemaModel):
    task: AgentTask
    result: AgentResult | None = None


class AgentDefinitionView(SchemaModel):
    agent_name: AgentName
    chinese_role: NonEmptyText
    responsibility: NonEmptyText
    boundaries: tuple[NonEmptyText, ...]
    allowed_skills: tuple[EntityId, ...]
    input_schema: EntityId
    output_schema: EntityId
    failure_behavior: NonEmptyText
    prompt_template_version: EntityId
