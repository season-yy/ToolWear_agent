"""数据准备、候选、审批、训练和报告动作路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

from toolwear_agent.backend.dependencies import get_workflow
from toolwear_agent.schemas import CandidateRecommendationSet
from toolwear_agent.schemas.api import (
    ActionRequest,
    DecisionRequest,
    EvaluationRequest,
    ExperimentActionResponse,
    PipelineApprovalRequest,
    PipelineValidationResponse,
    RecommendationRequest,
    RunStartRequest,
)
from toolwear_agent.services.workflow import ExperimentWorkflowService
from toolwear_agent.state import RunRecord


router = APIRouter(prefix="/api/v1/experiments", tags=["workflow"])
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]


@router.post("/{experiment_id}/profile", response_model=ExperimentActionResponse)
def profile(
    experiment_id: str,
    body: ActionRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.profile(
        experiment_id,
        rationale=body.rationale,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/labels", response_model=ExperimentActionResponse)
def labels(
    experiment_id: str,
    body: ActionRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.labels(
        experiment_id,
        rationale=body.rationale,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/split", response_model=ExperimentActionResponse)
def split(
    experiment_id: str,
    body: ActionRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.split(
        experiment_id,
        rationale=body.rationale,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/recommendations", response_model=CandidateRecommendationSet)
def recommendations(
    experiment_id: str,
    body: RecommendationRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> CandidateRecommendationSet:
    return workflow.recommendations(
        experiment_id,
        body,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/approve-pipeline", response_model=ExperimentActionResponse)
def approve_pipeline(
    experiment_id: str,
    body: PipelineApprovalRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.approve_pipeline(
        experiment_id,
        body,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/validate", response_model=PipelineValidationResponse)
def validate_pipeline(
    experiment_id: str,
    body: ActionRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> PipelineValidationResponse:
    return workflow.validate_pipeline(
        experiment_id,
        rationale=body.rationale,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/runs/mini", response_model=RunRecord)
def start_mini_run(
    experiment_id: str,
    body: RunStartRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> RunRecord:
    return workflow.start_mini_run(
        experiment_id,
        body,
        idempotency_key=idempotency_key,
    )


@router.get("/{experiment_id}/runs/{run_id}", response_model=RunRecord)
def get_run(
    experiment_id: str,
    run_id: str,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> RunRecord:
    run = workflow.get_run(run_id)
    if run.experiment_id != experiment_id:
        raise ValueError("run_id 不属于指定 experiment。")
    return run


@router.get("/{experiment_id}/runs/{run_id}/logs")
def get_run_logs(
    experiment_id: str,
    run_id: str,
    tail: int = Query(default=100, ge=1, le=500),
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> dict[str, object]:
    """供训练面板轮询结构化日志，最大返回最后 500 条。"""

    return workflow.run_logs(experiment_id, run_id, tail=tail)


@router.post("/{experiment_id}/evaluate", response_model=ExperimentActionResponse)
def evaluate(
    experiment_id: str,
    body: EvaluationRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.evaluate(
        experiment_id,
        rationale=body.rationale,
        idempotency_key=idempotency_key,
        force_refresh=body.force_refresh,
    )


@router.post("/{experiment_id}/decision", response_model=ExperimentActionResponse)
def decision(
    experiment_id: str,
    body: DecisionRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.decide(
        experiment_id,
        body,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/report", response_model=ExperimentActionResponse)
def report(
    experiment_id: str,
    body: ActionRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.report(
        experiment_id,
        rationale=body.rationale,
        idempotency_key=idempotency_key,
    )


@router.post("/{experiment_id}/cancel", response_model=ExperimentActionResponse)
def cancel(
    experiment_id: str,
    body: ActionRequest,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentActionResponse:
    return workflow.cancel(experiment_id, rationale=body.rationale)
