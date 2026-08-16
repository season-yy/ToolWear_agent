"""实验创建、查询、事件与证据路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, status

from toolwear_agent.backend.dependencies import get_workflow
from toolwear_agent.schemas import (
    CandidateRecommendationSet,
    EvidenceRef,
    ExperimentRevision,
    ExperimentState,
)
from toolwear_agent.schemas.api import CreateExperimentRequest
from toolwear_agent.services.workflow import ExperimentWorkflowService
from toolwear_agent.state import RunRecord, StateTransitionEvent


router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]


@router.post("", response_model=ExperimentState, status_code=status.HTTP_201_CREATED)
def create_experiment(
    body: CreateExperimentRequest,
    idempotency_key: IdempotencyKey = None,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentState:
    return workflow.create_experiment(body, idempotency_key=idempotency_key)


@router.get("", response_model=list[ExperimentState])
def list_experiments(
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> tuple[ExperimentState, ...]:
    return workflow.list_experiments()


@router.get("/{experiment_id}", response_model=ExperimentState)
def get_experiment(
    experiment_id: str,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentState:
    return workflow.get_experiment(experiment_id)


@router.get("/{experiment_id}/events", response_model=list[StateTransitionEvent])
def list_events(
    experiment_id: str,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> tuple[StateTransitionEvent, ...]:
    return workflow.events(experiment_id)


@router.get("/{experiment_id}/artifacts", response_model=list[EvidenceRef])
def list_artifacts(
    experiment_id: str,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> tuple[EvidenceRef, ...]:
    return workflow.artifacts(experiment_id)


@router.get(
    "/{experiment_id}/recommendations",
    response_model=CandidateRecommendationSet,
)
def get_latest_recommendations(
    experiment_id: str,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> CandidateRecommendationSet:
    """恢复页面刷新前最后一次候选集合。"""

    return workflow.get_latest_recommendations(experiment_id)


@router.get("/{experiment_id}/runs", response_model=list[RunRecord])
def list_runs(
    experiment_id: str,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> tuple[RunRecord, ...]:
    """恢复指定实验的真实运行列表，不读取全局 latest run。"""

    return workflow.list_runs(experiment_id)


@router.get(
    "/{experiment_id}/revisions/{revision}",
    response_model=ExperimentRevision,
)
def get_revision(
    experiment_id: str,
    revision: int,
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> ExperimentRevision:
    """返回页面刷新后恢复审批参数所需的不可变 revision。"""

    return workflow.get_revision(experiment_id, revision)
