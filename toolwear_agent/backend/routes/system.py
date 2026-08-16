"""健康、能力和数据集只读路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from toolwear_agent.agentteams.deployment_status import load_deployment_status
from toolwear_agent.backend.dependencies import ApplicationContainer, get_container, get_workflow
from toolwear_agent.schemas import DatasetManifest, RegistryCatalog
from toolwear_agent.services.workflow import ExperimentWorkflowService


router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health")
def health(
    container: ApplicationContainer = Depends(get_container),
) -> dict[str, object]:
    """返回 API、SQLite 和状态路径健康信息。"""

    settings = container.settings
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda = {
            "status": "available" if cuda_available else "unavailable",
            "available": cuda_available,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else "",
            "torch_version": torch.__version__,
        }
    except (ImportError, RuntimeError) as exc:
        cuda = {
            "status": "error",
            "available": False,
            "device_name": "",
            "error": type(exc).__name__,
        }
    database = container.repository.health_info()
    agentteams = load_deployment_status(settings)
    agentteams_component: dict[str, object] = {
        "status": agentteams.status,
        "worker_count": len(agentteams.workers),
        "verification_id": agentteams.verification_id,
        "verified_at": agentteams.verified_at,
    }
    if agentteams.team is not None:
        agentteams_component.update(
            {
                "framework_version": (
                    agentteams.framework.version if agentteams.framework is not None else ""
                ),
                "team": agentteams.team.runtime_name,
                "phase": agentteams.team.phase,
                "leader_ready": agentteams.team.leader_ready,
                "ready_workers": agentteams.team.ready_workers,
                "total_workers": agentteams.team.total_workers,
                "room_id": agentteams.team.room_id,
                "runtime": sorted({worker.runtime for worker in agentteams.workers}),
                "models": sorted({worker.model for worker in agentteams.workers}),
                "correlation_id": agentteams.toolwear_trace.correlation_id,
                "evidence_report": agentteams.evidence.report,
            }
        )
    return {
        "status": "ok",
        "service": "toolwear-agent-api",
        "database": database,
        "components": {
            "api": {"status": "ok", "auth_required": bool(settings.tool_api_token)},
            "sqlite": {"status": database["integrity"]},
            "llm": {
                "status": "configured" if settings.llm_api_key and settings.llm_model else "missing",
                "provider": settings.llm_provider,
                "model": settings.llm_model,
            },
            "cuda": cuda,
            "agents": {"status": "available", "count": 6},
            "agentteams": agentteams_component,
            "higress": agentteams.higress.model_dump(mode="json"),
        },
    }


@router.get("/capabilities", response_model=RegistryCatalog)
def capabilities(
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> RegistryCatalog:
    """返回页面可展示和校验的 Module/Trainer Registry。"""

    return workflow.capabilities()


@router.get("/datasets", response_model=list[DatasetManifest])
def datasets(
    workflow: ExperimentWorkflowService = Depends(get_workflow),
) -> tuple[DatasetManifest, ...]:
    """返回已体检并登记的数据集。"""

    return workflow.datasets()
