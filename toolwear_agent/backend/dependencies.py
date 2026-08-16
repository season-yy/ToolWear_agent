"""FastAPI 应用容器和请求级依赖。"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from toolwear_agent.agents.runtime import AgentRuntimeService
from toolwear_agent.core.settings import Settings
from toolwear_agent.services.candidate_service import CandidateProvider
from toolwear_agent.services.evaluation_diagnosis import DiagnosisProvider
from toolwear_agent.services.llm_chat import ChatClient
from toolwear_agent.services.workflow import ExperimentWorkflowService
from toolwear_agent.state import SQLiteExperimentRepository


@dataclass
class ApplicationContainer:
    """集中持有长生命周期资源，避免路由自行创建连接。"""

    settings: Settings
    repository: SQLiteExperimentRepository
    workflow: ExperimentWorkflowService
    agent_runtime: AgentRuntimeService

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        candidate_provider: CandidateProvider | None = None,
        diagnosis_provider: DiagnosisProvider | None = None,
        agent_chat_client: ChatClient | None = None,
    ) -> "ApplicationContainer":
        repository = SQLiteExperimentRepository(settings.state_db_path)
        repository.initialize()
        workflow = ExperimentWorkflowService(
            settings,
            repository,
            candidate_provider=candidate_provider,
            diagnosis_provider=diagnosis_provider,
        )
        agent_runtime = AgentRuntimeService(
            settings,
            repository,
            chat_client=agent_chat_client,
        )
        return cls(
            settings=settings,
            repository=repository,
            workflow=workflow,
            agent_runtime=agent_runtime,
        )

    def close(self) -> None:
        self.workflow.close()
        self.repository.close()


def get_container(request: Request) -> ApplicationContainer:
    """从 FastAPI app.state 取得唯一应用容器。"""

    return request.app.state.container


def get_workflow(request: Request) -> ExperimentWorkflowService:
    """路由只依赖工作流接口，不直接接触数据库连接。"""

    return get_container(request).workflow


def get_agent_runtime(request: Request) -> AgentRuntimeService:
    """返回与业务工作流共享 SQLite 的唯一 Agent Runtime。"""

    return get_container(request).agent_runtime
