"""ToolWear FastAPI Tool API 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from toolwear_agent.backend.dependencies import ApplicationContainer
from toolwear_agent.backend.errors import register_exception_handlers
from toolwear_agent.backend.security import install_tool_api_security
from toolwear_agent.backend.routes import (
    actions_router,
    agents_router,
    artifacts_router,
    experiments_router,
    system_router,
)
from toolwear_agent.core.settings import Settings, load_settings
from toolwear_agent.services.candidate_service import CandidateProvider
from toolwear_agent.services.evaluation_diagnosis import DiagnosisProvider
from toolwear_agent.services.llm_chat import ChatClient


def create_app(
    *,
    settings: Settings | None = None,
    candidate_provider: CandidateProvider | None = None,
    diagnosis_provider: DiagnosisProvider | None = None,
    agent_chat_client: ChatClient | None = None,
) -> FastAPI:
    """创建可注入配置的应用，生产和集成测试共用同一入口。"""

    resolved_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = ApplicationContainer.build(
            resolved_settings,
            candidate_provider=candidate_provider,
            diagnosis_provider=diagnosis_provider,
            agent_chat_client=agent_chat_client,
        )
        app.state.container = container
        try:
            yield
        finally:
            container.close()

    application = FastAPI(
        title="ToolWear Agent Tool API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{resolved_settings.streamlit_host}:{resolved_settings.streamlit_port}",
            "http://127.0.0.1:18101",
            "http://localhost:18101",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "Idempotency-Key",
            "X-ToolWear-AgentTeams-Skill",
            "X-ToolWear-AgentTeams-Agent",
            "X-ToolWear-Correlation-Id",
        ],
    )
    install_tool_api_security(application, resolved_settings)
    register_exception_handlers(application)
    application.include_router(system_router)
    application.include_router(agents_router)
    application.include_router(experiments_router)
    application.include_router(actions_router)
    application.include_router(artifacts_router)
    return application


app = create_app()
