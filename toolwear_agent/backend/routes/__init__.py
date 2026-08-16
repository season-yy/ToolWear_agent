"""FastAPI v1 路由集合。"""

from toolwear_agent.backend.routes.agents import router as agents_router
from toolwear_agent.backend.routes.actions import router as actions_router
from toolwear_agent.backend.routes.artifacts import router as artifacts_router
from toolwear_agent.backend.routes.experiments import router as experiments_router
from toolwear_agent.backend.routes.system import router as system_router

__all__ = [
    "actions_router",
    "agents_router",
    "artifacts_router",
    "experiments_router",
    "system_router",
]
