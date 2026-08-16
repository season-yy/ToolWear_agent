"""将领域错误映射为稳定 Tool API error_code。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from toolwear_agent.core.errors import ToolWearError
from toolwear_agent.schemas.api import ApiErrorDetail, ApiErrorResponse
from toolwear_agent.services.errors import InvalidWorkflowStateError
from toolwear_agent.state import (
    EntityNotFoundError,
    IdempotencyConflictError,
    InvalidStateTransitionError,
    RevisionLockedError,
    StateConflictError,
)


def _error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    trace_id: str | None = None,
) -> JSONResponse:
    payload = ApiErrorResponse(
        error=ApiErrorDetail(
            error_code=error_code,
            message=message or "请求失败。",
            trace_id=trace_id,
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    """注册一次全局异常映射，路由中不重复写 try/except。"""

    @app.exception_handler(ToolWearError)
    async def handle_toolwear_error(request: Request, exc: ToolWearError) -> JSONResponse:
        del request
        if isinstance(exc, EntityNotFoundError):
            status_code = 404
        elif isinstance(
            exc,
            (
                InvalidWorkflowStateError,
                InvalidStateTransitionError,
                StateConflictError,
                IdempotencyConflictError,
                RevisionLockedError,
            ),
        ):
            status_code = 409
        else:
            status_code = 400
        return _error_response(
            status_code=status_code,
            error_code=exc.error_code,
            message=exc.message,
        )

    @app.exception_handler(KeyError)
    async def handle_key_error(request: Request, exc: KeyError) -> JSONResponse:
        del request
        return _error_response(
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            message=str(exc).strip("'"),
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        del request
        return _error_response(
            status_code=422,
            error_code="INVALID_REQUEST",
            message=str(exc),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request
        first_error = exc.errors()[0] if exc.errors() else {"msg": "请求校验失败。"}
        return _error_response(
            status_code=422,
            error_code="REQUEST_VALIDATION_ERROR",
            message=str(first_error.get("msg", "请求校验失败。")),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """兜底返回稳定结构；详细堆栈只保留在服务端日志。"""

        del request, exc
        return _error_response(
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            message="服务执行失败，请查看服务端日志和当前实验状态。",
        )
