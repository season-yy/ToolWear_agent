"""状态驱动实验台共用的展示与错误组件。"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any
from uuid import uuid4

import streamlit as st

from toolwear_agent.frontend.api_client import ToolApiError
from toolwear_agent.frontend.ui_state import WORKFLOW_STAGES, workflow_stage_index


STATE_LABELS = {
    "DRAFT": "待准备",
    "DATA_VALIDATING": "数据准备中",
    "BLOCKED_DATA": "数据受阻",
    "WAITING_PLAN_SELECTION": "等待方案选择",
    "PIPELINE_VALIDATING": "方案校验中",
    "CODE_PREPARING": "等待训练",
    "MINI_TRAINING": "小样本训练中",
    "EVALUATING": "等待评估",
    "DECIDING": "等待决策",
    "WAITING_FULL_APPROVAL": "等待完整训练审批",
    "COMPLETED_MINI": "小样本闭环完成",
    "FAILED": "运行失败",
    "CANCELLED": "运行已取消",
}


def operation_key(action: str) -> str:
    """为一次明确的页面点击生成可审计幂等键。"""

    return f"ui-{action}-{uuid4().hex}"


def render_api_error(exc: ToolApiError) -> None:
    """显示机器错误码，同时给出不会破坏状态的恢复动作。"""

    trace = f" | trace: {exc.trace_id}" if exc.trace_id else ""
    st.error(f"{exc}  [{exc.error_code}{trace}]")
    if exc.error_code in {
        "INVALID_WORKFLOW_STATE",
        "STATE_CONFLICT",
        "IDEMPOTENCY_KEY_CONFLICT",
    }:
        st.caption("实验状态可能已经前进。刷新当前实验后再操作。")
    elif exc.error_code == "API_UNREACHABLE":
        st.caption("确认 FastAPI 已在 18100 端口运行，然后点击刷新。")


def set_flash(message: str, *, level: str = "success") -> None:
    """跨一次 st.rerun 保存操作结果。"""

    st.session_state["flash_message"] = message
    st.session_state["flash_level"] = level


def render_flash() -> None:
    message = st.session_state.pop("flash_message", "")
    level = st.session_state.pop("flash_level", "success")
    if not message:
        return
    renderer = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }.get(level, st.info)
    renderer(message)


def render_workflow_rail(state: str) -> None:
    """用加工流程轨道表达实验位置，分支状态映射到对应阶段。"""

    active = workflow_stage_index(state)
    steps = []
    for index, label in enumerate(WORKFLOW_STAGES):
        css_class = "done" if index < active else "current" if index == active else ""
        steps.append(
            "<div class='tw-rail-step {css}'>"
            "<span class='tw-rail-number'>{number:02d}</span>"
            "<span class='tw-rail-label'>{label}</span>"
            "</div>".format(
                css=css_class,
                number=index + 1,
                label=html.escape(label),
            )
        )
    st.html("<div class='tw-rail'>" + "".join(steps) + "</div>")


def status_html(label: str, status: str) -> str:
    """返回带文字的状态标记，不只依赖颜色。"""

    if status in {"ok", "available", "configured", "verified"}:
        css_class = "ok"
    elif status in {"error", "failed", "unavailable", "missing"}:
        css_class = "error"
    else:
        css_class = "warn"
    return f"<span class='tw-status {css_class}'>{html.escape(label)}</span>"


def readable_time(value: str | None) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime(
            "%m-%d %H:%M:%S"
        )
    except ValueError:
        return value


def compact_json_rows(items: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """为 DataFrame 提取稳定列，避免把大 JSON 直接塞进页面。"""

    return [{field: item.get(field, "") for field in fields} for item in items]
