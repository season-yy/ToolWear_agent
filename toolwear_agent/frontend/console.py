"""Streamlit 应用外壳：服务状态、实验切换和页面路由。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from toolwear_agent.core.settings import load_settings
from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient
from toolwear_agent.frontend.experiment_create import render_create_experiment
from toolwear_agent.frontend.experiment_workspace import render_experiment_workspace
from toolwear_agent.frontend.ui_components import render_api_error, render_flash, status_html
from toolwear_agent.frontend.ui_theme import apply_theme


STATUS_LABELS = {
    "ok": "正常",
    "configured": "已配置",
    "verified": "已验证",
    "available": "可用",
    "unavailable": "不可用",
    "missing": "未配置",
    "pending_integration": "待接入",
    "pending_verification": "待验证",
    "unknown": "未知",
}


def _component_status(health: dict[str, Any], key: str) -> str:
    components = health.get("components", {})
    return str(components.get(key, {}).get("status", "unknown"))


def _render_service_status(health: dict[str, Any]) -> None:
    st.caption("运行状态")
    labels = (
        ("API", "api"),
        ("SQLite", "sqlite"),
        ("LLM", "llm"),
        ("CUDA", "cuda"),
        ("Agent Runtime", "agents"),
        ("AgentTeams", "agentteams"),
        ("Higress", "higress"),
    )
    for label, key in labels:
        value = _component_status(health, key)
        st.html(
            "<div class='tw-agent-line'>"
            f"<span>{label}</span>{status_html(STATUS_LABELS.get(value, value), value)}"
            "</div>"
        )


def _agentteams_summary_lines(health: dict[str, Any]) -> tuple[str, ...]:
    """把健康接口中的 AgentTeams 证据压缩成侧边栏可扫描摘要。"""

    component = health.get("components", {}).get("agentteams", {})
    if component.get("status") != "verified":
        return ("尚无通过校验的 AgentTeams 部署证据。",)
    models = ", ".join(str(item) for item in component.get("models", [])) or "-"
    runtime = ", ".join(str(item) for item in component.get("runtime", [])) or "-"
    leader = "Leader Ready" if component.get("leader_ready") else "Leader Not Ready"
    return (
        f"Team：{component.get('team', '-')} · {component.get('phase', '-')}",
        (
            f"角色：{component.get('worker_count', 0)} · {leader} · "
            f"Workers {component.get('ready_workers', 0)}/{component.get('total_workers', 0)}"
        ),
        f"运行时：{runtime}",
        f"模型：{models}",
        f"证据：{component.get('verification_id', '-')}",
    )


def _activate_pending_experiment() -> None:
    """在选择器创建前同步刚创建的实验，避免修改已实例化控件。"""

    pending = st.session_state.pop("pending_experiment_id", "")
    if pending:
        st.session_state["experiment_id"] = pending
        st.session_state["experiment_picker"] = pending


def _reset_experiment_selection() -> None:
    """按钮回调在下一轮渲染前把页面切回新建实验。"""

    st.session_state["experiment_id"] = ""
    st.session_state["experiment_picker"] = ""


def _experiment_label(experiment_id: str, experiments: list[dict[str, Any]]) -> str:
    if not experiment_id:
        return "新建实验"
    item = next((value for value in experiments if value["experiment_id"] == experiment_id), None)
    if item is None:
        return experiment_id
    return f"{item['title']} · {item['state']}"


def _select_experiment(experiments: list[dict[str, Any]]) -> str:
    """显式切换实验；首次进入不猜测用户想打开哪一个历史运行。"""

    ids = ["", *[str(item["experiment_id"]) for item in experiments]]
    current = str(st.session_state.get("experiment_id", ""))
    index = ids.index(current) if current in ids else 0
    selected = st.selectbox(
        "当前实验",
        ids,
        index=index,
        format_func=lambda value: _experiment_label(value, experiments),
        key="experiment_picker",
    )
    st.session_state["experiment_id"] = selected
    return selected


def main() -> None:
    """启动只通过 FastAPI 读写状态的本地实验台。"""

    st.set_page_config(
        page_title="刃知 - 刀具磨损监测算法辅助平台",
        page_icon=":material/manufacturing:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_theme()
    _activate_pending_experiment()
    settings = load_settings()
    client = ToolWearApiClient(
        settings.tool_api_base_url,
        token=settings.tool_api_token,
        timeout_seconds=max(settings.llm_timeout_seconds + 30.0, 120.0),
    )
    try:
        health = client.health()
        datasets = client.datasets()
        capabilities = client.capabilities()
        agent_definitions = client.agent_definitions()
        experiments = client.list_experiments()
    except ToolApiError as exc:
        st.title("刃知：基于 AgentTeams 的刀具磨损监测算法辅助平台")
        render_api_error(exc)
        st.code(
            "python -m uvicorn toolwear_agent.backend.main:app --host 127.0.0.1 --port 18100",
            language="powershell",
        )
        return

    with st.sidebar:
        st.markdown("### ToolWear Console")
        st.caption("PHM2010 · 四阶段磨损分类")
        _render_service_status(health)
        st.divider()
        selected_id = _select_experiment(experiments)
        controls = st.columns(2)
        controls[0].button(
            "新建",
            icon=":material/add:",
            width="stretch",
            on_click=_reset_experiment_selection,
        )
        if controls[1].button("刷新", icon=":material/refresh:", width="stretch"):
            st.rerun()
        st.divider()
        st.caption("六个业务 Agent")
        for index, definition in enumerate(agent_definitions, start=1):
            st.markdown(f"`{index:02d}` {definition['agent_name']}")
        st.divider()
        st.caption("AgentTeams 协作部署")
        for line in _agentteams_summary_lines(health):
            st.caption(line)

    render_flash()
    selected_id = str(st.session_state.get("experiment_id", selected_id))
    if not selected_id:
        render_create_experiment(client, datasets, capabilities)
        return
    try:
        state = client.get_experiment(selected_id)
    except ToolApiError as exc:
        render_api_error(exc)
        return
    render_experiment_workspace(client, state, capabilities, agent_definitions)
