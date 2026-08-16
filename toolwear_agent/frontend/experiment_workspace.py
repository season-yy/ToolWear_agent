"""单个 ExperimentState 对应的完整七阶段工作台。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient
from toolwear_agent.frontend.ui_components import STATE_LABELS, render_api_error, render_workflow_rail
from toolwear_agent.frontend.ui_state import preparation_progress, state_actions
from toolwear_agent.frontend.workspace_agents import render_agents_tab
from toolwear_agent.frontend.workspace_data import render_data_tab
from toolwear_agent.frontend.workspace_evaluation import render_evaluation_tab
from toolwear_agent.frontend.workspace_evidence import render_evidence_tab
from toolwear_agent.frontend.workspace_pipeline import render_pipeline_tab
from toolwear_agent.frontend.workspace_training import render_training_tab


def _recover_optional_payloads(
    client: ToolWearApiClient,
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """只按当前实验指针恢复候选和 revision；404 表示尚未产生。"""

    recommendation = None
    revision = None
    try:
        if state.get("latest_recommendation_id"):
            recommendation = client.latest_recommendations(state["experiment_id"])
        if state.get("selected_pipeline_ref"):
            revision = client.get_revision(state["experiment_id"], int(state["revision"]))
    except ToolApiError as exc:
        if exc.status_code != 404:
            render_api_error(exc)
    return recommendation, revision


def render_experiment_workspace(
    client: ToolWearApiClient,
    state: dict[str, Any],
    capabilities: dict[str, Any],
    agent_definitions: list[dict[str, Any]],
) -> None:
    """读取一个实验的全部可恢复状态，并按页签组织真实业务动作。"""

    experiment_id = state["experiment_id"]
    try:
        artifacts = client.artifacts(experiment_id)
        events = client.events(experiment_id)
        runs = client.runs(experiment_id)
        agent_runs = client.agent_runs(experiment_id)
    except ToolApiError as exc:
        render_api_error(exc)
        return
    recommendation, revision = _recover_optional_payloads(client, state)
    progress = preparation_progress(experiment_id, int(state["revision"]), artifacts)
    has_succeeded_run = any(item.get("status") == "succeeded" for item in runs)
    actions = state_actions(
        str(state["state"]),
        preparation_ready=progress.complete,
        has_succeeded_run=has_succeeded_run,
    )

    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown("<div class='tw-kicker'>ACTIVE EXPERIMENT</div>", unsafe_allow_html=True)
        st.title(state["title"])
        st.markdown(
            f"<div class='tw-idline'>{experiment_id} · {state['trace_id']}</div>",
            unsafe_allow_html=True,
        )
    with header_right:
        st.caption("当前状态")
        st.markdown(f"**{STATE_LABELS.get(state['state'], state['state'])}**")
        st.caption(f"Revision {state['revision']} · 事件 {state['last_event_sequence']}")
    render_workflow_rail(str(state["state"]))

    if state.get("error_message"):
        st.error(f"{state.get('error_code', 'ERROR')}：{state['error_message']}")

    tabs = st.tabs(
        ["数据准备", "候选与参数", "真实训练", "评估与决策", "日志与证据", "Agent 协作"]
    )
    with tabs[0]:
        render_data_tab(client, state, progress, actions)
    with tabs[1]:
        render_pipeline_tab(client, state, capabilities, actions, recommendation, revision)
    with tabs[2]:
        render_training_tab(client, state, actions, runs, revision)
    with tabs[3]:
        render_evaluation_tab(client, state, actions, runs, artifacts)
    with tabs[4]:
        render_evidence_tab(client, state, events, artifacts, revision)
    with tabs[5]:
        render_agents_tab(agent_definitions, agent_runs)
