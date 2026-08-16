"""六 Agent Identity 与真实调用记录页签。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


STATUS_LABELS = {
    "success": "成功",
    "failed": "失败",
    "needs_human": "等待人工",
    "pending": "执行中",
}


def _latest_by_agent(agent_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in agent_runs:
        task = record.get("task", {})
        agent_name = str(task.get("assigned_to", ""))
        if agent_name:
            latest[agent_name] = record
    return latest


def _render_identity(
    definition: dict[str, Any],
    latest: dict[str, Any] | None,
) -> None:
    agent_name = str(definition.get("agent_name", "未知 Agent"))
    role = str(definition.get("chinese_role", ""))
    result = latest.get("result") if latest else None
    status = str(result.get("status", "pending")) if result else "not_run"
    label = STATUS_LABELS.get(status, "尚未调用")
    with st.expander(f"{agent_name} · {role} · {label}"):
        st.write(definition.get("responsibility", ""))
        st.caption(
            f"{definition.get('input_schema', '-')} -> "
            f"{definition.get('output_schema', '-')}"
        )
        st.markdown("**权限边界**")
        for item in definition.get("boundaries", []):
            st.write(f"- {item}")
        st.markdown("**授权 Skills**")
        st.code("\n".join(definition.get("allowed_skills", [])), language=None)
        if result:
            audit = result.get("llm_call") or {}
            metrics = st.columns(4)
            metrics[0].metric("状态", label)
            metrics[1].metric("Provider", audit.get("provider") or "-")
            metrics[2].metric("耗时", f"{audit.get('latency_ms', 0)} ms")
            metrics[3].metric("Token", audit.get("total_tokens") or "-")
            st.write(result.get("summary", ""))
            if result.get("error_message"):
                st.error(result["error_message"])


def render_agents_tab(
    definitions: list[dict[str, Any]],
    agent_runs: list[dict[str, Any]],
) -> None:
    """展示六 Agent 当前契约和本实验的真实调用历史。"""

    latest = _latest_by_agent(agent_runs)
    completed = [item for item in agent_runs if item.get("result")]
    failed = [item for item in completed if item["result"].get("status") == "failed"]
    metrics = st.columns(4)
    metrics[0].metric("固定 Agent", len(definitions))
    metrics[1].metric("调用记录", len(agent_runs))
    metrics[2].metric("已完成", len(completed))
    metrics[3].metric("失败", len(failed))

    for definition in definitions:
        _render_identity(
            definition,
            latest.get(str(definition.get("agent_name", ""))),
        )

    if not agent_runs:
        st.info("当前实验尚无 Agent 调用记录。")
        return
    rows = []
    for record in reversed(agent_runs):
        task = record.get("task", {})
        result = record.get("result") or {}
        audit = result.get("llm_call") or {}
        evidence = result.get("evidence") or []
        rows.append(
            {
                "时间": task.get("created_at"),
                "Agent": task.get("assigned_to"),
                "任务": task.get("task_type"),
                "状态": STATUS_LABELS.get(result.get("status"), "执行中"),
                "模型": audit.get("model") or "-",
                "耗时(ms)": audit.get("latency_ms"),
                "Token": audit.get("total_tokens"),
                "Evidence": evidence[0].get("evidence_id") if evidence else "-",
            }
        )
    st.markdown("**调用时间线**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
