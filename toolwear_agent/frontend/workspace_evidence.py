"""实验工作台的状态事件、日志和证据索引页签。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from toolwear_agent.frontend.api_client import ToolWearApiClient
from toolwear_agent.frontend.ui_components import readable_time


KIND_LABELS = {
    "config": "配置",
    "split": "切分",
    "metrics": "指标",
    "model": "模型",
    "figure": "图表",
    "log": "日志",
    "trace": "Trace",
    "report": "报告",
    "code": "代码",
    "approval": "审批",
}


def _event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "序号": event.get("sequence"),
            "时间": readable_time(event.get("created_at")),
            "状态变化": f"{event.get('before_state') or 'START'} → {event.get('after_state')}",
            "执行者": event.get("actor"),
            "原因": event.get("reason"),
            "证据数": len(event.get("evidence_ids", [])),
        }
        for event in events
    ]


def _artifact_rows(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "类型": KIND_LABELS.get(item.get("kind"), item.get("kind")),
            "说明": item.get("description"),
            "Run": item.get("run_id") or "实验级",
            "大小": f"{int(item.get('size_bytes', 0)) / 1024:.1f} KiB",
            "创建者": item.get("created_by"),
            "时间": readable_time(item.get("created_at")),
            "Evidence ID": item.get("evidence_id"),
        }
        for item in artifacts
    ]


def render_evidence_tab(
    client: ToolWearApiClient,
    state: dict[str, Any],
    events: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    revision: dict[str, Any] | None,
) -> None:
    """显示完整审计链，并为报告等小型证据提供受控访问入口。"""

    st.subheader("状态事件与证据索引")
    summary = st.columns(4)
    summary[0].metric("状态事件", len(events))
    summary[1].metric("证据条目", len(artifacts))
    summary[2].metric("当前 Revision", state["revision"])
    summary[3].metric("Trace", state["trace_id"][-12:])

    event_tab, artifact_tab, snapshot_tab = st.tabs(["状态时间线", "证据文件", "配置快照"])
    with event_tab:
        if events:
            st.dataframe(_event_rows(events), hide_index=True, width="stretch")
        else:
            st.caption("尚无状态事件。")

    with artifact_tab:
        if artifacts:
            st.dataframe(_artifact_rows(artifacts), hide_index=True, width="stretch")
            reports = [
                item
                for item in artifacts
                if item.get("kind") == "report" and item.get("media_type") == "text/markdown"
            ]
            if reports:
                st.markdown("**可直接打开的报告**")
                for report in reports:
                    st.link_button(
                        report.get("description") or report["evidence_id"],
                        client.artifact_url(report["evidence_id"]),
                        icon=":material/open_in_new:",
                    )
        else:
            st.caption("尚无证据文件。")

    with snapshot_tab:
        st.markdown("**ExperimentState**")
        st.json(state, expanded=False)
        if revision:
            st.markdown("**不可变 Revision**")
            st.json(revision, expanded=False)
        else:
            st.caption("方案审批后会在此显示不可变 revision。")
