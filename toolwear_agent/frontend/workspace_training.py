"""实验工作台的真实训练启动、进度和日志页签。"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient
from toolwear_agent.frontend.ui_components import operation_key, readable_time, render_api_error, set_flash
from toolwear_agent.frontend.ui_state import StateActions


RUN_STATUS_LABELS = {
    "queued": "排队中",
    "running": "训练中",
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}

ACTIVE_RUN_STATUSES = frozenset({"queued", "running"})


def latest_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """按创建时间选出当前实验最新 Run，不读取全局 latest 文件。"""

    if not runs:
        return None
    return max(runs, key=lambda item: str(item.get("created_at", "")))


def is_active_run(run: dict[str, Any] | None) -> bool:
    """只有排队和训练中的 Run 需要自动轮询。"""

    return bool(run and str(run.get("status")) in ACTIVE_RUN_STATUSES)


def _render_log_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        st.caption("训练日志尚未写入。")
        return
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in entries[-80:]]
    st.code("\n".join(lines), language="json")


def _render_run_snapshot(
    client: ToolWearApiClient,
    experiment_id: str,
    run: dict[str, Any],
) -> None:
    status = str(run["status"])
    columns = st.columns(5)
    columns[0].metric("Run 状态", RUN_STATUS_LABELS.get(status, status))
    columns[1].metric("Pipeline", run["pipeline_id"])
    columns[2].metric("进度", f"{float(run['progress']):.0%}")
    columns[3].metric("Epoch", f"{run['current_epoch']} / {run['total_epochs']}")
    columns[4].metric("创建时间", readable_time(run.get("created_at")))
    st.progress(float(run["progress"]), text=run.get("progress_message") or "正在更新运行状态…")
    if run.get("error_message"):
        st.error(f"{run.get('error_code', 'TRAINING_FAILED')}：{run['error_message']}")
    try:
        logs = client.run_logs(experiment_id, run["run_id"], tail=100)
    except ToolApiError as exc:
        render_api_error(exc)
    else:
        with st.expander("结构化训练日志", expanded=status in {"running", "failed"}):
            _render_log_entries(list(logs.get("entries", [])))


def _request_cancel(
    client: ToolWearApiClient,
    experiment_id: str,
) -> None:
    """向状态机发送取消请求，实际 Worker 会在安全检查点停止。"""

    try:
        client.action(
            experiment_id,
            "cancel",
            rationale="用户从实验台请求在安全检查点取消训练。",
            idempotency_key=operation_key(f"{experiment_id}-cancel"),
        )
    except ToolApiError as exc:
        render_api_error(exc)
    else:
        set_flash("取消请求已发送，Worker 会在安全检查点停止。", level="warning")
        st.rerun()


@st.fragment(run_every=2.0)
def _render_live_run(
    client: ToolWearApiClient,
    experiment_id: str,
    run_id: str,
    *,
    allow_cancel: bool,
) -> None:
    """按 Run ID 轮询 FastAPI；训练结束后刷新整个实验状态。"""

    try:
        run = client.get_run(experiment_id, run_id)
    except ToolApiError as exc:
        render_api_error(exc)
        st.caption("自动轮询暂时失败；页面会继续重试，也可手动刷新浏览器。")
        return

    _render_run_snapshot(client, experiment_id, run)
    if not is_active_run(run):
        # 全页刷新会重新获取 ExperimentState、Run、指标和按钮权限。
        st.rerun()

    if allow_cancel and st.button(
        "取消训练",
        icon=":material/stop_circle:",
        width="stretch",
        key=f"cancel-active-run-{run_id}",
    ):
        _request_cancel(client, experiment_id)


def render_training_tab(
    client: ToolWearApiClient,
    state: dict[str, Any],
    actions: StateActions,
    runs: list[dict[str, Any]],
    revision: dict[str, Any] | None,
) -> None:
    """启动由审批 revision 驱动的真实训练，并显示可恢复 Run。"""

    st.subheader("真实小样本训练")
    st.markdown(
        "<div class='tw-section-note'>训练只读取已锁定 split 的 train/validation。小样本从训练集内按磨损阶段分层抽取，"
        "生成的模型、代码快照、参数、指标和日志都会登记为证据。</div>",
        unsafe_allow_html=True,
    )
    if revision:
        config = revision["run_config"]
        config_columns = st.columns(6)
        config_columns[0].metric("方案", revision["pipeline"]["display_name"])
        config_columns[1].metric("Batch", config["batch_size"])
        config_columns[2].metric("Epoch", config["epochs"])
        config_columns[3].metric("学习率", f"{config['learning_rate']:.6g}")
        config_columns[4].metric("设备请求", config["device"])
        config_columns[5].metric("样本上限", config.get("max_samples") or "不限")
    else:
        st.info("尚未形成训练 revision。请先在“候选与参数”中确认方案并完成校验。")

    if actions.start_training:
        st.warning("点击后会启动真实训练 Worker，并实际占用 CPU 或 CUDA。")
        if st.button(
            "启动小样本训练",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        ):
            try:
                created = client.start_mini_run(
                    state["experiment_id"],
                    idempotency_key=operation_key(f"{state['experiment_id']}-mini-run"),
                )
            except ToolApiError as exc:
                render_api_error(exc)
            else:
                set_flash(f"训练任务 {created['run_id']} 已进入后台队列。")
                st.rerun()

    current = latest_run(runs)
    if current is None:
        st.caption("当前实验还没有训练 Run。")
        return

    st.divider()
    st.subheader("运行监控")
    if is_active_run(current):
        st.caption("页面每 2 秒从 FastAPI 同步一次当前 Run、Epoch、进度和结构化日志。")
        _render_live_run(
            client,
            state["experiment_id"],
            str(current["run_id"]),
            allow_cancel=actions.cancel_training,
        )
        return

    _render_run_snapshot(client, state["experiment_id"], current)
    controls = st.columns([1, 5])
    if controls[0].button("刷新运行", icon=":material/refresh:", width="stretch"):
        st.rerun()
