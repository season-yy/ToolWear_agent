"""实验工作台的数据准备页签。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient
from toolwear_agent.frontend.ui_components import operation_key, render_api_error, set_flash
from toolwear_agent.frontend.ui_state import PreparationProgress, StateActions


def _evidence_row(label: str, ready: bool, description: str) -> None:
    """显示一项数据治理证据是否已经落盘。"""

    left, right = st.columns([0.23, 0.77], vertical_alignment="center")
    left.markdown(f"**{'已完成' if ready else '待执行'} · {label}**")
    right.caption(description)


def _prepare_missing_steps(
    client: ToolWearApiClient,
    experiment_id: str,
    progress: PreparationProgress,
) -> None:
    """按依赖顺序补齐体检、标签和 split，已存在的证据不会重复计算。"""

    steps = (
        ("profile", "正在核对原始文件、通道与标签文件…", progress.profile),
        ("labels", "正在按 VB 规则生成四阶段标签…", progress.labels),
        ("split", "正在按 cut 分组切分并执行泄漏审计…", progress.split),
    )
    try:
        with st.status("正在准备实验数据", expanded=True) as status:
            for action, message, completed in steps:
                if completed:
                    st.write(f"跳过 {action}：证据已存在。")
                    continue
                st.write(message)
                client.action(
                    experiment_id,
                    action,
                    rationale="用户在实验台批准执行确定性数据准备。",
                    idempotency_key=operation_key(f"{experiment_id}-{action}"),
                )
            status.update(label="数据准备步骤执行完成。", state="complete")
    except ToolApiError as exc:
        render_api_error(exc)
        return
    set_flash("数据体检、标签和无泄漏切分证据已更新。")
    st.rerun()


def render_data_tab(
    client: ToolWearApiClient,
    state: dict[str, Any],
    progress: PreparationProgress,
    actions: StateActions,
) -> None:
    """展示创建时锁定的数据参数，并驱动三项确定性准备动作。"""

    st.subheader("数据与标签基线")
    st.markdown(
        "<div class='tw-section-note'>切分以 cut 为最小分组单位。同一 CSV 产生的窗口只会进入一个数据集，"
        "测试集不会参与候选选择、调参或停止决策。</div>",
        unsafe_allow_html=True,
    )
    dataset_ref = state["dataset_ref"]
    label_policy = state["label_policy"]
    split_spec = state["split_spec"]
    preferences = state["preferences"]

    columns = st.columns(4)
    columns[0].metric("数据集 / 刀具", f"{dataset_ref['dataset_id']} / {', '.join(dataset_ref['cutter_ids'])}")
    columns[1].metric("窗口", f"{preferences['window_length']} 点")
    columns[2].metric("重叠率", f"{float(preferences['overlap']):.0%}")
    columns[3].metric("训练抽样", f"{float(preferences['sample_fraction']):.0%}")

    details = st.columns([1.2, 1, 1])
    with details[0]:
        st.markdown("**输入通道**")
        st.write("、".join(preferences["input_channels"]))
    with details[1]:
        st.markdown("**VB 标签策略**")
        thresholds = " / ".join(str(value) for value in label_policy["stage_thresholds_um"])
        st.write(f"{label_policy['aggregation']} · {thresholds} μm")
    with details[2]:
        st.markdown("**切分策略**")
        st.write(
            f"{split_spec['strategy']} · "
            f"{split_spec['train_ratio']:.0%} / {split_spec['validation_ratio']:.0%} / "
            f"{split_spec['test_ratio']:.0%}"
        )

    st.divider()
    st.subheader("准备证据")
    _evidence_row("数据体检", progress.profile, "核对 C1 文件数、信号列数、磨损标签及清单哈希。")
    _evidence_row("标签生成", progress.labels, "三刃 VB 聚合后按阈值生成初期、正常、剧烈、失效四阶段。")
    _evidence_row("切分与泄漏审计", progress.split, "先按 cut 切分，再在训练集内部按阶段比例抽取小样本窗口。")

    if progress.complete:
        st.success("三项数据证据齐全，可以生成候选方案。")
    elif actions.prepare_data:
        if st.button(
            "执行尚未完成的数据准备",
            type="primary",
            icon=":material/dataset:",
            width="stretch",
        ):
            _prepare_missing_steps(client, state["experiment_id"], progress)
    else:
        st.info("当前状态不能修改数据证据；如需变更窗口或阈值，请建立新实验。")
