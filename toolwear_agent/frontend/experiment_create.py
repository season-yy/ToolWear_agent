"""新建实验页面：所有可见参数都会进入 ExperimentState。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient
from toolwear_agent.frontend.ui_components import operation_key, render_api_error, set_flash


def _available_cutters(dataset: dict[str, Any]) -> list[str]:
    cutters = dataset.get("cutters", {})
    return [
        cutter_id
        for cutter_id, detail in cutters.items()
        if detail.get("available") and detail.get("labeled")
    ]


def render_create_experiment(
    client: ToolWearApiClient,
    datasets: list[dict[str, Any]],
    capabilities: dict[str, Any],
) -> None:
    """渲染实验定义，并把参数作为一个 API 请求提交。"""

    st.markdown("<div class='tw-kicker'>NEW EXPERIMENT</div>", unsafe_allow_html=True)
    st.title("建立刀具磨损实验")
    st.markdown(
        "<div class='tw-subtitle'>先锁定数据、标签与窗口策略，再进入候选方案和真实训练。</div>",
        unsafe_allow_html=True,
    )
    if not datasets:
        st.error("Dataset Registry 当前没有可用数据集。")
        return

    dataset_ids = [str(item["dataset_id"]) for item in datasets]
    dataset_id = st.selectbox(
        "数据集",
        dataset_ids,
        format_func=lambda value: next(
            str(item.get("display_name", value)) for item in datasets if item["dataset_id"] == value
        ),
        key="create_dataset_id",
    )
    dataset = next(item for item in datasets if item["dataset_id"] == dataset_id)
    cutters = _available_cutters(dataset)
    if not cutters:
        st.error("该数据集没有同时具备信号与磨损标签的刀具。")
        return

    basics_left, basics_right = st.columns([1.35, 1])
    with basics_left:
        title = st.text_input(
            "实验名称",
            value="PHM2010 刀具磨损四阶段分类",
            key="create_title",
        )
        user_request = st.text_area(
            "实验目标",
            value=(
                "使用多通道铣削信号完成刀具磨损四阶段分类，先比较低成本基线与 "
                "1D-CNN，再依据 validation 指标决定是否调整或进入完整训练。"
            ),
            height=118,
            key="create_objective",
        )
    with basics_right:
        cutter_id = st.selectbox("刀具", cutters, key="create_cutter")
        mode = st.segmented_control(
            "实验模式",
            options=["quick", "balanced"],
            default="quick",
            format_func=lambda value: "快速验证" if value == "quick" else "均衡训练",
            key="create_mode",
        )
        enable_regression = st.toggle(
            "同时预测 VB 连续值",
            value=False,
            key="create_regression",
        )

    st.subheader("标签策略")
    aggregation = st.segmented_control(
        "VB 聚合",
        options=["max", "mean", "specified_flute"],
        default="max",
        format_func=lambda value: {
            "max": "三刃最大值",
            "mean": "三刃平均值",
            "specified_flute": "指定刀刃",
        }[value],
        key="create_aggregation",
    )
    threshold_columns = st.columns(4)
    threshold_1 = threshold_columns[0].number_input(
        "初期 / 正常 (μm)", min_value=1.0, value=90.0, step=5.0
    )
    threshold_2 = threshold_columns[1].number_input(
        "正常 / 剧烈 (μm)", min_value=1.0, value=130.0, step=5.0
    )
    threshold_3 = threshold_columns[2].number_input(
        "剧烈 / 失效 (μm)", min_value=1.0, value=160.0, step=5.0
    )
    specified_flute = None
    with threshold_columns[3]:
        if aggregation == "specified_flute":
            specified_flute = st.number_input(
                "刀刃编号",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
            )
        else:
            st.metric("阶段数", "4")

    st.subheader("信号与窗口")
    presets = {
        str(item["preset_id"]): item for item in capabilities.get("input_presets", [])
    }
    preset_ids = list(presets)
    default_preset = "all_7_channels" if "all_7_channels" in presets else preset_ids[0]
    if "create_channels" not in st.session_state:
        st.session_state["create_channels"] = list(presets[default_preset]["channel_ids"])

    def apply_preset() -> None:
        selected = st.session_state["create_preset"]
        st.session_state["create_channels"] = list(presets[selected]["channel_ids"])

    st.selectbox(
        "通道预设",
        preset_ids,
        index=preset_ids.index(default_preset),
        format_func=lambda value: str(presets[value]["display_name"]),
        key="create_preset",
        on_change=apply_preset,
    )
    dataset_channels = [str(item) for item in dataset.get("channels", [])]
    input_channels = st.multiselect(
        "输入通道",
        dataset_channels,
        key="create_channels",
    )
    signal_columns = st.columns(4)
    window_length = signal_columns[0].number_input(
        "窗口长度",
        min_value=256,
        max_value=65_536,
        value=4096,
        step=256,
    )
    overlap = signal_columns[1].slider(
        "重叠率",
        min_value=0.0,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )
    sample_fraction = signal_columns[2].slider(
        "训练集抽样比例",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )
    max_windows = signal_columns[3].number_input(
        "每个 cut 最大窗口",
        min_value=1,
        max_value=256,
        value=32,
        step=1,
    )

    with st.expander("切分与复现参数"):
        split_columns = st.columns(4)
        train_ratio = split_columns[0].number_input(
            "训练集比例", min_value=0.05, max_value=0.9, value=0.6, step=0.05
        )
        validation_ratio = split_columns[1].number_input(
            "验证集比例", min_value=0.05, max_value=0.9, value=0.2, step=0.05
        )
        test_ratio = split_columns[2].number_input(
            "测试集比例", min_value=0.05, max_value=0.9, value=0.2, step=0.05
        )
        random_seed = split_columns[3].number_input(
            "随机种子", min_value=0, value=42, step=1
        )

    if st.button(
        "创建实验",
        type="primary",
        icon=":material/add_circle:",
        width="stretch",
    ):
        thresholds = (float(threshold_1), float(threshold_2), float(threshold_3))
        if not thresholds[0] < thresholds[1] < thresholds[2]:
            st.error("三个 VB 阈值必须严格递增。")
            return
        if not input_channels:
            st.error("至少选择一个输入通道。")
            return
        if abs(float(train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-6:
            st.error("训练、验证和测试比例之和必须等于 1。")
            return
        payload = {
            "title": title,
            "user_request": user_request,
            "dataset_id": dataset_id,
            "cutter_ids": [cutter_id],
            "input_channels": input_channels,
            "vb_aggregation": aggregation,
            "vb_thresholds_um": thresholds,
            "enable_vb_regression": enable_regression,
            "specified_flute": int(specified_flute) if specified_flute else None,
            "train_ratio": float(train_ratio),
            "validation_ratio": float(validation_ratio),
            "test_ratio": float(test_ratio),
            "random_seed": int(random_seed),
            "window_length": int(window_length),
            "overlap": float(overlap),
            "sample_fraction": float(sample_fraction),
            "max_windows_per_cut": int(max_windows),
            "mode": mode,
        }
        try:
            with st.status("正在创建实验并登记状态…", expanded=True) as status:
                created = client.create_experiment(
                    payload,
                    idempotency_key=operation_key("create-experiment"),
                )
                status.update(label="实验已登记。", state="complete")
        except ToolApiError as exc:
            render_api_error(exc)
            return
        # 选择器已在本轮脚本中实例化，先写待切换值，下一轮渲染前再同步控件状态。
        st.session_state["pending_experiment_id"] = created["experiment_id"]
        set_flash("实验已创建。下一步进行数据体检、标签和无泄漏切分。")
        st.rerun()
