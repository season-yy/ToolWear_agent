"""实验工作台的 LLM 候选、模块参数和人工审批页签。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient
from toolwear_agent.frontend.ui_components import operation_key, render_api_error, set_flash
from toolwear_agent.frontend.ui_state import StateActions


COST_LABELS = {"low": "低", "medium": "中", "high": "高"}
KIND_LABELS = {
    "windowing": "滑窗",
    "preprocess": "预处理",
    "feature": "特征",
    "fusion": "融合",
    "model": "模型",
    "loss": "损失",
    "trainer": "训练器",
}


def _definition_maps(capabilities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = {
        str(item["module_id"]): item for item in capabilities.get("modules", [])
    }
    definitions.update(
        {str(item["trainer_id"]): item for item in capabilities.get("trainers", [])}
    )
    return definitions


def _parameter_widget(
    pipeline_id: str,
    module_id: str,
    parameter_name: str,
    rule: dict[str, Any],
    current_value: Any,
) -> Any:
    """把 Registry ParameterRule 转成类型匹配的 Streamlit 控件。"""

    key = f"param-{pipeline_id}-{module_id}-{parameter_name}"
    label = parameter_name.replace("_", " ")
    help_text = str(rule.get("description", ""))
    choices = list(rule.get("choices", []))
    value_type = rule.get("value_type")
    value = current_value if current_value is not None else rule.get("default")
    if choices:
        selected = value if value in choices else choices[0]
        return st.selectbox(label, choices, index=choices.index(selected), help=help_text, key=key)
    if value_type == "boolean":
        return st.toggle(label, value=bool(value), help=help_text, key=key)
    if value_type == "integer":
        minimum = int(rule.get("minimum")) if rule.get("minimum") is not None else None
        maximum = int(rule.get("maximum")) if rule.get("maximum") is not None else None
        return int(
            st.number_input(
                label,
                value=int(value or 0),
                min_value=minimum,
                max_value=maximum,
                step=1,
                help=help_text,
                key=key,
            )
        )
    if value_type == "number":
        minimum = float(rule.get("minimum")) if rule.get("minimum") is not None else None
        maximum = float(rule.get("maximum")) if rule.get("maximum") is not None else None
        numeric_value = float(value or 0.0)
        step = max(abs(numeric_value) / 10.0, 0.000001)
        return float(
            st.number_input(
                label,
                value=numeric_value,
                min_value=minimum,
                max_value=maximum,
                step=step,
                format="%.6g",
                help=help_text,
                key=key,
            )
        )
    return st.text_input(label, value=str(value or ""), help=help_text, key=key)


def _render_candidate_summary(
    pipeline: dict[str, Any],
    definitions: dict[str, dict[str, Any]],
) -> None:
    modules = [item for item in pipeline["modules"] if item.get("enabled", True)]
    module_names = [
        str(definitions.get(item["module_id"], {}).get("display_name", item["module_id"]))
        for item in modules
    ]
    columns = st.columns([0.8, 1, 2.2])
    columns[0].metric("预计成本", COST_LABELS.get(pipeline["expected_cost"], pipeline["expected_cost"]))
    columns[1].metric("可真实训练", "是" if pipeline.get("trainable") else "否")
    columns[2].markdown("**模块链**  \n" + " → ".join(module_names))
    st.write(pipeline["rationale"])
    with st.expander("风险与兼容性"):
        for risk in pipeline.get("risks", []):
            st.write(f"- {risk}")
        tags = " / ".join(pipeline.get("compatibility_tags", [])) or "无"
        st.caption(f"兼容标签：{tags}")


def _collect_module_parameters(
    pipeline: dict[str, Any],
    definitions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """只允许用户编辑 Registry 白名单中的参数，窗口参数保持创建时锁定。"""

    collected: dict[str, dict[str, Any]] = {}
    for module in pipeline["modules"]:
        module_id = str(module["module_id"])
        definition = definitions.get(module_id, {})
        if module.get("kind") == "windowing":
            continue
        parameter_rules = definition.get("parameters_schema", {})
        if not parameter_rules:
            continue
        display_name = definition.get("display_name", module_id)
        st.markdown(f"**{KIND_LABELS.get(module.get('kind'), module.get('kind'))} · {display_name}**")
        current = module.get("parameters", {})
        module_values: dict[str, Any] = {}
        columns = st.columns(min(len(parameter_rules), 3))
        for index, (name, rule) in enumerate(parameter_rules.items()):
            with columns[index % len(columns)]:
                module_values[name] = _parameter_widget(
                    pipeline["pipeline_id"],
                    module_id,
                    name,
                    rule,
                    current.get(name),
                )
        collected[module_id] = module_values
    return collected


def _approve_candidate(
    client: ToolWearApiClient,
    experiment_id: str,
    pipeline: dict[str, Any],
    module_parameters: dict[str, dict[str, Any]],
    training: dict[str, Any],
) -> None:
    payload = {
        "pipeline_id": pipeline["pipeline_id"],
        "rationale": "用户在实验台确认候选模块链和训练参数。",
        "input_channels": pipeline["input_channels"],
        "module_parameters": module_parameters,
        **training,
    }
    try:
        client.approve_pipeline(
            experiment_id,
            payload,
            idempotency_key=operation_key(f"{experiment_id}-approve"),
        )
    except ToolApiError as exc:
        render_api_error(exc)
        return
    set_flash("候选方案已审批并冻结为不可变 revision。下一步执行兼容性校验。")
    st.rerun()


def render_pipeline_tab(
    client: ToolWearApiClient,
    state: dict[str, Any],
    capabilities: dict[str, Any],
    actions: StateActions,
    recommendation: dict[str, Any] | None,
    revision: dict[str, Any] | None,
) -> None:
    """生成真实 LLM 候选，并让用户选择可执行模块和超参数。"""

    st.subheader("候选方案与人工确认")
    st.markdown(
        "<div class='tw-section-note'>LLM 负责结合实验背景提出 2–3 个候选；Registry 负责过滤不存在、"
        "未实现或不兼容的模块。只有用户确认后的方案才会进入训练。</div>",
        unsafe_allow_html=True,
    )
    request_text = st.text_area(
        "给算法架构 Agent 的要求",
        value=state["objective"],
        height=96,
        key=f"candidate-request-{state['experiment_id']}",
    )
    generate_columns = st.columns([1.2, 0.8, 3])
    force_refresh = generate_columns[1].toggle(
        "重新请求 LLM",
        value=False,
        help="关闭时会恢复上次候选；开启时才会产生新的模型调用费用。",
        key=f"force-recommend-{state['experiment_id']}",
    )
    if generate_columns[0].button(
        "调用 LLM 生成候选",
        type="primary",
        icon=":material/psychology:",
        disabled=not actions.generate_candidates,
        width="stretch",
    ):
        try:
            with st.status("AlgorithmArchitectAgent 正在分析数据约束和 Registry…", expanded=True) as status:
                st.write("提交项目目标、输入通道、样本比例和本机资源约束。")
                st.write("等待千问返回结构化 Pipeline，并执行本地兼容性校验。")
                result = client.generate_recommendations(
                    state["experiment_id"],
                    user_request=request_text,
                    force_refresh=force_refresh,
                    idempotency_key=operation_key(f"{state['experiment_id']}-recommend"),
                )
                status.update(
                    label=f"已生成 {len(result.get('pipelines', []))} 个可校验候选。",
                    state="complete",
                )
        except ToolApiError as exc:
            render_api_error(exc)
        else:
            set_flash("LLM 候选已保存到 SQLite，刷新或重启不会丢失。")
            st.rerun()

    if recommendation is None:
        if state.get("selected_pipeline_ref") and revision:
            st.info(f"当前已锁定方案：{state['selected_pipeline_ref']}")
        else:
            st.info("尚无候选。完成数据准备后，点击上方按钮调用 LLM。")
        return

    source_columns = st.columns(4)
    source_columns[0].metric("Provider", recommendation["provider"])
    source_columns[1].metric("Model", recommendation["model"] or "规则回退")
    source_columns[2].metric("候选数", len(recommendation["pipelines"]))
    source_columns[3].metric("使用回退", "是" if recommendation["used_fallback"] else "否")
    if recommendation.get("fallback_reason"):
        st.warning(f"LLM 回退原因：{recommendation['fallback_reason']}")

    definitions = _definition_maps(capabilities)
    pipelines = list(recommendation["pipelines"])
    selected_id = st.radio(
        "选择一个候选方案",
        options=[item["pipeline_id"] for item in pipelines],
        format_func=lambda value: next(item["display_name"] for item in pipelines if item["pipeline_id"] == value),
        key=f"pipeline-choice-{recommendation['recommendation_id']}",
    )
    selected = next(item for item in pipelines if item["pipeline_id"] == selected_id)
    _render_candidate_summary(selected, definitions)

    if actions.approve_pipeline:
        st.divider()
        st.subheader("模块与训练参数")
        preferences = state["preferences"]
        st.caption(
            f"窗口参数已由数据证据锁定：{preferences['window_length']} 点，"
            f"重叠率 {preferences['overlap']:.0%}。如需修改，必须新建实验并重新切分。"
        )
        module_parameters = _collect_module_parameters(selected, definitions)
        trainer_id = next(
            module["module_id"] for module in selected["modules"] if module["kind"] == "trainer"
        )
        is_pytorch = trainer_id == "pytorch"
        training_columns = st.columns(6)
        batch_size = training_columns[0].number_input("Batch size", 1, 1024, 64, 1)
        epochs = training_columns[1].number_input("Epoch", 1, 100, 5 if is_pytorch else 1, 1)
        learning_rate = training_columns[2].number_input(
            "学习率", 0.000001, 1.0, 0.001, format="%.6f"
        )
        device_options = ["cuda", "auto", "cpu"] if is_pytorch else ["cpu", "auto"]
        device = training_columns[3].selectbox("设备", device_options)
        num_workers = training_columns[4].number_input("Workers", 0, 16, 0, 1)
        max_samples_value = training_columns[5].number_input(
            "样本上限 (0=不限)", 0, 100_000, 0, 100
        )
        training = {
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "device": device,
            "num_workers": int(num_workers),
            "max_samples": int(max_samples_value) or None,
        }
        if st.button(
            "确认方案与参数",
            type="primary",
            icon=":material/verified:",
            width="stretch",
            disabled=not selected.get("trainable", False),
        ):
            _approve_candidate(
                client,
                state["experiment_id"],
                selected,
                module_parameters,
                training,
            )

    if revision:
        st.divider()
        st.subheader("已冻结 revision")
        run_config = revision["run_config"]
        columns = st.columns(4)
        columns[0].metric("Revision", revision["revision"])
        columns[1].metric("Pipeline", revision["pipeline"]["display_name"])
        columns[2].metric("训练样本比例", f"{run_config['sample_fraction']:.0%}")
        columns[3].metric("设备", run_config["device"])

    if actions.validate_pipeline:
        if st.button(
            "校验模块链并准备训练",
            type="primary",
            icon=":material/fact_check:",
            width="stretch",
        ):
            try:
                response = client.action(
                    state["experiment_id"],
                    "validate",
                    rationale="用户确认后执行 Registry 兼容性和可训练性校验。",
                    idempotency_key=operation_key(f"{state['experiment_id']}-validate"),
                )
            except ToolApiError as exc:
                render_api_error(exc)
            else:
                validation = response.get("validation", {})
                set_flash(
                    "模块链校验通过，可以启动真实小样本训练。"
                    if validation.get("valid")
                    else "模块链校验失败，请返回候选方案调整。",
                    level="success" if validation.get("valid") else "warning",
                )
                st.rerun()
