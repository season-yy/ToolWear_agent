"""ToolWear AgentTeams 统一实验台。

第 12 步目标：让用户能在页面选择 C1、查看六 Agent、确认候选方案、
触发已有 P0 小样本训练流程，并查看指标、图表、诊断、报告和 Trace。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from toolwear_agent.frontend.console import main as run_state_driven_console


# Streamlit 直接执行本文件时只启动新工作台；下方旧函数暂留给迁移期导入兼容。
if __name__ == "__main__":
    run_state_driven_console()
    st.stop()

from toolwear_agent.common.config import load_settings
from toolwear_agent.frontend.dashboard_data import (
    build_dashboard_paths,
    build_result_snapshot,
    build_candidate_choices,
    generate_llm_candidates_for_request,
    run_p0_training_flow,
    run_toolwear_command,
)
from toolwear_agent.training.selection import (
    load_candidate_set,
    select_candidate_plan,
    write_selected_plan_json,
    write_selected_plan_log,
    write_selected_plan_report,
)


def _default_paths() -> tuple[Path, Path, Path, Path]:
    """兼容旧测试：返回候选方案和确认结果默认路径。"""

    settings = load_settings()
    paths = build_dashboard_paths(settings)
    return (
        paths.candidate_file,
        paths.selected_plan_file,
        paths.selected_plan_report,
        settings.log_root / "phm2010_c1_selected_plan.log",
    )


def _show_command_results(results: list[object]) -> None:
    """展示按钮触发的命令结果。"""

    for result in results:
        ok = result.return_code == 0
        with st.expander(f"{'完成' if ok else '失败'}：{result.command}", expanded=not ok):
            if result.stdout.strip():
                st.code(result.stdout.strip(), language="text")
            if result.stderr.strip():
                st.code(result.stderr.strip(), language="text")


def _render_plan_card(plan: dict[str, object], selected: bool) -> None:
    """渲染候选方案卡片。"""

    status = "当前选择" if selected else "候选方案"
    source_label = "LLM Agent" if plan.get("source") == "llm" else "固定模板"
    st.markdown(f"#### {plan['display_name']}")
    st.caption(f"{status} | {source_label} | {plan['plan_id']}")
    st.write(plan["summary"])
    cols = st.columns([1, 1, 1, 1])
    cols[0].metric("当前可训练", "是" if plan.get("suitable_for_p0", False) else "否")
    cols[1].write("**模型结构**")
    cols[1].write(plan["model_structure"])
    cols[2].write("**预计成本**")
    cols[2].write(plan["expected_cost"])
    cols[3].write("**训练入口**")
    train_entries = []
    if plan.get("single_train_supported", False):
        train_entries.append("单方案")
    if plan.get("compare_train_supported", False):
        train_entries.append("多候选对比")
    cols[3].write(" / ".join(train_entries) if train_entries else "后续扩展")
    with st.expander("推荐理由、优点和风险"):
        st.write(plan["recommended_reason"])
        st.markdown("**优点**")
        for item in plan["advantages"]:
            st.write(f"- {item}")
        st.markdown("**风险**")
        for item in plan["risks"]:
            st.write(f"- {item}")


def _render_metrics(snapshot: dict[str, object]) -> None:
    """渲染训练指标和图表。"""

    metrics = snapshot["metrics"]
    visual_manifest = snapshot["visual_manifest"]
    if not metrics:
        st.info("还没有训练结果。请先点击“小样本训练并刷新证据”。")
        return

    cols = st.columns(4)
    cols[0].metric("验证 Macro-F1", f"{metrics.get('validation_macro_f1', 0):.4f}")
    cols[1].metric("验证 Balanced Acc", f"{metrics.get('validation_balanced_accuracy', 0):.4f}")
    cols[2].metric("训练小样本", int(metrics.get("sample_count", 0)))
    final_test_status = str(metrics.get("final_test_status", "not_run_pipeline_not_frozen"))
    cols[3].metric("最终测试", "未执行" if final_test_status.startswith("not_run") else "已完成")

    st.caption(f"运行目录：{snapshot['run_dir']}")
    image_cols = st.columns(3)
    image_items = [
        ("验证集混淆矩阵", visual_manifest.get("validation_confusion_matrix_png", "")),
        ("t-SNE", visual_manifest.get("tsne_png", "")),
        ("阶段分布", visual_manifest.get("stage_distribution_png", "")),
    ]
    for column, (title, image_path) in zip(image_cols, image_items, strict=False):
        with column:
            st.markdown(f"**{title}**")
            if image_path and Path(str(image_path)).exists():
                st.image(str(image_path), width="stretch")
            else:
                st.warning("图表暂未生成")


def _render_agent_outputs(snapshot: dict[str, object]) -> None:
    """渲染诊断、决策和证据路径。"""

    diagnosis = snapshot["diagnosis"]
    decision = snapshot["decision"]
    paths = snapshot["paths"]
    st.markdown("### Agent 诊断")
    st.write(diagnosis.get("overall_conclusion", "暂无诊断。"))
    for item in diagnosis.get("recommendations", [])[:5]:
        st.write(f"- {item}")

    st.markdown("### Agent 决策")
    st.write(decision.get("overall_decision", "暂无决策。"))

    st.markdown("### 报告与证据")
    for label, path in [
        ("P0 实验报告", paths.p0_report),
        ("AgentTeams Trace", paths.trace_report),
        ("Agent Identity", paths.identity_report),
        ("Skill Manifest", paths.skill_manifest),
    ]:
        st.write(label)
        st.code(str(path), language="text")


def _render_llm_candidates(snapshot: dict[str, object]) -> None:
    """渲染 LLM 候选方案。"""

    llm_candidates = snapshot["llm_candidates"]
    if not llm_candidates:
        st.info("尚未生成 LLM 候选方案。")
        return
    st.caption(
        f"Provider: {llm_candidates.get('provider')} | Model: {llm_candidates.get('model') or '未使用'} | "
        f"Fallback: {llm_candidates.get('used_fallback')}"
    )
    if llm_candidates.get("fallback_reason"):
        st.warning(f"Fallback 原因：{llm_candidates['fallback_reason']}")
    st.info("当前页面会优先使用这批 LLM 候选作为下方单选来源；如果没有 LLM 候选，才使用固定模板。")


def _render_compare_result(snapshot: dict[str, object]) -> None:
    """渲染多候选训练对比结果。"""

    compare_result = snapshot["compare_result"]
    if not compare_result:
        st.info("尚未生成多候选训练对比结果。")
        return
    rows = []
    for metric in compare_result.get("metrics", []):
        rows.append(
            {
                "候选": metric["display_name"],
                "分类器": metric["classifier_name"],
                "Validation Macro-F1": metric["validation_macro_f1"],
                "Validation Balanced Acc": metric["validation_balanced_accuracy"],
                "建议": metric["recommendation"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.success(f"当前推荐候选：{compare_result.get('best_plan_id', '')}")
    st.code(str(snapshot["paths"].candidate_compare_report), language="text")


def _render_official_agentteams(snapshot: dict[str, object]) -> None:
    """渲染官方 AgentTeams 最小接入证据。"""

    paths = snapshot["paths"]
    package = snapshot.get("official_agentteams", {})
    if not package:
        st.info("尚未生成官方 AgentTeams 最小接入包。")
        st.code("python -m toolwear_agent official-agentteams-c1", language="powershell")
        return

    st.success(f"AgentTeams 接入包已生成：{package.get('team_name', '')}")
    cols = st.columns(3)
    cols[0].metric("Team Leader", package.get("leader_name", ""))
    cols[1].metric("业务 Worker", len(package.get("worker_names", [])))
    cols[2].metric("Docker 证据", len(package.get("docker_items", [])))
    st.write("Element 创建消息")
    st.code(str(paths.official_agentteams_message), language="text")
    st.write("接入报告")
    st.code(str(paths.official_agentteams_report), language="text")


def main() -> None:
    """渲染统一实验台。"""

    st.set_page_config(page_title="刃知 - 刀具磨损监测算法辅助平台", layout="wide", initial_sidebar_state="expanded")
    settings = load_settings()
    paths = build_dashboard_paths(settings)
    snapshot = build_result_snapshot(settings)

    st.title("刃知：基于 AgentTeams 的刀具磨损监测算法辅助平台")
    st.caption("PHM2010 C1 初赛 PoC：模块选择、小样本训练、结果诊断和证据归档。")

    with st.sidebar:
        st.header("实验配置")
        dataset = st.selectbox("数据集", ["PHM2010 C1"], index=0)
        task = st.selectbox("任务", ["四阶段磨损分类"], index=0)
        vb_policy = st.selectbox("VB 聚合", ["max", "mean（预留）"], index=0)
        window_size = st.number_input("窗口长度", min_value=1024, max_value=8192, value=4096, step=512)
        overlap = st.slider("重叠率", min_value=0.0, max_value=0.75, value=0.5, step=0.05)
        sample_fraction = st.slider("小样本比例", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
        st.divider()
        st.write("当前选择")
        st.code(
            f"数据集: {dataset}\n任务: {task}\nVB: {vb_policy}\n窗口: {window_size}\n重叠率: {overlap}\n小样本比例: {sample_fraction}",
            language="text",
        )

    top_cols = st.columns(3)
    top_cols[0].metric("核心 Agent", len(snapshot["core_agents"]))
    top_cols[1].metric("当前候选", len(snapshot["candidate_choices"]))
    top_cols[2].metric("最新运行", "已生成" if snapshot["run_dir"] else "暂无")

    st.markdown("## 六个核心 Agent")
    st.write(" / ".join(snapshot["core_agents"]))

    st.markdown("## 候选方案与用户确认")
    user_request = st.text_area(
        "用户需求",
        value="我想用 PHM2010 C1 做四阶段刀具磨损分类，优先快速验证，并比较传统模型和后续深度模型路线。",
        height=90,
    )
    action_cols = st.columns([1, 1, 1, 1])
    if action_cols[0].button("生成/刷新候选方案", type="secondary"):
        _show_command_results([run_toolwear_command("candidates-c1", settings.app_root)])
        st.rerun()
    if action_cols[1].button("调用 LLM 生成候选", type="secondary"):
        candidate_file, report_file, log_file = generate_llm_candidates_for_request(settings, user_request)
        st.success("AlgorithmArchitectAgent 已生成候选方案")
        st.code(f"{candidate_file}\n{report_file}\n{log_file}", language="text")
        st.rerun()
    if action_cols[2].button("多候选训练对比", type="secondary"):
        _show_command_results([run_toolwear_command("compare-candidates-c1", settings.app_root)])
        st.rerun()
    if action_cols[3].button("刷新页面数据", type="secondary"):
        st.rerun()

    st.markdown("### 候选来源")
    _render_llm_candidates(snapshot)

    candidate_choices = snapshot.get("candidate_choices") or build_candidate_choices(snapshot)
    if not candidate_choices:
        st.warning("尚未生成候选方案，请点击“生成/刷新候选方案”。")
    else:
        source_name = "LLM Agent 候选" if candidate_choices[0].get("source") == "llm" else "固定模板候选"
        st.markdown(f"### 方案选择（当前来源：{source_name}）")
        plan_options = [card["plan_id"] for card in candidate_choices]
        selected_plan_id = st.radio(
            "选择一个候选方案",
            options=plan_options,
            format_func=lambda plan_id: next(str(card["display_name"]) for card in candidate_choices if card["plan_id"] == plan_id),
            horizontal=True,
        )
        selected_choice = next(card for card in candidate_choices if card["plan_id"] == selected_plan_id)
        if not selected_choice.get("single_train_supported", False):
            st.warning(
                "当前选中的候选不能进入单方案小样本训练。"
                "如果它支持多候选对比，请点击上方“多候选训练对比”；否则需要后续补训练后端。"
            )
        for card in candidate_choices:
            _render_plan_card(card, selected=card["plan_id"] == selected_plan_id)
            st.divider()

        if st.button("确认当前方案", type="primary"):
            if not selected_choice.get("single_train_supported", False):
                st.error("该方案暂不支持单方案确认训练，系统不会写入错误的 selected_plan 文件。")
            else:
                candidate_set = load_candidate_set(paths.candidate_file)
                selected_plan = select_candidate_plan(candidate_set, selected_plan_id, "local_user", str(paths.candidate_file))
                selected_log = settings.log_root / "phm2010_c1_selected_plan.log"
                write_selected_plan_json(selected_plan, paths.selected_plan_file)
                write_selected_plan_report(selected_plan, paths.selected_plan_report)
                write_selected_plan_log(selected_plan, selected_log, paths.selected_plan_file, paths.selected_plan_report)
                st.success(f"已确认方案：{selected_plan.selected_plan.display_name}")
                st.code(str(paths.selected_plan_file), language="text")

    st.markdown("## 单方案小样本训练")
    st.warning("当前单方案训练入口只支持 RandomForest；ExtraTrees 请使用“多候选训练对比”。")
    if st.button("小样本训练并刷新证据", type="primary"):
        with st.status("正在执行 P0 小样本训练流程", expanded=True):
            results = run_p0_training_flow(settings.app_root)
            _show_command_results(results)
        st.rerun()

    st.markdown("## 训练结果")
    _render_metrics(snapshot)

    st.markdown("## 多候选训练对比")
    _render_compare_result(snapshot)

    st.markdown("## 官方 AgentTeams 接入")
    if st.button("生成 AgentTeams 最小接入包", type="secondary"):
        _show_command_results([run_toolwear_command("official-agentteams-c1", settings.app_root)])
        st.rerun()
    _render_official_agentteams(snapshot)

    st.markdown("## 诊断、决策与证据")
    _render_agent_outputs(snapshot)
