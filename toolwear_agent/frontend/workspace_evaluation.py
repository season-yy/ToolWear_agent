"""实验工作台的指标、诊断、决策与报告页签。"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from toolwear_agent.frontend.api_client import ToolApiError, ToolWearApiClient
from toolwear_agent.frontend.ui_components import operation_key, render_api_error, set_flash
from toolwear_agent.frontend.ui_state import StateActions
from toolwear_agent.frontend.workspace_training import latest_run


def _find_artifact(
    artifacts: list[dict[str, Any]],
    *,
    run_id: str,
    description: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in reversed(artifacts)
            if item.get("run_id") == run_id and item.get("description") == description
        ),
        None,
    )


def _load_json_evidence(
    client: ToolWearApiClient,
    artifact: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if artifact is None:
        return None
    try:
        payload = client.artifact_json(str(artifact["evidence_id"]))
    except ToolApiError as exc:
        render_api_error(exc)
        return None
    return payload if isinstance(payload, dict) else None


def _render_metrics(metrics_payload: dict[str, Any]) -> None:
    validation = metrics_payload.get("metrics_by_split", {}).get("validation")
    train = metrics_payload.get("metrics_by_split", {}).get("train")
    if not validation:
        st.info("指标证据存在，但还没有 validation 结果。")
        return
    columns = st.columns(5)
    columns[0].metric("Validation Macro-F1", f"{validation['macro_f1']:.4f}")
    columns[1].metric("Balanced Accuracy", f"{validation['balanced_accuracy']:.4f}")
    columns[2].metric("Validation 样本", validation["sample_count"])
    columns[3].metric("Train Macro-F1", f"{train['macro_f1']:.4f}" if train else "-")
    runtime = metrics_payload.get("runtime", {})
    device = runtime.get("resolved_device", "-")
    columns[4].metric("实际设备", device)

    class_labels = list(validation.get("per_class", {}))
    report_rows = []
    for label, values in validation.get("per_class", {}).items():
        report_rows.append(
            {
                "磨损阶段": label,
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "F1": values.get("f1-score"),
                "Support": values.get("support"),
            }
        )
    st.markdown("**分类报告**")
    st.table(pd.DataFrame(report_rows))
    st.markdown("**Validation 混淆矩阵**")
    matrix = validation.get("confusion_matrix", [])
    if matrix and class_labels:
        frame = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
        st.table(frame)
    else:
        st.caption("暂无混淆矩阵。")

    history = metrics_payload.get("epoch_history", [])
    if history:
        history_frame = pd.DataFrame(history).set_index("epoch")
        st.markdown("**损失曲线**")
        st.line_chart(history_frame[["train_loss", "validation_loss"]])
    st.caption(
        f"运行耗时 {float(runtime.get('elapsed_seconds', 0.0)):.2f} 秒；"
        f"CUDA {'已实际使用' if runtime.get('cuda_used') else '未用于本次后端'}。"
    )


def _render_diagnosis(payload: dict[str, Any] | None) -> None:
    if payload is None:
        st.caption("尚未生成结构化评估诊断。")
        return
    st.markdown("**EvaluationGovernorAgent 诊断**")
    if "advice" in payload and "facts" in payload:
        facts = payload["facts"]
        advice = payload["advice"]
        llm_call = payload.get("llm_call", {})
        score = float(facts.get("validation_macro_f1", 0.0))
        balanced = float(facts.get("validation_balanced_accuracy", 0.0))
        columns = st.columns(4)
        columns[0].metric("Macro-F1", f"{score:.4f}")
        columns[1].metric("Balanced Accuracy", f"{balanced:.4f}")
        columns[2].metric(
            "最弱阶段 / F1",
            (
                f"{facts.get('weakest_class', '-')} / "
                f"{float(facts.get('weakest_class_f1', 0.0)):.4f}"
            ),
        )
        columns[3].metric("建议动作", advice.get("recommended_action", "-"))
        st.write(advice.get("overall_conclusion", "诊断未给出总体结论。"))
        if llm_call.get("used_fallback"):
            st.warning(
                "本次 LLM 调用未成功，页面展示的是确定性规则诊断。"
                f"原因：{llm_call.get('fallback_reason', '未知')}"
            )
        else:
            st.success(
                f"真实 LLM 诊断：{llm_call.get('provider', '-')}/"
                f"{llm_call.get('model', '-')}，耗时 {llm_call.get('latency_ms', 0)} ms。"
            )
        st.markdown("**诊断发现**")
        for finding in advice.get("findings", []):
            message = f"{finding.get('title', '-')}：{finding.get('detail', '-')}"
            severity = finding.get("severity")
            if severity == "critical":
                st.error(message)
            elif severity == "warning":
                st.warning(message)
            else:
                st.info(message)
        recommendations = advice.get("recommendations", [])
        if recommendations:
            st.markdown("**下一 revision 建议**")
            st.table(
                pd.DataFrame(
                    [
                        {
                            "优先级": item.get("priority"),
                            "目标": item.get("target"),
                            "建议": item.get("suggestion"),
                            "理由": item.get("rationale"),
                        }
                        for item in recommendations
                    ]
                )
            )
        st.caption(
            f"诊断只使用 {facts.get('basis_split', 'validation')}；"
            "Final test 未运行，也未进入 LLM 上下文。所有建议都需要用户审批。"
        )
        return

    metrics = payload.get("metrics", {})
    score = float(metrics.get("macro_f1", 0.0))
    balanced = float(metrics.get("balanced_accuracy", 0.0))
    st.write(
        f"本次判断只基于 validation：Macro-F1={score:.4f}，"
        f"Balanced Accuracy={balanced:.4f}。Final test 未运行，也未参与任何选择。"
    )
    per_class = metrics.get("per_class", {})
    if per_class:
        weakest = min(per_class.items(), key=lambda item: float(item[1].get("f1-score", 0.0)))
        st.info(
            f"当前最弱阶段是“{weakest[0]}”，F1={float(weakest[1].get('f1-score', 0.0)):.4f}。"
            "下一轮应优先检查该阶段的样本量和相邻阶段混淆。"
        )


def _render_decision(payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    decision = payload.get("decision", payload)
    st.markdown("**已归档决策**")
    st.success(f"{decision.get('action', '-')}：{decision.get('reason', '无说明')}")


def render_evaluation_tab(
    client: ToolWearApiClient,
    state: dict[str, Any],
    actions: StateActions,
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> None:
    """展示真实训练结果，并驱动评估、停止/换方案/完整训练决策。"""

    st.subheader("评估、诊断与下一步决策")
    run = latest_run(runs)
    if run is None or run.get("status") != "succeeded":
        st.info("完成一次真实小样本训练后，这里会出现指标、分类报告和决策入口。")
        return

    metrics_ref = _find_artifact(
        artifacts,
        run_id=run["run_id"],
        description="train/validation 指标",
    )
    metrics_payload = _load_json_evidence(client, metrics_ref)
    if metrics_payload:
        _render_metrics(metrics_payload)
    else:
        summary = run.get("result_summary", {})
        columns = st.columns(2)
        columns[0].metric("Validation Macro-F1", f"{float(summary.get('validation_macro_f1', 0.0)):.4f}")
        columns[1].metric(
            "Balanced Accuracy",
            f"{float(summary.get('validation_balanced_accuracy', 0.0)):.4f}",
        )

    tsne_ref = _find_artifact(
        artifacts,
        run_id=run["run_id"],
        description="仅使用 validation 的 t-SNE 特征分布",
    )
    if tsne_ref is not None:
        try:
            tsne_image = client.artifact_bytes(str(tsne_ref["evidence_id"]))
        except ToolApiError as exc:
            render_api_error(exc)
        else:
            st.markdown("#### Validation t-SNE")
            st.image(tsne_image, width="stretch")
            st.caption("该图只使用 validation 特征生成，未读取 final test 数据。")

    if actions.evaluate:
        if st.button(
            "调用 LLM 生成结构化评估诊断",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        ):
            try:
                with st.status(
                    "正在汇总 validation 事实并请求 EvaluationGovernorAgent…",
                    expanded=True,
                ) as status:
                    client.evaluate(
                        state["experiment_id"],
                        rationale="仅使用 train/validation 事实生成诊断，保持 final test 隔离。",
                        force_refresh=False,
                        idempotency_key=operation_key(f"{state['experiment_id']}-evaluate"),
                    )
                    status.update(label="评估诊断与 LLM 调用证据已归档。", state="complete")
            except ToolApiError as exc:
                render_api_error(exc)
            else:
                set_flash("结构化评估已归档；可查看 LLM 状态、诊断建议并人工确认下一步。")
                st.rerun()

    diagnosis_ref = _find_artifact(
        artifacts,
        run_id=run["run_id"],
        description="EvaluationGovernorAgent 的结构化 LLM 诊断",
    )
    if diagnosis_ref is None:
        diagnosis_ref = _find_artifact(
            artifacts,
            run_id=run["run_id"],
            description="仅基于 validation 的结构化评估摘要",
        )
    diagnosis = _load_json_evidence(client, diagnosis_ref)
    _render_diagnosis(diagnosis)
    if (
        diagnosis
        and diagnosis.get("llm_call", {}).get("used_fallback")
        and state.get("state") == "DECIDING"
    ):
        if st.button(
            "只重试 LLM 诊断",
            icon=":material/refresh:",
            width="stretch",
        ):
            try:
                with st.status("正在重试千问；训练结果和旧诊断证据不会被覆盖…") as status:
                    client.evaluate(
                        state["experiment_id"],
                        rationale="保留既有训练和失败证据，仅重试结构化 LLM 诊断。",
                        force_refresh=True,
                        idempotency_key=operation_key(
                            f"{state['experiment_id']}-retry-evaluation"
                        ),
                    )
                    status.update(label="新的诊断版本已归档。", state="complete")
            except ToolApiError as exc:
                render_api_error(exc)
            else:
                set_flash("LLM 诊断已重试；历史失败版本仍保留在证据时间线中。")
                st.rerun()

    decision_ref = _find_artifact(
        artifacts,
        run_id=run["run_id"],
        description="validation 指标、用户选择和预算共同形成的决策",
    )
    decision = _load_json_evidence(client, decision_ref)
    _render_decision(decision)

    if actions.decide:
        st.divider()
        st.subheader("人工确认下一步")
        decision_action = st.segmented_control(
            "决策",
            options=["auto", "approve_full", "change_pipeline", "stop"],
            default="auto",
            format_func=lambda value: {
                "auto": "采用诊断建议",
                "approve_full": "申请完整训练",
                "change_pipeline": "返回更换方案",
                "stop": "停止本实验",
            }[value],
            key=f"decision-{state['experiment_id']}-{state['revision']}",
        )
        rationale = st.text_area(
            "决策说明",
            value="依据 validation 指标、类别混淆和剩余训练预算决定下一步。",
            height=76,
        )
        if st.button(
            "确认并归档决策",
            type="primary",
            icon=":material/account_balance:",
            width="stretch",
        ):
            try:
                client.decide(
                    state["experiment_id"],
                    action=str(decision_action),
                    rationale=rationale,
                    idempotency_key=operation_key(f"{state['experiment_id']}-decision"),
                )
            except ToolApiError as exc:
                render_api_error(exc)
            else:
                set_flash("决策已写入状态事件和证据索引。")
                st.rerun()

    if actions.generate_report:
        st.divider()
        if st.button(
            "生成 / 恢复 Markdown 实验报告",
            icon=":material/description:",
            width="stretch",
        ):
            try:
                client.action(
                    state["experiment_id"],
                    "report",
                    rationale="汇总实验目标、Pipeline、validation 指标、决策和证据索引。",
                    idempotency_key=operation_key(f"{state['experiment_id']}-report"),
                )
            except ToolApiError as exc:
                render_api_error(exc)
            else:
                set_flash("Markdown 实验报告已经生成或从既有证据恢复。")
                st.rerun()
