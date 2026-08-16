"""将真实实验状态、validation 诊断和证据渲染为 Markdown。"""

from __future__ import annotations

from collections.abc import Sequence

from toolwear_agent.schemas import (
    DecisionRecord,
    EvaluationDiagnosis,
    EvidenceRef,
    ExperimentState,
    TrainingRunResult,
)
from toolwear_agent.state import RunRecord, StateTransitionEvent


def _metric_table(result: TrainingRunResult) -> list[str]:
    validation = next(
        item for item in result.evaluation.metrics if item.split.value == "validation"
    )
    lines = [
        "| 磨损阶段 | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in result.class_labels:
        values = validation.per_class.get(label, {})
        if not isinstance(values, dict):
            values = {}
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    f"{float(values.get('precision', 0.0)):.4f}",
                    f"{float(values.get('recall', 0.0)):.4f}",
                    f"{float(values.get('f1-score', 0.0)):.4f}",
                    str(int(values.get("support", 0))),
                )
            )
            + " |"
        )
    return lines


def render_experiment_report(
    *,
    state: ExperimentState,
    run: RunRecord,
    result: TrainingRunResult,
    diagnosis: EvaluationDiagnosis | None,
    decision: DecisionRecord | None,
    events: Sequence[StateTransitionEvent],
    evidence_refs: Sequence[EvidenceRef],
    rationale: str,
) -> str:
    """生成可直接展示且保留工程边界的实验报告。"""

    validation = next(
        item for item in result.evaluation.metrics if item.split.value == "validation"
    )
    lines = [
        f"# {state.title}",
        "",
        "## 实验结论",
        "",
        f"- 实验 ID：`{state.experiment_id}`",
        f"- Trace ID：`{state.trace_id}`",
        f"- Run ID：`{run.run_id}`",
        f"- Pipeline：`{run.pipeline_id}`",
        f"- Validation Macro-F1：{validation.macro_f1:.4f}",
        f"- Validation Balanced Accuracy：{validation.balanced_accuracy:.4f}",
        "- Final test：未运行，未参与候选选择、调参、诊断或停止决策",
        "",
        "## 用户目标",
        "",
        state.objective,
        "",
        "## Validation 分类表现",
        "",
        *_metric_table(result),
        "",
    ]
    if diagnosis is not None:
        audit = diagnosis.llm_call
        lines.extend(
            [
                "## EvaluationGovernorAgent 诊断",
                "",
                diagnosis.advice.overall_conclusion,
                "",
                f"- 风险级别：`{diagnosis.advice.risk_level}`",
                f"- 建议动作：`{diagnosis.advice.recommended_action}`",
                f"- 诊断来源：`{audit.provider or '未配置'}/{audit.model or '未配置'}`",
                f"- LLM 状态：`{'规则降级' if audit.used_fallback else '真实调用成功'}`",
                f"- 调用耗时：{audit.latency_ms} ms",
                f"- Token：{audit.total_tokens if audit.total_tokens is not None else 'Provider 未返回'}",
            ]
        )
        if audit.used_fallback:
            lines.append(f"- 降级原因：{audit.fallback_reason}")
        lines.extend(["", "### 诊断发现", ""])
        for finding in diagnosis.advice.findings:
            lines.append(
                f"- **[{finding.severity}] {finding.title}**：{finding.detail}（证据：{finding.evidence}）"
            )
        lines.extend(["", "### 下一轮建议", ""])
        for item in diagnosis.advice.recommendations:
            lines.append(
                f"- **{item.suggestion}** 目标：`{item.target}`；理由：{item.rationale}；"
                f"优先级：`{item.priority}`；需人工审批。"
            )
        lines.append("")
    else:
        lines.extend(["## EvaluationGovernorAgent 诊断", "", "尚未生成结构化诊断。", ""])

    lines.extend(["## 已归档决策", ""])
    if decision is None:
        lines.extend(["尚未由用户确认下一步决策。", ""])
    else:
        lines.extend(
            [
                f"- 动作：`{decision.action.value}`",
                f"- 依据：`{decision.basis_split}`",
                f"- 理由：{decision.reason}",
                "- 高影响动作仍需用户在实验台明确确认。",
                "",
            ]
        )
    lines.extend(
        [
            "## 配置与运行边界",
            "",
            f"- 输入通道：{', '.join(state.preferences.input_channels)}",
            f"- 窗口长度 / 重叠率：{state.preferences.window_length} / {state.preferences.overlap}",
            f"- 小样本比例：{state.preferences.sample_fraction}",
            f"- Train / Validation 窗口：{result.train_sample_count} / {result.validation_sample_count}",
            f"- 实际后端 / 设备：{result.runtime.backend} / {result.runtime.resolved_device}",
            f"- CUDA 实际使用：{'是' if result.runtime.cuda_used else '否'}",
            f"- 小样本预算：{state.budget.completed_mini_runs}/{state.budget.max_mini_runs}",
            "- 当前结论只代表本实验切分上的 validation 表现，不等价于跨刀具、跨工况泛化。",
            "",
            "## 证据索引",
            "",
            "| Evidence ID | 类型 | SHA-256 前 12 位 | 说明 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for evidence in evidence_refs:
        lines.append(
            f"| `{evidence.evidence_id}` | `{evidence.kind.value}` | `{evidence.sha256[:12]}` | "
            f"{evidence.description or '-'} |"
        )
    lines.extend(
        [
            "",
            "## 审计摘要",
            "",
            f"- 状态事件：{len(events)} 条",
            f"- 报告生成前证据：{len(evidence_refs)} 条",
            f"- 报告说明：{rationale}",
            "",
        ]
    )
    return "\n".join(lines)
