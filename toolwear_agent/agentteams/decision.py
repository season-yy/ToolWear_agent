"""规则型决策 Agent。

本模块对应 P0 第 8 步：在诊断 Agent 已经给出风险和建议后，
进一步输出“继续、调参、切换模型或停止”的结构化执行决策。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from toolwear_agent.agentteams.diagnosis import find_latest_visualized_run, load_json
from toolwear_agent.common.config import Settings


@dataclass(frozen=True)
class DecisionAction:
    """单个可执行动作。"""

    action_id: str
    action_type: str
    priority: str
    title: str
    detail: str
    reason: str


@dataclass(frozen=True)
class AgentDecision:
    """第 8 步结构化决策结果。"""

    decision_id: str
    run_id: str
    created_at: str
    agent_name: str
    should_continue_current_plan: bool
    should_stop_current_plan: bool
    should_adjust_window_params: bool
    should_adjust_sample_fraction: bool
    should_start_cnn_baseline: bool
    should_start_cross_cutter_validation: bool
    overall_decision: str
    actions: list[DecisionAction]
    parameter_suggestions: dict[str, object]
    stop_conditions: list[str]
    source_files: dict[str, str]


def _now_shanghai() -> str:
    """返回上海时间。"""

    shanghai_timezone = timezone(timedelta(hours=8))
    return datetime.now(shanghai_timezone).isoformat(timespec="seconds")


def build_parameter_suggestions(train_config: dict[str, object], metrics_summary: dict[str, object]) -> dict[str, object]:
    """生成参数调整建议。

    当前指标已经很好，因此不建议盲目改窗口参数；如果要进入完整训练，
    优先把 sample_fraction 从 20% 提到 100%，而不是先动滑窗。
    """

    return {
        "window_size": {
            "current": train_config["window_size"],
            "suggested": train_config["window_size"],
            "should_change_now": False,
            "reason": "当前 C1 内部验证已经达到高指标，窗口长度不是优先风险点。",
        },
        "overlap_ratio": {
            "current": train_config["overlap_ratio"],
            "suggested": train_config["overlap_ratio"],
            "should_change_now": False,
            "reason": "0.5 重叠率能兼顾样本数量和相邻窗口差异，暂不建议盲目提高重叠率。",
        },
        "max_windows_per_cut": {
            "current": train_config["max_windows_per_cut"],
            "suggested": train_config["max_windows_per_cut"],
            "should_change_now": False,
            "reason": "当前全量窗口约 10080 条，已满足 P0 小范围训练和后续深度学习起步需求。",
        },
        "sample_fraction": {
            "current": metrics_summary["sample_fraction"],
            "suggested": 1.0,
            "should_change_now": True,
            "reason": "如果要形成更完整的 C1 内部训练结论，下一轮可使用 100% 窗口样本完整训练。",
        },
    }


def build_decision_actions(agent_diagnosis: dict[str, object]) -> list[DecisionAction]:
    """根据诊断结果生成可执行动作清单。"""

    decision = agent_diagnosis["decision"]
    continue_current = bool(decision["continue_current_plan"])
    if not continue_current:
        return [
            DecisionAction(
                action_id="stop_and_debug_current_plan",
                action_type="stop",
                priority="P0",
                title="暂停当前方案并排查数据与参数",
                detail="当前指标不足以支持继续展示，应先检查窗口参数、标签边界、特征提取和数据泄露。",
                reason=str(decision["reason"]),
            )
        ]

    return [
        DecisionAction(
            action_id="keep_random_forest_baseline",
            action_type="continue",
            priority="P0",
            title="保留 RandomForest 作为稳定基线",
            detail="继续保留统计特征 + RandomForest，不需要因为指标高就删除或替换它。",
            reason="当前方案训练快、解释性强、C1 内部 cut 级别验证结果好，适合作为后续模型的对照基线。",
        ),
        DecisionAction(
            action_id="run_full_window_training",
            action_type="adjust_sample_fraction",
            priority="P0",
            title="下一轮可把小范围训练提升为全量窗口训练",
            detail="把 sample_fraction 从 0.2 提升到 1.0，验证完整窗口样本下指标是否稳定。",
            reason="当前只是 20% 小范围训练，全量训练能给第 9 步实验报告提供更完整证据。",
        ),
        DecisionAction(
            action_id="start_cross_cutter_validation",
            action_type="cross_validation",
            priority="P1",
            title="启动跨刀具验证设计",
            detail="优先准备 C1 -> C4、C1 -> C6、C1+C4 -> C6 等验证组合。",
            reason="诊断结果已经明确指出 C1 内部验证不等价于跨刀具泛化。",
        ),
        DecisionAction(
            action_id="start_cnn_comparison",
            action_type="switch_or_compare_model",
            priority="P1",
            title="启动 CNN 对照方案",
            detail="在复用同一窗口 manifest 和 cut split 的前提下，实现一个轻量 1D CNN 对照模型。",
            reason="CNN 可以输出真实 loss 曲线，并验证深度模型是否比统计特征基线更有价值。",
        ),
    ]


def build_stop_conditions() -> list[str]:
    """定义后续应该停止当前方案的条件。"""

    return [
        "跨刀具验证 Macro-F1 连续两轮低于 0.60，且调参后无明显提升。",
        "t-SNE 显示不同阶段严重混叠，同时混淆矩阵出现系统性相邻阶段误判。",
        "全量窗口训练相比 20% 小范围训练明显退化，需要先排查窗口抽样或标签边界。",
        "深度模型和传统基线都无法稳定超过随机或多数类基线时，应暂停模型堆叠，回到数据和标签质量检查。",
    ]


def build_agent_decision(run_dir: Path) -> AgentDecision:
    """从运行目录构建第 8 步决策。"""

    agent_diagnosis_file = run_dir / "agent_diagnosis.json"
    metrics_summary_file = run_dir / "metrics_summary.json"
    train_config_file = run_dir / "train_config.json"
    agent_diagnosis = load_json(agent_diagnosis_file)
    metrics_summary = load_json(metrics_summary_file)
    train_config = load_json(train_config_file)
    actions = build_decision_actions(agent_diagnosis)
    continue_current = bool(agent_diagnosis["decision"]["continue_current_plan"])

    return AgentDecision(
        decision_id=f"{agent_diagnosis['run_id']}_decision",
        run_id=str(agent_diagnosis["run_id"]),
        created_at=_now_shanghai(),
        agent_name="ExperimentDecisionAgent",
        should_continue_current_plan=continue_current,
        should_stop_current_plan=not continue_current,
        should_adjust_window_params=False,
        should_adjust_sample_fraction=continue_current,
        should_start_cnn_baseline=continue_current,
        should_start_cross_cutter_validation=continue_current,
        overall_decision=(
            "不停止当前 RandomForest 基线；下一步优先做全量窗口训练、跨刀具验证和 CNN 对照。"
            if continue_current
            else "暂停当前方案，先排查数据、标签和窗口参数。"
        ),
        actions=actions,
        parameter_suggestions=build_parameter_suggestions(train_config, metrics_summary),
        stop_conditions=build_stop_conditions(),
        source_files={
            "agent_diagnosis": str(agent_diagnosis_file),
            "metrics_summary": str(metrics_summary_file),
            "train_config": str(train_config_file),
            "window_manifest": str(train_config.get("window_manifest_file", "")),
        },
    )


def write_decision_json(decision: AgentDecision, output_file: Path) -> Path:
    """写出结构化决策 JSON。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(asdict(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def render_decision_markdown(decision: AgentDecision) -> str:
    """渲染决策 Markdown 报告。"""

    lines = [
        "# PHM2010 C1 Agent 参数调整与停止决策报告",
        "",
        "## 1. 决策结论",
        "",
        decision.overall_decision,
        "",
        "## 2. 决策开关",
        "",
        f"- 是否继续当前方案：{decision.should_continue_current_plan}",
        f"- 是否停止当前方案：{decision.should_stop_current_plan}",
        f"- 是否调整窗口参数：{decision.should_adjust_window_params}",
        f"- 是否调整小样本比例：{decision.should_adjust_sample_fraction}",
        f"- 是否启动 CNN 基线：{decision.should_start_cnn_baseline}",
        f"- 是否启动跨刀具验证：{decision.should_start_cross_cutter_validation}",
        "",
        "## 3. 可执行动作",
        "",
    ]
    for action in decision.actions:
        lines.extend(
            [
                f"### {action.title}",
                "",
                f"- 动作 ID：`{action.action_id}`",
                f"- 类型：`{action.action_type}`",
                f"- 优先级：`{action.priority}`",
                f"- 说明：{action.detail}",
                f"- 原因：{action.reason}",
                "",
            ]
        )
    lines.extend(["## 4. 参数建议", ""])
    for name, suggestion in decision.parameter_suggestions.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 当前值：{suggestion['current']}",
                f"- 建议值：{suggestion['suggested']}",
                f"- 是否现在调整：{suggestion['should_change_now']}",
                f"- 原因：{suggestion['reason']}",
                "",
            ]
        )
    lines.extend(["## 5. 停止条件", ""])
    lines.extend(f"{index}. {item}" for index, item in enumerate(decision.stop_conditions, start=1))
    lines.extend(["", "## 6. 证据文件", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in decision.source_files.items())
    lines.append("")
    return "\n".join(lines)


def write_decision_markdown(decision: AgentDecision, output_file: Path) -> Path:
    """写出决策 Markdown 报告。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_decision_markdown(decision), encoding="utf-8")
    return output_file


def run_c1_agent_decision(settings: Settings, run_dir: Path | None = None) -> AgentDecision:
    """执行 C1 第 8 步参数调整或停止决策。"""

    selected_run_dir = run_dir or find_latest_visualized_run(settings.experiment_root)
    decision = build_agent_decision(selected_run_dir)
    decision_json = selected_run_dir / "agent_decision.json"
    decision_report = settings.ai_infra_root / "reports" / "phm2010_c1_agent_decision.md"
    decision_log = settings.log_root / "phm2010_c1_agent_decision.log"

    write_decision_json(decision, decision_json)
    write_decision_markdown(decision, decision_report)
    decision_log.parent.mkdir(parents=True, exist_ok=True)
    decision_log.write_text(
        "\n".join(
            [
                "PHM2010 C1 Agent 参数调整与停止决策日志",
                f"决策编号: {decision.decision_id}",
                f"运行编号: {decision.run_id}",
                f"是否继续当前方案: {decision.should_continue_current_plan}",
                f"是否停止当前方案: {decision.should_stop_current_plan}",
                f"是否调整窗口参数: {decision.should_adjust_window_params}",
                f"是否调整小样本比例: {decision.should_adjust_sample_fraction}",
                f"决策 JSON: {decision_json}",
                f"决策报告: {decision_report}",
            ]
        ),
        encoding="utf-8",
    )
    return decision
