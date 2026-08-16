"""AgentTeams 协作记录生成器。

本模块对应 P0 第 10 步：把已经完成的 P0 流程整理成可追溯的协作记录。
当前先生成工程可落地的 Trace JSON 和 Markdown，后续接入官方 AgentTeams SDK 时，
可以把这里的步骤结构映射到框架的真实运行事件。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from toolwear_agent.agentteams.identity import build_core_agent_identities
from toolwear_agent.agentteams.reporting import find_latest_decided_run
from toolwear_agent.common.config import Settings


@dataclass(frozen=True)
class TraceStep:
    """单个 Agent 协作步骤。"""

    step_no: int
    step_name: str
    agent_name: str
    agent_role: str
    status: str
    input_files: list[str]
    output_files: list[str]
    log_files: list[str]
    framework_mapping: list[str]
    contribution: str


@dataclass(frozen=True)
class AgentTeamsTrace:
    """P0 阶段的 AgentTeams 协作记录。"""

    trace_id: str
    run_id: str
    created_at: str
    status: str
    framework_note: str
    core_agents: list[str]
    steps: list[TraceStep]


@dataclass(frozen=True)
class TraceResult:
    """Trace 文件生成结果。"""

    trace_id: str
    trace_json: str
    trace_report: str
    trace_log: str


def _now_shanghai() -> str:
    """返回上海时间字符串。"""

    shanghai_timezone = timezone(timedelta(hours=8))
    return datetime.now(shanghai_timezone).isoformat(timespec="seconds")


def _status_for_outputs(paths: list[Path]) -> str:
    """根据输出文件是否存在判断步骤状态。"""

    return "completed" if all(path.exists() for path in paths) else "planned_or_missing_evidence"


def _strings(paths: list[Path]) -> list[str]:
    """把 Path 列表转换为字符串列表。"""

    return [str(path) for path in paths]


def build_p0_trace(ai_infra_root: Path, run_dir: Path, run_id: str) -> AgentTeamsTrace:
    """构建 P0 的 10 步 AgentTeams 协作记录。"""

    reports = ai_infra_root / "reports"
    logs = ai_infra_root / "logs"
    processed = ai_infra_root / "datasets" / "processed" / "phm2010"
    candidates = ai_infra_root / "experiments" / "candidates"
    trace_steps = [
        (
            "PHM2010 C1 数据登记与体检",
            "DataStewardAgent",
            "负责登记原始数据、检查信号文件和磨损标签是否完整。",
            [],
            [reports / "phm2010_c1_data_profile.md", ai_infra_root / "datasets" / "manifests" / "phm2010_c1_inventory.json"],
            [logs / "phm2010_c1_data_profile.log"],
            ["角色编排", "任务输入"],
            "确认 C1 数据可用于后续标签生成和训练。",
        ),
        (
            "VB max 四阶段标签生成",
            "DataStewardAgent",
            "负责把三刀刃 VB 最大值转换成初期、正常、剧烈、失效四阶段标签。",
            [ai_infra_root / "datasets" / "raw" / "phm2010" / "c1" / "c1_wear.csv"],
            [processed / "phm2010_c1_stage_labels.csv", reports / "phm2010_c1_stage_labels.md"],
            [logs / "phm2010_c1_stage_labels.log"],
            ["任务拆解", "上下文传递"],
            "把连续磨损值变成分类训练所需的监督标签。",
        ),
        (
            "候选算法方案生成",
            "AlgorithmArchitectAgent",
            "负责生成互相兼容的候选算法路线，并说明理由、风险和成本。",
            [processed / "phm2010_c1_stage_labels.csv"],
            [candidates / "phm2010_c1_candidate_plans.json", reports / "phm2010_c1_candidate_plans.md"],
            [logs / "phm2010_c1_candidate_plans.log"],
            ["角色编排", "任务拆解"],
            "为用户选择算法模块提供结构化备选方案。",
        ),
        (
            "页面展示与用户确认",
            "ExperimentManagerAgent",
            "负责保存用户确认的候选方案，形成训练前的人机协同证据。",
            [reports / "phm2010_c1_candidate_plans.md"],
            [reports / "phm2010_c1_selected_plan.md"],
            [logs / "phm2010_c1_selected_plan.log"],
            ["审批与回滚", "上下文传递"],
            "保证训练方案不是黑箱自动选择，而是经过用户确认。",
        ),
        (
            "窗口样本构建",
            "DataStewardAgent",
            "负责先按刀次划分 train/val/test，再在刀次内部切窗口，避免数据泄露。",
            [processed / "phm2010_c1_stage_labels.csv"],
            [processed / "phm2010_c1_window_manifest.csv", reports / "phm2010_c1_window_split_report.md"],
            [logs / "phm2010_c1_window_split.log"],
            ["任务拆解", "状态追踪"],
            "把 315 个刀次扩展为可训练的窗口样本，并保留切分证据。",
        ),
        (
            "小样本训练",
            "CodeTrainingEngineerAgent",
            "负责按确认方案抽取 20% 窗口样本训练模型，并保存代码快照。",
            [processed / "phm2010_c1_window_manifest.csv", reports / "phm2010_c1_selected_plan.md"],
            [run_dir / "metrics_summary.json", run_dir / "code_snapshot", reports / "phm2010_c1_mini_train_report.md"],
            [logs / "phm2010_c1_mini_train.log"],
            ["工具调用", "执行证据沉淀"],
            "用低成本训练先验证算法路线是否能跑通。",
        ),
        (
            "指标与图表输出",
            "EvaluationGovernorAgent",
            "负责输出分类报告、混淆矩阵、t-SNE 和可视化 manifest。",
            [run_dir / "metrics_summary.json", run_dir / "feature_table.csv"],
            [run_dir / "visual_report_manifest.json", reports / "phm2010_c1_visual_report.md"],
            [logs / "phm2010_c1_visual_report.log"],
            ["状态追踪", "执行证据沉淀"],
            "把训练结果转换为可解释、可展示的图表证据。",
        ),
        (
            "Agent 结构化诊断",
            "EvaluationGovernorAgent",
            "负责解释指标、识别风险，并输出下一步建议。",
            [run_dir / "metrics_summary.json", run_dir / "visual_report_manifest.json"],
            [run_dir / "agent_diagnosis.json", reports / "phm2010_c1_agent_diagnosis.md"],
            [logs / "phm2010_c1_agent_diagnosis.log"],
            ["协同执行", "经验沉淀"],
            "让系统能说明训练结果为什么好或不好。",
        ),
        (
            "参数调整或停止决策",
            "EvaluationGovernorAgent",
            "负责把诊断结论转成继续、调参、换模型或停止条件。",
            [run_dir / "agent_diagnosis.json"],
            [run_dir / "agent_decision.json", reports / "phm2010_c1_agent_decision.md"],
            [logs / "phm2010_c1_agent_decision.log"],
            ["审批与回滚", "状态追踪"],
            "让多轮训练不会无限循环，形成可控闭环。",
        ),
        (
            "报告与证据归档",
            "ReportMemoryCuratorAgent",
            "负责汇总总报告、Trace、日志、配置和证据索引。",
            [reports / "phm2010_c1_p0_experiment_report.md"],
            [
                run_dir / "p0_report_manifest.json",
                reports / "phm2010_c1_p0_experiment_report.md",
                ai_infra_root / "traces" / "phm2010_c1_agentteams_trace.json",
                reports / "phm2010_c1_agentteams_trace.md",
            ],
            [logs / "phm2010_c1_p0_experiment_report.log"],
            ["执行证据沉淀", "状态追踪"],
            "把 P0 闭环整理成可提交、可复盘的参赛证据链。",
        ),
    ]

    steps = [
        TraceStep(
            step_no=index,
            step_name=item[0],
            agent_name=item[1],
            agent_role=item[2],
            status="completed" if item[1] == "ReportMemoryCuratorAgent" else _status_for_outputs(item[4]),
            input_files=_strings(item[3]),
            output_files=_strings(item[4]),
            log_files=_strings(item[5]),
            framework_mapping=item[6],
            contribution=item[7],
        )
        for index, item in enumerate(trace_steps, start=1)
    ]
    return AgentTeamsTrace(
        trace_id=f"{run_id}_agentteams_trace",
        run_id=run_id,
        created_at=_now_shanghai(),
        status="completed",
        framework_note="当前 Trace 是工程证据层，后续可映射到官方 AgentTeams SDK 的真实运行事件。",
        core_agents=[identity.agent_name for identity in build_core_agent_identities()],
        steps=steps,
    )


def render_trace_markdown(trace: AgentTeamsTrace) -> str:
    """渲染 AgentTeams 协作记录 Markdown。"""

    lines = [
        "# PHM2010 C1 AgentTeams 协作记录",
        "",
        "## 1. 说明",
        "",
        trace.framework_note,
        "",
        "本记录用于说明 6 个核心 Agent 如何通过 10 个流程步骤完成角色编排、任务拆解、上下文传递、协同执行、状态追踪和执行证据沉淀。",
        "",
        "## 2. 六个核心 Agent",
        "",
    ]
    lines.extend(f"- {agent_name}" for agent_name in trace.core_agents)
    lines.extend(
        [
            "",
            "## 3. 协作步骤",
            "",
        ]
    )
    for step in trace.steps:
        lines.extend(
            [
                f"### {step.step_no}. {step.step_name}",
                "",
                f"- Agent：{step.agent_name}",
                f"- 角色：{step.agent_role}",
                f"- 状态：{step.status}",
                f"- AgentTeams 能力映射：{', '.join(step.framework_mapping)}",
                f"- 本步贡献：{step.contribution}",
                "- 输入文件：" if step.input_files else "- 输入文件：无",
            ]
        )
        lines.extend(f"  - `{path}`" for path in step.input_files)
        lines.append("- 输出文件：")
        lines.extend(f"  - `{path}`" for path in step.output_files)
        lines.append("- 日志文件：")
        lines.extend(f"  - `{path}`" for path in step.log_files)
        lines.append("")
    lines.extend(
        [
            "## 3. 边界",
            "",
            "当前记录证明 P0 工程已经具备 AgentTeams 风格的多 Agent 协作设计和证据链。",
            "它还不是官方 AgentTeams SDK 自动采集的运行 Trace，后续接入 SDK 后可替换或增强本记录。",
            "",
        ]
    )
    return "\n".join(lines)


def run_c1_agentteams_trace(settings: Settings, run_dir: Path | None = None) -> TraceResult:
    """生成 C1 P0 AgentTeams Trace 文件。"""

    selected_run_dir = run_dir or find_latest_decided_run(settings.experiment_root)
    run_id = selected_run_dir.name
    trace = build_p0_trace(settings.ai_infra_root, selected_run_dir, run_id)
    trace_root = settings.ai_infra_root / "traces"
    trace_json = trace_root / "phm2010_c1_agentteams_trace.json"
    trace_report = settings.ai_infra_root / "reports" / "phm2010_c1_agentteams_trace.md"
    trace_log = settings.log_root / "phm2010_c1_agentteams_trace.log"

    trace_root.mkdir(parents=True, exist_ok=True)
    trace_json.write_text(json.dumps(asdict(trace), ensure_ascii=False, indent=2), encoding="utf-8")
    trace_report.parent.mkdir(parents=True, exist_ok=True)
    trace_report.write_text(render_trace_markdown(trace), encoding="utf-8")
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    trace_log.write_text(
        "\n".join(
            [
                "PHM2010 C1 AgentTeams 协作记录生成日志",
                f"Trace ID: {trace.trace_id}",
                f"运行编号: {trace.run_id}",
                f"协作步骤数量: {len(trace.steps)}",
                f"Trace JSON: {trace_json}",
                f"Trace Markdown: {trace_report}",
            ]
        ),
        encoding="utf-8",
    )
    return TraceResult(str(trace.trace_id), str(trace_json), str(trace_report), str(trace_log))
