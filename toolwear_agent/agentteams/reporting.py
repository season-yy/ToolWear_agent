"""P0 实验总报告生成器。

本模块对应 P0 第 9 步：把前面 1-8 步留下的数据体检、标签、候选方案、
训练指标、可视化、诊断和决策汇总成一份可交付的 Markdown 实验报告。
这里不重新训练模型，只整理已经产生的证据。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from toolwear_agent.agentteams.diagnosis import load_json
from toolwear_agent.agentteams.identity import build_core_agent_identities
from toolwear_agent.common.config import Settings


REQUIRED_RUN_FILES = [
    "metrics_summary.json",
    "train_config.json",
    "visual_report_manifest.json",
    "agent_diagnosis.json",
    "agent_decision.json",
]


@dataclass(frozen=True)
class P0ReportResult:
    """第 9 步报告生成结果。"""

    run_id: str
    report_file: str
    manifest_file: str
    log_file: str
    source_files: dict[str, str]


def find_latest_decided_run(experiment_root: Path) -> Path:
    """查找最新的、已经完成第 8 步决策的运行目录。"""

    runs = [
        item
        for item in experiment_root.glob("phm2010_c1_window_mini_train_*")
        if item.is_dir() and all((item / file_name).exists() for file_name in REQUIRED_RUN_FILES)
    ]
    if not runs:
        raise FileNotFoundError(f"未找到已经完成第 8 步决策的运行目录: {experiment_root}")
    return max(runs, key=lambda item: item.stat().st_mtime)


def build_report_sources(ai_infra_root: Path, run_dir: Path) -> dict[str, Path]:
    """整理 P0 总报告需要引用的证据文件。"""

    report_root = ai_infra_root / "reports"
    return {
        "data_profile": report_root / "phm2010_c1_data_profile.md",
        "stage_labels": report_root / "phm2010_c1_stage_labels.md",
        "candidate_plans": report_root / "phm2010_c1_candidate_plans.md",
        "selected_plan": report_root / "phm2010_c1_selected_plan.md",
        "window_split": report_root / "phm2010_c1_window_split_report.md",
        "mini_train": report_root / "phm2010_c1_mini_train_report.md",
        "visual_report": report_root / "phm2010_c1_visual_report.md",
        "agent_diagnosis": report_root / "phm2010_c1_agent_diagnosis.md",
        "agent_decision": report_root / "phm2010_c1_agent_decision.md",
        "metrics_summary": run_dir / "metrics_summary.json",
        "train_config": run_dir / "train_config.json",
        "visual_manifest": run_dir / "visual_report_manifest.json",
        "diagnosis_json": run_dir / "agent_diagnosis.json",
        "decision_json": run_dir / "agent_decision.json",
        "code_snapshot": run_dir / "code_snapshot",
    }


def _format_metric(value: object) -> str:
    """把指标格式化为适合报告阅读的字符串。"""

    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_source_index(source_files: dict[str, Path]) -> list[str]:
    """渲染证据索引。"""

    lines: list[str] = []
    for key, path in source_files.items():
        status = "存在" if path.exists() else "缺失"
        lines.append(f"- {key}: `{path}`（{status}）")
    return lines


def render_p0_experiment_report(
    run_id: str,
    metrics_summary: dict[str, object],
    train_config: dict[str, object],
    diagnosis: dict[str, object],
    decision: dict[str, object],
    visual_manifest: dict[str, object],
    source_files: dict[str, Path],
) -> str:
    """渲染 P0 Markdown 实验总报告正文。"""

    model_family = str(train_config.get("model_family", train_config.get("model_name", "RandomForest")))
    recommendations = diagnosis.get("recommendations", [])
    stop_conditions = decision.get("stop_conditions", [])
    core_agents = build_core_agent_identities()

    lines = [
        "# 刀具磨损监测算法辅助 Agent P0 实验报告",
        "",
        "## 1. 作品简介",
        "",
        (
            "本作品面向刀具铣削过程中的磨损阶段识别，当前 P0 使用 PHM2010 C1 数据，"
            "输入信号包括力、振动和声发射。多 Agent 协同完成数据检查、标签生成、候选算法方案、"
            "小样本训练、指标可视化、结果诊断和下一步决策。"
        ),
        "",
        "后续扩展到自采铣削数据时，可接入主轴电流等额外工业信号，但主轴电流不属于 PHM2010 当前输入。",
        "",
        "当前报告是初赛 P0 可验证 PoC：它已经跑通 C1 内部 cut 级别闭环，"
        "但 C1 内部结果不等价于跨刀具泛化，后续还必须补 C1 -> C4、C1 -> C6 等验证。",
        "",
        "## 2. P0 闭环概览",
        "",
        "1. PHM2010 C1 数据登记与体检。",
        "2. 三刀刃 VB 取最大值，并按 90/130/160 um 生成四阶段标签。",
        "3. 生成多套候选算法方案，并保留用户确认记录。",
        "4. 按 cut 先划分 train/val/test，再生成滑窗样本，避免同一刀次泄露。",
        "5. 使用 20% 窗口样本执行快速训练，并保存训练代码快照。",
        "6. 输出 Macro-F1、Balanced Accuracy、分类报告、混淆矩阵和 t-SNE。",
        "7. 诊断 Agent 给出结构化风险解释和下一步建议。",
        "8. 决策 Agent 给出继续、调参、切换模型或停止条件。",
        "",
        "## 3. 数据、标签与样本",
        "",
        f"- 全量窗口数：{metrics_summary.get('full_window_count')}",
        f"- 小样本窗口数：{metrics_summary.get('sample_count')}",
        f"- 小样本比例：{metrics_summary.get('sample_fraction')}",
        f"- 窗口长度：{train_config.get('window_size')}",
        f"- 重叠率：{train_config.get('overlap_ratio')}",
        f"- 每个刀次最大窗口数：{train_config.get('max_windows_per_cut')}",
        f"- 窗口 manifest：`{train_config.get('window_manifest_file', '')}`",
        "",
        "## 4. AgentTeams 协作映射",
        "",
        "本项目初赛口径固定为 6 个核心 Agent；P0 的 10 个动作是流程步骤，会归属到这 6 个 Agent。",
        "",
    ]
    lines.extend(f"- {identity.agent_name}：{identity.chinese_role}，{identity.responsibility}" for identity in core_agents)
    lines.extend(
        [
            "",
            "P0 流程映射：数据体检、标签和窗口切分归属 DataStewardAgent；候选方案归属 AlgorithmArchitectAgent；用户确认归属 ExperimentManagerAgent；训练归属 CodeTrainingEngineerAgent；可视化、诊断和决策归属 EvaluationGovernorAgent；报告和 Trace 归属 ReportMemoryCuratorAgent。",
            "",
            "## 5. 训练方案与指标",
        ]
    )
    lines.extend(
        [
            "",
            f"- 当前模型：{model_family}",
            f"- 验证集 Macro-F1：{_format_metric(metrics_summary.get('validation_macro_f1'))}",
            f"- 验证集 Balanced Accuracy：{_format_metric(metrics_summary.get('validation_balanced_accuracy'))}",
            f"- 最终测试状态：{metrics_summary.get('final_test_status', 'not_run_pipeline_not_frozen')}",
            "",
            "候选排序、调参和停止判断只使用 validation；test 会在方案与参数冻结后通过独立最终评估读取。",
            "这些验证指标说明当前基线在 C1 内部任务上可作为 P0 展示证据，但不能直接写成已经解决跨刀具、跨工况问题。",
            "",
            "## 6. 可视化证据",
            "",
            f"- t-SNE 图：`{visual_manifest.get('tsne_png', '')}`",
            f"- 验证集混淆矩阵：`{visual_manifest.get('validation_confusion_matrix_png', '')}`",
            f"- 损失曲线说明：`{visual_manifest.get('loss_curve_note_file', '')}`",
            "",
            "## 7. Agent 诊断结论",
            "",
            str(diagnosis.get("overall_conclusion", "")),
            "",
            "### 诊断建议",
            "",
        ]
    )
    lines.extend(f"{index}. {item}" for index, item in enumerate(recommendations, start=1))
    lines.extend(
        [
            "",
            "## 8. Agent 决策输出",
            "",
            str(decision.get("overall_decision", "")),
            "",
            "### 停止条件",
            "",
        ]
    )
    lines.extend(f"{index}. {item}" for index, item in enumerate(stop_conditions, start=1))
    lines.extend(
        [
            "",
            "## 9. 风险与边界",
            "",
            "- 当前只完成 PHM2010 C1 内部 cut 级别验证，不等价于跨刀具泛化。",
            "- 当前候选阶段未读取最终测试集，不能把 validation 指标写成 final test 结论。",
            "- 当前 RandomForest 基线没有 epoch loss 曲线，因此报告中只保存损失曲线说明，不伪造神经网络训练曲线。",
            "- 当前小样本训练用于快速筛选方案，后续完整结论需要全量窗口训练和跨刀具验证支撑。",
            "",
            "## 10. 下一步工作",
            "",
            "1. 使用 100% 窗口样本运行完整 C1 训练，验证指标是否稳定。",
            "2. 准备 C1 -> C4、C1 -> C6、C1+C4 -> C6 等跨刀具验证。",
            "3. 在同一窗口 manifest 约束下实现轻量 1D CNN，对比传统基线和深度模型。",
            "4. 第 10 步补齐 AgentTeams 协作记录、Trace、日志、配置和证据索引。",
            "",
            "## 11. 证据索引",
            "",
        ]
    )
    lines.extend(_render_source_index(source_files))
    lines.append("")
    return "\n".join(lines)


def run_c1_p0_report(settings: Settings, run_dir: Path | None = None) -> P0ReportResult:
    """执行第 9 步：生成 C1 P0 Markdown 实验总报告。"""

    selected_run_dir = run_dir or find_latest_decided_run(settings.experiment_root)
    metrics_summary = load_json(selected_run_dir / "metrics_summary.json")
    train_config = load_json(selected_run_dir / "train_config.json")
    visual_manifest = load_json(selected_run_dir / "visual_report_manifest.json")
    diagnosis = load_json(selected_run_dir / "agent_diagnosis.json")
    decision = load_json(selected_run_dir / "agent_decision.json")
    source_files = build_report_sources(settings.ai_infra_root, selected_run_dir)

    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_p0_experiment_report.md"
    manifest_file = selected_run_dir / "p0_report_manifest.json"
    log_file = settings.log_root / "phm2010_c1_p0_experiment_report.log"
    report_text = render_p0_experiment_report(
        run_id=str(metrics_summary["run_id"]),
        metrics_summary=metrics_summary,
        train_config=train_config,
        diagnosis=diagnosis,
        decision=decision,
        visual_manifest=visual_manifest,
        source_files=source_files,
    )

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_text, encoding="utf-8")
    result = P0ReportResult(
        run_id=str(metrics_summary["run_id"]),
        report_file=str(report_file),
        manifest_file=str(manifest_file),
        log_file=str(log_file),
        source_files={key: str(value) for key, value in source_files.items()},
    )
    manifest_file.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 P0 Markdown 实验总报告生成日志",
                f"运行编号: {result.run_id}",
                f"总报告: {result.report_file}",
                f"报告 manifest: {result.manifest_file}",
                f"证据文件数量: {len(result.source_files)}",
            ]
        ),
        encoding="utf-8",
    )
    return result
