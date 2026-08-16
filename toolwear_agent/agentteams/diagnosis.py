"""规则型诊断 Agent。

本模块对应 P0 第 7 步：读取训练指标、图表索引和数据切分证据，
输出结构化诊断和下一步建议。当前先用规则型诊断，保证结果稳定可复现；
后续接入 AgentTeams/LLM 时，可以把这里的 JSON 作为上下文输入。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from toolwear_agent.common.config import Settings


@dataclass(frozen=True)
class DiagnosisFinding:
    """诊断中的单条发现。"""

    level: str
    title: str
    detail: str
    evidence: str


@dataclass(frozen=True)
class DiagnosisDecision:
    """诊断后的执行决策。"""

    continue_current_plan: bool
    recommended_next_action: str
    reason: str


@dataclass(frozen=True)
class AgentDiagnosis:
    """第 7 步结构化诊断结果。"""

    diagnosis_id: str
    run_id: str
    created_at: str
    agent_name: str
    agent_role: str
    overall_conclusion: str
    metric_summary: dict[str, float | int]
    findings: list[DiagnosisFinding]
    recommendations: list[str]
    decision: DiagnosisDecision
    source_files: dict[str, str]


def _now_shanghai() -> str:
    """返回上海时间。"""

    shanghai_timezone = timezone(timedelta(hours=8))
    return datetime.now(shanghai_timezone).isoformat(timespec="seconds")


def load_json(json_file: Path) -> dict[str, object]:
    """读取 JSON 文件。"""

    if not json_file.exists():
        raise FileNotFoundError(f"诊断输入文件不存在: {json_file}")
    return json.loads(json_file.read_text(encoding="utf-8"))


def find_latest_visualized_run(experiment_root: Path) -> Path:
    """查找最新的、已经生成可视化报告的窗口训练目录。"""

    runs = [
        item
        for item in experiment_root.glob("phm2010_c1_window_mini_train_*")
        if item.is_dir()
        and (item / "metrics_summary.json").exists()
        and (item / "visual_report_manifest.json").exists()
        and (item / "metrics.json").exists()
    ]
    if not runs:
        raise FileNotFoundError(f"未找到已可视化的窗口训练目录: {experiment_root}")
    return max(runs, key=lambda item: item.stat().st_mtime)


def build_metric_findings(metrics_summary: dict[str, object]) -> list[DiagnosisFinding]:
    """只根据 train/validation 摘要生成诊断发现。"""

    findings: list[DiagnosisFinding] = []
    validation_macro_f1 = float(metrics_summary["validation_macro_f1"])
    validation_balanced_accuracy = float(metrics_summary["validation_balanced_accuracy"])

    if validation_macro_f1 >= 0.95:
        findings.append(
            DiagnosisFinding(
                level="positive",
                title="当前窗口样本验证效果很高",
                detail="Validation Macro-F1 达到高水平，说明当前方案值得保留为候选，但这不是最终测试结论。",
                evidence=f"validation_macro_f1={validation_macro_f1:.4f}",
            )
        )

    if validation_balanced_accuracy >= 0.95:
        findings.append(
            DiagnosisFinding(
                level="positive",
                title="类别不均衡下的平均召回表现很好",
                detail="Validation Balanced Accuracy 同样很高，说明当前结果不是只偏向多数阶段。",
                evidence=f"validation_balanced_accuracy={validation_balanced_accuracy:.4f}",
            )
        )

    if float(metrics_summary["sample_fraction"]) < 1.0:
        findings.append(
            DiagnosisFinding(
                level="caution",
                title="当前只是 20% 小范围训练",
                detail="本次训练只使用窗口 manifest 中约 20% 的样本，适合快速验证模块，但还不是完整训练结论。",
                evidence=f"sample_count={metrics_summary['sample_count']}, full_window_count={metrics_summary['full_window_count']}",
            )
        )

    return findings


def build_risk_findings(metrics_summary: dict[str, object], visual_manifest: dict[str, object]) -> list[DiagnosisFinding]:
    """生成风险与边界诊断。"""

    findings = [
        DiagnosisFinding(
            level="risk",
            title="C1 内部验证不等价于跨刀具泛化",
            detail="当前 train/val/test 都来自 C1，同一把刀内部的信号分布可能相近，因此不能直接证明 C4/C6 或跨工况下仍然有效。",
            evidence=str(visual_manifest.get("report_file", "")),
        ),
        DiagnosisFinding(
            level="risk",
            title="Validation 高分需要谨慎解读",
            detail="高分可能来自特征区分度确实很强，也可能说明 C1 内部任务较容易。test 仍被隔离，后续还需最终评估和跨刀具验证。",
            evidence=f"validation_macro_f1={metrics_summary['validation_macro_f1']}",
        ),
        DiagnosisFinding(
            level="boundary",
            title="当前 RandomForest 没有 loss 曲线",
            detail="RandomForest 不是按 epoch 训练的神经网络，因此不应伪造 loss 曲线。后续 CNN 方案再输出真实训练/验证 loss。",
            evidence=str(visual_manifest.get("loss_curve_note_file", "")),
        ),
    ]
    return findings


def build_recommendations(metrics_summary: dict[str, object]) -> list[str]:
    """根据当前结果生成下一步建议。"""

    return [
        "保留统计特征 + RandomForest 作为 P0 基线方案，因为它训练快、指标高、解释性强。",
        "下一轮不要只继续追求 C1 内部指标，应优先设计 C1 -> C4、C1 -> C6、C1+C4 -> C6 等跨刀具验证。",
        "进入深度学习方案前，先复用当前窗口 manifest 和 cut 级别 split，保证不同模型之间数据边界一致。",
        "如果继续做 CNN，需要补充真实 epoch loss 曲线、特征嵌入 t-SNE 和训练耗时记录。",
        "报告展示时必须标明：当前结果是 C1 内部 cut 级别验证，不代表跨工况泛化已经完成。",
    ]


def build_decision(metrics_summary: dict[str, object]) -> DiagnosisDecision:
    """只根据 validation 生成是否继续当前方案的决策。"""

    validation_macro_f1 = float(metrics_summary["validation_macro_f1"])
    if validation_macro_f1 >= 0.90:
        return DiagnosisDecision(
            continue_current_plan=True,
            recommended_next_action="保留当前 RandomForest 基线，同时进入跨刀具验证或 CNN 对照方案。",
            reason="当前 C1 内部窗口验证效果足够好，适合作为稳定基线；但泛化风险未验证，不能只停留在当前方案。",
        )
    return DiagnosisDecision(
        continue_current_plan=False,
        recommended_next_action="暂停当前方案，优先检查窗口参数、标签边界和特征提取方式。",
        reason="当前指标不足以支持进入展示阶段，需要先定位训练效果差的原因。",
    )


def build_agent_diagnosis(run_dir: Path) -> AgentDiagnosis:
    """从一个训练运行目录构建结构化诊断。"""

    metrics_summary_file = run_dir / "metrics_summary.json"
    metrics_file = run_dir / "metrics.json"
    visual_manifest_file = run_dir / "visual_report_manifest.json"
    metrics_summary = load_json(metrics_summary_file)
    visual_manifest = load_json(visual_manifest_file)
    metrics = load_json(metrics_file)

    findings = [
        *build_metric_findings(metrics_summary),
        *build_risk_findings(metrics_summary, visual_manifest),
    ]
    recommendations = build_recommendations(metrics_summary)
    decision = build_decision(metrics_summary)

    return AgentDiagnosis(
        diagnosis_id=f"{metrics_summary['run_id']}_diagnosis",
        run_id=str(metrics_summary["run_id"]),
        created_at=_now_shanghai(),
        agent_name="AlgorithmDiagnosisAgent",
        agent_role="刀具磨损监测算法诊断专家，负责解释训练结果、识别风险并给出下一步建议。",
        overall_conclusion=(
            "当前统计特征 + RandomForest 在 PHM2010 C1 窗口样本 cut 级别划分下表现很好，"
            "可以作为 P0 基线证据；但它尚未证明跨刀具、跨工况泛化能力。"
        ),
        metric_summary={
            "full_window_count": int(metrics_summary["full_window_count"]),
            "sample_count": int(metrics_summary["sample_count"]),
            "validation_macro_f1": float(metrics_summary["validation_macro_f1"]),
            "validation_balanced_accuracy": float(metrics_summary["validation_balanced_accuracy"]),
        },
        findings=findings,
        recommendations=recommendations,
        decision=decision,
        source_files={
            "metrics_summary": str(metrics_summary_file),
            "metrics": str(metrics_file),
            "visual_manifest": str(visual_manifest_file),
            "feature_table": str(run_dir / "feature_table.csv"),
            "visual_report": str(visual_manifest.get("report_file", "")),
            "tsne": str(visual_manifest.get("tsne_png", "")),
            "validation_confusion_matrix": str(visual_manifest.get("validation_confusion_matrix_png", "")),
            "final_test_status": str(metrics_summary.get("final_test_status", "not_run")),
            "loss_curve_note": str(visual_manifest.get("loss_curve_note_file", "")),
            "raw_metrics_keys": ", ".join(sorted(metrics.keys())),
        },
    )


def write_diagnosis_json(diagnosis: AgentDiagnosis, output_file: Path) -> Path:
    """写出结构化诊断 JSON。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(asdict(diagnosis), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file


def render_diagnosis_markdown(diagnosis: AgentDiagnosis) -> str:
    """渲染诊断 Markdown 报告。"""

    lines = [
        "# PHM2010 C1 Agent 结构化诊断报告",
        "",
        "## 1. 诊断结论",
        "",
        diagnosis.overall_conclusion,
        "",
        "## 2. 指标摘要",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in diagnosis.metric_summary.items())
    lines.extend(["", "## 3. 关键发现", ""])
    for finding in diagnosis.findings:
        lines.extend(
            [
                f"### {finding.title}",
                "",
                f"- 等级：`{finding.level}`",
                f"- 说明：{finding.detail}",
                f"- 证据：`{finding.evidence}`",
                "",
            ]
        )
    lines.extend(["## 4. 下一步建议", ""])
    lines.extend(f"{index}. {item}" for index, item in enumerate(diagnosis.recommendations, start=1))
    lines.extend(
        [
            "",
            "## 5. 决策输出",
            "",
            f"- 是否继续当前方案：{diagnosis.decision.continue_current_plan}",
            f"- 推荐动作：{diagnosis.decision.recommended_next_action}",
            f"- 原因：{diagnosis.decision.reason}",
            "",
            "## 6. 证据文件",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in diagnosis.source_files.items())
    lines.append("")
    return "\n".join(lines)


def write_diagnosis_markdown(diagnosis: AgentDiagnosis, output_file: Path) -> Path:
    """写出诊断 Markdown 报告。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_diagnosis_markdown(diagnosis), encoding="utf-8")
    return output_file


def run_c1_agent_diagnosis(settings: Settings, run_dir: Path | None = None) -> AgentDiagnosis:
    """执行 C1 第 7 步结构化诊断。"""

    selected_run_dir = run_dir or find_latest_visualized_run(settings.experiment_root)
    diagnosis = build_agent_diagnosis(selected_run_dir)
    diagnosis_json = selected_run_dir / "agent_diagnosis.json"
    diagnosis_report = settings.ai_infra_root / "reports" / "phm2010_c1_agent_diagnosis.md"
    diagnosis_log = settings.log_root / "phm2010_c1_agent_diagnosis.log"

    write_diagnosis_json(diagnosis, diagnosis_json)
    write_diagnosis_markdown(diagnosis, diagnosis_report)
    diagnosis_log.parent.mkdir(parents=True, exist_ok=True)
    diagnosis_log.write_text(
        "\n".join(
            [
                "PHM2010 C1 Agent 结构化诊断日志",
                f"诊断编号: {diagnosis.diagnosis_id}",
                f"运行编号: {diagnosis.run_id}",
                f"诊断 Agent: {diagnosis.agent_name}",
                f"发现数量: {len(diagnosis.findings)}",
                f"是否继续当前方案: {diagnosis.decision.continue_current_plan}",
                f"推荐动作: {diagnosis.decision.recommended_next_action}",
                f"诊断 JSON: {diagnosis_json}",
                f"诊断报告: {diagnosis_report}",
            ]
        ),
        encoding="utf-8",
    )
    return diagnosis
