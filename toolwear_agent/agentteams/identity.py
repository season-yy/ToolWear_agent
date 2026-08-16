"""六个核心 Agent 与 Skill 清单。

本模块是初赛材料的统一口径来源。P0 有 10 个流程步骤，但核心 Agent
固定为 6 个；每个流程步骤都必须归属到这 6 个 Agent 之一。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from toolwear_agent.common.config import Settings


@dataclass(frozen=True)
class AgentIdentity:
    """Agent 身份定义。"""

    agent_name: str
    chinese_role: str
    responsibility: str
    inputs: list[str]
    outputs: list[str]
    boundaries: list[str]
    prompt_summary: str


@dataclass(frozen=True)
class SkillManifestItem:
    """P0 Skill 清单项。"""

    skill_name: str
    version: str
    owner_agent: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    safety_notes: list[str]


def build_core_agent_identities() -> list[AgentIdentity]:
    """返回六个核心 Agent 身份清单。"""

    return [
        AgentIdentity(
            agent_name="ExperimentManagerAgent",
            chinese_role="实验主控 Agent",
            responsibility="维护实验状态、拆解任务、协调各专业 Agent，并管理用户审批点。",
            inputs=["用户目标", "实验状态", "Agent 输出", "审批结果"],
            outputs=["任务计划", "Agent 路由", "等待项", "最终汇总"],
            boundaries=["不能绕过用户选择", "不能伪造成功", "不能直接修改原始数据"],
            prompt_summary="你是刀具磨损算法实验主控，负责状态、任务和审批，不替代专业 Agent 做细节判断。",
        ),
        AgentIdentity(
            agent_name="DataStewardAgent",
            chinese_role="数据治理 Agent",
            responsibility="负责数据体检、VB 标签、窗口样本切分和防泄漏检查。",
            inputs=["Dataset Manifest", "原始文件只读路径", "标签策略", "切分策略"],
            outputs=["数据体检报告", "阶段标签", "窗口 manifest", "阻断项和风险项"],
            boundaries=["不原地修改 CSV", "不用测试集拟合预处理参数", "禁止同一 cut 跨集合泄漏"],
            prompt_summary="你只根据数据证据判断，先区分阻断错误、可修复风险和普通提示。",
        ),
        AgentIdentity(
            agent_name="AlgorithmArchitectAgent",
            chinese_role="算法方案 Agent",
            responsibility="根据数据、目标和算力生成兼容候选方案，并说明理由、风险和成本。",
            inputs=["用户目标", "数据报告", "资源预算", "模块目录", "历史案例"],
            outputs=["候选 PipelineSpec", "方案理由", "风险说明", "成本预估"],
            boundaries=["不推荐未注册模块", "不承诺准确率", "最终方案必须由用户确认"],
            prompt_summary="你的推荐必须落到已注册模块和合法参数，每个候选都要说明结构、风险和适用条件。",
        ),
        AgentIdentity(
            agent_name="CodeTrainingEngineerAgent",
            chinese_role="代码训练 Agent",
            responsibility="执行已批准方案的小样本训练，保存代码快照、配置、日志、模型和指标。",
            inputs=["已批准 PipelineSpec", "窗口 manifest", "训练预算", "运行环境"],
            outputs=["代码快照", "训练日志", "模型文件", "指标文件", "训练报告"],
            boundaries=["不执行非白名单命令", "不覆盖最佳模型旧版本", "不访问白名单外路径"],
            prompt_summary="你是受控执行者，优先复用模板和稳定训练函数，失败时返回可定位错误。",
        ),
        AgentIdentity(
            agent_name="EvaluationGovernorAgent",
            chinese_role="评估治理 Agent",
            responsibility="汇总指标和图表，诊断失败原因，并给出调参、换方案或停止建议。",
            inputs=["指标", "曲线", "混淆矩阵", "t-SNE", "日志", "历史基线"],
            outputs=["结构化诊断", "下一步建议", "参数建议", "停止条件"],
            boundaries=["不只凭 t-SNE 下结论", "不突破训练预算", "不把用户不满意等同于模型失败"],
            prompt_summary="先运行确定性评估规则，再基于证据解释结果；证据不足时降低置信度。",
        ),
        AgentIdentity(
            agent_name="ReportMemoryCuratorAgent",
            chinese_role="报告记忆 Agent",
            responsibility="整理报告、Trace、证据索引和实验经验，保存成功与失败案例。",
            inputs=["实验状态", "配置 revision", "指标", "决策", "用户反馈", "产物路径"],
            outputs=["Markdown 报告", "Trace", "证据清单", "经验条目"],
            boundaries=["不修改原始指标", "不删除失败实验", "不把推测写成事实"],
            prompt_summary="你只能基于可索引证据写报告，成功和失败配置都要保留适用条件与限制。",
        ),
    ]


def build_p0_skill_manifest() -> list[SkillManifestItem]:
    """返回当前 P0 已实现或已映射的 Skill 清单。"""

    return [
        SkillManifestItem("DataProfileSkill", "1.0.0", "DataStewardAgent", "登记并体检 PHM2010 C1 数据。", ["C1 原始目录"], ["数据体检报告", "inventory.json"], ["只读原始数据"]),
        SkillManifestItem("StageLabelSkill", "1.0.0", "DataStewardAgent", "按 VB max 和阈值生成四阶段标签。", ["c1_wear.csv", "阈值配置"], ["stage_labels.csv", "标签报告"], ["不改写原始标签文件"]),
        SkillManifestItem("WindowSplitSkill", "1.0.0", "DataStewardAgent", "按 cut 划分集合并生成窗口样本。", ["stage_labels.csv", "C1 信号 CSV"], ["window_manifest.csv", "切分报告"], ["断言同一 cut 不跨集合"]),
        SkillManifestItem("PipelineRecommendSkill", "1.0.0", "AlgorithmArchitectAgent", "生成 2-3 个兼容候选算法方案。", ["数据报告", "任务目标"], ["candidate_plans.json", "候选方案报告"], ["不推荐未实现模块"]),
        SkillManifestItem("HumanSelectionSkill", "1.0.0", "ExperimentManagerAgent", "保存用户确认的候选方案。", ["候选方案", "用户选择"], ["selected_plan.md"], ["用户未确认不得训练"]),
        SkillManifestItem("MiniTrainSkill", "1.0.0", "CodeTrainingEngineerAgent", "执行窗口小样本训练并保存代码快照。", ["selected_plan", "window_manifest"], ["metrics_summary.json", "model.joblib", "code_snapshot"], ["限制在白名单路径内运行"]),
        SkillManifestItem("VisualizationSkill", "1.0.0", "EvaluationGovernorAgent", "生成混淆矩阵、t-SNE、分类报告和图表索引。", ["metrics", "feature_table"], ["figures", "visual_report.md"], ["不伪造 loss 曲线"]),
        SkillManifestItem("DiagnosisSkill", "1.0.0", "EvaluationGovernorAgent", "基于指标和图表输出结构化诊断。", ["metrics_summary", "visual_manifest"], ["agent_diagnosis.json", "诊断报告"], ["明确 C1 内部验证边界"]),
        SkillManifestItem("DecisionSkill", "1.0.0", "EvaluationGovernorAgent", "输出继续、调参、换方案或停止决策。", ["agent_diagnosis"], ["agent_decision.json", "决策报告"], ["不得无限循环调参"]),
        SkillManifestItem("ReportTraceSkill", "1.0.0", "ReportMemoryCuratorAgent", "生成总报告、Trace 和证据索引。", ["全部 P0 证据"], ["p0_report.md", "trace.json", "trace.md"], ["不删除失败证据"]),
    ]


def render_agent_identity_markdown(identities: list[AgentIdentity]) -> str:
    """渲染 Agent Identity Markdown。"""

    lines = ["# PHM2010 C1 Agent Identity 清单", "", "本清单固定项目初赛口径：6 个核心 Agent。", ""]
    for index, identity in enumerate(identities, start=1):
        lines.extend(
            [
                f"## {index}. {identity.agent_name}",
                "",
                f"- 中文角色：{identity.chinese_role}",
                f"- 职责：{identity.responsibility}",
                "- 输入：",
                *[f"  - {item}" for item in identity.inputs],
                "- 输出：",
                *[f"  - {item}" for item in identity.outputs],
                "- 边界：",
                *[f"  - {item}" for item in identity.boundaries],
                f"- 提示词摘要：{identity.prompt_summary}",
                "",
            ]
        )
    return "\n".join(lines)


def render_skill_manifest_markdown(skills: list[SkillManifestItem]) -> str:
    """渲染 Skill Manifest Markdown。"""

    lines = ["# PHM2010 C1 Skill Manifest", "", "本清单记录 P0 当前已实现或已映射的核心 Skill。", ""]
    for index, skill in enumerate(skills, start=1):
        lines.extend(
            [
                f"## {index}. {skill.skill_name}@{skill.version}",
                "",
                f"- 负责 Agent：{skill.owner_agent}",
                f"- 用途：{skill.purpose}",
                "- 输入：",
                *[f"  - {item}" for item in skill.inputs],
                "- 输出：",
                *[f"  - {item}" for item in skill.outputs],
                "- 安全说明：",
                *[f"  - {item}" for item in skill.safety_notes],
                "",
            ]
        )
    return "\n".join(lines)


def write_identity_and_skill_reports(settings: Settings) -> tuple[Path, Path]:
    """写出 Agent Identity 清单和 Skill Manifest。"""

    report_root = settings.ai_infra_root / "reports"
    identity_file = report_root / "phm2010_c1_agent_identity.md"
    skill_file = report_root / "phm2010_c1_skill_manifest.md"
    report_root.mkdir(parents=True, exist_ok=True)
    identity_file.write_text(render_agent_identity_markdown(build_core_agent_identities()), encoding="utf-8")
    skill_file.write_text(render_skill_manifest_markdown(build_p0_skill_manifest()), encoding="utf-8")
    return identity_file, skill_file
