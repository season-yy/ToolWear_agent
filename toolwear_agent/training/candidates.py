"""候选算法方案生成。

本模块对应初赛 P0 的第 3 步：
在真正训练前，先生成 2-3 个互相兼容、可解释、可落地的候选算法方案。

这里暂时不调用 LLM，而是先使用工程上可控的规则化方案。
原因：
- P0 首要目标是跑通闭环，候选方案必须能被后续训练代码执行。
- 规则化方案更稳定，也方便测试和复现。
- 后续可以把这些方案作为 LLM 推荐的基础模板。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_INPUT_CHANNELS = (
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "acoustic_emission_rms",
)


@dataclass(frozen=True)
class CandidatePlan:
    """单个候选算法方案。

    这个结构既要给人看，也要给后续训练代码读取。
    所以字段会比普通文字说明更细一些。
    """

    plan_id: str
    display_name: str
    summary: str
    model_family: str
    input_channels: list[str]
    preprocess_steps: list[str]
    feature_strategy: str
    model_structure: str
    training_strategy: str
    expected_cost: str
    advantages: list[str]
    risks: list[str]
    recommended_reason: str
    user_confirm_params: list[str]
    suitable_for_p0: bool
    recommended_order: int


@dataclass(frozen=True)
class CandidateSet:
    """某个数据集和刀具对应的一组候选方案。"""

    dataset_id: str
    cutter: str
    source_label_file: str
    primary_task: str
    plans: list[CandidatePlan]


def validate_candidate_plan(plan: CandidatePlan) -> None:
    """校验候选方案是否足够完整。

    如果方案缺字段，后续页面展示和训练代码都会很难处理。
    所以这里提前报错，而不是生成“看起来有方案、实际不能用”的配置。
    """

    if not plan.plan_id:
        raise ValueError("候选方案缺少 plan_id")
    if not plan.display_name:
        raise ValueError(f"{plan.plan_id} 缺少 display_name")
    if not plan.summary:
        raise ValueError(f"{plan.plan_id} 缺少 summary")
    if not plan.model_family:
        raise ValueError(f"{plan.plan_id} 缺少 model_family")
    if not plan.input_channels:
        raise ValueError(f"{plan.plan_id} 缺少 input_channels")
    if not plan.preprocess_steps:
        raise ValueError(f"{plan.plan_id} 缺少 preprocess_steps")
    if not plan.feature_strategy:
        raise ValueError(f"{plan.plan_id} 缺少 feature_strategy")
    if not plan.model_structure:
        raise ValueError(f"{plan.plan_id} 缺少 model_structure")
    if not plan.training_strategy:
        raise ValueError(f"{plan.plan_id} 缺少 training_strategy")
    if not plan.expected_cost:
        raise ValueError(f"{plan.plan_id} 缺少 expected_cost")
    if not plan.advantages:
        raise ValueError(f"{plan.plan_id} 缺少 advantages")
    if not plan.risks:
        raise ValueError(f"{plan.plan_id} 缺少 risks")
    if not plan.recommended_reason:
        raise ValueError(f"{plan.plan_id} 缺少 recommended_reason")
    if not plan.user_confirm_params:
        raise ValueError(f"{plan.plan_id} 缺少 user_confirm_params")
    if plan.recommended_order <= 0:
        raise ValueError(f"{plan.plan_id} recommended_order 必须大于 0")


def _statistical_baseline_plan() -> CandidatePlan:
    """方案 1：统计特征 + 传统分类器。

    这是 P0 最稳的 baseline。它训练快、解释性强，适合作为后续深度模型的对照组。
    """

    return CandidatePlan(
        plan_id="statistical_features_random_forest",
        display_name="统计特征 + RandomForest 基线",
        summary="从每刀七通道信号中提取均值、标准差、峰峰值、RMS、峭度等统计特征，再训练传统分类器。",
        model_family="traditional_ml",
        input_channels=list(DEFAULT_INPUT_CHANNELS),
        preprocess_steps=[
            "按刀次读取 7 通道信号 CSV",
            "每个通道抽取固定长度或全刀次统计特征",
            "只在训练集上拟合标准化参数，避免数据泄露",
            "使用四阶段 stage_id 作为分类标签",
        ],
        feature_strategy="时域统计特征为主，可补充少量频域能量特征",
        model_structure="RandomForestClassifier 或 ExtraTreesClassifier",
        training_strategy="先用小样本快速训练，评估 Macro-F1 和 Balanced Accuracy，作为后续深度模型下限参考",
        expected_cost="低：CPU 也能快速运行，适合作为 P0 第一候选",
        advantages=[
            "训练速度快，失败成本低",
            "结果解释性强，方便报告展示",
            "对小样本更友好",
            "可以作为深度模型是否真正有效的对照基线",
        ],
        risks=[
            "对复杂时序模式表达能力有限",
            "特征设计会影响上限",
            "如果阶段边界附近样本混叠严重，传统模型可能区分不明显",
        ],
        recommended_reason="P0 阶段需要先建立可复现、可解释、能快速跑通的基线，因此该方案推荐顺序最高。",
        user_confirm_params=[
            "每刀使用全量信号还是抽样窗口",
            "是否加入频域特征",
            "分类器使用 RandomForest 还是 ExtraTrees",
        ],
        suitable_for_p0=True,
        recommended_order=1,
    )


def _one_dimensional_cnn_plan() -> CandidatePlan:
    """方案 2：多通道时序切片 + 1D CNN。"""

    return CandidatePlan(
        plan_id="multichannel_window_1d_cnn",
        display_name="多通道时序切片 + 1D CNN",
        summary="把每刀信号切成固定长度窗口，保留七通道时序结构，用轻量 1D CNN 学习局部磨损相关模式。",
        model_family="deep_learning",
        input_channels=list(DEFAULT_INPUT_CHANNELS),
        preprocess_steps=[
            "按刀次读取 7 通道信号 CSV",
            "从每刀中抽取固定数量窗口，降低小样本训练成本",
            "对每个通道做训练集标准化",
            "窗口继承所属刀次的四阶段标签",
        ],
        feature_strategy="不手工构造复杂特征，由 1D CNN 从多通道时序窗口中学习局部模式",
        model_structure="Conv1D + BatchNorm + ReLU + GlobalAveragePooling + Linear",
        training_strategy="CUDA 小样本训练，先限制窗口数量和 epoch，观察 loss、Macro-F1、混淆矩阵",
        expected_cost="中：需要 CUDA，训练时间高于传统模型，但仍适合 P0 小样本粗训练",
        advantages=[
            "能利用原始时序信号",
            "比 Transformer 更轻，适合当前 12GB 显存",
            "后续容易扩展到多分支或注意力模块",
        ],
        risks=[
            "窗口标签继承刀次标签，会引入一定弱标签噪声",
            "窗口采样策略会影响训练结果",
            "小样本下可能过拟合，需要早停和正则化",
        ],
        recommended_reason="该方案是深度学习路线的最小可行版本，适合证明系统具备 CUDA 小样本训练能力。",
        user_confirm_params=[
            "窗口长度",
            "每刀抽取窗口数量",
            "epoch 数",
            "学习率",
        ],
        suitable_for_p0=True,
        recommended_order=2,
    )


def _lightweight_multibranch_plan() -> CandidatePlan:
    """方案 3：轻量多分支 CNN 或简单注意力。"""

    return CandidatePlan(
        plan_id="light_multibranch_cnn_attention",
        display_name="轻量多分支 CNN + 简单通道注意力",
        summary="把力、振动、声发射按信号来源分支处理，再用轻量通道注意力融合多源信息。",
        model_family="deep_learning",
        input_channels=list(DEFAULT_INPUT_CHANNELS),
        preprocess_steps=[
            "按力信号、振动信号、声发射信号分组",
            "每组信号分别标准化",
            "每刀抽取固定窗口用于小样本训练",
            "用四阶段 stage_id 作为监督标签",
        ],
        feature_strategy="分支 CNN 提取不同传感器特征，再进行轻量融合",
        model_structure="ForceBranchCNN + VibrationBranchCNN + AEBranchCNN + ChannelAttention + Linear",
        training_strategy="作为 P0 第三候选，优先小规模试跑；如果前两个方案已经足够，可暂缓完整训练",
        expected_cost="中高：比 1D CNN 复杂，仍低于 Transformer/DANN，适合作为展示系统可扩展性的候选",
        advantages=[
            "更贴合多传感器融合问题",
            "能体现项目中多通道融合和注意力机制的想法",
            "后续可扩展到跨刀具迁移学习",
        ],
        risks=[
            "结构更复杂，小样本下更容易过拟合",
            "需要更谨慎的参数控制和停止策略",
            "P0 时间紧时可能只适合做候选，不适合优先训练",
        ],
        recommended_reason="该方案体现你的核心想法：多通道融合、分支结构和注意力机制，但 P0 中建议作为第三候选。",
        user_confirm_params=[
            "传感器分组方式",
            "是否启用通道注意力",
            "每个分支卷积层数",
            "最大训练轮数",
        ],
        suitable_for_p0=True,
        recommended_order=3,
    )


def build_default_candidate_set(
    dataset_id: str,
    cutter: str,
    source_label_file: str = "",
    primary_task: str = "four_stage_classification",
) -> CandidateSet:
    """生成 P0 默认候选方案集合。"""

    plans = [
        _statistical_baseline_plan(),
        _one_dimensional_cnn_plan(),
        _lightweight_multibranch_plan(),
    ]
    for plan in plans:
        validate_candidate_plan(plan)

    return CandidateSet(
        dataset_id=dataset_id,
        cutter=cutter,
        source_label_file=source_label_file,
        primary_task=primary_task,
        plans=plans,
    )


def write_candidate_json(candidate_set: CandidateSet, output_file: Path) -> Path:
    """写出候选方案 JSON，并附带统一的 PipelineSpec。"""

    # 延迟导入避免迁移期 CandidatePlan 与转换器在模块加载时形成循环依赖。
    from toolwear_agent.registry import validate_pipeline_with_default_registries
    from toolwear_agent.schemas.converters import candidate_plan_to_pipeline

    payload = asdict(candidate_set)
    pipelines = [candidate_plan_to_pipeline(plan) for plan in candidate_set.plans]
    validations = [validate_pipeline_with_default_registries(pipeline) for pipeline in pipelines]
    invalid = [result for result in validations if not result.valid]
    if invalid:
        messages = [issue.message for result in invalid for issue in result.issues if issue.severity.value == "error"]
        raise ValueError("候选方案未通过 Module Registry：" + "；".join(messages))
    payload["pipeline_specs"] = [pipeline.model_dump(mode="json") for pipeline in pipelines]
    payload["registry_validations"] = [result.model_dump(mode="json") for result in validations]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def render_candidate_report(candidate_set: CandidateSet) -> str:
    """生成候选方案 Markdown 报告。"""

    lines = [
        f"# {candidate_set.dataset_id.upper()} {candidate_set.cutter.upper()} 候选算法方案报告",
        "",
        "## 1. 生成目的",
        "",
        "本报告用于 P0 初赛 PoC 的第 3 步：在训练前给出 2-3 个相互兼容的候选算法方案。",
        "这些方案会在下一步展示给用户，由用户确认一个方案进入 CUDA 小样本训练。",
        "",
        "## 2. 输入基础",
        "",
        f"- 数据集：`{candidate_set.dataset_id}`",
        f"- 刀具：`{candidate_set.cutter}`",
        f"- 任务：`{candidate_set.primary_task}`",
        f"- 标签文件：`{candidate_set.source_label_file or '未指定'}`",
        "",
        "## 3. 候选方案总览",
        "",
        "| 推荐顺序 | 方案 ID | 方案名称 | 模型族 | 预计成本 |",
        "|---:|---|---|---|---|",
    ]

    for plan in sorted(candidate_set.plans, key=lambda item: item.recommended_order):
        lines.append(
            f"| {plan.recommended_order} | `{plan.plan_id}` | {plan.display_name} | "
            f"{plan.model_family} | {plan.expected_cost} |"
        )

    for plan in sorted(candidate_set.plans, key=lambda item: item.recommended_order):
        lines.extend(
            [
                "",
                f"## 4.{plan.recommended_order} {plan.display_name}",
                "",
                f"- 方案 ID：`{plan.plan_id}`",
                f"- 简要说明：{plan.summary}",
                f"- 模型结构：{plan.model_structure}",
                f"- 特征策略：{plan.feature_strategy}",
                f"- 训练策略：{plan.training_strategy}",
                f"- 推荐理由：{plan.recommended_reason}",
                "",
                "### 优点",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in plan.advantages)
        lines.extend(["", "### 风险", ""])
        lines.extend(f"- {item}" for item in plan.risks)
        lines.extend(["", "### 需要用户确认的参数", ""])
        lines.extend(f"- {item}" for item in plan.user_confirm_params)

    lines.extend(
        [
            "",
            "## 5. 对下一步的意义",
            "",
            "下一步会把这些候选方案展示给用户，说明理由、风险和预计成本，"
            "然后由用户确认其中一个方案进入小样本训练。",
            "",
        ]
    )

    return "\n".join(lines)


def write_candidate_report(candidate_set: CandidateSet, output_file: Path) -> Path:
    """写出候选方案 Markdown 报告。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_candidate_report(candidate_set), encoding="utf-8")
    return output_file
