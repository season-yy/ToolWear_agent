"""命令行入口。

当前入口支持以下主要动作：
- 默认环境自检
- PHM2010 C1 数据登记与体检
- PHM2010 C1 四阶段标签生成
- PHM2010 C1 候选算法方案生成
- Registry 驱动的 PHM2010 C1 真实训练

后续会扩展为更多子命令，例如数据检查、标签生成、小样本训练。
"""

from __future__ import annotations

import argparse

from toolwear_agent.agentteams.decision import run_c1_agent_decision
from toolwear_agent.agentteams.diagnosis import run_c1_agent_diagnosis
from toolwear_agent.agentteams.identity import write_identity_and_skill_reports
from toolwear_agent.agentteams.llm_candidates import generate_llm_candidate_set, write_llm_candidate_outputs
from toolwear_agent.agentteams.official_adapter import run_c1_official_agentteams_minimal
from toolwear_agent.agentteams.reporting import run_c1_p0_report
from toolwear_agent.agentteams.trace import run_c1_agentteams_trace
from toolwear_agent.common.config import load_settings
from toolwear_agent.common.logging import get_logger
from toolwear_agent.common.paths import ensure_runtime_dirs
from toolwear_agent.data.adapters.phm2010 import PHM2010Adapter
from toolwear_agent.data.registry import DatasetRegistry
from toolwear_agent.registry import build_default_registry_catalog, write_registry_catalog
from toolwear_agent.training.candidates import (
    build_default_candidate_set,
    write_candidate_json,
    write_candidate_report,
)
from toolwear_agent.training.c1_runs import run_c1_training
from toolwear_agent.training.data_profile import write_profile_report
from toolwear_agent.training.labels import (
    build_label_dataset,
    write_label_csv,
    write_label_json,
    write_label_report,
)
from toolwear_agent.training.mini_train import run_c1_mini_train
from toolwear_agent.training.phm2010 import build_cutter_inventory, write_inventory_json
from toolwear_agent.training.candidate_compare import run_c1_candidate_compare
from toolwear_agent.training.visualization import run_c1_visual_report
from toolwear_agent.training.windows import build_c1_window_manifest


def run_health_check() -> None:
    """执行一次轻量环境自检。"""

    logger = get_logger("toolwear_agent")
    settings = load_settings()
    ensured_dirs = ensure_runtime_dirs(settings)

    logger.info("ToolWear Agent App 环境自检完成")
    logger.info("项目根目录: %s", settings.project_root)
    logger.info("应用代码目录: %s", settings.app_root)
    logger.info("数据清单路径: %s", settings.dataset_manifest)
    logger.info("训练设备默认值: %s", settings.train_device)
    logger.info("已确认运行目录数量: %s", len(ensured_dirs))


def run_register_datasets() -> None:
    """发现、体检并注册当前配置的 PHM2010 数据集。"""

    logger = get_logger("toolwear_agent.register_datasets")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始发现 PHM2010 C1-C6 数据")
    inspection = PHM2010Adapter().inspect(settings.phm2010_raw_root)
    if not inspection.validation.valid:
        error_messages = [
            issue.message
            for issue in inspection.validation.issues
            if issue.severity.value == "error"
        ]
        raise ValueError("PHM2010 数据体检未通过：" + "；".join(error_messages))

    registry = DatasetRegistry(settings.dataset_manifest)
    manifest = registry.register(inspection.manifest)
    logger.info("Dataset Registry 已写出: %s", settings.dataset_manifest)
    logger.info("可用刀具: %s", ", ".join(manifest.available_cutter_ids))
    logger.info("有标签刀具: %s", ", ".join(manifest.labeled_cutter_ids))
    logger.info("无标签刀具: %s", ", ".join(manifest.unlabeled_cutter_ids))
    logger.info("Manifest hash: %s", manifest.manifest_hash)


def run_register_capabilities() -> None:
    """写出供 API、页面和 Agent 共用的 Module/Trainer Catalog。"""

    logger = get_logger("toolwear_agent.register_capabilities")
    settings = load_settings()
    ensure_runtime_dirs(settings)
    output_file = settings.state_root / "registries" / "module_trainer_catalog.json"

    catalog = build_default_registry_catalog()
    write_registry_catalog(catalog, output_file)
    logger.info("Module/Trainer Catalog 已写出: %s", output_file)
    logger.info("输入预设数量: %s", len(catalog.input_presets))
    logger.info("模块数量: %s", len(catalog.modules))
    logger.info("训练器数量: %s", len(catalog.trainers))
    logger.info("Catalog hash: %s", catalog.catalog_hash)


def run_profile_c1() -> None:
    """执行 PHM2010 C1 数据登记与体检。"""

    logger = get_logger("toolwear_agent.profile_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    cutter = "c1"
    cutter_dir = settings.phm2010_raw_root / cutter
    inventory_file = settings.dataset_manifest.parent / "phm2010_c1_inventory.json"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_data_profile.md"
    log_file = settings.log_root / "phm2010_c1_data_profile.log"

    logger.info("开始登记 PHM2010 %s 数据", cutter.upper())
    logger.info("刀具目录: %s", cutter_dir)

    inventory = build_cutter_inventory(cutter_dir, cutter=cutter)
    write_inventory_json(inventory, inventory_file)
    write_profile_report(inventory, report_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 数据登记与体检运行日志",
                f"刀具目录: {cutter_dir}",
                f"信号 CSV 数量: {inventory.signal_file_count}",
                f"磨损标签数量: {inventory.label_count}",
                f"缺失信号刀次数量: {len(inventory.missing_signal_cuts)}",
                f"缺失标签刀次数量: {len(inventory.missing_label_cuts)}",
                f"数据清单: {inventory_file}",
                f"体检报告: {report_file}",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("C1 信号 CSV 数量: %s", inventory.signal_file_count)
    logger.info("C1 磨损标签数量: %s", inventory.label_count)
    logger.info("数据清单已写出: %s", inventory_file)
    logger.info("体检报告已写出: %s", report_file)
    logger.info("运行日志已写出: %s", log_file)


def run_label_c1() -> None:
    """执行 PHM2010 C1 四阶段标签生成。"""

    logger = get_logger("toolwear_agent.label_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    cutter = "c1"
    wear_file = settings.phm2010_raw_root / cutter / "c1_wear.csv"
    output_root = settings.ai_infra_root / "datasets" / "processed" / "phm2010"
    label_json = output_root / "phm2010_c1_stage_labels.json"
    label_csv = output_root / "phm2010_c1_stage_labels.csv"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_stage_labels.md"
    log_file = settings.log_root / "phm2010_c1_stage_labels.log"

    logger.info("开始生成 PHM2010 %s 四阶段标签", cutter.upper())
    logger.info("磨损标签文件: %s", wear_file)
    logger.info("VB 聚合方式: %s", settings.vb_aggregation)
    logger.info("阶段阈值 um: %s", settings.vb_stage_thresholds_um)

    label_dataset = build_label_dataset(
        wear_file=wear_file,
        cutter=cutter,
        aggregation=settings.vb_aggregation,
        thresholds=settings.vb_stage_thresholds_um,
    )
    write_label_json(label_dataset, label_json)
    write_label_csv(label_dataset, label_csv)
    write_label_report(label_dataset, report_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 四阶段标签生成运行日志",
                f"磨损标签文件: {wear_file}",
                f"VB 聚合方式: {label_dataset.aggregation}",
                f"阶段阈值 um: {label_dataset.thresholds_um}",
                f"标签记录数量: {label_dataset.record_count}",
                f"阶段分布: {label_dataset.stage_distribution}",
                f"JSON 标签文件: {label_json}",
                f"CSV 标签文件: {label_csv}",
                f"标签报告: {report_file}",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("标签记录数量: %s", label_dataset.record_count)
    logger.info("阶段分布: %s", label_dataset.stage_distribution)
    logger.info("JSON 标签文件已写出: %s", label_json)
    logger.info("CSV 标签文件已写出: %s", label_csv)
    logger.info("标签报告已写出: %s", report_file)
    logger.info("运行日志已写出: %s", log_file)


def run_candidates_c1() -> None:
    """生成 PHM2010 C1 候选算法方案。"""

    logger = get_logger("toolwear_agent.candidates_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    cutter = "c1"
    label_file = settings.ai_infra_root / "datasets" / "processed" / "phm2010" / "phm2010_c1_stage_labels.csv"
    output_root = settings.ai_infra_root / "experiments" / "candidates"
    candidate_json = output_root / "phm2010_c1_candidate_plans.json"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_candidate_plans.md"
    log_file = settings.log_root / "phm2010_c1_candidate_plans.log"

    logger.info("开始生成 PHM2010 %s 候选算法方案", cutter.upper())
    logger.info("标签文件: %s", label_file)

    candidate_set = build_default_candidate_set(
        dataset_id="phm2010",
        cutter=cutter,
        source_label_file=str(label_file),
        primary_task=settings.primary_task,
    )
    write_candidate_json(candidate_set, candidate_json)
    write_candidate_report(candidate_set, report_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 候选算法方案生成运行日志",
                f"标签文件: {label_file}",
                f"候选方案数量: {len(candidate_set.plans)}",
                f"候选方案 JSON: {candidate_json}",
                f"候选方案报告: {report_file}",
                "推荐顺序: "
                + ", ".join(
                    f"{plan.recommended_order}:{plan.plan_id}"
                    for plan in sorted(candidate_set.plans, key=lambda item: item.recommended_order)
                ),
            ]
        ),
        encoding="utf-8",
    )

    logger.info("候选方案数量: %s", len(candidate_set.plans))
    logger.info("候选方案 JSON 已写出: %s", candidate_json)
    logger.info("候选方案报告已写出: %s", report_file)
    logger.info("运行日志已写出: %s", log_file)


def run_mini_train_c1() -> None:
    """执行 PHM2010 C1 小样本训练。

    这一命令对应 P0 第 5 步：读取第 4 步已确认的方案，抽取少量样本完成一次训练闭环。
    """

    logger = get_logger("toolwear_agent.mini_train_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始执行 PHM2010 C1 小样本训练")
    result = run_c1_mini_train(settings)
    logger.info("小样本训练完成，运行编号: %s", result.run_id)
    logger.info("Macro-F1: %.4f", result.macro_f1)
    logger.info("Balanced Accuracy: %.4f", result.balanced_accuracy)
    logger.info("模型文件: %s", result.model_file)
    logger.info("训练报告: %s", result.report_file)


def run_training_service_c1(args: argparse.Namespace) -> None:
    """通过统一 TrainingService 执行 C1 传统模型或 1D-CNN。"""

    logger = get_logger("toolwear_agent.train_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)
    logger.info("开始执行统一 C1 训练，方案: %s", args.plan_id)
    logger.info("训练上限: %s", args.max_samples if args.max_samples is not None else "完整 20% train")
    logger.info("epoch/batch/lr: %s/%s/%s", args.epochs, args.batch_size, args.learning_rate)
    result = run_c1_training(
        settings,
        plan_id=args.plan_id,
        run_id=args.run_id,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        base_channels=args.base_channels,
        dropout=args.dropout,
        loss_id=args.loss_id,
    )
    validation = next(
        metric for metric in result.evaluation.metrics if metric.split.value == "validation"
    )
    logger.info("训练完成，运行编号: %s", result.run_id)
    logger.info("后端/设备: %s/%s", result.runtime.backend, result.runtime.resolved_device)
    logger.info("Validation Macro-F1: %.4f", validation.macro_f1)
    logger.info("Validation Balanced Accuracy: %.4f", validation.balanced_accuracy)
    logger.info("模型文件: %s", result.artifacts.model_file)
    logger.info("证据索引: %s", result.artifacts.evidence_index_file)


def run_windows_c1() -> None:
    """构建 PHM2010 C1 窗口样本索引。

    这一命令会先按 cut 划分 train/val/test，再在每个 cut 内生成窗口索引，避免数据泄露。
    """

    logger = get_logger("toolwear_agent.windows_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    output_root = settings.ai_infra_root / "datasets" / "processed" / "phm2010"
    label_file = output_root / "phm2010_c1_stage_labels.csv"
    cutter_dir = settings.phm2010_raw_root / "c1"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_window_split_report.md"
    log_file = settings.log_root / "phm2010_c1_window_split.log"

    logger.info("开始构建 PHM2010 C1 窗口样本索引")
    result = build_c1_window_manifest(
        label_file=label_file,
        cutter_dir=cutter_dir,
        output_root=output_root,
        report_file=report_file,
        log_file=log_file,
        split_lock_file=settings.state_root / "splits" / "phm2010_c1_p0" / "r0001" / "split_lock.json",
        experiment_id="phm2010_c1_p0",
        revision=1,
    )
    logger.info("窗口样本数量: %s", result.window_count)
    logger.info("窗口 manifest: %s", result.window_manifest_file)
    logger.info("切分报告: %s", result.report_file)


def run_visualize_c1() -> None:
    """生成 PHM2010 C1 窗口训练图表和报告。"""

    logger = get_logger("toolwear_agent.visualize_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始生成 PHM2010 C1 可视化报告")
    result = run_c1_visual_report(settings)
    logger.info("可视化报告完成，运行编号: %s", result.run_id)
    logger.info("图表目录: %s", result.figure_dir)
    logger.info("Markdown 报告: %s", result.report_file)


def run_diagnose_c1() -> None:
    """生成 PHM2010 C1 Agent 结构化诊断报告。"""

    logger = get_logger("toolwear_agent.diagnose_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始生成 PHM2010 C1 Agent 结构化诊断")
    diagnosis = run_c1_agent_diagnosis(settings)
    logger.info("诊断完成，诊断编号: %s", diagnosis.diagnosis_id)
    logger.info("是否继续当前方案: %s", diagnosis.decision.continue_current_plan)
    logger.info("推荐动作: %s", diagnosis.decision.recommended_next_action)


def run_decide_c1() -> None:
    """生成 PHM2010 C1 Agent 参数调整或停止决策。"""

    logger = get_logger("toolwear_agent.decide_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始生成 PHM2010 C1 Agent 参数调整或停止决策")
    decision = run_c1_agent_decision(settings)
    logger.info("决策完成，决策编号: %s", decision.decision_id)
    logger.info("是否继续当前方案: %s", decision.should_continue_current_plan)
    logger.info("是否停止当前方案: %s", decision.should_stop_current_plan)
    logger.info("是否调整小样本比例: %s", decision.should_adjust_sample_fraction)


def run_report_c1() -> None:
    """生成 PHM2010 C1 P0 Markdown 实验总报告。"""

    logger = get_logger("toolwear_agent.report_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始生成 PHM2010 C1 P0 Markdown 实验总报告")
    result = run_c1_p0_report(settings)
    logger.info("总报告生成完成，运行编号: %s", result.run_id)
    logger.info("Markdown 总报告: %s", result.report_file)
    logger.info("报告 manifest: %s", result.manifest_file)


def run_trace_c1() -> None:
    """生成 PHM2010 C1 AgentTeams 协作记录。"""

    logger = get_logger("toolwear_agent.trace_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始生成 PHM2010 C1 AgentTeams 协作记录")
    result = run_c1_agentteams_trace(settings)
    logger.info("Trace 生成完成，Trace ID: %s", result.trace_id)
    logger.info("Trace JSON: %s", result.trace_json)
    logger.info("Trace Markdown: %s", result.trace_report)


def run_identity_c1() -> None:
    """生成 PHM2010 C1 Agent Identity 和 Skill Manifest。"""

    logger = get_logger("toolwear_agent.identity_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始生成 PHM2010 C1 Agent Identity 和 Skill Manifest")
    identity_file, skill_file = write_identity_and_skill_reports(settings)
    logger.info("Agent Identity 清单: %s", identity_file)
    logger.info("Skill Manifest: %s", skill_file)


def run_llm_candidates_c1() -> None:
    """调用 AlgorithmArchitectAgent 生成 LLM 候选方案。"""

    logger = get_logger("toolwear_agent.llm_candidates_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    user_request = "我想用 PHM2010 C1 做四阶段刀具磨损分类，优先快速验证，并比较传统模型和后续深度模型路线。"
    logger.info("开始生成 PHM2010 C1 LLM 候选方案")
    candidate_set = generate_llm_candidate_set(settings, user_request)
    candidate_file, report_file, log_file = write_llm_candidate_outputs(candidate_set, settings)
    logger.info("候选方案数量: %s", len(candidate_set.plans))
    logger.info("是否 fallback: %s", candidate_set.used_fallback)
    logger.info("候选方案 JSON: %s", candidate_file)
    logger.info("候选方案报告: %s", report_file)
    logger.info("日志: %s", log_file)


def run_compare_candidates_c1() -> None:
    """执行 PHM2010 C1 多候选训练对比。"""

    logger = get_logger("toolwear_agent.compare_candidates_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始执行 PHM2010 C1 多候选训练对比")
    result = run_c1_candidate_compare(settings)
    logger.info("对比完成，运行编号: %s", result.compare_run_id)
    logger.info("推荐候选: %s", result.best_plan_id)
    logger.info("对比报告: %s", result.report_file)


def run_official_agentteams_c1() -> None:
    """生成 PHM2010 C1 官方 AgentTeams 最小接入包。"""

    logger = get_logger("toolwear_agent.official_agentteams_c1")
    settings = load_settings()
    ensure_runtime_dirs(settings)

    logger.info("开始生成 PHM2010 C1 官方 AgentTeams 最小接入包")
    package = run_c1_official_agentteams_minimal(settings)
    logger.info("接入包生成完成: %s", package.package_id)
    logger.info("Team: %s", package.team_name)
    logger.info("Team Leader: %s", package.leader_name)
    logger.info("Worker 数量: %s", len(package.worker_names))
    logger.info("Element 创建消息: %s", package.output_files["element_message"])
    logger.info("接入报告: %s", package.output_files["report"])


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="python -m toolwear_agent",
        description="ToolWear Agent App 本地命令行工具",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="health",
        choices=[
            "health",
            "register-datasets",
            "register-capabilities",
            "profile-c1",
            "label-c1",
            "candidates-c1",
            "windows-c1",
            "mini-train-c1",
            "train-c1",
            "visualize-c1",
            "diagnose-c1",
            "decide-c1",
            "report-c1",
            "trace-c1",
            "identity-c1",
            "llm-candidates-c1",
            "compare-candidates-c1",
            "official-agentteams-c1",
        ],
        help=(
            "要执行的动作：health=环境自检，profile-c1=数据体检，"
            "label-c1=四阶段标签生成，candidates-c1=候选方案生成"
        ),
    )
    parser.add_argument(
        "--plan-id",
        choices=[
            "statistical_features_random_forest",
            "statistical_features_extra_trees",
            "multichannel_window_1d_cnn",
        ],
        default="statistical_features_random_forest",
        help="train-c1 使用的规范候选方案 ID。",
    )
    parser.add_argument("--run-id", default=None, help="可选的唯一运行编号；默认按当前时间生成。")
    parser.add_argument("--max-samples", type=int, default=None, help="smoke 时每个 train/validation 的样本上限。")
    parser.add_argument("--epochs", type=int, default=2, help="PyTorch 训练轮数；传统模型会记录但不使用。")
    parser.add_argument("--batch-size", type=int, default=64, help="PyTorch batch size。")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="PyTorch 初始学习率。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="训练设备。")
    parser.add_argument("--n-estimators", type=int, default=300, help="树模型数量。")
    parser.add_argument("--max-depth", type=int, default=0, help="树最大深度；0 表示不限制。")
    parser.add_argument("--base-channels", type=int, default=32, help="1D-CNN 首层卷积通道数。")
    parser.add_argument("--dropout", type=float, default=0.2, help="1D-CNN 分类头 Dropout。")
    parser.add_argument(
        "--loss-id",
        choices=["cross_entropy", "weighted_cross_entropy"],
        default="cross_entropy",
        help="1D-CNN 损失模块。",
    )
    return parser


def main() -> None:
    """命令行入口。"""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "health":
        run_health_check()
        return
    if args.command == "register-datasets":
        run_register_datasets()
        return
    if args.command == "register-capabilities":
        run_register_capabilities()
        return
    if args.command == "profile-c1":
        run_profile_c1()
        return
    if args.command == "label-c1":
        run_label_c1()
        return
    if args.command == "candidates-c1":
        run_candidates_c1()
        return
    if args.command == "windows-c1":
        run_windows_c1()
        return
    if args.command == "mini-train-c1":
        run_mini_train_c1()
        return
    if args.command == "train-c1":
        run_training_service_c1(args)
        return
    if args.command == "visualize-c1":
        run_visualize_c1()
        return
    if args.command == "diagnose-c1":
        run_diagnose_c1()
        return
    if args.command == "decide-c1":
        run_decide_c1()
        return
    if args.command == "report-c1":
        run_report_c1()
        return
    if args.command == "trace-c1":
        run_trace_c1()
        return
    if args.command == "identity-c1":
        run_identity_c1()
        return
    if args.command == "llm-candidates-c1":
        run_llm_candidates_c1()
        return
    if args.command == "compare-candidates-c1":
        run_compare_candidates_c1()
        return
    if args.command == "official-agentteams-c1":
        run_official_agentteams_c1()
        return

    # argparse 的 choices 已经限制了合法值，这里只是防御性兜底。
    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
