"""Streamlit 实验台的数据读取与命令执行辅助。

页面层只负责展示和交互；这里负责读取候选方案、训练结果、图表路径、
Agent 诊断和证据文件，避免 Streamlit 页面里堆太多业务逻辑。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from toolwear_agent.agentteams.llm_candidates import (
    generate_llm_candidate_set,
    validate_candidate_payload,
    write_llm_candidate_outputs,
)
from toolwear_agent.agentteams.identity import build_core_agent_identities
from toolwear_agent.agentteams.reporting import find_latest_decided_run
from toolwear_agent.common.config import Settings
from toolwear_agent.registry import validate_pipeline_with_default_registries
from toolwear_agent.schemas import PipelineSpec
from toolwear_agent.schemas.converters import llm_candidate_plan_to_pipeline
from toolwear_agent.training.selection import load_candidate_set


@dataclass(frozen=True)
class CommandResult:
    """页面按钮触发命令后的结果。"""

    command: str
    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class DashboardPaths:
    """实验台固定使用的关键路径。"""

    candidate_file: Path
    selected_plan_file: Path
    selected_plan_report: Path
    identity_report: Path
    skill_manifest: Path
    p0_report: Path
    trace_report: Path
    trace_json: Path
    llm_candidate_file: Path
    candidate_compare_result: Path
    candidate_compare_report: Path
    official_agentteams_package: Path
    official_agentteams_report: Path
    official_agentteams_message: Path


def build_dashboard_paths(settings: Settings) -> DashboardPaths:
    """根据配置生成实验台需要展示的路径。"""

    reports = settings.ai_infra_root / "reports"
    return DashboardPaths(
        candidate_file=settings.ai_infra_root / "experiments" / "candidates" / "phm2010_c1_candidate_plans.json",
        selected_plan_file=settings.ai_infra_root / "experiments" / "selected" / "phm2010_c1_selected_plan.json",
        selected_plan_report=reports / "phm2010_c1_selected_plan.md",
        identity_report=reports / "phm2010_c1_agent_identity.md",
        skill_manifest=reports / "phm2010_c1_skill_manifest.md",
        p0_report=reports / "phm2010_c1_p0_experiment_report.md",
        trace_report=reports / "phm2010_c1_agentteams_trace.md",
        trace_json=settings.ai_infra_root / "traces" / "phm2010_c1_agentteams_trace.json",
        llm_candidate_file=(
            settings.ai_infra_root
            / "experiments"
            / "candidates"
            / "phm2010_c1_llm_candidate_plans.json"
        ),
        candidate_compare_result=_find_latest_compare_result(settings.experiment_root),
        candidate_compare_report=reports / "phm2010_c1_candidate_compare_report.md",
        official_agentteams_package=settings.ai_infra_root
        / "agentteams"
        / "phm2010_c1_minimal"
        / "toolwear_c1_agentteams_minimal_package.json",
        official_agentteams_report=reports / "phm2010_c1_official_agentteams_minimal.md",
        official_agentteams_message=settings.ai_infra_root
        / "agentteams"
        / "phm2010_c1_minimal"
        / "element_manager_create_team_message.md",
    )


def _find_latest_compare_result(experiment_root: Path) -> Path:
    """查找最新候选对比结果。"""

    runs = [
        item / "candidate_compare_result.json"
        for item in experiment_root.glob("phm2010_c1_candidate_compare_*")
        if item.is_dir() and (item / "candidate_compare_result.json").exists()
    ]
    if not runs:
        return experiment_root / "candidate_compare_result_missing.json"
    return max(runs, key=lambda item: item.stat().st_mtime)


def load_json_file(json_file: Path) -> dict[str, object]:
    """读取 JSON 文件；不存在时返回空字典。"""

    if not json_file.exists():
        return {}
    return json.loads(json_file.read_text(encoding="utf-8"))


def load_candidate_cards(candidate_file: Path) -> list[dict[str, object]]:
    """读取候选方案并转换为页面卡片数据。"""

    if not candidate_file.exists():
        return []
    payload = load_json_file(candidate_file)
    canonical = payload.get("pipeline_specs", [])
    if isinstance(canonical, list):
        choices = [
            choice
            for item in canonical
            if isinstance(item, dict)
            for choice in [_validated_pipeline_choice(item)]
            if choice is not None
        ]
        if choices:
            return choices

    # 兼容迁移前没有 PipelineSpec 的固定候选文件。
    candidate_set = load_candidate_set(candidate_file)
    plans = sorted(candidate_set.plans, key=lambda item: item.recommended_order)
    return [
        {
            "plan_id": plan.plan_id,
            "display_name": plan.display_name,
            "summary": plan.summary,
            "model_structure": plan.model_structure,
            "expected_cost": plan.expected_cost,
            "recommended_reason": plan.recommended_reason,
            "advantages": plan.advantages,
            "risks": plan.risks,
            "suitable_for_p0": plan.suitable_for_p0,
            "source": "static",
            "single_train_supported": plan.plan_id == "statistical_features_random_forest",
            "compare_train_supported": plan.plan_id == "statistical_features_random_forest",
        }
        for plan in plans
    ]


def build_candidate_choices(snapshot: dict[str, object]) -> list[dict[str, object]]:
    """生成页面统一候选列表。

    页面上半部分和单选框必须使用同一批候选，否则用户会看到“LLM 生成一套、
    下面选择另一套”的错位。这里优先使用 LLM 候选；如果还没有 LLM 候选，
    才退回固定模板候选。
    """

    llm_candidates = snapshot.get("llm_candidates", {})
    if isinstance(llm_candidates, dict) and llm_candidates.get("pipeline_specs"):
        choices = [
            choice
            for plan in llm_candidates["pipeline_specs"]
            if isinstance(plan, dict)
            for choice in [_validated_pipeline_choice(plan)]
            if choice is not None
        ]
        if choices:
            return choices
    if isinstance(llm_candidates, dict) and llm_candidates.get("plans"):
        choices = [
            choice
            for plan in llm_candidates["plans"]
            if isinstance(plan, dict)
            for choice in [_validated_legacy_llm_choice(plan)]
            if choice is not None
        ]
        if choices:
            return choices
    return list(snapshot.get("candidate_cards", []))


def _validated_pipeline_choice(pipeline_payload: dict[str, object]) -> dict[str, object] | None:
    """只把 Schema 和 Registry 均通过的 Pipeline 转成页面选项。"""

    try:
        pipeline = PipelineSpec.model_validate(pipeline_payload)
    except ValueError:
        return None
    validation = validate_pipeline_with_default_registries(pipeline)
    if not validation.valid:
        return None
    choice = _pipeline_spec_to_choice(pipeline.model_dump(mode="json"))
    choice["registry_validated"] = True
    choice["registry_warnings"] = [
        issue.message for issue in validation.issues if issue.severity.value == "warning"
    ]
    return choice


def _validated_legacy_llm_choice(plan_payload: dict[str, object]) -> dict[str, object] | None:
    """把旧 LLM 文本方案重新转换成当前 Registry 的 canonical Pipeline。"""

    try:
        plan = validate_candidate_payload([plan_payload])[0]
        pipeline = llm_candidate_plan_to_pipeline(plan)
    except (ValueError, KeyError, TypeError, IndexError):
        return None
    choice = _validated_pipeline_choice(pipeline.model_dump(mode="json"))
    if choice is None:
        return None

    # 推荐理由沿用旧 LLM 输出，但模块结构、可训练状态和后端只信任 canonical Pipeline。
    legacy_display = _llm_plan_to_choice(plan_payload)
    for field_name in ("summary", "recommended_reason", "advantages", "risks"):
        choice[field_name] = legacy_display[field_name]
    return choice


def _pipeline_spec_to_choice(pipeline: dict[str, object]) -> dict[str, object]:
    """把统一 PipelineSpec 转成页面卡片和单选框共用的数据。"""

    plan_id = str(pipeline["pipeline_id"])
    rationale = str(pipeline.get("rationale", ""))
    risks = [str(item) for item in pipeline.get("risks", [])]
    modules = pipeline.get("modules", [])
    module_ids = [
        str(module.get("module_id", ""))
        for module in modules
        if isinstance(module, dict) and bool(module.get("enabled", True))
    ]
    trainable = bool(pipeline.get("trainable", False))
    source = str(pipeline.get("source", "unknown"))
    return {
        "plan_id": plan_id,
        "display_name": str(pipeline["display_name"]),
        "summary": rationale,
        "model_structure": " -> ".join(module_ids),
        "expected_cost": str(pipeline.get("expected_cost", "")),
        "recommended_reason": rationale,
        "advantages": [rationale] if rationale else [],
        "risks": risks,
        "suitable_for_p0": trainable,
        "trainable_now": trainable,
        "training_backend": module_ids[-1] if module_ids else "",
        "source": "static" if source == "fixed" else source,
        "single_train_supported": plan_id == "statistical_features_random_forest",
        "compare_train_supported": plan_id
        in {"statistical_features_random_forest", "statistical_features_extra_trees"},
    }


def _llm_plan_to_choice(plan: dict[str, object]) -> dict[str, object]:
    """把 LLM 候选转换成页面卡片和单选框共用的数据结构。"""

    plan_id = str(plan["plan_id"])
    reason = str(plan.get("reason", ""))
    risk = str(plan.get("risk", ""))
    pipeline = [str(item) for item in plan.get("module_pipeline", [])]
    trainable_now = bool(plan.get("trainable_now", False))
    return {
        "plan_id": plan_id,
        "display_name": str(plan["display_name"]),
        "summary": reason,
        "model_structure": " -> ".join(pipeline),
        "expected_cost": str(plan.get("expected_cost", "")),
        "recommended_reason": reason,
        "advantages": [reason] if reason else [],
        "risks": [risk] if risk else [],
        "suitable_for_p0": trainable_now,
        "trainable_now": trainable_now,
        "training_backend": str(plan.get("training_backend", "")),
        "source": "llm",
        "single_train_supported": plan_id == "statistical_features_random_forest",
        "compare_train_supported": plan_id
        in {"statistical_features_random_forest", "statistical_features_extra_trees"},
    }


def find_latest_run_or_none(settings: Settings) -> Path | None:
    """查找最新完成诊断决策的运行目录，找不到时返回 None。"""

    try:
        return find_latest_decided_run(settings.experiment_root)
    except FileNotFoundError:
        return None


def build_result_snapshot(settings: Settings) -> dict[str, object]:
    """汇总页面需要展示的当前实验结果。"""

    paths = build_dashboard_paths(settings)
    run_dir = find_latest_run_or_none(settings)
    metrics = load_json_file(run_dir / "metrics_summary.json") if run_dir else {}
    diagnosis = load_json_file(run_dir / "agent_diagnosis.json") if run_dir else {}
    decision = load_json_file(run_dir / "agent_decision.json") if run_dir else {}
    visual_manifest = load_json_file(run_dir / "visual_report_manifest.json") if run_dir else {}
    llm_candidates = load_json_file(paths.llm_candidate_file)
    compare_result = load_json_file(paths.candidate_compare_result)
    official_agentteams = load_json_file(paths.official_agentteams_package)
    snapshot = {
        "core_agents": [identity.agent_name for identity in build_core_agent_identities()],
        "candidate_cards": load_candidate_cards(paths.candidate_file),
        "run_dir": str(run_dir) if run_dir else "",
        "metrics": metrics,
        "diagnosis": diagnosis,
        "decision": decision,
        "visual_manifest": visual_manifest,
        "llm_candidates": llm_candidates,
        "compare_result": compare_result,
        "official_agentteams": official_agentteams,
        "paths": paths,
    }
    snapshot["candidate_choices"] = build_candidate_choices(snapshot)
    return snapshot


def run_toolwear_command(command: str, app_root: Path) -> CommandResult:
    """用当前 Python 环境执行 `python -m toolwear_agent <command>`。"""

    process = subprocess.run(
        [sys.executable, "-m", "toolwear_agent", command],
        cwd=app_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandResult(
        command=command,
        return_code=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def run_p0_training_flow(app_root: Path) -> list[CommandResult]:
    """执行页面按钮对应的 P0 小样本训练与证据刷新流程。"""

    commands = [
        "mini-train-c1",
        "visualize-c1",
        "diagnose-c1",
        "decide-c1",
        "report-c1",
        "identity-c1",
        "trace-c1",
    ]
    results: list[CommandResult] = []
    for command in commands:
        result = run_toolwear_command(command, app_root)
        results.append(result)
        if result.return_code != 0:
            break
    return results


def generate_llm_candidates_for_request(settings: Settings, user_request: str) -> tuple[Path, Path, Path]:
    """根据页面输入调用 AlgorithmArchitectAgent 生成候选方案。"""

    candidate_set = generate_llm_candidate_set(settings, user_request)
    return write_llm_candidate_outputs(candidate_set, settings)
