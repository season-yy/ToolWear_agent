"""候选方案确认与保存。

本模块对应 P0 第 4 步的业务核心：
用户在页面中确认一个候选算法方案后，系统需要把选择结果保存下来，
供第 5 步小样本训练直接读取。

前端页面只负责展示和触发选择；真正的读写逻辑放在这里，方便测试和复用。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from toolwear_agent.training.candidates import CandidatePlan, CandidateSet


@dataclass(frozen=True)
class SelectedPlan:
    """用户确认后的候选方案结果。"""

    dataset_id: str
    cutter: str
    primary_task: str
    confirmed_by: str
    confirmed_at: str
    selected_plan: CandidatePlan
    source_candidate_file: str


def _now_shanghai() -> str:
    """返回上海时区的确认时间。

    记录时区是为了后续报告、日志、Trace 对齐。
    这里使用固定 UTC+8，避免旧 Python 环境缺少 `zoneinfo`。
    """

    shanghai_timezone = timezone(timedelta(hours=8))
    return datetime.now(shanghai_timezone).isoformat(timespec="seconds")


def _candidate_plan_from_dict(data: dict[str, object]) -> CandidatePlan:
    """把 JSON 字典还原成 `CandidatePlan`。

    JSON 读取出来没有 dataclass 类型，需要显式转换，后续代码才能稳定使用字段。
    """

    return CandidatePlan(
        plan_id=str(data["plan_id"]),
        display_name=str(data["display_name"]),
        summary=str(data["summary"]),
        model_family=str(data["model_family"]),
        input_channels=list(data["input_channels"]),
        preprocess_steps=list(data["preprocess_steps"]),
        feature_strategy=str(data["feature_strategy"]),
        model_structure=str(data["model_structure"]),
        training_strategy=str(data["training_strategy"]),
        expected_cost=str(data["expected_cost"]),
        advantages=list(data["advantages"]),
        risks=list(data["risks"]),
        recommended_reason=str(data["recommended_reason"]),
        user_confirm_params=list(data["user_confirm_params"]),
        suitable_for_p0=bool(data["suitable_for_p0"]),
        recommended_order=int(data["recommended_order"]),
    )


def load_candidate_set(candidate_file: Path) -> CandidateSet:
    """从 JSON 文件读取候选方案集合。"""

    if not candidate_file.exists():
        raise FileNotFoundError(f"候选方案文件不存在: {candidate_file}")

    data = json.loads(candidate_file.read_text(encoding="utf-8"))
    plans = [_candidate_plan_from_dict(item) for item in data["plans"]]
    return CandidateSet(
        dataset_id=str(data["dataset_id"]),
        cutter=str(data["cutter"]),
        source_label_file=str(data["source_label_file"]),
        primary_task=str(data["primary_task"]),
        plans=plans,
    )


def find_candidate_plan(candidate_set: CandidateSet, plan_id: str) -> CandidatePlan:
    """按 `plan_id` 查找候选方案。"""

    for plan in candidate_set.plans:
        if plan.plan_id == plan_id:
            return plan

    available = ", ".join(plan.plan_id for plan in candidate_set.plans)
    raise ValueError(f"未找到候选方案: {plan_id}。可选方案: {available}")


def select_candidate_plan(
    candidate_set: CandidateSet,
    plan_id: str,
    confirmed_by: str = "local_user",
    source_candidate_file: str = "",
) -> SelectedPlan:
    """生成用户确认结果。"""

    plan = find_candidate_plan(candidate_set, plan_id)
    return SelectedPlan(
        dataset_id=candidate_set.dataset_id,
        cutter=candidate_set.cutter,
        primary_task=candidate_set.primary_task,
        confirmed_by=confirmed_by,
        confirmed_at=_now_shanghai(),
        selected_plan=plan,
        source_candidate_file=source_candidate_file,
    )


def write_selected_plan_json(selected_plan: SelectedPlan, output_file: Path) -> Path:
    """写出用户确认方案 JSON。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(asdict(selected_plan), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file


def render_selected_plan_report(selected_plan: SelectedPlan) -> str:
    """生成用户确认方案 Markdown 报告。"""

    plan = selected_plan.selected_plan
    lines = [
        f"# {selected_plan.dataset_id.upper()} {selected_plan.cutter.upper()} 方案确认报告",
        "",
        "## 1. 确认结果",
        "",
        f"- 选择方案：`{plan.plan_id}`",
        f"- 方案名称：{plan.display_name}",
        f"- 确认人：{selected_plan.confirmed_by}",
        f"- 确认时间：{selected_plan.confirmed_at}",
        f"- 任务类型：`{selected_plan.primary_task}`",
        "",
        "## 2. 方案摘要",
        "",
        f"{plan.summary}",
        "",
        "## 3. 预计成本",
        "",
        f"{plan.expected_cost}",
        "",
        "## 4. 推荐理由",
        "",
        f"{plan.recommended_reason}",
        "",
        "## 5. 优点",
        "",
    ]
    lines.extend(f"- {item}" for item in plan.advantages)
    lines.extend(["", "## 6. 风险", ""])
    lines.extend(f"- {item}" for item in plan.risks)
    lines.extend(["", "## 7. 需要确认的参数", ""])
    lines.extend(f"- {item}" for item in plan.user_confirm_params)
    lines.extend(
        [
            "",
            "## 8. 对下一步的意义",
            "",
            "第 5 步小样本训练会读取本确认结果，并按选中方案准备训练配置。",
            "",
        ]
    )
    return "\n".join(lines)


def write_selected_plan_report(selected_plan: SelectedPlan, output_file: Path) -> Path:
    """写出用户确认方案 Markdown 报告。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_selected_plan_report(selected_plan), encoding="utf-8")
    return output_file


def write_selected_plan_log(selected_plan: SelectedPlan, output_file: Path, json_file: Path, report_file: Path) -> Path:
    """写出用户确认运行日志。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 候选方案确认运行日志",
                f"确认人: {selected_plan.confirmed_by}",
                f"确认时间: {selected_plan.confirmed_at}",
                f"选择方案: {selected_plan.selected_plan.plan_id}",
                f"方案名称: {selected_plan.selected_plan.display_name}",
                f"来源候选文件: {selected_plan.source_candidate_file}",
                f"选择结果 JSON: {json_file}",
                f"选择报告: {report_file}",
            ]
        ),
        encoding="utf-8",
    )
    return output_file
