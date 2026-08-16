"""把 ExperimentState 转换成页面阶段、按钮和数据准备进度。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WORKFLOW_STAGES = (
    "实验定义",
    "数据准备",
    "候选方案",
    "方案校验",
    "模型训练",
    "结果评估",
    "决策归档",
)


_STATE_STAGE = {
    "DRAFT": 0,
    "DATA_VALIDATING": 1,
    "BLOCKED_DATA": 1,
    "WAITING_PLAN_SELECTION": 2,
    "PIPELINE_VALIDATING": 3,
    "CODE_PREPARING": 3,
    "MINI_TRAINING": 4,
    "FULL_TRAINING": 4,
    "FAILED": 4,
    "CANCELLED": 4,
    "EVALUATING": 5,
    "EVALUATING_FULL": 5,
    "DECIDING": 6,
    "WAITING_FULL_APPROVAL": 6,
    "WAITING_USER_REVIEW": 6,
    "COMPLETED_MINI": 6,
    "COMPLETED_FULL": 6,
}


@dataclass(frozen=True)
class PreparationProgress:
    """三项确定性数据证据是否已存在。"""

    profile: bool
    labels: bool
    split: bool

    @property
    def complete(self) -> bool:
        return self.profile and self.labels and self.split


@dataclass(frozen=True)
class StateActions:
    """当前状态下页面允许触发的命令。"""

    prepare_data: bool = False
    generate_candidates: bool = False
    approve_pipeline: bool = False
    validate_pipeline: bool = False
    start_training: bool = False
    cancel_training: bool = False
    evaluate: bool = False
    decide: bool = False
    generate_report: bool = False


def workflow_stage_index(state: str) -> int:
    """返回状态在七段实验轨道中的位置。"""

    return _STATE_STAGE.get(state, 0)


def preparation_progress(
    experiment_id: str,
    revision: int,
    artifacts: list[dict[str, Any]],
) -> PreparationProgress:
    """按稳定 Evidence ID 判断页面刷新后哪些数据动作已经完成。"""

    evidence_ids = {str(item.get("evidence_id", "")) for item in artifacts}
    return PreparationProgress(
        profile=f"{experiment_id}-profile-r{revision}" in evidence_ids,
        labels=f"{experiment_id}-labels-r{revision}" in evidence_ids,
        split=f"{experiment_id}-split-r{revision}" in evidence_ids,
    )


def state_actions(
    state: str,
    *,
    preparation_ready: bool,
    has_succeeded_run: bool,
) -> StateActions:
    """集中定义按钮开关，避免各页面分支自行猜测状态机。"""

    return StateActions(
        prepare_data=(
            state in {"DRAFT", "DATA_VALIDATING", "BLOCKED_DATA"}
            and not preparation_ready
        ),
        generate_candidates=(
            state == "WAITING_PLAN_SELECTION"
            or (state == "DATA_VALIDATING" and preparation_ready)
        ),
        approve_pipeline=state == "WAITING_PLAN_SELECTION",
        validate_pipeline=state == "PIPELINE_VALIDATING",
        start_training=state == "CODE_PREPARING",
        cancel_training=state == "MINI_TRAINING",
        evaluate=state == "EVALUATING",
        decide=state == "DECIDING",
        generate_report=(
            has_succeeded_run
            and state
            in {
                "DECIDING",
                "WAITING_PLAN_SELECTION",
                "WAITING_FULL_APPROVAL",
                "WAITING_USER_REVIEW",
                "COMPLETED_MINI",
                "COMPLETED_FULL",
            }
        ),
    )
