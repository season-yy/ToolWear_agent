"""master spec 实验状态机的纯逻辑实现。"""

from __future__ import annotations

from toolwear_agent.schemas.experiment import ExperimentStatus
from toolwear_agent.state.models import InvalidStateTransitionError


ALLOWED_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.DRAFT: frozenset({ExperimentStatus.DATA_VALIDATING}),
    ExperimentStatus.DATA_VALIDATING: frozenset(
        {ExperimentStatus.BLOCKED_DATA, ExperimentStatus.WAITING_PLAN_SELECTION}
    ),
    ExperimentStatus.BLOCKED_DATA: frozenset({ExperimentStatus.DATA_VALIDATING}),
    ExperimentStatus.WAITING_PLAN_SELECTION: frozenset({ExperimentStatus.PIPELINE_VALIDATING}),
    ExperimentStatus.PIPELINE_VALIDATING: frozenset(
        {ExperimentStatus.WAITING_PLAN_SELECTION, ExperimentStatus.CODE_PREPARING}
    ),
    ExperimentStatus.CODE_PREPARING: frozenset(
        {ExperimentStatus.FAILED, ExperimentStatus.MINI_TRAINING}
    ),
    ExperimentStatus.MINI_TRAINING: frozenset(
        {ExperimentStatus.EVALUATING, ExperimentStatus.CANCELLED, ExperimentStatus.FAILED}
    ),
    ExperimentStatus.EVALUATING: frozenset({ExperimentStatus.DECIDING}),
    ExperimentStatus.DECIDING: frozenset(
        {
            ExperimentStatus.MINI_TRAINING,
            ExperimentStatus.WAITING_PLAN_SELECTION,
            ExperimentStatus.WAITING_FULL_APPROVAL,
            ExperimentStatus.COMPLETED_MINI,
        }
    ),
    ExperimentStatus.WAITING_FULL_APPROVAL: frozenset(
        {ExperimentStatus.FULL_TRAINING, ExperimentStatus.COMPLETED_MINI}
    ),
    ExperimentStatus.FULL_TRAINING: frozenset({ExperimentStatus.EVALUATING_FULL}),
    ExperimentStatus.EVALUATING_FULL: frozenset({ExperimentStatus.WAITING_USER_REVIEW}),
    ExperimentStatus.WAITING_USER_REVIEW: frozenset(
        {ExperimentStatus.COMPLETED_FULL, ExperimentStatus.DECIDING}
    ),
    ExperimentStatus.COMPLETED_MINI: frozenset(),
    ExperimentStatus.COMPLETED_FULL: frozenset(),
    ExperimentStatus.FAILED: frozenset(),
    ExperimentStatus.CANCELLED: frozenset(),
}


REVISION_LOCKED_STATES = frozenset(
    {
        ExperimentStatus.CODE_PREPARING,
        ExperimentStatus.MINI_TRAINING,
        ExperimentStatus.EVALUATING,
        ExperimentStatus.FULL_TRAINING,
        ExperimentStatus.EVALUATING_FULL,
    }
)


def parse_state(value: ExperimentStatus | str) -> ExperimentStatus:
    """把 API 文本或枚举统一为严格状态枚举。"""

    if isinstance(value, ExperimentStatus):
        return value
    try:
        return ExperimentStatus(value)
    except ValueError as exc:
        raise InvalidStateTransitionError(f"未知实验状态：{value}") from exc


def validate_transition(
    before: ExperimentStatus | str,
    after: ExperimentStatus | str,
) -> tuple[ExperimentStatus, ExperimentStatus]:
    """验证一条转换边并返回规范化状态。"""

    normalized_before = parse_state(before)
    normalized_after = parse_state(after)
    if normalized_after not in ALLOWED_TRANSITIONS[normalized_before]:
        raise InvalidStateTransitionError(
            f"不允许从 {normalized_before.value} 转换到 {normalized_after.value}。"
        )
    return normalized_before, normalized_after


def revision_is_locked(state: ExperimentStatus | str) -> bool:
    """判断当前状态是否禁止切换 revision 指针。"""

    return parse_state(state) in REVISION_LOCKED_STATES
