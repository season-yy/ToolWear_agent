"""ToolWear 状态、审批、运行和证据的唯一 SQLite 写入口。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, JsonValue

from toolwear_agent.schemas import (
    AgentResult,
    AgentTask,
    ApprovalRecord,
    CandidateRecommendationSet,
    EvidenceRef,
    ExperimentRevision,
    ExperimentState,
    MemoryCase,
)
from toolwear_agent.schemas.agent import AgentName
from toolwear_agent.schemas.base import EntityId, utc_now
from toolwear_agent.schemas.experiment import ApprovalStatus, ExperimentStatus
from toolwear_agent.state.database import SQLiteDatabase
from toolwear_agent.state.models import (
    EntityNotFoundError,
    IdempotencyConflictError,
    RevisionLockedError,
    RunRecord,
    RunStatus,
    StateConflictError,
    StateTransitionEvent,
)
from toolwear_agent.state.state_machine import parse_state, revision_is_locked, validate_transition


ModelT = TypeVar("ModelT", bound=BaseModel)
_VOLATILE_FIELDS = {
    "created_at",
    "updated_at",
    "requested_at",
    "decided_at",
    "started_at",
    "completed_at",
    "last_event_sequence",
}
_AGENT_NAMES = {
    "ExperimentManagerAgent",
    "DataStewardAgent",
    "AlgorithmArchitectAgent",
    "CodeTrainingEngineerAgent",
    "EvaluationGovernorAgent",
    "ReportMemoryCuratorAgent",
}
_WAITING_OR_TERMINAL_STATES = {
    ExperimentStatus.BLOCKED_DATA,
    ExperimentStatus.WAITING_PLAN_SELECTION,
    ExperimentStatus.WAITING_FULL_APPROVAL,
    ExperimentStatus.WAITING_USER_REVIEW,
    ExperimentStatus.COMPLETED_MINI,
    ExperimentStatus.COMPLETED_FULL,
    ExperimentStatus.FAILED,
    ExperimentStatus.CANCELLED,
}


def _model_json(model: BaseModel) -> str:
    """生成可由对应 Pydantic 模型稳定恢复的 JSON。"""

    return model.model_dump_json()


def _strip_volatile(value: Any) -> Any:
    """幂等请求哈希忽略由服务端生成的时间与事件序号。"""

    if isinstance(value, Mapping):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in _VOLATILE_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_strip_volatile(item) for item in value]
    return value


def _request_hash(payload: Mapping[str, Any]) -> str:
    normalized = _strip_volatile(payload)
    canonical = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_hash(revision: ExperimentRevision) -> str:
    payload = {
        "pipeline": revision.pipeline.model_dump(mode="json"),
        "run_config": revision.run_config.model_dump(mode="json"),
    }
    return _request_hash(payload)


class SQLiteExperimentRepository:
    """集中维护数据库事务，业务模块不能直接散写 SQLite。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db = SQLiteDatabase(db_path)

    @property
    def db_path(self) -> Path:
        return self.db.path

    @property
    def fts5_enabled(self) -> bool:
        return self.db.fts5_enabled

    def initialize(self) -> None:
        self.db.initialize()

    def close(self) -> None:
        self.db.close()

    def health_info(self) -> dict[str, object]:
        """返回 API 健康检查需要的只读数据库信息。"""

        with self.db.read() as connection:
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        return {
            "path": str(self.db_path),
            "user_version": user_version,
            "integrity": integrity,
            "fts5_enabled": self.fts5_enabled,
        }

    def _replay(
        self,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str | None,
        operation: str,
        request_payload: Mapping[str, Any],
        response_model: type[ModelT],
    ) -> tuple[ModelT | None, str]:
        digest = _request_hash(request_payload)
        if idempotency_key is None:
            return None, digest
        row = connection.execute(
            "SELECT operation, request_hash, response_json FROM idempotency_records "
            "WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None, digest
        if row["operation"] != operation or row["request_hash"] != digest:
            raise IdempotencyConflictError(
                f"幂等键 {idempotency_key} 已用于另一份写请求。"
            )
        return response_model.model_validate_json(row["response_json"]), digest

    @staticmethod
    def _remember(
        connection: sqlite3.Connection,
        *,
        idempotency_key: str | None,
        operation: str,
        request_hash: str,
        response: BaseModel,
    ) -> None:
        if idempotency_key is None:
            return
        connection.execute(
            "INSERT INTO idempotency_records "
            "(idempotency_key, operation, request_hash, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                idempotency_key,
                operation,
                request_hash,
                _model_json(response),
                utc_now().isoformat(),
            ),
        )

    @staticmethod
    def _load_experiment(connection: sqlite3.Connection, experiment_id: str) -> ExperimentState:
        row = connection.execute(
            "SELECT payload_json FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"实验不存在：{experiment_id}")
        return ExperimentState.model_validate_json(row["payload_json"])

    @staticmethod
    def _save_experiment(connection: sqlite3.Connection, state: ExperimentState) -> None:
        connection.execute(
            "UPDATE experiments SET trace_id = ?, state = ?, revision = ?, "
            "pending_approval = ?, best_run_id = ?, updated_at = ?, payload_json = ? "
            "WHERE experiment_id = ?",
            (
                state.trace_id,
                state.state.value,
                state.revision,
                state.pending_approval,
                state.best_run_id,
                state.updated_at.isoformat(),
                _model_json(state),
                state.experiment_id,
            ),
        )

    @staticmethod
    def _append_state_event(
        connection: sqlite3.Connection,
        *,
        state: ExperimentState,
        before_state: ExperimentStatus | None,
        after_state: ExperimentStatus,
        actor: str,
        reason: str,
        evidence_ids: tuple[str, ...],
    ) -> StateTransitionEvent:
        event_id = f"event-{uuid4().hex}"
        created_at = utc_now()
        cursor = connection.execute(
            "INSERT INTO state_events "
            "(event_id, experiment_id, before_state, after_state, actor, trace_id, "
            "created_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                state.experiment_id,
                before_state.value if before_state is not None else None,
                after_state.value,
                actor,
                state.trace_id,
                created_at.isoformat(),
                "{}",
            ),
        )
        event = StateTransitionEvent(
            event_id=event_id,
            sequence=int(cursor.lastrowid),
            experiment_id=state.experiment_id,
            revision=state.revision,
            before_state=before_state,
            after_state=after_state,
            actor=actor,
            reason=reason,
            trace_id=state.trace_id,
            evidence_ids=evidence_ids,
            created_at=created_at,
        )
        connection.execute(
            "UPDATE state_events SET payload_json = ? WHERE sequence = ?",
            (_model_json(event), event.sequence),
        )
        return event

    def create_experiment(
        self,
        state: ExperimentState,
        *,
        actor: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> ExperimentState:
        """创建 DRAFT 快照和首条审计事件；重复请求安全重放。"""

        operation = "create_experiment"
        state_payload = state.model_dump(mode="json")
        # trace_id 是首次创建时由服务端生成的响应字段，不属于客户端幂等请求语义。
        state_payload.pop("trace_id", None)
        request_payload = {
            "state": state_payload,
            "actor": actor,
            "reason": reason,
        }
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=ExperimentState,
            )
            if replayed is not None:
                return replayed
            if state.state is not ExperimentStatus.DRAFT:
                raise StateConflictError("新实验必须从 DRAFT 状态创建。")
            if connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?", (state.experiment_id,)
            ).fetchone():
                raise StateConflictError(f"实验已存在：{state.experiment_id}")

            now = utc_now()
            initial = state.model_copy(
                update={"last_event_sequence": 0, "created_at": now, "updated_at": now}
            )
            connection.execute(
                "INSERT INTO experiments "
                "(experiment_id, trace_id, state, revision, pending_approval, best_run_id, "
                "updated_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    initial.experiment_id,
                    initial.trace_id,
                    initial.state.value,
                    initial.revision,
                    initial.pending_approval,
                    initial.best_run_id,
                    initial.updated_at.isoformat(),
                    _model_json(initial),
                ),
            )
            event = self._append_state_event(
                connection,
                state=initial,
                before_state=None,
                after_state=ExperimentStatus.DRAFT,
                actor=actor,
                reason=reason,
                evidence_ids=(),
            )
            created = initial.model_copy(
                update={"last_event_sequence": event.sequence, "updated_at": event.created_at}
            )
            self._save_experiment(connection, created)
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=created,
            )
            return created

    def get_experiment(self, experiment_id: str) -> ExperimentState:
        with self.db.read() as connection:
            return self._load_experiment(connection, experiment_id)

    def list_experiments(self, *, limit: int = 100) -> tuple[ExperimentState, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit 必须在 1 到 1000 之间。")
        with self.db.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM experiments ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(ExperimentState.model_validate_json(row["payload_json"]) for row in rows)

    def transition_state(
        self,
        experiment_id: str,
        target_state: ExperimentStatus | str,
        *,
        actor: str,
        reason: str,
        evidence_ids: tuple[EntityId, ...] = (),
        idempotency_key: str | None = None,
    ) -> ExperimentState:
        """原子更新状态快照并追加 before/after 事件。"""

        normalized_target = parse_state(target_state)
        operation = "transition_state"
        request_payload = {
            "experiment_id": experiment_id,
            "target_state": normalized_target.value,
            "actor": actor,
            "reason": reason,
            "evidence_ids": evidence_ids,
        }
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=ExperimentState,
            )
            if replayed is not None:
                return replayed
            current = self._load_experiment(connection, experiment_id)
            before, after = validate_transition(current.state, normalized_target)
            event = self._append_state_event(
                connection,
                state=current,
                before_state=before,
                after_state=after,
                actor=actor,
                reason=reason,
                evidence_ids=evidence_ids,
            )
            current_agent: AgentName | None = current.current_agent
            if after in _WAITING_OR_TERMINAL_STATES:
                current_agent = None
            elif actor in _AGENT_NAMES:
                current_agent = actor  # type: ignore[assignment]
            transitioned = current.model_copy(
                update={
                    "state": after,
                    "current_agent": current_agent,
                    "last_event_sequence": event.sequence,
                    "updated_at": event.created_at,
                }
            )
            self._save_experiment(connection, transitioned)
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=transitioned,
            )
            return transitioned

    def list_state_events(self, experiment_id: str) -> tuple[StateTransitionEvent, ...]:
        with self.db.read() as connection:
            self._load_experiment(connection, experiment_id)
            rows = connection.execute(
                "SELECT payload_json FROM state_events WHERE experiment_id = ? ORDER BY sequence",
                (experiment_id,),
            ).fetchall()
        return tuple(StateTransitionEvent.model_validate_json(row["payload_json"]) for row in rows)

    def create_revision(
        self,
        revision: ExperimentRevision,
        *,
        idempotency_key: str | None = None,
    ) -> ExperimentRevision:
        """新增不可变 revision，并原子切换实验的当前配置指针。"""

        operation = "create_revision"
        request_payload = {"revision": revision.model_dump(mode="json")}
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=ExperimentRevision,
            )
            if replayed is not None:
                return replayed
            state = self._load_experiment(connection, revision.experiment_id)
            if revision_is_locked(state.state):
                raise RevisionLockedError(
                    f"{state.state.value} 状态下不能切换当前 revision。"
                )
            row = connection.execute(
                "SELECT MAX(revision) AS latest FROM experiment_revisions WHERE experiment_id = ?",
                (revision.experiment_id,),
            ).fetchone()
            latest = row["latest"]
            expected = 1 if latest is None else int(latest) + 1
            if revision.revision != expected:
                raise StateConflictError(
                    f"下一 revision 应为 {expected}，收到 {revision.revision}。"
                )
            expected_parent = None if expected == 1 else expected - 1
            if revision.parent_revision != expected_parent:
                raise StateConflictError(
                    f"revision {expected} 的 parent_revision 应为 {expected_parent}。"
                )
            calculated_hash = _content_hash(revision)
            if revision.content_hash is not None and revision.content_hash != calculated_hash:
                raise StateConflictError("revision.content_hash 与配置内容不一致。")
            created = revision.model_copy(update={"content_hash": calculated_hash})
            connection.execute(
                "INSERT INTO experiment_revisions "
                "(experiment_id, revision, pipeline_id, content_hash, created_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    created.experiment_id,
                    created.revision,
                    created.pipeline.pipeline_id,
                    created.content_hash,
                    created.created_at.isoformat(),
                    _model_json(created),
                ),
            )
            updated_state = state.model_copy(
                update={
                    "revision": created.revision,
                    "selected_pipeline_ref": created.pipeline.pipeline_id,
                    "updated_at": utc_now(),
                }
            )
            self._save_experiment(connection, updated_state)
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=created,
            )
            return created

    def get_revision(self, experiment_id: str, revision: int) -> ExperimentRevision:
        with self.db.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM experiment_revisions "
                "WHERE experiment_id = ? AND revision = ?",
                (experiment_id, revision),
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"实验 {experiment_id} 不存在 revision {revision}。")
        return ExperimentRevision.model_validate_json(row["payload_json"])

    def save_recommendations(
        self,
        recommendations: CandidateRecommendationSet,
        *,
        idempotency_key: str | None = None,
    ) -> CandidateRecommendationSet:
        """保存候选集合并更新实验的最新候选指针。"""

        operation = "save_recommendations"
        request_payload = {"recommendations": recommendations.model_dump(mode="json")}
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=CandidateRecommendationSet,
            )
            if replayed is not None:
                return replayed
            state = self._load_experiment(connection, recommendations.experiment_id)
            if recommendations.revision != state.revision:
                raise StateConflictError("候选集合 revision 必须等于实验当前 revision。")
            connection.execute(
                "INSERT INTO candidate_recommendations "
                "(recommendation_id, experiment_id, revision, created_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    recommendations.recommendation_id,
                    recommendations.experiment_id,
                    recommendations.revision,
                    recommendations.created_at.isoformat(),
                    _model_json(recommendations),
                ),
            )
            self._save_experiment(
                connection,
                state.model_copy(
                    update={
                        "latest_recommendation_id": recommendations.recommendation_id,
                        "updated_at": utc_now(),
                    }
                ),
            )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=recommendations,
            )
            return recommendations

    def get_recommendations(self, recommendation_id: str) -> CandidateRecommendationSet:
        with self.db.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM candidate_recommendations WHERE recommendation_id = ?",
                (recommendation_id,),
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"候选集合不存在：{recommendation_id}")
        return CandidateRecommendationSet.model_validate_json(row["payload_json"])

    def get_latest_recommendations(self, experiment_id: str) -> CandidateRecommendationSet:
        state = self.get_experiment(experiment_id)
        if state.latest_recommendation_id is None:
            raise EntityNotFoundError(f"实验尚未生成候选：{experiment_id}")
        return self.get_recommendations(state.latest_recommendation_id)

    def create_approval(
        self,
        approval: ApprovalRecord,
        *,
        idempotency_key: str | None = None,
    ) -> ApprovalRecord:
        """创建真实 pending 审批，并在实验快照中留下等待指针。"""

        operation = "create_approval"
        request_payload = {"approval": approval.model_dump(mode="json")}
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=ApprovalRecord,
            )
            if replayed is not None:
                return replayed
            state = self._load_experiment(connection, approval.experiment_id)
            if approval.status is not ApprovalStatus.PENDING:
                raise StateConflictError("新审批必须是 pending 状态。")
            if approval.revision != state.revision:
                raise StateConflictError("审批 revision 必须等于实验当前 revision。")
            if state.pending_approval is not None:
                raise StateConflictError(f"实验仍有待处理审批：{state.pending_approval}")
            connection.execute(
                "INSERT INTO approvals "
                "(approval_id, experiment_id, revision, status, requested_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    approval.approval_id,
                    approval.experiment_id,
                    approval.revision,
                    approval.status.value,
                    approval.requested_at.isoformat(),
                    _model_json(approval),
                ),
            )
            self._save_experiment(
                connection,
                state.model_copy(
                    update={"pending_approval": approval.approval_id, "updated_at": utc_now()}
                ),
            )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=approval,
            )
            return approval

    def get_approval(self, approval_id: str) -> ApprovalRecord:
        with self.db.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"审批不存在：{approval_id}")
        return ApprovalRecord.model_validate_json(row["payload_json"])

    def decide_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus | str,
        decided_by: str,
        rationale: str,
        idempotency_key: str | None = None,
    ) -> ApprovalRecord:
        """审批只允许从 pending 一次性进入结束状态。"""

        normalized_status = ApprovalStatus(status)
        if normalized_status is ApprovalStatus.PENDING:
            raise StateConflictError("审批决定不能仍为 pending。")
        operation = "decide_approval"
        request_payload = {
            "approval_id": approval_id,
            "status": normalized_status.value,
            "decided_by": decided_by,
            "rationale": rationale,
        }
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=ApprovalRecord,
            )
            if replayed is not None:
                return replayed
            row = connection.execute(
                "SELECT payload_json FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise EntityNotFoundError(f"审批不存在：{approval_id}")
            pending = ApprovalRecord.model_validate_json(row["payload_json"])
            if pending.status is not ApprovalStatus.PENDING:
                raise StateConflictError(f"审批已经结束：{approval_id}")
            combined_rationale = pending.rationale
            if rationale:
                combined_rationale = f"{combined_rationale}\n审批决定：{rationale}".strip()
            decided = pending.model_copy(
                update={
                    "status": normalized_status,
                    "decided_by": decided_by,
                    "rationale": combined_rationale,
                    "decided_at": utc_now(),
                }
            )
            decided = ApprovalRecord.model_validate(decided.model_dump())
            connection.execute(
                "UPDATE approvals SET status = ?, payload_json = ? WHERE approval_id = ?",
                (decided.status.value, _model_json(decided), approval_id),
            )
            state = self._load_experiment(connection, pending.experiment_id)
            if state.pending_approval == approval_id:
                self._save_experiment(
                    connection,
                    state.model_copy(update={"pending_approval": None, "updated_at": utc_now()}),
                )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=decided,
            )
            return decided

    def create_run(
        self,
        run: RunRecord,
        *,
        idempotency_key: str | None = None,
    ) -> RunRecord:
        operation = "create_run"
        request_payload = {"run": run.model_dump(mode="json")}
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=RunRecord,
            )
            if replayed is not None:
                return replayed
            state = self._load_experiment(connection, run.experiment_id)
            if run.revision > state.revision:
                raise StateConflictError("运行不能引用尚未创建的 revision。")
            connection.execute(
                "INSERT INTO runs "
                "(run_id, experiment_id, revision, status, updated_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.experiment_id,
                    run.revision,
                    run.status.value,
                    run.updated_at.isoformat(),
                    _model_json(run),
                ),
            )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=run,
            )
            return run

    def get_run(self, run_id: str) -> RunRecord:
        with self.db.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"运行不存在：{run_id}")
        return RunRecord.model_validate_json(row["payload_json"])

    def list_runs(self, experiment_id: str) -> tuple[RunRecord, ...]:
        """按更新时间倒序返回一个实验的运行摘要。"""

        with self.db.read() as connection:
            self._load_experiment(connection, experiment_id)
            rows = connection.execute(
                "SELECT payload_json FROM runs WHERE experiment_id = ? ORDER BY updated_at DESC",
                (experiment_id,),
            ).fetchall()
        return tuple(RunRecord.model_validate_json(row["payload_json"]) for row in rows)

    def update_run_progress(
        self,
        run_id: str,
        *,
        progress: float,
        message: str,
        current_epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> RunRecord:
        """保存前端轮询需要的轻量进度，不改写终态结果。"""

        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise EntityNotFoundError(f"运行不存在：{run_id}")
            current = RunRecord.model_validate_json(row["payload_json"])
            if current.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                return current
            updated = current.model_copy(
                update={
                    "progress": progress,
                    "progress_message": message,
                    "current_epoch": (
                        current.current_epoch if current_epoch is None else current_epoch
                    ),
                    "total_epochs": current.total_epochs if total_epochs is None else total_epochs,
                    "updated_at": utc_now(),
                }
            )
            updated = RunRecord.model_validate(updated.model_dump())
            connection.execute(
                "UPDATE runs SET updated_at = ?, payload_json = ? WHERE run_id = ?",
                (updated.updated_at.isoformat(), _model_json(updated), run_id),
            )
            return updated

    def request_run_cancel(self, run_id: str) -> RunRecord:
        """持久化取消意图，Worker 会在下一个安全检查点停止。"""

        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise EntityNotFoundError(f"运行不存在：{run_id}")
            current = RunRecord.model_validate_json(row["payload_json"])
            if current.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
                return current
            updated = current.model_copy(
                update={
                    "cancel_requested": True,
                    "progress_message": "已请求取消，等待训练到达安全检查点。",
                    "updated_at": utc_now(),
                }
            )
            connection.execute(
                "UPDATE runs SET updated_at = ?, payload_json = ? WHERE run_id = ?",
                (updated.updated_at.isoformat(), _model_json(updated), run_id),
            )
            return updated

    def record_successful_run_budget(
        self,
        run_id: str,
        *,
        consumed_epochs: int,
    ) -> ExperimentState:
        """成功 Run 只记账一次，并按 validation Macro-F1 更新 best_run 指针。"""

        if consumed_epochs < 0:
            raise ValueError("consumed_epochs 不能小于 0。")
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise EntityNotFoundError(f"运行不存在：{run_id}")
            run = RunRecord.model_validate_json(row["payload_json"])
            if run.status is not RunStatus.SUCCEEDED:
                raise StateConflictError("只有 succeeded Run 可以计入实验预算。")
            state = self._load_experiment(connection, run.experiment_id)
            if run.budget_accounted:
                return state

            budget = state.budget
            if run.run_kind in {"mini_train", "smoke"}:
                budget = budget.model_copy(
                    update={"completed_mini_runs": budget.completed_mini_runs + 1}
                )
            elif run.run_kind == "full_train":
                budget = budget.model_copy(
                    update={"completed_full_runs": budget.completed_full_runs + 1}
                )
            budget = budget.model_copy(
                update={"consumed_epochs": budget.consumed_epochs + consumed_epochs}
            )
            budget = type(state.budget).model_validate(budget.model_dump())

            best_run_id = state.best_run_id
            current_score = float(run.result_summary.get("validation_macro_f1", -1.0))
            if best_run_id is None:
                best_run_id = run_id
            else:
                best_row = connection.execute(
                    "SELECT payload_json FROM runs WHERE run_id = ?", (best_run_id,)
                ).fetchone()
                best_score = -1.0
                if best_row is not None:
                    best_run = RunRecord.model_validate_json(best_row["payload_json"])
                    best_score = float(
                        best_run.result_summary.get("validation_macro_f1", -1.0)
                    )
                if current_score > best_score:
                    best_run_id = run_id
            updated = state.model_copy(
                update={
                    "budget": budget,
                    "best_run_id": best_run_id,
                    "updated_at": utc_now(),
                }
            )
            accounted_run = run.model_copy(
                update={
                    "budget_accounted": True,
                    "consumed_epochs": consumed_epochs,
                    "updated_at": utc_now(),
                }
            )
            accounted_run = RunRecord.model_validate(accounted_run.model_dump())
            connection.execute(
                "UPDATE runs SET updated_at = ?, payload_json = ? WHERE run_id = ?",
                (
                    accounted_run.updated_at.isoformat(),
                    _model_json(accounted_run),
                    run_id,
                ),
            )
            self._save_experiment(connection, updated)
            return updated

    def update_run_status(
        self,
        run_id: str,
        *,
        status: RunStatus | str,
        result_summary: dict[str, JsonValue] | None = None,
        artifact_uri: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunRecord:
        """更新运行摘要；终态记录不可再次改写。"""

        normalized_status = RunStatus(status)
        operation = "update_run_status"
        request_payload = {
            "run_id": run_id,
            "status": normalized_status.value,
            "result_summary": result_summary or {},
            "artifact_uri": artifact_uri,
            "error_code": error_code,
            "error_message": error_message,
        }
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=RunRecord,
            )
            if replayed is not None:
                return replayed
            row = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise EntityNotFoundError(f"运行不存在：{run_id}")
            current = RunRecord.model_validate_json(row["payload_json"])
            terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
            if current.status in terminal:
                raise StateConflictError(f"终态运行不能再次修改：{run_id}")
            allowed = {
                RunStatus.QUEUED: {RunStatus.RUNNING, *terminal},
                RunStatus.RUNNING: terminal,
            }
            if normalized_status not in allowed[current.status]:
                raise StateConflictError(
                    f"不允许从 {current.status.value} 转换到 {normalized_status.value}。"
                )
            now = utc_now()
            started_at = current.started_at
            if normalized_status in {RunStatus.RUNNING, *terminal} and started_at is None:
                started_at = now
            completed_at = now if normalized_status in terminal else None
            updated = current.model_copy(
                update={
                    "status": normalized_status,
                    "progress": 1.0 if normalized_status is RunStatus.SUCCEEDED else current.progress,
                    "progress_message": {
                        RunStatus.SUCCEEDED: "训练完成。",
                        RunStatus.FAILED: "训练失败。",
                        RunStatus.CANCELLED: "训练已取消。",
                    }.get(normalized_status, current.progress_message),
                    "result_summary": result_summary or current.result_summary,
                    "artifact_uri": artifact_uri or current.artifact_uri,
                    "error_code": error_code,
                    "error_message": error_message,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "updated_at": now,
                }
            )
            updated = RunRecord.model_validate(updated.model_dump())
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ?, payload_json = ? WHERE run_id = ?",
                (updated.status.value, updated.updated_at.isoformat(), _model_json(updated), run_id),
            )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=updated,
            )
            return updated

    def register_evidence(
        self,
        evidence: EvidenceRef,
        *,
        idempotency_key: str | None = None,
    ) -> EvidenceRef:
        operation = "register_evidence"
        request_payload = {"evidence": evidence.model_dump(mode="json")}
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=EvidenceRef,
            )
            if replayed is not None:
                return replayed
            self._load_experiment(connection, evidence.experiment_id)
            if evidence.run_id is not None:
                run_row = connection.execute(
                    "SELECT experiment_id FROM runs WHERE run_id = ?", (evidence.run_id,)
                ).fetchone()
                if run_row is None or run_row["experiment_id"] != evidence.experiment_id:
                    raise StateConflictError("EvidenceRef.run_id 不属于指定实验。")
            connection.execute(
                "INSERT INTO evidence_refs "
                "(evidence_id, experiment_id, run_id, kind, created_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    evidence.evidence_id,
                    evidence.experiment_id,
                    evidence.run_id,
                    evidence.kind.value,
                    evidence.created_at.isoformat(),
                    _model_json(evidence),
                ),
            )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=evidence,
            )
            return evidence

    def list_evidence(self, experiment_id: str) -> tuple[EvidenceRef, ...]:
        with self.db.read() as connection:
            self._load_experiment(connection, experiment_id)
            rows = connection.execute(
                "SELECT payload_json FROM evidence_refs WHERE experiment_id = ? "
                "ORDER BY created_at, evidence_id",
                (experiment_id,),
            ).fetchall()
        return tuple(EvidenceRef.model_validate_json(row["payload_json"]) for row in rows)

    def get_evidence(self, evidence_id: str) -> EvidenceRef:
        """按稳定 ID 恢复单条证据引用。"""

        with self.db.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM evidence_refs WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"EvidenceRef 不存在：{evidence_id}")
        return EvidenceRef.model_validate_json(row["payload_json"])

    def save_agent_task(
        self,
        task: AgentTask,
        *,
        idempotency_key: str | None = None,
    ) -> AgentTask:
        operation = "save_agent_task"
        request_payload = {"task": task.model_dump(mode="json")}
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=AgentTask,
            )
            if replayed is not None:
                return replayed
            state = self._load_experiment(connection, task.experiment_id)
            if task.revision > state.revision:
                raise StateConflictError("AgentTask 不能引用尚未创建的 revision。")
            connection.execute(
                "INSERT INTO agent_tasks "
                "(task_id, experiment_id, revision, created_at, payload_json) VALUES (?, ?, ?, ?, ?)",
                (
                    task.task_id,
                    task.experiment_id,
                    task.revision,
                    task.created_at.isoformat(),
                    _model_json(task),
                ),
            )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=task,
            )
            return task

    def get_agent_task(self, task_id: str) -> AgentTask:
        """按稳定 task_id 恢复 AgentTask。"""

        with self.db.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"AgentTask 不存在：{task_id}")
        return AgentTask.model_validate_json(row["payload_json"])

    def list_agent_tasks(self, experiment_id: str) -> tuple[AgentTask, ...]:
        """按创建顺序列出一个实验的 AgentTask。"""

        with self.db.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM agent_tasks WHERE experiment_id = ? "
                "ORDER BY created_at, task_id",
                (experiment_id,),
            ).fetchall()
        return tuple(AgentTask.model_validate_json(row["payload_json"]) for row in rows)

    def save_agent_result(
        self,
        result: AgentResult,
        *,
        idempotency_key: str | None = None,
    ) -> AgentResult:
        operation = "save_agent_result"
        request_payload = {"result": result.model_dump(mode="json")}
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=AgentResult,
            )
            if replayed is not None:
                return replayed
            task_row = connection.execute(
                "SELECT payload_json FROM agent_tasks WHERE task_id = ?", (result.task_id,)
            ).fetchone()
            if task_row is None:
                raise EntityNotFoundError(f"AgentTask 不存在：{result.task_id}")
            task = AgentTask.model_validate_json(task_row["payload_json"])
            if task.assigned_to != result.agent_name:
                raise StateConflictError("AgentResult.agent_name 与任务接收者不一致。")
            connection.execute(
                "INSERT INTO agent_results (task_id, status, completed_at, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    result.task_id,
                    result.status.value,
                    result.completed_at.isoformat(),
                    _model_json(result),
                ),
            )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=result,
            )
            return result

    def get_agent_result(self, task_id: str) -> AgentResult:
        with self.db.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM agent_results WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise EntityNotFoundError(f"AgentResult 不存在：{task_id}")
        return AgentResult.model_validate_json(row["payload_json"])

    def save_memory_case(
        self,
        memory: MemoryCase,
        *,
        idempotency_key: str | None = None,
    ) -> MemoryCase:
        operation = "save_memory_case"
        request_payload = {"memory": memory.model_dump(mode="json")}
        searchable_text = " ".join(
            (
                memory.problem,
                memory.intervention,
                memory.outcome,
                memory.summary,
                *memory.tags,
            )
        )
        with self.db.transaction() as connection:
            replayed, digest = self._replay(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_payload=request_payload,
                response_model=MemoryCase,
            )
            if replayed is not None:
                return replayed
            connection.execute(
                "INSERT INTO memory_cases "
                "(memory_id, dataset_id, task_type, searchable_text, created_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory.memory_id,
                    memory.dataset_id,
                    memory.task_type,
                    searchable_text,
                    memory.created_at.isoformat(),
                    _model_json(memory),
                ),
            )
            if self.fts5_enabled:
                connection.execute(
                    "INSERT INTO memory_cases_fts (memory_id, searchable_text) VALUES (?, ?)",
                    (memory.memory_id, searchable_text),
                )
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                operation=operation,
                request_hash=digest,
                response=memory,
            )
            return memory

    def search_memory(
        self,
        query: str,
        *,
        dataset_id: str | None = None,
        task_type: str | None = None,
        limit: int = 10,
    ) -> tuple[MemoryCase, ...]:
        """优先使用 FTS5；不可用或无结果时退化为参数化 LIKE。"""

        normalized_query = query.strip()
        if not normalized_query:
            return ()
        if limit < 1 or limit > 100:
            raise ValueError("limit 必须在 1 到 100 之间。")
        filters: list[str] = []
        parameters: list[Any] = []
        if dataset_id is not None:
            filters.append("m.dataset_id = ?")
            parameters.append(dataset_id)
        if task_type is not None:
            filters.append("m.task_type = ?")
            parameters.append(task_type)
        filter_sql = f" AND {' AND '.join(filters)}" if filters else ""

        with self.db.read() as connection:
            rows: list[sqlite3.Row] = []
            if self.fts5_enabled:
                escaped = normalized_query.replace('"', '""')
                try:
                    rows = connection.execute(
                        "SELECT m.payload_json FROM memory_cases_fts f "
                        "JOIN memory_cases m ON m.memory_id = f.memory_id "
                        f"WHERE memory_cases_fts MATCH ?{filter_sql} "
                        "ORDER BY bm25(memory_cases_fts) LIMIT ?",
                        (f'"{escaped}"', *parameters, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                rows = connection.execute(
                    "SELECT m.payload_json FROM memory_cases m "
                    f"WHERE m.searchable_text LIKE ?{filter_sql} "
                    "ORDER BY m.created_at DESC LIMIT ?",
                    (f"%{normalized_query}%", *parameters, limit),
                ).fetchall()
        return tuple(MemoryCase.model_validate_json(row["payload_json"]) for row in rows)
