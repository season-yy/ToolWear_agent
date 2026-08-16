"""本机后台训练队列、进度、取消和中断恢复。"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock

from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import EvidenceRef, ExperimentRevision, TrainingDataRef
from toolwear_agent.schemas.experiment import ExperimentStatus
from toolwear_agent.services.errors import InvalidWorkflowStateError, TrainingCancelledError
from toolwear_agent.state import RunRecord, RunStatus, SQLiteExperimentRepository
from toolwear_agent.training.service import TrainingService


class TrainingJobService:
    """用单 Worker 串行使用本机 GPU，并把进度写回 SQLite。"""

    def __init__(self, settings: Settings, repository: SQLiteExperimentRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.training_service = TrainingService(settings)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="toolwear-train")
        self._lock = RLock()
        self._futures: dict[str, Future[None]] = {}
        self._cancel_events: dict[str, Event] = {}

    def close(self) -> None:
        """API 正常关闭时等待当前训练收尾，避免关闭数据库后线程仍写入。"""

        self._executor.shutdown(wait=True, cancel_futures=False)

    def submit(
        self,
        revision: ExperimentRevision,
        data_ref: TrainingDataRef,
        *,
        idempotency_key: str | None,
    ) -> RunRecord:
        """幂等创建 Run，并且同一进程内只提交一次 Future。"""

        state = self.repository.get_experiment(revision.experiment_id)
        if state.state not in {ExperimentStatus.CODE_PREPARING, ExperimentStatus.MINI_TRAINING}:
            raise InvalidWorkflowStateError(
                f"{state.state.value} 状态下不能启动小样本训练。"
            )
        run_config = revision.run_config
        if state.budget.completed_mini_runs >= state.budget.max_mini_runs:
            raise InvalidWorkflowStateError("小样本训练次数预算已经耗尽。")
        projected_epochs = state.budget.consumed_epochs + run_config.epochs
        if projected_epochs > state.budget.max_total_epochs:
            raise InvalidWorkflowStateError(
                f"本次训练会使累计 epoch 达到 {projected_epochs}，"
                f"超过预算上限 {state.budget.max_total_epochs}。"
            )
        run = RunRecord(
            run_id=run_config.run_id,
            experiment_id=run_config.experiment_id,
            revision=run_config.revision,
            pipeline_id=run_config.pipeline_id,
            run_kind=run_config.run_kind.value,
            total_epochs=run_config.epochs,
        )
        persisted = self.repository.create_run(
            run,
            idempotency_key=(f"{idempotency_key}:run" if idempotency_key else None),
        )
        current_run = self.repository.get_run(run.run_id)
        if current_run.status is not RunStatus.QUEUED:
            return current_run
        if state.state is ExperimentStatus.CODE_PREPARING:
            self.repository.transition_state(
                state.experiment_id,
                ExperimentStatus.MINI_TRAINING,
                actor="CodeTrainingEngineerAgent",
                reason="代码和配置预检通过，后台训练任务已入队。",
                idempotency_key=(f"{idempotency_key}:state" if idempotency_key else None),
            )
        with self._lock:
            existing = self._futures.get(run.run_id)
            if existing is None or existing.done():
                cancel_event = Event()
                self._cancel_events[run.run_id] = cancel_event
                self._futures[run.run_id] = self._executor.submit(
                    self._execute,
                    revision,
                    data_ref,
                    cancel_event,
                )
        return persisted

    def _progress_callback(self, run_id: str, cancel_event: Event):
        def callback(event: str, payload: dict[str, object]) -> None:
            if cancel_event.is_set():
                raise TrainingCancelledError("用户请求取消训练。")
            current_epoch = int(payload.get("epoch", 0) or 0)
            total_epochs = int(payload.get("epochs", 0) or 0)
            if event == "training_started":
                progress = 0.20
                message = "模型训练已开始。"
            elif event == "epoch_completed":
                denominator = max(total_epochs, 1)
                progress = 0.20 + 0.65 * current_epoch / denominator
                message = f"已完成 epoch {current_epoch}/{denominator}。"
            elif event == "training_completed":
                progress = 0.90
                message = "训练结束，正在归档证据。"
            elif event == "windows_loaded":
                progress = 0.15
                message = "训练和验证窗口已加载。"
            elif event == "data_validated":
                progress = 0.08
                message = "split、样本和泄漏证据校验通过。"
            else:
                progress = 0.03
                message = f"正在执行：{event}"
            self.repository.update_run_progress(
                run_id,
                progress=min(progress, 0.99),
                message=message,
                current_epoch=current_epoch or None,
                total_epochs=total_epochs or None,
            )

        return callback

    def _register_training_evidence(self, evidence_index_file: Path) -> None:
        payload = json.loads(evidence_index_file.read_text(encoding="utf-8"))
        for raw_evidence in payload.get("evidence", []):
            evidence = EvidenceRef.model_validate(raw_evidence)
            self.repository.register_evidence(
                evidence,
                idempotency_key=f"training-evidence:{evidence.evidence_id}",
            )

    def _transition_if_current(
        self,
        experiment_id: str,
        expected: ExperimentStatus,
        target: ExperimentStatus,
        *,
        reason: str,
        idempotency_key: str,
    ) -> None:
        state = self.repository.get_experiment(experiment_id)
        if state.state is expected:
            self.repository.transition_state(
                experiment_id,
                target,
                actor="CodeTrainingEngineerAgent",
                reason=reason,
                idempotency_key=idempotency_key,
            )

    def _execute(
        self,
        revision: ExperimentRevision,
        data_ref: TrainingDataRef,
        cancel_event: Event,
    ) -> None:
        run_id = revision.run_config.run_id
        try:
            self.repository.update_run_status(run_id, status=RunStatus.RUNNING)
            result = self.training_service.train(
                pipeline=revision.pipeline,
                run_config=revision.run_config,
                data_ref=data_ref,
                event_sink=self._progress_callback(run_id, cancel_event),
            )
            if cancel_event.is_set():
                raise TrainingCancelledError("用户请求取消训练。")
            self._register_training_evidence(result.artifacts.evidence_index_file)
            validation = next(
                item for item in result.evaluation.metrics if item.split.value == "validation"
            )
            self.repository.update_run_status(
                run_id,
                status=RunStatus.SUCCEEDED,
                artifact_uri=str(result.artifacts.run_dir),
                result_summary={
                    "validation_macro_f1": validation.macro_f1,
                    "validation_balanced_accuracy": validation.balanced_accuracy,
                    "result_file": str(result.artifacts.result_file),
                    "evidence_index_file": str(result.artifacts.evidence_index_file),
                    "resolved_device": result.runtime.resolved_device,
                },
                idempotency_key=f"{run_id}:succeeded",
            )
            self.repository.record_successful_run_budget(
                run_id,
                consumed_epochs=len(result.epoch_history),
            )
            self._transition_if_current(
                revision.experiment_id,
                ExperimentStatus.MINI_TRAINING,
                ExperimentStatus.EVALUATING,
                reason="真实训练结束，进入 validation 评估。",
                idempotency_key=f"{run_id}:evaluating",
            )
        except TrainingCancelledError:
            self.repository.update_run_status(
                run_id,
                status=RunStatus.CANCELLED,
                idempotency_key=f"{run_id}:cancelled",
            )
            self._transition_if_current(
                revision.experiment_id,
                ExperimentStatus.MINI_TRAINING,
                ExperimentStatus.CANCELLED,
                reason="用户取消了训练。",
                idempotency_key=f"{run_id}:state-cancelled",
            )
        except Exception as exc:  # 后台线程必须把失败落库，不能静默退出
            self.repository.update_run_status(
                run_id,
                status=RunStatus.FAILED,
                error_code="TRAINING_FAILED",
                error_message=str(exc),
                idempotency_key=f"{run_id}:failed",
            )
            self._transition_if_current(
                revision.experiment_id,
                ExperimentStatus.MINI_TRAINING,
                ExperimentStatus.FAILED,
                reason=f"训练失败：{type(exc).__name__}: {exc}",
                idempotency_key=f"{run_id}:state-failed",
            )
        finally:
            with self._lock:
                self._cancel_events.pop(run_id, None)

    def request_cancel(self, experiment_id: str) -> RunRecord:
        """请求取消该实验最近的活动 Run。"""

        active = [
            run
            for run in self.repository.list_runs(experiment_id)
            if run.status in {RunStatus.QUEUED, RunStatus.RUNNING}
        ]
        if not active:
            raise InvalidWorkflowStateError("当前实验没有可取消的训练 Run。")
        run = active[0]
        persisted = self.repository.request_run_cancel(run.run_id)
        with self._lock:
            cancel_event = self._cancel_events.get(run.run_id)
            if cancel_event is not None:
                cancel_event.set()
        return persisted
