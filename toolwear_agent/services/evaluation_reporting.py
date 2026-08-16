"""validation 评估、停止治理与 Markdown 报告服务。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import (
    DecisionRecord,
    EvaluationDiagnosis,
    EvidenceRef,
    TrainingRunResult,
)
from toolwear_agent.schemas.api import DecisionRequest, ExperimentActionResponse
from toolwear_agent.schemas.experiment import DecisionAction, ExperimentStatus
from toolwear_agent.services.evaluation_diagnosis import (
    DefaultDiagnosisProvider,
    DiagnosisProvider,
    build_evaluation_facts,
)
from toolwear_agent.services.errors import InvalidWorkflowStateError
from toolwear_agent.services.report_renderer import render_experiment_report
from toolwear_agent.state import (
    EntityNotFoundError,
    RunRecord,
    RunStatus,
    SQLiteExperimentRepository,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_text(path: Path, content: str) -> Path:
    """原子写入报告，避免进程中断留下不完整 Markdown。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


class EvaluationReportingService:
    """只依据 validation 和真实 Run 证据产生评估、决策与报告。"""

    def __init__(
        self,
        settings: Settings,
        repository: SQLiteExperimentRepository,
        *,
        diagnosis_provider: DiagnosisProvider | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.path_resolver = PathResolver(settings)
        self.diagnosis_provider = diagnosis_provider or DefaultDiagnosisProvider(settings)

    def _best_or_latest_run(self, experiment_id: str) -> RunRecord:
        state = self.repository.get_experiment(experiment_id)
        if state.best_run_id is not None:
            run = self.repository.get_run(state.best_run_id)
            if run.status is RunStatus.SUCCEEDED:
                return run
        succeeded = [
            run
            for run in self.repository.list_runs(experiment_id)
            if run.status is RunStatus.SUCCEEDED
        ]
        if not succeeded:
            raise InvalidWorkflowStateError("实验没有可评估的 succeeded Run。")
        return succeeded[0]

    @staticmethod
    def _load_training_result(run: RunRecord) -> TrainingRunResult:
        result_file = run.result_summary.get("result_file")
        if not isinstance(result_file, str) or not result_file:
            raise FileNotFoundError("Run 摘要缺少 result_file。")
        path = Path(result_file)
        if not path.is_file():
            raise FileNotFoundError(f"训练结果文件不存在：{path}")
        return TrainingRunResult.model_validate_json(path.read_text(encoding="utf-8"))

    def _register_evidence(
        self,
        *,
        evidence_id: str,
        experiment_id: str,
        run_id: str | None,
        kind: str,
        path: Path,
        description: str,
        idempotency_key: str | None,
        created_by: str | None = None,
    ) -> EvidenceRef:
        evidence = EvidenceRef(
            evidence_id=evidence_id,
            experiment_id=experiment_id,
            run_id=run_id,
            kind=kind,
            uri=str(path),
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
            media_type="text/markdown" if path.suffix.lower() == ".md" else "application/json",
            description=description,
            created_by=(
                created_by
                or (
                    "EvaluationGovernorAgent"
                    if kind != "report"
                    else "ReportMemoryCuratorAgent"
                )
            ),
        )
        try:
            return self.repository.get_evidence(evidence_id)
        except EntityNotFoundError:
            return self.repository.register_evidence(
                evidence,
                idempotency_key=idempotency_key,
            )

    def _existing_evidence(self, evidence_id: str) -> EvidenceRef | None:
        try:
            return self.repository.get_evidence(evidence_id)
        except EntityNotFoundError:
            return None

    def _diagnosis_evidence(self, experiment_id: str, run_id: str) -> tuple[EvidenceRef, ...]:
        """按创建顺序返回同一 Run 的不可变诊断版本。"""

        return tuple(
            item
            for item in self.repository.list_evidence(experiment_id)
            if item.run_id == run_id
            and item.description == "EvaluationGovernorAgent 的结构化 LLM 诊断"
        )

    def _latest_diagnosis(
        self,
        experiment_id: str,
        run_id: str,
    ) -> tuple[EvaluationDiagnosis, EvidenceRef] | None:
        versions = self._diagnosis_evidence(experiment_id, run_id)
        if not versions:
            return None
        evidence = max(versions, key=lambda item: item.created_at)
        path = self.path_resolver.assert_within(
            evidence.uri,
            (self.settings.ai_infra_root,),
        )
        if not path.is_file():
            raise FileNotFoundError(f"诊断证据文件不存在：{path}")
        return (
            EvaluationDiagnosis.model_validate_json(path.read_text(encoding="utf-8")),
            evidence,
        )

    @staticmethod
    def _decision_target(action: DecisionAction) -> ExperimentStatus:
        targets = {
            DecisionAction.ADJUST_PARAMETERS: ExperimentStatus.WAITING_PLAN_SELECTION,
            DecisionAction.STOP: ExperimentStatus.COMPLETED_MINI,
            DecisionAction.CHANGE_PIPELINE: ExperimentStatus.WAITING_PLAN_SELECTION,
            DecisionAction.APPROVE_FULL_TRAIN: ExperimentStatus.WAITING_FULL_APPROVAL,
        }
        try:
            return targets[action]
        except KeyError as exc:  # pragma: no cover - P0 请求契约只产生三种动作
            raise ValueError(f"P0 不支持决策动作：{action.value}") from exc

    def evaluate(
        self,
        experiment_id: str,
        *,
        rationale: str,
        idempotency_key: str | None,
        force_refresh: bool = False,
    ) -> ExperimentActionResponse:
        state = self.repository.get_experiment(experiment_id)
        if state.state not in {ExperimentStatus.EVALUATING, ExperimentStatus.DECIDING}:
            raise InvalidWorkflowStateError(
                f"{state.state.value} 状态下不能执行 validation 评估。"
            )
        run = self._best_or_latest_run(experiment_id)
        result = self._load_training_result(run)
        train = next(
            (item for item in result.evaluation.metrics if item.split.value == "train"),
            None,
        )
        validation = next(
            item for item in result.evaluation.metrics if item.split.value == "validation"
        )
        revision = self.repository.get_revision(experiment_id, run.revision)
        evidence_id = f"{run.run_id}-evaluation-summary"
        evidence = self._existing_evidence(evidence_id)
        facts = build_evaluation_facts(
            experiment_id=experiment_id,
            run_id=run.run_id,
            pipeline_id=run.pipeline_id,
            train=train,
            validation=validation,
            class_labels=result.class_labels,
            epoch_history=result.epoch_history,
            module_ids=revision.pipeline.module_ids,
            completed_mini_runs=state.budget.completed_mini_runs,
            max_mini_runs=state.budget.max_mini_runs,
            source_evidence_ids=(evidence_id, *result.evaluation.evidence_ids),
        )
        if evidence is None:
            output_file = _write_json(
                Path(result.artifacts.run_dir) / "evaluation_summary.json",
                {
                    "schema_version": "1.0",
                    "experiment_id": experiment_id,
                    "run_id": run.run_id,
                    "basis_split": "validation",
                    "final_test": False,
                    "metrics": validation.model_dump(mode="json"),
                    "class_labels": result.class_labels,
                    "facts": facts.model_dump(mode="json"),
                    "rationale": rationale,
                },
            )
            evidence = self._register_evidence(
                evidence_id=evidence_id,
                experiment_id=experiment_id,
                run_id=run.run_id,
                kind="metrics",
                path=output_file,
                description="仅基于 validation 的结构化评估摘要",
                idempotency_key=(
                    f"{idempotency_key}:summary" if idempotency_key else None
                ),
            )

        existing_diagnosis = self._latest_diagnosis(experiment_id, run.run_id)
        call_evidence: EvidenceRef | None = None
        if existing_diagnosis is not None and not force_refresh:
            diagnosis, diagnosis_evidence = existing_diagnosis
            call_evidence_id = diagnosis_evidence.evidence_id.replace(
                "evaluation-diagnosis",
                "evaluation-llm-call",
            )
            call_evidence = self._existing_evidence(call_evidence_id)
        else:
            attempt = len(self._diagnosis_evidence(experiment_id, run.run_id)) + 1
            diagnosis_evidence_id = f"{run.run_id}-evaluation-diagnosis-v{attempt}"
            diagnosis_file = (
                Path(result.artifacts.run_dir) / f"evaluation_diagnosis_v{attempt}.json"
            )
            call_evidence_id = f"{run.run_id}-evaluation-llm-call-v{attempt}"
            call_file = (
                Path(result.artifacts.run_dir) / f"evaluation_llm_call_v{attempt}.json"
            )
            diagnosis = self.diagnosis_provider.diagnose(facts)
            _write_json(diagnosis_file, diagnosis.model_dump(mode="json"))
            _write_json(call_file, diagnosis.llm_call.model_dump(mode="json"))
            call_evidence = self._register_evidence(
                evidence_id=call_evidence_id,
                experiment_id=experiment_id,
                run_id=run.run_id,
                kind="log",
                path=call_file,
                description="EvaluationGovernorAgent LLM 调用审计",
                idempotency_key=(
                    f"{idempotency_key}:llm-call-v{attempt}" if idempotency_key else None
                ),
            )
            diagnosis_evidence = self._register_evidence(
                evidence_id=diagnosis_evidence_id,
                experiment_id=experiment_id,
                run_id=run.run_id,
                kind="report",
                path=diagnosis_file,
                description="EvaluationGovernorAgent 的结构化 LLM 诊断",
                idempotency_key=(
                    f"{idempotency_key}:diagnosis-v{attempt}" if idempotency_key else None
                ),
                created_by="EvaluationGovernorAgent",
            )
        evidence_bundle = tuple(
            item for item in (evidence, diagnosis_evidence, call_evidence) if item is not None
        )
        if state.state is ExperimentStatus.EVALUATING:
            state = self.repository.transition_state(
                experiment_id,
                ExperimentStatus.DECIDING,
                actor="EvaluationGovernorAgent",
                reason="validation 评估完成，进入受预算约束的决策。",
                evidence_ids=tuple(item.evidence_id for item in evidence_bundle),
                idempotency_key=(f"{idempotency_key}:state" if idempotency_key else None),
            )
        return ExperimentActionResponse(
            operation="evaluate",
            summary=(
                "validation 指标与 LLM 诊断完成；final test 仍未运行。"
                if not diagnosis.llm_call.used_fallback
                else "validation 指标完成；LLM 不可用，已显式使用规则诊断。"
            ),
            state=state,
            payload={
                "run_id": run.run_id,
                "basis_split": "validation",
                "final_test": False,
                "metrics": validation.model_dump(mode="json"),
                "diagnosis": diagnosis.model_dump(mode="json"),
            },
            evidence=evidence_bundle,
        )

    def decide(
        self,
        experiment_id: str,
        request: DecisionRequest,
        *,
        idempotency_key: str | None,
    ) -> ExperimentActionResponse:
        state = self.repository.get_experiment(experiment_id)
        if state.state not in {
            ExperimentStatus.DECIDING,
            ExperimentStatus.WAITING_FULL_APPROVAL,
            ExperimentStatus.COMPLETED_MINI,
            ExperimentStatus.WAITING_PLAN_SELECTION,
        }:
            raise InvalidWorkflowStateError(f"{state.state.value} 状态下不能形成实验决策。")
        run = self._best_or_latest_run(experiment_id)
        evidence_id = f"{run.run_id}-decision"
        decision_file = self.path_resolver.run_path(
            experiment_id,
            run.revision,
            run.run_id,
        ) / "decision.json"
        existing_evidence = self._existing_evidence(evidence_id)
        if existing_evidence is not None and decision_file.is_file():
            decision = DecisionRecord.model_validate_json(
                decision_file.read_text(encoding="utf-8")
            )
            target = self._decision_target(decision.action)
            if state.state is ExperimentStatus.DECIDING:
                state = self.repository.transition_state(
                    experiment_id,
                    target,
                    actor="EvaluationGovernorAgent",
                    reason=decision.reason,
                    evidence_ids=(existing_evidence.evidence_id,),
                    idempotency_key=(
                        f"{idempotency_key}:state" if idempotency_key else None
                    ),
                )
            return ExperimentActionResponse(
                operation="decision",
                summary=decision.reason,
                state=state,
                payload={"decision": decision.model_dump(mode="json")},
                evidence=(existing_evidence,),
            )
        score = float(run.result_summary.get("validation_macro_f1", 0.0))
        latest_diagnosis = self._latest_diagnosis(experiment_id, run.run_id)
        diagnosis = latest_diagnosis[0] if latest_diagnosis is not None else None
        recommended_changes: dict[str, object] = {}
        if request.action == "stop":
            action = DecisionAction.STOP
            target = ExperimentStatus.COMPLETED_MINI
            reason = request.rationale
        elif request.action == "change_pipeline":
            action = DecisionAction.CHANGE_PIPELINE
            target = ExperimentStatus.WAITING_PLAN_SELECTION
            reason = request.rationale
        elif request.action == "approve_full":
            action = DecisionAction.APPROVE_FULL_TRAIN
            target = ExperimentStatus.WAITING_FULL_APPROVAL
            reason = request.rationale
        elif diagnosis is not None:
            advised_action = diagnosis.advice.recommended_action
            action = {
                "approve_full": DecisionAction.APPROVE_FULL_TRAIN,
                "adjust_parameters": DecisionAction.ADJUST_PARAMETERS,
                "change_pipeline": DecisionAction.CHANGE_PIPELINE,
                "stop": DecisionAction.STOP,
            }[advised_action]
            target = self._decision_target(action)
            reason = diagnosis.advice.overall_conclusion
            recommended_changes = {
                "diagnosis_id": diagnosis.diagnosis_id,
                "recommendations": [
                    item.model_dump(mode="json")
                    for item in diagnosis.advice.recommendations
                ],
            }
        elif score >= 0.75:
            action = DecisionAction.APPROVE_FULL_TRAIN
            target = ExperimentStatus.WAITING_FULL_APPROVAL
            reason = f"Validation Macro-F1={score:.4f}，达到 0.75 的首轮通过阈值。"
        elif state.budget.completed_mini_runs >= state.budget.max_mini_runs:
            action = DecisionAction.STOP
            target = ExperimentStatus.COMPLETED_MINI
            reason = "小样本运行预算已耗尽，停止自动迭代。"
        else:
            action = DecisionAction.CHANGE_PIPELINE
            target = ExperimentStatus.WAITING_PLAN_SELECTION
            reason = f"Validation Macro-F1={score:.4f}，建议更换模块或参数后新建 revision。"

        decision = DecisionRecord(
            decision_id=(
                "decision-"
                + hashlib.sha256(
                    f"{experiment_id}:{run.run_id}:{action.value}:{reason}".encode("utf-8")
                ).hexdigest()[:24]
            ),
            experiment_id=experiment_id,
            run_id=run.run_id,
            action=action,
            reason=reason,
            decided_by="EvaluationGovernorAgent",
            recommended_changes=recommended_changes,
        )
        decision_file = _write_json(decision_file, decision.model_dump(mode="json"))
        evidence = self._register_evidence(
            evidence_id=evidence_id,
            experiment_id=experiment_id,
            run_id=run.run_id,
            kind="approval",
            path=decision_file,
            description="validation 指标、用户选择和预算共同形成的决策",
            idempotency_key=idempotency_key,
        )
        if state.state is ExperimentStatus.DECIDING:
            state = self.repository.transition_state(
                experiment_id,
                target,
                actor="EvaluationGovernorAgent",
                reason=reason,
                evidence_ids=(evidence.evidence_id,),
                idempotency_key=(f"{idempotency_key}:state" if idempotency_key else None),
            )
        return ExperimentActionResponse(
            operation="decision",
            summary=reason,
            state=state,
            payload={"decision": decision.model_dump(mode="json")},
            evidence=(evidence,),
        )

    def report(
        self,
        experiment_id: str,
        *,
        rationale: str,
        idempotency_key: str | None,
    ) -> ExperimentActionResponse:
        state = self.repository.get_experiment(experiment_id)
        run = self._best_or_latest_run(experiment_id)
        events = self.repository.list_state_events(experiment_id)
        evidence_refs = self.repository.list_evidence(experiment_id)
        result = self._load_training_result(run)
        latest_diagnosis = self._latest_diagnosis(experiment_id, run.run_id)
        diagnosis = latest_diagnosis[0] if latest_diagnosis is not None else None
        decision_file = self.path_resolver.run_path(
            experiment_id,
            run.revision,
            run.run_id,
        ) / "decision.json"
        decision = (
            DecisionRecord.model_validate_json(decision_file.read_text(encoding="utf-8"))
            if decision_file.is_file()
            else None
        )
        report_file = self.path_resolver.report_path(
            experiment_id,
            run.revision,
            run.run_id,
        ) / "experiment_report_v2.md"
        evidence_id = f"{run.run_id}-markdown-report-v2"
        existing_evidence = self._existing_evidence(evidence_id)
        if existing_evidence is not None and report_file.is_file():
            return ExperimentActionResponse(
                operation="report",
                summary="Markdown 实验报告已存在，已按证据 ID 恢复。",
                state=state,
                payload={"report_file": str(report_file)},
                evidence=(existing_evidence,),
            )
        _write_text(
            report_file,
            render_experiment_report(
                state=state,
                run=run,
                result=result,
                diagnosis=diagnosis,
                decision=decision,
                events=events,
                evidence_refs=evidence_refs,
                rationale=rationale,
            ),
        )
        evidence = self._register_evidence(
            evidence_id=evidence_id,
            experiment_id=experiment_id,
            run_id=run.run_id,
            kind="report",
            path=report_file,
            description="结构化评估、决策与证据 Markdown 实验报告",
            idempotency_key=idempotency_key,
        )
        return ExperimentActionResponse(
            operation="report",
            summary="Markdown 实验报告已生成并登记证据。",
            state=state,
            payload={
                "report_file": str(report_file),
                "diagnosis_id": diagnosis.diagnosis_id if diagnosis else None,
            },
            evidence=(evidence,),
        )
