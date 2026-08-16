"""FastAPI、前端与后续 AgentTeams 共用的实验工作流服务。"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from uuid import uuid4

from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.settings import Settings
from toolwear_agent.data.registry import DatasetRegistry
from toolwear_agent.registry import build_default_registry_catalog
from toolwear_agent.schemas import (
    CandidateRecommendationSet,
    DatasetManifest,
    DatasetRef,
    EvidenceRef,
    ExperimentRevision,
    ExperimentPreferences,
    ExperimentState,
    LabelPolicy,
    ModuleSpec,
    PipelineSpec,
    RunConfig,
    SplitSpec,
)
from toolwear_agent.schemas.api import (
    CreateExperimentRequest,
    DecisionRequest,
    ExperimentActionResponse,
    PipelineApprovalRequest,
    PipelineValidationResponse,
    RecommendationRequest,
    RunStartRequest,
)
from toolwear_agent.schemas.experiment import ApprovalRecord
from toolwear_agent.schemas.experiment import ExperimentStatus
from toolwear_agent.data.splitting import load_split_manifest
from toolwear_agent.registry import validate_pipeline_with_default_registries
from toolwear_agent.services.candidate_service import CandidateProvider, DefaultCandidateProvider
from toolwear_agent.services.preparation import DataPreparationService
from toolwear_agent.services.evaluation_reporting import EvaluationReportingService
from toolwear_agent.services.evaluation_diagnosis import DiagnosisProvider
from toolwear_agent.services.training_jobs import TrainingJobService
from toolwear_agent.services.errors import InvalidWorkflowStateError
from toolwear_agent.state import (
    EntityNotFoundError,
    RunRecord,
    SQLiteExperimentRepository,
    StateTransitionEvent,
)


class ExperimentWorkflowService:
    """维护状态机并协调 Registry、训练和报告等确定性能力。"""

    def __init__(
        self,
        settings: Settings,
        repository: SQLiteExperimentRepository,
        *,
        candidate_provider: CandidateProvider | None = None,
        diagnosis_provider: DiagnosisProvider | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.path_resolver = PathResolver(settings)
        self.dataset_registry = DatasetRegistry(settings.dataset_manifest)
        self.candidate_provider = candidate_provider or DefaultCandidateProvider(settings)
        self.preparation = DataPreparationService(settings, repository)
        self.training_jobs = TrainingJobService(settings, repository)
        self.evaluation_reporting = EvaluationReportingService(
            settings,
            repository,
            diagnosis_provider=diagnosis_provider,
        )

    def close(self) -> None:
        self.training_jobs.close()

    def capabilities(self):
        """返回页面和 Agent 共用的真实模块/训练器目录。"""

        return build_default_registry_catalog()

    def datasets(self) -> tuple[DatasetManifest, ...]:
        return self.dataset_registry.list()

    def create_experiment(
        self,
        request: CreateExperimentRequest,
        *,
        idempotency_key: str | None,
    ) -> ExperimentState:
        manifest = self.dataset_registry.get(request.dataset_id)
        unavailable = [
            cutter_id
            for cutter_id in request.cutter_ids
            if cutter_id not in manifest.cutters or not manifest.cutters[cutter_id].available
        ]
        if unavailable:
            raise ValueError(f"数据集不可用刀具: {', '.join(unavailable)}")
        unlabeled = [
            cutter_id for cutter_id in request.cutter_ids if not manifest.cutters[cutter_id].labeled
        ]
        if unlabeled:
            raise ValueError(f"四阶段分类需要磨损标签，以下刀具无标签: {', '.join(unlabeled)}")
        unknown_channels = set(request.input_channels) - set(manifest.channels)
        if unknown_channels:
            raise ValueError(f"数据集不包含输入通道: {sorted(unknown_channels)}")
        experiment_id = request.experiment_id or f"experiment-{uuid4().hex}"
        state = ExperimentState(
            experiment_id=experiment_id,
            title=request.title,
            objective=request.user_request,
            dataset_ref=DatasetRef(
                dataset_id=request.dataset_id,
                cutter_ids=request.cutter_ids,
                manifest_hash=manifest.manifest_hash,
            ),
            label_policy=LabelPolicy(
                aggregation=request.vb_aggregation,
                stage_thresholds_um=request.vb_thresholds_um,
                enable_regression=request.enable_vb_regression,
                specified_flute=request.specified_flute,
            ),
            split_spec=SplitSpec(
                train_ratio=request.train_ratio,
                validation_ratio=request.validation_ratio,
                test_ratio=request.test_ratio,
                random_seed=request.random_seed,
            ),
            preferences=ExperimentPreferences(
                input_channels=request.input_channels,
                window_length=request.window_length,
                overlap=request.overlap,
                sample_fraction=request.sample_fraction,
                max_windows_per_cut=request.max_windows_per_cut,
                mode=request.mode,
            ),
        )
        return self.repository.create_experiment(
            state,
            actor="human",
            reason="用户通过 Tool API 创建实验。",
            idempotency_key=idempotency_key,
        )

    def list_experiments(self) -> tuple[ExperimentState, ...]:
        return self.repository.list_experiments()

    def get_experiment(self, experiment_id: str) -> ExperimentState:
        return self.repository.get_experiment(experiment_id)

    def events(self, experiment_id: str) -> tuple[StateTransitionEvent, ...]:
        return self.repository.list_state_events(experiment_id)

    def artifacts(self, experiment_id: str) -> tuple[EvidenceRef, ...]:
        return self.repository.list_evidence(experiment_id)

    def artifact_content(self, evidence_id: str) -> tuple[EvidenceRef, Path]:
        """解析数据库登记的证据路径，并限制在 AI 运行目录和 8 MiB 内。"""

        evidence = self.repository.get_evidence(evidence_id)
        path = self.path_resolver.assert_within(
            evidence.uri,
            (self.settings.ai_infra_root,),
        )
        if not path.is_file():
            raise EntityNotFoundError(f"证据文件不存在：{path.name}")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("证据内容超过 8 MiB，只允许通过产物路径离线查看。")
        return evidence, path

    def get_latest_recommendations(self, experiment_id: str) -> CandidateRecommendationSet:
        return self.repository.get_latest_recommendations(experiment_id)

    def list_runs(self, experiment_id: str) -> tuple[RunRecord, ...]:
        return self.repository.list_runs(experiment_id)

    def get_revision(self, experiment_id: str, revision: int) -> ExperimentRevision:
        return self.repository.get_revision(experiment_id, revision)

    def run_logs(
        self,
        experiment_id: str,
        run_id: str,
        *,
        tail: int,
    ) -> dict[str, object]:
        """读取指定 Run 的结构化日志尾部，不允许页面自行访问文件系统。"""

        run = self.repository.get_run(run_id)
        if run.experiment_id != experiment_id:
            raise ValueError("run_id 不属于指定 experiment。")
        log_file = self.path_resolver.run_path(
            experiment_id,
            run.revision,
            run_id,
        ) / "run.jsonl"
        if not log_file.is_file():
            return {"run_id": run_id, "entries": []}
        entries: deque[dict[str, object]] = deque(maxlen=tail)
        with log_file.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))
        return {"run_id": run_id, "entries": list(entries)}

    def profile(self, experiment_id: str, *, rationale: str, idempotency_key: str | None):
        return self.preparation.profile(
            experiment_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
        )

    def labels(self, experiment_id: str, *, rationale: str, idempotency_key: str | None):
        return self.preparation.labels(
            experiment_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
        )

    def split(self, experiment_id: str, *, rationale: str, idempotency_key: str | None):
        return self.preparation.split(
            experiment_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
        )

    def recommendations(
        self,
        experiment_id: str,
        request: RecommendationRequest,
        *,
        idempotency_key: str | None,
    ) -> CandidateRecommendationSet:
        state = self.repository.get_experiment(experiment_id)
        if state.state is ExperimentStatus.DATA_VALIDATING:
            evidence_ids = {item.evidence_id for item in self.repository.list_evidence(experiment_id)}
            required = {
                f"{experiment_id}-profile-r{state.revision}",
                f"{experiment_id}-labels-r{state.revision}",
                f"{experiment_id}-split-r{state.revision}",
            }
            missing = required - evidence_ids
            if missing:
                raise InvalidWorkflowStateError(
                    "数据准备尚未完成，缺少证据：" + ", ".join(sorted(missing))
                )
            state = self.repository.transition_state(
                experiment_id,
                ExperimentStatus.WAITING_PLAN_SELECTION,
                actor="AlgorithmArchitectAgent",
                reason="数据准备证据齐全，进入候选方案选择。",
                evidence_ids=tuple(sorted(required)),
                idempotency_key=(
                    f"{idempotency_key}:ready" if idempotency_key is not None else None
                ),
            )
        if state.state is not ExperimentStatus.WAITING_PLAN_SELECTION:
            raise InvalidWorkflowStateError(
                f"{state.state.value} 状态下不能生成候选方案。"
            )
        if state.latest_recommendation_id is not None and not request.force_refresh:
            return self.repository.get_recommendations(state.latest_recommendation_id)
        generated = self.candidate_provider.recommend(state, request.user_request)
        return self.repository.save_recommendations(
            generated,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _stable_token(*parts: str, prefix: str) -> str:
        digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:24]
        return f"{prefix}-{digest}"

    @staticmethod
    def _apply_pipeline_parameters(
        pipeline: PipelineSpec,
        state: ExperimentState,
        request: PipelineApprovalRequest,
    ) -> PipelineSpec:
        available_ids = {module.module_id for module in pipeline.modules}
        unknown = set(request.module_parameters) - available_ids
        if unknown:
            raise ValueError(f"参数引用了候选中不存在的模块: {sorted(unknown)}")
        modules: list[ModuleSpec] = []
        for module in pipeline.modules:
            parameters = dict(module.parameters)
            if module.kind.value == "windowing":
                parameters.update(
                    {
                        "window_length": state.preferences.window_length,
                        "overlap": state.preferences.overlap,
                    }
                )
            parameters.update(request.module_parameters.get(module.module_id, {}))
            if module.kind.value == "windowing":
                requested_length = int(parameters["window_length"])
                requested_overlap = float(parameters["overlap"])
                if requested_length != state.preferences.window_length:
                    raise ValueError(
                        "窗口长度已经写入切分证据，审批阶段不能修改；"
                        "请用新参数创建实验并重新切分。"
                    )
                if requested_overlap != state.preferences.overlap:
                    raise ValueError(
                        "重叠率已经写入切分证据，审批阶段不能修改；"
                        "请用新参数创建实验并重新切分。"
                    )
            payload = module.model_dump(mode="python")
            payload["parameters"] = parameters
            modules.append(ModuleSpec.model_validate(payload))
        payload = pipeline.model_dump(mode="python")
        payload["modules"] = tuple(modules)
        selected_channels = request.input_channels or state.preferences.input_channels
        if selected_channels:
            payload["input_channels"] = selected_channels
        return PipelineSpec.model_validate(payload)

    def approve_pipeline(
        self,
        experiment_id: str,
        request: PipelineApprovalRequest,
        *,
        idempotency_key: str | None,
    ) -> ExperimentActionResponse:
        state = self.repository.get_experiment(experiment_id)
        if state.state not in {
            ExperimentStatus.WAITING_PLAN_SELECTION,
            ExperimentStatus.PIPELINE_VALIDATING,
        }:
            raise InvalidWorkflowStateError(
                f"{state.state.value} 状态下不能审批候选 Pipeline。"
            )
        recommendations = self.repository.get_latest_recommendations(experiment_id)
        try:
            candidate = next(
                item for item in recommendations.pipelines if item.pipeline_id == request.pipeline_id
            )
        except StopIteration as exc:
            raise ValueError(f"候选集合中不存在 Pipeline：{request.pipeline_id}") from exc
        pipeline = self._apply_pipeline_parameters(candidate, state, request)
        validation = validate_pipeline_with_default_registries(pipeline)
        if not validation.valid:
            messages = [
                issue.message for issue in validation.issues if issue.severity.value == "error"
            ]
            raise ValueError("用户选择的 Pipeline 不兼容：" + "；".join(messages))

        operation_key = idempotency_key or f"approval-{uuid4().hex}"
        approval_id = self._stable_token(experiment_id, operation_key, prefix="approval")
        approval = ApprovalRecord(
            approval_id=approval_id,
            experiment_id=experiment_id,
            revision=state.revision,
            action="approve_pipeline",
            requested_by="AlgorithmArchitectAgent",
            rationale="候选生成完成，等待用户选择。",
        )
        self.repository.create_approval(
            approval,
            idempotency_key=f"{operation_key}:request",
        )
        decided = self.repository.decide_approval(
            approval_id,
            status="approved",
            decided_by="human",
            rationale=request.rationale,
            idempotency_key=f"{operation_key}:decision",
        )

        try:
            self.repository.get_revision(experiment_id, state.revision)
            revision_number = (
                state.revision
                if state.state is ExperimentStatus.PIPELINE_VALIDATING
                else state.revision + 1
            )
        except EntityNotFoundError:
            revision_number = state.revision
        data_ref = self.preparation.bind_data_ref_to_revision(state, revision_number)
        split_manifest = load_split_manifest(data_ref.split_manifest_file)
        if split_manifest.split_hash is None:  # pragma: no cover - loader 已验证
            raise ValueError("split_hash 不能为空。")
        run_id = self._stable_token(experiment_id, operation_key, prefix="run")
        run_config = RunConfig(
            run_id=run_id,
            experiment_id=experiment_id,
            revision=revision_number,
            pipeline_id=pipeline.pipeline_id,
            run_kind="mini_train",
            split_hash=split_manifest.split_hash,
            sample_fraction=state.preferences.sample_fraction,
            max_samples=request.max_samples,
            batch_size=request.batch_size,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            device=request.device,
            random_seed=state.split_spec.random_seed,
            num_workers=request.num_workers,
            evaluate_test=False,
        )
        revision = ExperimentRevision(
            experiment_id=experiment_id,
            revision=revision_number,
            pipeline=pipeline,
            run_config=run_config,
            created_by="human",
            change_reason=request.rationale,
            parent_revision=None if revision_number == 1 else revision_number - 1,
        )
        created_revision = self.repository.create_revision(
            revision,
            idempotency_key=f"{operation_key}:revision",
        )
        transitioned = self.repository.transition_state(
            experiment_id,
            ExperimentStatus.PIPELINE_VALIDATING,
            actor="ExperimentManagerAgent",
            reason="用户已批准候选和训练参数，开始组合校验。",
            evidence_ids=(),
            idempotency_key=f"{operation_key}:state",
        )
        return ExperimentActionResponse(
            operation="approve_pipeline",
            summary="候选方案和训练参数已形成不可变 revision。",
            state=transitioned,
            payload={
                "approval": decided.model_dump(mode="json"),
                "revision": created_revision.model_dump(mode="json"),
            },
        )

    def validate_pipeline(
        self,
        experiment_id: str,
        *,
        rationale: str,
        idempotency_key: str | None,
    ) -> PipelineValidationResponse:
        state = self.repository.get_experiment(experiment_id)
        if state.state is ExperimentStatus.CODE_PREPARING:
            revision = self.repository.get_revision(experiment_id, state.revision)
            return PipelineValidationResponse(
                state=state,
                validation=validate_pipeline_with_default_registries(revision.pipeline),
            )
        if state.state is not ExperimentStatus.PIPELINE_VALIDATING:
            raise InvalidWorkflowStateError(
                f"{state.state.value} 状态下不能执行 Pipeline 校验。"
            )
        revision = self.repository.get_revision(experiment_id, state.revision)
        validation = validate_pipeline_with_default_registries(revision.pipeline)
        target = (
            ExperimentStatus.CODE_PREPARING
            if validation.valid
            else ExperimentStatus.WAITING_PLAN_SELECTION
        )
        messages = [issue.message for issue in validation.issues]
        transitioned = self.repository.transition_state(
            experiment_id,
            target,
            actor="CodeTrainingEngineerAgent",
            reason=rationale if validation.valid else "Pipeline 校验失败：" + "；".join(messages),
            idempotency_key=idempotency_key,
        )
        return PipelineValidationResponse(state=transitioned, validation=validation)

    def start_mini_run(
        self,
        experiment_id: str,
        request: RunStartRequest,
        *,
        idempotency_key: str | None,
    ) -> RunRecord:
        state = self.repository.get_experiment(experiment_id)
        if state.state not in {ExperimentStatus.CODE_PREPARING, ExperimentStatus.MINI_TRAINING}:
            raise InvalidWorkflowStateError(
                f"{state.state.value} 状态下不能启动小样本训练。"
            )
        revision = self.repository.get_revision(experiment_id, state.revision)
        if request.run_id is not None and request.run_id != revision.run_config.run_id:
            raise ValueError("run_id 已在审批 revision 中锁定，启动时不能替换。")
        if (
            request.max_samples is not None
            and request.max_samples != revision.run_config.max_samples
        ):
            raise ValueError("max_samples 必须在方案审批时锁定，启动时不能替换。")
        data_ref = self.preparation.load_training_data_ref(state)
        return self.training_jobs.submit(
            revision,
            data_ref,
            idempotency_key=idempotency_key,
        )

    def cancel(
        self,
        experiment_id: str,
        *,
        rationale: str,
    ) -> ExperimentActionResponse:
        run = self.training_jobs.request_cancel(experiment_id)
        return ExperimentActionResponse(
            operation="cancel",
            summary=rationale,
            state=self.repository.get_experiment(experiment_id),
            payload={"run": run.model_dump(mode="json")},
        )

    def evaluate(
        self,
        experiment_id: str,
        *,
        rationale: str,
        idempotency_key: str | None,
        force_refresh: bool = False,
    ):
        return self.evaluation_reporting.evaluate(
            experiment_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
            force_refresh=force_refresh,
        )

    def decide(
        self,
        experiment_id: str,
        request: DecisionRequest,
        *,
        idempotency_key: str | None,
    ):
        return self.evaluation_reporting.decide(
            experiment_id,
            request,
            idempotency_key=idempotency_key,
        )

    def report(self, experiment_id: str, *, rationale: str, idempotency_key: str | None):
        return self.evaluation_reporting.report(
            experiment_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
        )

    def get_run(self, run_id: str) -> RunRecord:
        return self.repository.get_run(run_id)
