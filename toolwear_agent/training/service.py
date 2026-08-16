"""统一训练服务：验证 Pipeline、数据证据，调度后端并归档产物。"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.settings import Settings
from toolwear_agent.registry import (
    ModuleRegistry,
    TrainerRegistry,
    build_default_module_registry,
    build_default_trainer_registry,
    validate_pipeline_against_registries,
)
from toolwear_agent.schemas import (
    EvaluationReport,
    EvidenceRef,
    PipelineSpec,
    RunConfig,
    TrainingArtifacts,
    TrainingDataRef,
    TrainingRunResult,
)
from toolwear_agent.training.backends import (
    BackendTrainingContext,
    PytorchTrainingBackend,
    SklearnTrainingBackend,
    TrainingBackend,
)
from toolwear_agent.training.data_loading import load_raw_window_batch, prepare_training_data


class JsonlRunLogger:
    """为单次 Run 保存带序号和时间的结构化事件日志。"""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.sequence = 0

    def emit(self, event: str, payload: dict[str, object]) -> None:
        """追加一条 JSONL 事件；不记录环境变量或密钥。"""

        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.log_file.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_json(output_file: Path, payload: object) -> Path:
    """使用同目录临时文件原子写出 JSON，避免中断留下半个文件。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return output_file


def _sha256_file(input_file: Path) -> str:
    """流式计算证据文件哈希。"""

    digest = hashlib.sha256()
    with input_file.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_training_code(run_dir: Path, app_root: Path) -> tuple[Path, Path]:
    """保存真正参与训练的代码，便于结果复现和赛后审计。"""

    relative_sources = (
        Path("toolwear_agent/training/models.py"),
        Path("toolwear_agent/training/data_loading.py"),
        Path("toolwear_agent/training/backends.py"),
        Path("toolwear_agent/training/service.py"),
        Path("toolwear_agent/registry/module_registry.py"),
        Path("toolwear_agent/registry/trainer_registry.py"),
        Path("toolwear_agent/registry/validation.py"),
        Path("toolwear_agent/schemas/pipeline.py"),
        Path("toolwear_agent/schemas/training.py"),
        Path("pyproject.toml"),
    )
    snapshot_dir = run_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for relative_source in relative_sources:
        source_file = app_root / relative_source
        if not source_file.is_file():
            raise FileNotFoundError(f"训练代码快照缺少源文件: {source_file}")
        target_file = snapshot_dir / relative_source
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        manifest.append(
            {
                "source": str(source_file),
                "snapshot": str(target_file),
                "sha256": _sha256_file(target_file),
                "size_bytes": target_file.stat().st_size,
            }
        )
    manifest_file = snapshot_dir / "code_snapshot_manifest.json"
    _write_json(manifest_file, manifest)
    return snapshot_dir, manifest_file


class TrainingService:
    """所有页面、API 和 Agent 共用的唯一训练入口。"""

    def __init__(
        self,
        settings: Settings,
        *,
        module_registry: ModuleRegistry | None = None,
        trainer_registry: TrainerRegistry | None = None,
        backends: dict[str, TrainingBackend] | None = None,
    ) -> None:
        self.settings = settings
        self.path_resolver = PathResolver(settings)
        self.module_registry = module_registry or build_default_module_registry()
        self.trainer_registry = trainer_registry or build_default_trainer_registry()
        self.backends: dict[str, TrainingBackend] = backends or {
            "sklearn": SklearnTrainingBackend(),
            "pytorch": PytorchTrainingBackend(),
        }

    @staticmethod
    def _trainer_id(pipeline: PipelineSpec) -> str:
        """从模块链取得训练器 ID，不依赖候选方案名称。"""

        trainers = [
            module.module_id
            for module in pipeline.modules
            if module.enabled and module.kind.value == "trainer"
        ]
        if len(trainers) != 1:
            raise ValueError("Pipeline 必须且只能有一个 trainer 模块。")
        return trainers[0]

    def _validate_request(self, pipeline: PipelineSpec, run_config: RunConfig) -> str:
        """在创建运行目录前完成静态契约校验。"""

        if pipeline.pipeline_id != run_config.pipeline_id:
            raise ValueError("RunConfig.pipeline_id 与 PipelineSpec 不一致。")
        if run_config.evaluate_test or run_config.run_kind.value == "final_evaluation":
            raise ValueError("TrainingService 候选训练入口不允许执行 test；请使用最终评估服务。")
        validation = validate_pipeline_against_registries(
            pipeline,
            self.module_registry,
            self.trainer_registry,
        )
        if not validation.valid:
            messages = [issue.message for issue in validation.issues if issue.severity.value == "error"]
            raise ValueError("Pipeline 未通过 Registry 校验: " + "；".join(messages))
        trainer_id = self._trainer_id(pipeline)
        trainer = self.trainer_registry.get(trainer_id)
        backend_id = trainer.backend.value
        if backend_id not in self.backends:
            raise ValueError(f"训练后端未注册实现: {backend_id}")
        return backend_id

    def _assert_window_contract(self, pipeline: PipelineSpec, records: tuple[object, ...]) -> None:
        """确认 Manifest 窗口长度与用户审批后的 Pipeline 参数一致。"""

        window_modules = [
            module
            for module in pipeline.modules
            if module.enabled and module.kind.value == "windowing"
        ]
        window_module = window_modules[0]
        definition = self.module_registry.get(window_module.module_id)
        configured_length = window_module.parameters.get(
            "window_length",
            definition.parameters_schema["window_length"].default,
        )
        actual_lengths = {getattr(record, "window_size") for record in records}
        if actual_lengths != {int(configured_length)}:
            raise ValueError(
                f"Pipeline window_length={configured_length}，但 Manifest 窗口长度为 {sorted(actual_lengths)}。"
            )

    def _build_evidence_index(
        self,
        *,
        run_config: RunConfig,
        data_ref: TrainingDataRef,
        files: list[tuple[str, Path, str, str]],
        output_file: Path,
    ) -> Path:
        """为 Run 的本地产物和输入证据生成可校验索引。"""

        evidence: list[EvidenceRef] = []
        for index, (kind, file_path, media_type, description) in enumerate(files, start=1):
            evidence.append(
                EvidenceRef(
                    evidence_id=f"{run_config.run_id}-{kind}-{index}",
                    experiment_id=run_config.experiment_id,
                    run_id=run_config.run_id,
                    kind=kind,
                    uri=str(file_path),
                    sha256=_sha256_file(file_path),
                    size_bytes=file_path.stat().st_size,
                    media_type=media_type,
                    description=description,
                    created_by="training_service",
                )
            )
        payload = {
            "schema_version": "1.0",
            "experiment_id": run_config.experiment_id,
            "run_id": run_config.run_id,
            "dataset_id": data_ref.dataset_id,
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
        return _write_json(output_file, payload)

    def train(
        self,
        *,
        pipeline: PipelineSpec,
        run_config: RunConfig,
        data_ref: TrainingDataRef,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
    ) -> TrainingRunResult:
        """执行一次只使用 train/validation 的可追溯真实训练。"""

        backend_id = self._validate_request(pipeline, run_config)
        run_dir = self.path_resolver.run_path(
            run_config.experiment_id,
            run_config.revision,
            run_config.run_id,
        )
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run_id 已存在产物，拒绝覆盖: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)

        log_file = run_dir / "run.jsonl"
        config_file = run_dir / "run_config.json"
        pipeline_file = run_dir / "pipeline.json"
        data_ref_file = run_dir / "data_ref.json"
        metrics_file = run_dir / "metrics.json"
        result_file = run_dir / "result.json"
        evidence_index_file = run_dir / "evidence_index.json"
        logger = JsonlRunLogger(log_file)

        def emit(event: str, payload: dict[str, object]) -> None:
            """同时写 Run 日志，并把进度传给 API job 状态。"""

            logger.emit(event, payload)
            if event_sink is not None:
                event_sink(event, payload)

        _write_json(config_file, run_config.model_dump(mode="json"))
        _write_json(pipeline_file, pipeline.model_dump(mode="json"))
        _write_json(data_ref_file, data_ref.model_dump(mode="json"))
        emit(
            "run_created",
            {
                "run_id": run_config.run_id,
                "pipeline_id": pipeline.pipeline_id,
                "backend": backend_id,
                "split_hash": run_config.split_hash,
            },
        )

        try:
            prepared = prepare_training_data(
                data_ref=data_ref,
                run_config=run_config,
                path_resolver=self.path_resolver,
            )
            records = (*prepared.train_records, *prepared.validation_records)
            self._assert_window_contract(pipeline, records)
            emit(
                "data_validated",
                {
                    "train_samples": len(prepared.train_records),
                    "validation_samples": len(prepared.validation_records),
                    "sample_hash": prepared.sample_manifest.sample_hash,
                    "test_loaded": False,
                },
            )
            train_batch = load_raw_window_batch(prepared.train_records, pipeline.input_channels)
            validation_batch = load_raw_window_batch(prepared.validation_records, pipeline.input_channels)
            emit(
                "windows_loaded",
                {
                    "train_shape": list(train_batch.values.shape),
                    "validation_shape": list(validation_batch.values.shape),
                    "channels": list(pipeline.input_channels),
                },
            )
            output = self.backends[backend_id].train(
                BackendTrainingContext(
                    pipeline=pipeline,
                    run_config=run_config,
                    class_labels=prepared.class_labels,
                    train_batch=train_batch,
                    validation_batch=validation_batch,
                    run_dir=run_dir,
                    progress=emit,
                )
            )
            evaluation = EvaluationReport(
                evaluation_id=f"{run_config.run_id}-evaluation",
                experiment_id=run_config.experiment_id,
                run_id=run_config.run_id,
                pipeline_id=pipeline.pipeline_id,
                metrics=(output.metrics_by_split["train"], output.metrics_by_split["validation"]),
                class_labels=prepared.class_labels,
                final_test=False,
            )
            metrics_payload = {
                "schema_version": "1.0",
                "run_id": run_config.run_id,
                "pipeline_id": pipeline.pipeline_id,
                "metrics_by_split": {
                    split: bundle.model_dump(mode="json")
                    for split, bundle in output.metrics_by_split.items()
                },
                "epoch_history": [item.model_dump(mode="json") for item in output.epoch_history],
                "runtime": output.runtime.model_dump(mode="json"),
                "feature_names": list(output.feature_names),
                "final_test_status": "not_run_pipeline_not_frozen",
            }
            _write_json(metrics_file, metrics_payload)
            snapshot_dir, snapshot_manifest = _snapshot_training_code(run_dir, self.settings.app_root)
            artifacts = TrainingArtifacts(
                run_dir=run_dir,
                model_file=output.model_file,
                metrics_file=metrics_file,
                config_file=config_file,
                pipeline_file=pipeline_file,
                data_ref_file=data_ref_file,
                log_file=log_file,
                code_snapshot_dir=snapshot_dir,
                evidence_index_file=evidence_index_file,
                result_file=result_file,
                loss_history_file=output.loss_history_file,
                loss_curve_file=output.loss_curve_file,
                validation_tsne_file=output.validation_tsne_file,
            )
            if prepared.sample_manifest.sample_hash is None:  # pragma: no cover - loader 已验证
                raise ValueError("sample_hash 不能为空。")
            if run_config.split_hash is None:  # pragma: no cover - 训练数据校验已阻断
                raise ValueError("split_hash 不能为空。")
            result = TrainingRunResult(
                run_id=run_config.run_id,
                experiment_id=run_config.experiment_id,
                revision=run_config.revision,
                pipeline_id=pipeline.pipeline_id,
                split_hash=run_config.split_hash,
                sample_hash=prepared.sample_manifest.sample_hash,
                train_sample_count=len(prepared.train_records),
                validation_sample_count=len(prepared.validation_records),
                class_labels=prepared.class_labels,
                runtime=output.runtime,
                evaluation=evaluation,
                epoch_history=output.epoch_history,
                artifacts=artifacts,
            )
            _write_json(result_file, result.model_dump(mode="json"))
            evidence_files: list[tuple[str, Path, str, str]] = [
                ("config", config_file, "application/json", "运行参数快照"),
                ("config", pipeline_file, "application/json", "Pipeline 快照"),
                ("config", data_ref_file, "application/json", "训练数据引用"),
                ("metrics", metrics_file, "application/json", "train/validation 指标"),
                ("model", output.model_file, "application/octet-stream", "训练模型或 checkpoint"),
                ("log", log_file, "application/x-ndjson", "结构化训练事件"),
                ("code", snapshot_manifest, "application/json", "训练代码快照清单"),
                ("report", result_file, "application/json", "统一训练结果"),
                ("split", data_ref.split_manifest_file, "application/json", "锁定 split Manifest"),
                ("split", data_ref.split_lock_file, "application/json", "实验修订 split lock"),
                ("split", data_ref.training_sample_manifest_file, "application/json", "训练小样本 Manifest"),
                ("split", data_ref.leakage_audit_file, "application/json", "数据泄漏审计"),
            ]
            if output.loss_history_file is not None:
                evidence_files.append(
                    ("metrics", output.loss_history_file, "application/json", "真实 epoch 损失历史")
                )
            if output.loss_curve_file is not None:
                evidence_files.append(
                    ("figure", output.loss_curve_file, "image/png", "真实 train/validation 损失曲线")
                )
            if output.validation_tsne_file is not None:
                evidence_files.append(
                    (
                        "figure",
                        output.validation_tsne_file,
                        "image/png",
                        "仅使用 validation 的 t-SNE 特征分布",
                    )
                )
            emit(
                "run_completed",
                {
                    "result_file": str(result_file),
                    "evidence_index_file": str(evidence_index_file),
                },
            )
            # 日志完成后再计算证据哈希；此后不再追加日志，索引中的 SHA-256 才稳定。
            self._build_evidence_index(
                run_config=run_config,
                data_ref=data_ref,
                files=evidence_files,
                output_file=evidence_index_file,
            )
            return result
        except Exception as exc:
            emit(
                "run_failed",
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
