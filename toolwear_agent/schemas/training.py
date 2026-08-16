"""训练服务的输入引用、运行过程和产物契约。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex, utc_now
from toolwear_agent.schemas.evaluation import EvaluationReport


class TrainingDataRef(SchemaModel):
    """一次训练所依赖的只读数据证据。

    这里保存的是 Manifest 路径而不是一批游离的窗口列表。训练服务会重新验证
    split hash、split lock 和小样本 hash，避免前端绕过数据治理直接喂入 test。
    """

    dataset_id: EntityId
    cutter_id: EntityId
    window_manifest_file: Path
    training_sample_manifest_file: Path
    split_manifest_file: Path
    split_lock_file: Path
    leakage_audit_file: Path


class EpochLoss(SchemaModel):
    """深度模型一轮真实训练与验证损失。"""

    epoch: int = Field(ge=1)
    train_loss: float = Field(ge=0)
    validation_loss: float = Field(ge=0)
    learning_rate: float = Field(gt=0)


class TrainingRuntimeInfo(SchemaModel):
    """后端、设备和耗时证据，防止 CPU 运行被误写成 CUDA。"""

    backend: Literal["sklearn", "pytorch"]
    requested_device: NonEmptyText
    resolved_device: NonEmptyText
    cuda_available: bool = False
    cuda_used: bool = False
    torch_version: str | None = None
    cuda_device_name: str | None = None
    cuda_peak_memory_bytes: int | None = Field(default=None, ge=0)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)
    elapsed_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _cuda_fields_are_consistent(self) -> "TrainingRuntimeInfo":
        if self.cuda_used and (not self.cuda_available or not self.resolved_device.startswith("cuda")):
            raise ValueError("cuda_used=true 时必须实际解析到可用 CUDA 设备。")
        if not self.cuda_used and self.cuda_peak_memory_bytes not in (None, 0):
            raise ValueError("未使用 CUDA 时不能记录非零 CUDA 峰值显存。")
        return self


class TrainingArtifacts(SchemaModel):
    """一次 Run 的可定位产物，不在数据库或 API 中内嵌大文件。"""

    run_dir: Path
    model_file: Path
    metrics_file: Path
    config_file: Path
    pipeline_file: Path
    data_ref_file: Path
    log_file: Path
    code_snapshot_dir: Path
    evidence_index_file: Path
    result_file: Path
    loss_history_file: Path | None = None
    loss_curve_file: Path | None = None
    validation_tsne_file: Path | None = None


class TrainingRunResult(SchemaModel):
    """sklearn 与 PyTorch 后端统一返回的训练结果。"""

    run_id: EntityId
    experiment_id: EntityId
    revision: int = Field(ge=1)
    pipeline_id: EntityId
    split_hash: Sha256Hex
    sample_hash: Sha256Hex
    train_sample_count: int = Field(ge=1)
    validation_sample_count: int = Field(ge=1)
    class_labels: tuple[NonEmptyText, ...] = Field(min_length=2)
    runtime: TrainingRuntimeInfo
    evaluation: EvaluationReport
    epoch_history: tuple[EpochLoss, ...] = ()
    artifacts: TrainingArtifacts
    final_test_status: Literal["not_run_pipeline_not_frozen"] = "not_run_pipeline_not_frozen"

    @model_validator(mode="after")
    def _identities_match_evaluation(self) -> "TrainingRunResult":
        if self.evaluation.run_id != self.run_id:
            raise ValueError("EvaluationReport.run_id 与训练结果不一致。")
        if self.evaluation.experiment_id != self.experiment_id:
            raise ValueError("EvaluationReport.experiment_id 与训练结果不一致。")
        if self.evaluation.pipeline_id != self.pipeline_id:
            raise ValueError("EvaluationReport.pipeline_id 与训练结果不一致。")
        if self.evaluation.final_test:
            raise ValueError("候选训练结果不能包含最终 test 评估。")
        return self
