"""Registry 驱动的 sklearn 与 PyTorch 训练后端。"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from toolwear_agent.schemas import (
    EpochLoss,
    MetricBundle,
    ModuleSpec,
    PipelineSpec,
    RunConfig,
    TrainingRuntimeInfo,
)
from toolwear_agent.training.data_loading import (
    RawWindowBatch,
    apply_channel_normalizer,
    fit_channel_normalizer,
    statistical_feature_matrix,
)
from toolwear_agent.training.embedding import write_validation_tsne
from toolwear_agent.training.models import LightweightCNN1D


ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class BackendTrainingContext:
    """训练后端需要的全部运行时输入。"""

    pipeline: PipelineSpec
    run_config: RunConfig
    class_labels: tuple[str, ...]
    train_batch: RawWindowBatch
    validation_batch: RawWindowBatch
    run_dir: Path
    progress: ProgressCallback


@dataclass(frozen=True)
class BackendTrainingOutput:
    """两个后端交给 TrainingService 的统一结果。"""

    runtime: TrainingRuntimeInfo
    metrics_by_split: dict[str, MetricBundle]
    epoch_history: tuple[EpochLoss, ...]
    model_file: Path
    loss_history_file: Path | None = None
    loss_curve_file: Path | None = None
    validation_tsne_file: Path | None = None
    feature_names: tuple[str, ...] = ()


class TrainingBackend(Protocol):
    """TrainingService 可调度的最小后端接口。"""

    backend_id: str

    def train(self, context: BackendTrainingContext) -> BackendTrainingOutput:
        """执行一次真实训练并返回统一结果。"""


def _module_by_kind(pipeline: PipelineSpec, kind: str) -> ModuleSpec:
    """返回唯一启用模块；Pipeline Schema 和 Registry 已先做结构校验。"""

    matches = [module for module in pipeline.modules if module.enabled and module.kind.value == kind]
    if len(matches) != 1:
        raise ValueError(f"Pipeline 必须且只能有一个 {kind} 模块。")
    return matches[0]


def _optional_module_by_kind(pipeline: PipelineSpec, kind: str) -> ModuleSpec | None:
    """返回可选的唯一模块。"""

    matches = [module for module in pipeline.modules if module.enabled and module.kind.value == kind]
    if len(matches) > 1:
        raise ValueError(f"Pipeline 最多只能有一个 {kind} 模块。")
    return matches[0] if matches else None


def _classification_metric(
    *,
    split: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_labels: tuple[str, ...],
    loss: float | None,
) -> MetricBundle:
    """生成固定四分类顺序的指标，缺失类别也保留混淆矩阵行列。"""

    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    class_ids = list(range(len(class_labels)))
    report = classification_report(
        labels,
        predictions,
        labels=class_ids,
        target_names=list(class_labels),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1-score": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in class_labels
    }
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    return MetricBundle(
        split=split,
        sample_count=int(labels.size),
        macro_f1=float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        loss=loss,
        per_class=per_class,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix.tolist()),
    )


def _seed_everything(random_seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机源。"""

    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        import torch

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    except ImportError:  # pragma: no cover - sklearn-only 环境的防御路径
        return


class SklearnTrainingBackend:
    """统计特征 + 树模型的低成本训练后端。"""

    backend_id = "sklearn"

    def train(self, context: BackendTrainingContext) -> BackendTrainingOutput:
        """训练 RandomForest 或 ExtraTrees，并只评估 train/validation。"""

        from joblib import dump
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        started_at = datetime.now(timezone.utc)
        _seed_everything(context.run_config.random_seed)
        model_module = _module_by_kind(context.pipeline, "model")
        trainer_module = _module_by_kind(context.pipeline, "trainer")
        train_features, feature_names = statistical_feature_matrix(
            context.train_batch.values,
            context.pipeline.input_channels,
        )
        validation_features, _ = statistical_feature_matrix(
            context.validation_batch.values,
            context.pipeline.input_channels,
        )

        parameters = model_module.parameters
        class_weight = parameters.get("class_weight", "balanced")
        if class_weight == "none":
            class_weight = None
        max_depth = int(parameters.get("max_depth", 0))
        common_parameters = {
            "n_estimators": int(parameters.get("n_estimators", 300)),
            "max_depth": None if max_depth == 0 else max_depth,
            "class_weight": class_weight,
            "random_state": context.run_config.random_seed,
            "n_jobs": int(trainer_module.parameters.get("n_jobs", -1)),
        }
        factories = {
            "random_forest": RandomForestClassifier,
            "extra_trees": ExtraTreesClassifier,
        }
        try:
            classifier = factories[model_module.module_id](**common_parameters)
        except KeyError as exc:
            raise ValueError(f"sklearn 后端不支持模型: {model_module.module_id}") from exc

        context.progress(
            "training_started",
            {
                "backend": self.backend_id,
                "model_id": model_module.module_id,
                "train_samples": len(context.train_batch.labels),
                "validation_samples": len(context.validation_batch.labels),
            },
        )
        classifier.fit(train_features, context.train_batch.labels)
        train_predictions = classifier.predict(train_features)
        validation_predictions = classifier.predict(validation_features)
        validation_tsne_file = write_validation_tsne(
            validation_features,
            context.validation_batch.labels,
            context.class_labels,
            context.run_dir / "validation_tsne.png",
            random_seed=context.run_config.random_seed,
        )
        model_file = context.run_dir / "model.joblib"
        dump(
            {
                "model": classifier,
                "feature_names": feature_names,
                "input_channels": context.pipeline.input_channels,
                "class_labels": context.class_labels,
            },
            model_file,
        )
        finished_at = datetime.now(timezone.utc)
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_version = str(torch.__version__)
            cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
        except ImportError:  # pragma: no cover - 项目正式依赖已包含 torch
            cuda_available = False
            torch_version = None
            cuda_device_name = None
        runtime = TrainingRuntimeInfo(
            backend=self.backend_id,
            requested_device=context.run_config.device,
            resolved_device="cpu",
            cuda_available=cuda_available,
            cuda_used=False,
            torch_version=torch_version,
            cuda_device_name=cuda_device_name,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=(finished_at - started_at).total_seconds(),
        )
        metrics = {
            "train": _classification_metric(
                split="train",
                labels=context.train_batch.labels,
                predictions=train_predictions,
                class_labels=context.class_labels,
                loss=None,
            ),
            "validation": _classification_metric(
                split="validation",
                labels=context.validation_batch.labels,
                predictions=validation_predictions,
                class_labels=context.class_labels,
                loss=None,
            ),
        }
        context.progress(
            "training_completed",
            {
                "backend": self.backend_id,
                "validation_macro_f1": metrics["validation"].macro_f1,
                "elapsed_seconds": runtime.elapsed_seconds,
            },
        )
        return BackendTrainingOutput(
            runtime=runtime,
            metrics_by_split=metrics,
            epoch_history=(),
            model_file=model_file,
            validation_tsne_file=validation_tsne_file,
            feature_names=feature_names,
        )


class PytorchTrainingBackend:
    """原始窗口 + 轻量 1D-CNN 的真实反向传播后端。"""

    backend_id = "pytorch"

    @staticmethod
    def _resolve_device(requested: str):
        """解析设备；CUDA 不可用时允许诚实回退 CPU。"""

        import torch

        cuda_available = bool(torch.cuda.is_available())
        if requested == "cpu":
            return torch.device("cpu"), cuda_available
        if requested == "auto":
            return torch.device("cuda:0" if cuda_available else "cpu"), cuda_available
        if requested == "cuda" and cuda_available:
            return torch.device("cuda:0"), cuda_available
        if requested.startswith("cuda") and cuda_available:
            return torch.device(requested), cuda_available
        return torch.device("cpu"), cuda_available

    @staticmethod
    def _class_weights(labels: np.ndarray, class_count: int):
        """按训练集类别频次计算均衡权重。"""

        import torch

        counts = np.bincount(labels, minlength=class_count).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError("加权交叉熵要求训练小样本覆盖全部类别。")
        weights = labels.size / (class_count * counts)
        return torch.as_tensor(weights, dtype=torch.float32)

    @staticmethod
    def _evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
        """关闭梯度后计算真实平均损失和预测。"""

        import torch

        model.eval()
        total_loss = 0.0
        total_count = 0
        labels: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                total_loss += float(loss.item()) * int(targets.size(0))
                total_count += int(targets.size(0))
                labels.append(targets.detach().cpu().numpy())
                predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        return total_loss / total_count, np.concatenate(labels), np.concatenate(predictions)

    def train(self, context: BackendTrainingContext) -> BackendTrainingOutput:
        """执行 1~N 轮 CNN 训练，保存最佳验证损失 checkpoint。"""

        import torch
        from torch import nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset

        started_at = datetime.now(timezone.utc)
        _seed_everything(context.run_config.random_seed)
        device, cuda_available = self._resolve_device(context.run_config.device)
        cuda_used = device.type == "cuda"
        if cuda_used:
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        preprocess_module = _optional_module_by_kind(context.pipeline, "preprocess")
        epsilon = float(preprocess_module.parameters.get("epsilon", 1e-8)) if preprocess_module else 1e-8
        means, standard_deviations = fit_channel_normalizer(context.train_batch.values, epsilon)
        normalized_train = apply_channel_normalizer(context.train_batch.values, means, standard_deviations)
        normalized_validation = apply_channel_normalizer(
            context.validation_batch.values,
            means,
            standard_deviations,
        )

        train_dataset = TensorDataset(
            torch.from_numpy(normalized_train),
            torch.from_numpy(context.train_batch.labels),
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(normalized_validation),
            torch.from_numpy(context.validation_batch.labels),
        )
        generator = torch.Generator().manual_seed(context.run_config.random_seed)
        loader_options = {
            "batch_size": context.run_config.batch_size,
            "num_workers": context.run_config.num_workers,
            "pin_memory": cuda_used,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
        validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

        model_module = _module_by_kind(context.pipeline, "model")
        loss_module = _module_by_kind(context.pipeline, "loss")
        model = LightweightCNN1D(
            input_channels=len(context.pipeline.input_channels),
            base_channels=int(model_module.parameters.get("base_channels", 32)),
            class_count=len(context.class_labels),
            dropout=float(model_module.parameters.get("dropout", 0.2)),
        ).to(device)
        loss_kwargs: dict[str, object] = {}
        if loss_module.module_id == "weighted_cross_entropy":
            loss_kwargs["weight"] = self._class_weights(
                context.train_batch.labels,
                len(context.class_labels),
            ).to(device)
        elif loss_module.module_id != "cross_entropy":
            raise ValueError(f"PyTorch 后端不支持损失模块: {loss_module.module_id}")
        loss_kwargs["label_smoothing"] = float(loss_module.parameters.get("label_smoothing", 0.0))
        criterion = nn.CrossEntropyLoss(**loss_kwargs)
        optimizer = AdamW(model.parameters(), lr=context.run_config.learning_rate)
        checkpoint_file = context.run_dir / "checkpoint.pt"
        loss_history: list[EpochLoss] = []
        best_validation_loss = float("inf")

        context.progress(
            "training_started",
            {
                "backend": self.backend_id,
                "requested_device": context.run_config.device,
                "resolved_device": str(device),
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "epochs": context.run_config.epochs,
            },
        )
        for epoch_index in range(context.run_config.epochs):
            model.train()
            running_loss = 0.0
            seen = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * int(targets.size(0))
                seen += int(targets.size(0))
            train_loss = running_loss / seen
            validation_loss, _validation_labels, _validation_predictions = self._evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )
            epoch_loss = EpochLoss(
                epoch=epoch_index + 1,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            loss_history.append(epoch_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "epoch": epoch_index + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_loss": validation_loss,
                        "input_channels": len(context.pipeline.input_channels),
                        "channel_ids": context.pipeline.input_channels,
                        "class_labels": context.class_labels,
                        "channel_mean": means.tolist(),
                        "channel_std": standard_deviations.tolist(),
                        "pipeline": context.pipeline.model_dump(mode="json"),
                        "run_config": context.run_config.model_dump(mode="json"),
                    },
                    checkpoint_file,
                )
            context.progress(
                "epoch_completed",
                {
                    "epoch": epoch_index + 1,
                    "epochs": context.run_config.epochs,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
            )

        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        train_loss, train_labels, train_predictions = self._evaluate(model, train_loader, criterion, device)
        validation_loss, validation_labels, validation_predictions = self._evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        if cuda_used:
            torch.cuda.synchronize(device)
        finished_at = datetime.now(timezone.utc)
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if cuda_used else None
        runtime = TrainingRuntimeInfo(
            backend=self.backend_id,
            requested_device=context.run_config.device,
            resolved_device=str(device),
            cuda_available=cuda_available,
            cuda_used=cuda_used,
            torch_version=str(torch.__version__),
            cuda_device_name=torch.cuda.get_device_name(device) if cuda_used else None,
            cuda_peak_memory_bytes=peak_memory,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=(finished_at - started_at).total_seconds(),
        )
        metrics = {
            "train": _classification_metric(
                split="train",
                labels=train_labels,
                predictions=train_predictions,
                class_labels=context.class_labels,
                loss=train_loss,
            ),
            "validation": _classification_metric(
                split="validation",
                labels=validation_labels,
                predictions=validation_predictions,
                class_labels=context.class_labels,
                loss=validation_loss,
            ),
        }
        loss_history_file = context.run_dir / "loss_history.json"
        loss_history_file.write_text(
            json.dumps([item.model_dump(mode="json") for item in loss_history], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        loss_curve_file = context.run_dir / "loss_curve.png"
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 5), dpi=140)
        axis = figure.subplots()
        epoch_numbers = [item.epoch for item in loss_history]
        axis.plot(epoch_numbers, [item.train_loss for item in loss_history], marker="o", label="Train")
        axis.plot(
            epoch_numbers,
            [item.validation_loss for item in loss_history],
            marker="s",
            label="Validation",
        )
        axis.set_title("1D-CNN Loss Curve")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-Entropy Loss")
        axis.set_xticks(epoch_numbers)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(loss_curve_file)
        context.progress(
            "training_completed",
            {
                "backend": self.backend_id,
                "resolved_device": str(device),
                "validation_macro_f1": metrics["validation"].macro_f1,
                "elapsed_seconds": runtime.elapsed_seconds,
            },
        )
        return BackendTrainingOutput(
            runtime=runtime,
            metrics_by_split=metrics,
            epoch_history=tuple(loss_history),
            model_file=checkpoint_file,
            loss_history_file=loss_history_file,
            loss_curve_file=loss_curve_file,
        )
