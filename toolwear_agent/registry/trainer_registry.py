"""P0 Trainer Registry。"""

from __future__ import annotations

from toolwear_agent.schemas import ParameterRule, TrainerDefinition


class TrainerRegistry:
    """保存训练后端以及模型、损失和资源兼容关系。"""

    def __init__(self, trainers: tuple[TrainerDefinition, ...] = ()) -> None:
        self._trainers: dict[str, TrainerDefinition] = {}
        for trainer in trainers:
            self.register(trainer)

    def register(self, trainer: TrainerDefinition) -> None:
        """登记训练器；重复 ID 必须明确失败。"""

        if trainer.trainer_id in self._trainers:
            raise ValueError(f"Trainer Registry 已存在训练器: {trainer.trainer_id}")
        self._trainers[trainer.trainer_id] = trainer

    def get(self, trainer_id: str) -> TrainerDefinition:
        """按稳定 ID 返回训练器定义。"""

        try:
            return self._trainers[trainer_id]
        except KeyError as exc:
            raise KeyError(f"Trainer Registry 中不存在训练器: {trainer_id}") from exc

    def list_trainers(self) -> tuple[TrainerDefinition, ...]:
        """按 ID 稳定返回所有训练器。"""

        return tuple(self._trainers[key] for key in sorted(self._trainers))


def _integer(
    description: str,
    default: int,
    minimum: int,
    maximum: int,
) -> ParameterRule:
    """构造训练器整数参数约束。"""

    return ParameterRule(
        value_type="integer",
        description=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _number(
    description: str,
    default: float,
    minimum: float,
    maximum: float,
) -> ParameterRule:
    """构造训练器浮点参数约束。"""

    return ParameterRule(
        value_type="number",
        description=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def build_default_trainer_registry() -> TrainerRegistry:
    """返回 sklearn 与 PyTorch 两个明确边界的训练后端。"""

    return TrainerRegistry(
        trainers=(
            TrainerDefinition(
                trainer_id="sklearn",
                display_name="scikit-learn 训练器",
                backend="sklearn",
                supported_model_ids=("random_forest", "extra_trees"),
                parameters_schema={
                    "n_jobs": _integer("并行 CPU 任务数；-1 表示使用全部核心。", -1, -1, 64),
                },
                resource_class="low",
                implemented=True,
            ),
            TrainerDefinition(
                trainer_id="pytorch",
                display_name="PyTorch CUDA 训练器",
                backend="pytorch",
                supported_model_ids=("cnn_1d",),
                supported_loss_ids=("cross_entropy", "weighted_cross_entropy"),
                parameters_schema={
                    "batch_size": _integer("每个优化步骤的窗口数量。", 64, 1, 1024),
                    "epochs": _integer("最大训练轮数。", 5, 1, 10000),
                    "learning_rate": _number("优化器初始学习率。", 0.001, 1e-6, 1.0),
                    "num_workers": _integer("DataLoader 子进程数量。", 0, 0, 16),
                },
                resource_class="medium",
                requires_cuda=True,
                implemented=True,
            ),
        )
    )
