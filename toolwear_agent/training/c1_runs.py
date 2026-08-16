"""PHM2010 C1 的可复现统一训练入口。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from toolwear_agent.core.settings import Settings
from toolwear_agent.data.sampling import load_sample_manifest
from toolwear_agent.data.splitting import load_split_manifest
from toolwear_agent.schemas import ModuleSpec, PipelineSpec, RunConfig, TrainingDataRef, TrainingRunResult
from toolwear_agent.schemas.converters import candidate_plan_to_pipeline
from toolwear_agent.training.candidates import build_default_candidate_set
from toolwear_agent.training.service import TrainingService


C1_EXPERIMENT_ID = "phm2010_c1_p0"
C1_REVISION = 1


def build_c1_training_data_ref(settings: Settings) -> TrainingDataRef:
    """返回 C1 已锁定窗口、split 和训练小样本的稳定引用。"""

    processed_root = settings.ai_infra_root / "datasets" / "processed" / "phm2010"
    return TrainingDataRef(
        dataset_id="phm2010",
        cutter_id="c1",
        window_manifest_file=processed_root / "phm2010_c1_window_manifest.csv",
        training_sample_manifest_file=processed_root / "phm2010_c1_train_sample_20pct.json",
        split_manifest_file=processed_root / "phm2010_c1_split_manifest.json",
        split_lock_file=(
            settings.state_root
            / "splits"
            / C1_EXPERIMENT_ID
            / f"r{C1_REVISION:04d}"
            / "split_lock.json"
        ),
        leakage_audit_file=processed_root / "phm2010_c1_leakage_audit.json",
    )


def _replace_module(
    modules: tuple[ModuleSpec, ...],
    *,
    kind: str,
    module_id: str | None = None,
    parameters: dict[str, object] | None = None,
) -> tuple[ModuleSpec, ...]:
    """替换指定类别模块的 ID 或参数，并保持原执行顺序。"""

    replaced = False
    updated: list[ModuleSpec] = []
    for module in modules:
        if module.kind.value != kind:
            updated.append(module)
            continue
        if replaced:
            raise ValueError(f"Pipeline 中存在多个 {kind} 模块。")
        payload = module.model_dump(mode="python")
        if module_id is not None:
            payload["module_id"] = module_id
        if parameters is not None:
            payload["parameters"] = parameters
        updated.append(ModuleSpec.model_validate(payload))
        replaced = True
    if not replaced:
        raise ValueError(f"Pipeline 中不存在 {kind} 模块。")
    return tuple(updated)


def build_c1_pipeline(
    plan_id: str,
    *,
    input_channels: tuple[str, ...] | None = None,
    n_estimators: int = 300,
    max_depth: int = 0,
    class_weight: str = "balanced",
    base_channels: int = 32,
    dropout: float = 0.2,
    loss_id: str = "cross_entropy",
) -> PipelineSpec:
    """从默认候选构建参数已落入 ModuleSpec 的可执行 Pipeline。"""

    candidate_set = build_default_candidate_set("phm2010", "c1")
    try:
        candidate = next(plan for plan in candidate_set.plans if plan.plan_id == plan_id)
    except StopIteration as exc:
        available = ", ".join(plan.plan_id for plan in candidate_set.plans)
        raise ValueError(f"未知 C1 方案 {plan_id}，可选方案: {available}") from exc
    pipeline = candidate_plan_to_pipeline(candidate)
    if not pipeline.trainable:
        raise ValueError(f"方案 {plan_id} 当前仅可展示，尚不能进入训练服务。")

    modules = _replace_module(
        pipeline.modules,
        kind="windowing",
        parameters={"window_length": 4096, "overlap": 0.5},
    )
    model_id = next(module.module_id for module in modules if module.kind.value == "model")
    if model_id in {"random_forest", "extra_trees"}:
        modules = _replace_module(
            modules,
            kind="model",
            parameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "class_weight": class_weight,
            },
        )
    elif model_id == "cnn_1d":
        if loss_id not in {"cross_entropy", "weighted_cross_entropy"}:
            raise ValueError("CNN loss_id 只能是 cross_entropy 或 weighted_cross_entropy。")
        modules = _replace_module(
            modules,
            kind="model",
            parameters={"base_channels": base_channels, "dropout": dropout},
        )
        modules = _replace_module(modules, kind="loss", module_id=loss_id, parameters={})
    else:  # pragma: no cover - 固定候选和 Registry 已限制
        raise ValueError(f"当前 C1 入口不支持模型: {model_id}")

    payload = pipeline.model_dump(mode="python")
    payload["modules"] = modules
    if input_channels is not None:
        payload["input_channels"] = input_channels
    return PipelineSpec.model_validate(payload)


def _default_run_id(plan_id: str) -> str:
    """生成上海时区、可读且不会复用旧产物的运行编号。"""

    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    model_token = {
        "statistical_features_random_forest": "rf",
        "statistical_features_extra_trees": "extra_trees",
        "multichannel_window_1d_cnn": "cnn1d",
    }[plan_id]
    return f"c1_{model_token}_{now}"


def run_c1_training(
    settings: Settings,
    *,
    plan_id: str,
    run_id: str | None = None,
    input_channels: tuple[str, ...] | None = None,
    max_samples: int | None = None,
    batch_size: int = 64,
    epochs: int = 2,
    learning_rate: float = 0.001,
    device: str | None = None,
    n_estimators: int = 300,
    max_depth: int = 0,
    class_weight: str = "balanced",
    base_channels: int = 32,
    dropout: float = 0.2,
    loss_id: str = "cross_entropy",
) -> TrainingRunResult:
    """使用既有 20% 训练样本和锁定 validation 执行一次 C1 训练。"""

    data_ref = build_c1_training_data_ref(settings)
    sample_manifest = load_sample_manifest(data_ref.training_sample_manifest_file)
    split_manifest = load_split_manifest(data_ref.split_manifest_file)
    if split_manifest.split_hash is None:  # pragma: no cover - loader 已校验
        raise ValueError("C1 split_hash 不能为空。")
    pipeline = build_c1_pipeline(
        plan_id,
        input_channels=input_channels,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        base_channels=base_channels,
        dropout=dropout,
        loss_id=loss_id,
    )
    run_config = RunConfig(
        run_id=run_id or _default_run_id(plan_id),
        experiment_id=C1_EXPERIMENT_ID,
        revision=C1_REVISION,
        pipeline_id=pipeline.pipeline_id,
        run_kind="smoke" if max_samples is not None else "mini_train",
        split_hash=split_manifest.split_hash,
        sample_fraction=sample_manifest.fraction,
        max_samples=max_samples,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device or settings.train_device,
        random_seed=settings.random_seed,
        num_workers=0,
        evaluate_test=False,
    )
    return TrainingService(settings).train(
        pipeline=pipeline,
        run_config=run_config,
        data_ref=data_ref,
    )
