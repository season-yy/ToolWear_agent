"""PHM2010 C1 小范围窗口样本训练编排。

本模块对应 P0 第 5 步的修正版：
先把每个 CSV 切成窗口样本，再从窗口样本中按阶段比例抽取 20% 做小范围训练。
这样样本数量不再受限于 315 个刀次文件，同时仍然避免同一个 cut 泄露到不同数据集。
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from toolwear_agent.common.config import Settings
from toolwear_agent.data.leakage import assert_no_window_leakage, assert_windows_match_split_manifest
from toolwear_agent.data.sampling import build_training_sample, write_sample_manifest
from toolwear_agent.data.splitting import (
    assert_manifest_matches_lock,
    load_split_lock,
    load_split_manifest,
    normalize_split_name,
)
from toolwear_agent.training.features import SignalFeatureRow, extract_signal_statistics
from toolwear_agent.training.selection import SelectedPlan, _candidate_plan_from_dict
from toolwear_agent.training.windows import (
    DEFAULT_MAX_WINDOWS_PER_CUT,
    DEFAULT_OVERLAP_RATIO,
    DEFAULT_WINDOW_SIZE,
    WindowRecord,
    build_c1_window_manifest,
    load_window_manifest,
)


DEFAULT_SAMPLE_FRACTION = 0.20


@dataclass(frozen=True)
class CudaStatus:
    """CUDA 环境检测结果。"""

    requested_device: str
    torch_available: bool
    cuda_available: bool
    torch_version: str
    cuda_device_name: str
    note: str


@dataclass(frozen=True)
class MiniTrainConfig:
    """一次小范围训练的可回溯配置。"""

    run_id: str
    dataset_id: str
    cutter: str
    selected_plan_id: str
    selected_plan_name: str
    classifier_name: str
    sample_fraction: float
    window_size: int
    overlap_ratio: float
    max_windows_per_cut: int
    random_seed: int
    selected_plan_file: str
    window_manifest_file: str
    split_manifest_file: str
    split_hash: str
    split_lock_file: str
    sample_manifest_file: str
    leakage_audit_file: str


@dataclass(frozen=True)
class MiniTrainResult:
    """一次小范围训练完成后的主要产物索引。"""

    run_id: str
    selected_plan_id: str
    full_window_count: int
    sample_count: int
    train_count: int
    validation_count: int
    feature_count: int
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    stage_distribution: dict[str, int]
    split_distribution: dict[str, int]
    cuda_status: CudaStatus
    run_dir: str
    model_file: str
    metrics_file: str
    config_file: str
    feature_table_file: str
    report_file: str
    log_file: str
    code_snapshot_dir: str
    sample_manifest_file: str


def _now_shanghai_compact() -> str:
    """生成适合作为目录名的上海时间戳。"""

    shanghai_timezone = timezone(timedelta(hours=8))
    return datetime.now(shanghai_timezone).strftime("%Y%m%d_%H%M%S")


def load_selected_plan(selected_plan_file: Path) -> SelectedPlan:
    """读取第 4 步保存的用户确认方案。"""

    if not selected_plan_file.exists():
        raise FileNotFoundError(f"用户确认方案不存在: {selected_plan_file}")
    data = json.loads(selected_plan_file.read_text(encoding="utf-8"))
    return SelectedPlan(
        dataset_id=str(data["dataset_id"]),
        cutter=str(data["cutter"]),
        primary_task=str(data["primary_task"]),
        confirmed_by=str(data["confirmed_by"]),
        confirmed_at=str(data["confirmed_at"]),
        selected_plan=_candidate_plan_from_dict(data["selected_plan"]),
        source_candidate_file=str(data["source_candidate_file"]),
    )


def detect_cuda_status(requested_device: str) -> CudaStatus:
    """检测当前 Python 环境是否可以使用 CUDA。

    PyTorch 官方建议通过 `torch.cuda.is_available()` 判断 CUDA 是否可用：
    https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html
    """

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - 环境检测要保存异常文字，方便复盘
        return CudaStatus(
            requested_device=requested_device,
            torch_available=False,
            cuda_available=False,
            torch_version="",
            cuda_device_name="",
            note=f"torch 导入失败: {exc}",
        )

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else ""
    note = "CUDA 可用" if cuda_available else "CUDA 不可用，当前训练将回退到 CPU 或使用非 CUDA 模型"
    return CudaStatus(
        requested_device=requested_device,
        torch_available=True,
        cuda_available=cuda_available,
        torch_version=str(torch.__version__),
        cuda_device_name=cuda_device_name,
        note=note,
    )


def ensure_c1_window_manifest(settings: Settings) -> Path:
    """确保 C1 窗口 manifest 已存在。

    如果 manifest 不存在，就按固定参数构建一次。后续训练会复用这份文件，不再重复切分。
    """

    output_root = settings.ai_infra_root / "datasets" / "processed" / "phm2010"
    window_manifest_file = output_root / "phm2010_c1_window_manifest.csv"
    split_manifest_file = output_root / "phm2010_c1_split_manifest.json"
    leakage_audit_file = output_root / "phm2010_c1_leakage_audit.json"
    split_lock_file = settings.state_root / "splits" / "phm2010_c1_p0" / "r0001" / "split_lock.json"
    required_files = (window_manifest_file, split_manifest_file, leakage_audit_file, split_lock_file)
    if all(path.exists() for path in required_files):
        split_manifest = load_split_manifest(split_manifest_file)
        split_lock = load_split_lock(split_lock_file)
        assert_manifest_matches_lock(split_manifest, split_lock)
        existing_windows = load_window_manifest(window_manifest_file)
        assert_no_window_leakage(existing_windows)
        assert_windows_match_split_manifest(existing_windows, split_manifest)
        return window_manifest_file

    label_file = output_root / "phm2010_c1_stage_labels.csv"
    cutter_dir = settings.phm2010_raw_root / "c1"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_window_split_report.md"
    log_file = settings.log_root / "phm2010_c1_window_split.log"
    build_c1_window_manifest(
        label_file=label_file,
        cutter_dir=cutter_dir,
        output_root=output_root,
        report_file=report_file,
        log_file=log_file,
        cutter="c1",
        window_size=DEFAULT_WINDOW_SIZE,
        overlap_ratio=DEFAULT_OVERLAP_RATIO,
        max_windows_per_cut=DEFAULT_MAX_WINDOWS_PER_CUT,
        split_lock_file=split_lock_file,
        experiment_id="phm2010_c1_p0",
        revision=1,
    )
    return window_manifest_file


def _count_by(records: Iterable[WindowRecord], field_name: str) -> dict[str, int]:
    """按窗口记录字段统计数量。"""

    counts: dict[str, int] = {}
    for record in records:
        value = str(getattr(record, field_name))
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_window_feature_table(records: list[WindowRecord]) -> tuple[list[SignalFeatureRow], list[int]]:
    """根据窗口索引读取原始 CSV，并提取统计特征。"""

    feature_rows: list[SignalFeatureRow] = []
    y_stage: list[int] = []
    for record in records:
        feature_row = extract_signal_statistics(
            signal_file=Path(record.file_path),
            cut=record.cut,
            max_rows=record.window_size,
            start_row=record.start_row,
        )
        feature_rows.append(feature_row)
        y_stage.append(record.stage_id)
    return feature_rows, y_stage


def write_feature_table_csv(
    feature_rows: list[SignalFeatureRow],
    window_records: list[WindowRecord],
    output_file: Path,
) -> Path:
    """保存训练所用窗口特征表，方便复盘和复现实验。"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_names = feature_rows[0].feature_names if feature_rows else []
    with output_file.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "window_id",
                "cut",
                "split",
                "stage_id",
                "stage_name",
                "vb_value",
                "start_row",
                "end_row",
                "window_size",
                "sampled_rows",
                *feature_names,
            ]
        )
        for feature_row, record in zip(feature_rows, window_records):
            writer.writerow(
                [
                    record.window_id,
                    record.cut,
                    record.split,
                    record.stage_id,
                    record.stage_name,
                    record.vb_value,
                    record.start_row,
                    record.end_row,
                    record.window_size,
                    feature_row.sampled_rows,
                    *feature_row.features,
                ]
            )
    return output_file


def _evaluate_classifier(classifier: object, x_values: list[list[float]], y_stage: list[int]) -> dict[str, object]:
    """对一个 split 计算分类指标。"""

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    predictions = classifier.predict(x_values)
    return {
        "count": len(y_stage),
        "accuracy": float(accuracy_score(y_stage, predictions)),
        "macro_f1": float(f1_score(y_stage, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_stage, predictions)),
        "classification_report": classification_report(y_stage, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_stage, predictions).tolist(),
        "true": [int(item) for item in y_stage],
        "pred": [int(item) for item in predictions],
    }


def train_random_forest_baseline(
    feature_rows: list[SignalFeatureRow],
    window_records: list[WindowRecord],
    y_stage: list[int],
    random_seed: int,
) -> tuple[object, dict[str, object]]:
    """训练随机森林窗口样本基线并只返回 validation 指标。

    scikit-learn 官方说明 RandomForestClassifier 会拟合多棵决策树并做平均/投票：
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    from sklearn.ensemble import RandomForestClassifier

    x_values = [row.features for row in feature_rows]
    normalized_splits = [normalize_split_name(record.split) for record in window_records]
    train_indices = [index for index, split in enumerate(normalized_splits) if split == "train"]
    val_indices = [index for index, split in enumerate(normalized_splits) if split == "validation"]
    if not train_indices:
        raise ValueError("小样本训练缺少 train 窗口。")
    if not val_indices:
        raise ValueError("小样本训练缺少 validation 窗口。")

    classifier = RandomForestClassifier(
        n_estimators=160,
        random_state=random_seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit([x_values[index] for index in train_indices], [y_stage[index] for index in train_indices])

    validation_metrics = _evaluate_classifier(
        classifier,
        [x_values[index] for index in val_indices],
        [y_stage[index] for index in val_indices],
    )
    return classifier, {
        "train_count": len(train_indices),
        "validation": validation_metrics,
        "final_test_status": "not_run_pipeline_not_frozen",
    }


def save_training_code_snapshot(run_dir: Path, app_root: Path) -> Path:
    """保存本次训练用到的关键代码文件快照。"""

    snapshot_dir = run_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        app_root / "toolwear_agent" / "__main__.py",
        app_root / "toolwear_agent" / "training" / "features.py",
        app_root / "toolwear_agent" / "training" / "mini_train.py",
        app_root / "toolwear_agent" / "training" / "windows.py",
        app_root / "toolwear_agent" / "data" / "leakage.py",
        app_root / "toolwear_agent" / "data" / "sampling.py",
        app_root / "toolwear_agent" / "data" / "splitting.py",
        app_root / "pyproject.toml",
    ]
    manifest: list[dict[str, str]] = []
    for source_file in source_files:
        target_file = snapshot_dir / source_file.name
        shutil.copy2(source_file, target_file)
        manifest.append({"source": str(source_file), "snapshot": str(target_file)})
    (snapshot_dir / "code_snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snapshot_dir


def write_mini_train_report(result: MiniTrainResult, config: MiniTrainConfig, output_file: Path) -> Path:
    """写出第 5 步窗口小范围训练 Markdown 报告。"""

    cuda_text = "可用" if result.cuda_status.cuda_available else "不可用"
    lines = [
        "# PHM2010 C1 窗口小范围训练报告",
        "",
        "## 1. 本次训练结论",
        "",
        f"- 运行编号：`{result.run_id}`",
        f"- 选择方案：`{result.selected_plan_id}`",
        f"- 全量窗口样本数：{result.full_window_count}",
        f"- 本次抽样比例：{config.sample_fraction:.0%}",
        f"- 本次训练窗口样本数：{result.sample_count}",
        f"- 训练小样本 / 完整验证集：{result.train_count} / {result.validation_count}",
        "- 最终测试集：未执行（方案尚未冻结，test 保持隔离）",
        f"- 特征数量：{result.feature_count}",
        f"- 验证集 Macro-F1：{result.macro_f1:.4f}",
        f"- 验证集 Balanced Accuracy：{result.balanced_accuracy:.4f}",
        f"- 验证集 Accuracy：{result.accuracy:.4f}",
        "",
        "## 2. 窗口参数",
        "",
        f"- window_size：{config.window_size}",
        f"- overlap_ratio：{config.overlap_ratio}",
        f"- max_windows_per_cut：{config.max_windows_per_cut}",
        "",
        "## 3. CUDA 检测",
        "",
        f"- 期望设备：`{result.cuda_status.requested_device}`",
        f"- torch 版本：`{result.cuda_status.torch_version}`",
        f"- CUDA 状态：{cuda_text}",
        f"- CUDA 设备：{result.cuda_status.cuda_device_name or '无'}",
        f"- 说明：{result.cuda_status.note}",
        "",
        "## 4. 阶段分布",
        "",
    ]
    lines.extend(f"- {stage_name}：{count}" for stage_name, count in result.stage_distribution.items())
    lines.extend(["", "## 5. split 分布", ""])
    lines.extend(f"- {split}：{count}" for split, count in result.split_distribution.items())
    lines.extend(
        [
            "",
            "## 6. 产物索引",
            "",
            f"- 模型文件：`{result.model_file}`",
            f"- 指标文件：`{result.metrics_file}`",
            f"- 配置文件：`{result.config_file}`",
            f"- 特征表：`{result.feature_table_file}`",
            f"- 小样本 Manifest：`{result.sample_manifest_file}`",
            f"- split hash：`{config.split_hash}`",
            f"- split lock：`{config.split_lock_file}`",
            f"- 泄漏审计：`{config.leakage_audit_file}`",
            f"- 代码快照：`{result.code_snapshot_dir}`",
            f"- 日志文件：`{result.log_file}`",
            "",
            "## 7. 对下一步的意义",
            "",
            "本报告证明当前项目已经从 cut 文件级样本升级为窗口级样本。训练前先按 cut 划分并锁定 split，训练小样本只从 train 抽取，validation 用于候选判断，test 在方案冻结前不读取。",
            "",
        ]
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def run_c1_mini_train(settings: Settings) -> MiniTrainResult:
    """执行 PHM2010 C1 窗口小范围训练闭环。"""

    selected_plan_file = settings.ai_infra_root / "experiments" / "selected" / "phm2010_c1_selected_plan.json"
    selected_plan = load_selected_plan(selected_plan_file)
    selected_plan_id = selected_plan.selected_plan.plan_id
    if selected_plan_id != "statistical_features_random_forest":
        raise ValueError(f"当前第 5 步暂只支持 RandomForest 基线方案，实际选择为: {selected_plan_id}")

    cutter = selected_plan.cutter.lower()
    window_manifest_file = ensure_c1_window_manifest(settings)
    processed_root = settings.ai_infra_root / "datasets" / "processed" / "phm2010"
    split_manifest_file = processed_root / "phm2010_c1_split_manifest.json"
    leakage_audit_file = processed_root / "phm2010_c1_leakage_audit.json"
    split_lock_file = settings.state_root / "splits" / "phm2010_c1_p0" / "r0001" / "split_lock.json"
    split_manifest = load_split_manifest(split_manifest_file)
    split_lock = load_split_lock(split_lock_file)
    assert_manifest_matches_lock(split_manifest, split_lock)
    if split_manifest.split_hash is None:  # pragma: no cover - 加载校验后的类型防御
        raise ValueError("split_hash 不能为空。")
    all_windows = load_window_manifest(window_manifest_file)
    assert_no_window_leakage(all_windows)
    assert_windows_match_split_manifest(all_windows, split_manifest)
    training_sample = build_training_sample(
        all_windows,
        dataset_id=selected_plan.dataset_id,
        cutter_id=cutter,
        split_hash=split_manifest.split_hash,
        fraction=DEFAULT_SAMPLE_FRACTION,
        random_seed=settings.random_seed,
    )
    sampled_train_windows = list(training_sample.records)
    validation_windows = [
        record for record in all_windows if normalize_split_name(record.split) == "validation"
    ]
    model_windows = [*sampled_train_windows, *validation_windows]
    assert_no_window_leakage(model_windows)

    run_id = f"phm2010_{cutter}_window_mini_train_{_now_shanghai_compact()}"
    run_dir = settings.experiment_root / run_id
    model_file = settings.artifact_root / "models" / run_id / "model.joblib"
    metrics_file = run_dir / "metrics.json"
    config_file = run_dir / "train_config.json"
    feature_table_file = run_dir / "feature_table.csv"
    sample_manifest_file = run_dir / "training_sample_manifest.json"
    report_file = settings.ai_infra_root / "reports" / "phm2010_c1_mini_train_report.md"
    log_file = settings.log_root / "phm2010_c1_mini_train.log"

    run_dir.mkdir(parents=True, exist_ok=True)
    write_sample_manifest(training_sample.manifest, sample_manifest_file)
    feature_rows, y_stage = build_window_feature_table(model_windows)
    classifier, metrics = train_random_forest_baseline(
        feature_rows=feature_rows,
        window_records=model_windows,
        y_stage=y_stage,
        random_seed=settings.random_seed,
    )
    cuda_status = detect_cuda_status(settings.train_device)

    import joblib

    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_file)
    write_feature_table_csv(feature_rows, model_windows, feature_table_file)
    code_snapshot_dir = save_training_code_snapshot(run_dir, settings.app_root)

    config = MiniTrainConfig(
        run_id=run_id,
        dataset_id=selected_plan.dataset_id,
        cutter=cutter,
        selected_plan_id=selected_plan_id,
        selected_plan_name=selected_plan.selected_plan.display_name,
        classifier_name="RandomForestClassifier",
        sample_fraction=DEFAULT_SAMPLE_FRACTION,
        window_size=DEFAULT_WINDOW_SIZE,
        overlap_ratio=DEFAULT_OVERLAP_RATIO,
        max_windows_per_cut=DEFAULT_MAX_WINDOWS_PER_CUT,
        random_seed=settings.random_seed,
        selected_plan_file=str(selected_plan_file),
        window_manifest_file=str(window_manifest_file),
        split_manifest_file=str(split_manifest_file),
        split_hash=split_manifest.split_hash,
        split_lock_file=str(split_lock_file),
        sample_manifest_file=str(sample_manifest_file),
        leakage_audit_file=str(leakage_audit_file),
    )
    config_file.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")

    validation_metrics = metrics["validation"]
    metrics_payload = {
        **metrics,
        "run_id": run_id,
        "selected_plan_id": selected_plan_id,
        "full_window_count": len(all_windows),
        "full_train_count": training_sample.manifest.full_train_count,
        "sample_count": len(sampled_train_windows),
        "sample_fraction": DEFAULT_SAMPLE_FRACTION,
        "feature_count": len(feature_rows[0].feature_names) if feature_rows else 0,
        "stage_distribution": _count_by(sampled_train_windows, "stage_name"),
        "split_distribution": _count_by(model_windows, "split"),
        "cuda_status": asdict(cuda_status),
        "window_manifest_file": str(window_manifest_file),
        "split_manifest_file": str(split_manifest_file),
        "split_hash": split_manifest.split_hash,
        "split_lock_file": str(split_lock_file),
        "sample_manifest_file": str(sample_manifest_file),
        "sample_hash": training_sample.manifest.sample_hash,
        "leakage_audit_file": str(leakage_audit_file),
        "code_snapshot_dir": str(code_snapshot_dir),
        "source_docs": {
            "random_forest": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
            "classification_report": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html",
            "torch_cuda": "https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html",
        },
    }
    metrics_file.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = MiniTrainResult(
        run_id=run_id,
        selected_plan_id=selected_plan_id,
        full_window_count=len(all_windows),
        sample_count=len(sampled_train_windows),
        train_count=int(metrics["train_count"]),
        validation_count=int(validation_metrics["count"]),
        feature_count=len(feature_rows[0].feature_names) if feature_rows else 0,
        macro_f1=float(validation_metrics["macro_f1"]),
        balanced_accuracy=float(validation_metrics["balanced_accuracy"]),
        accuracy=float(validation_metrics["accuracy"]),
        stage_distribution=_count_by(sampled_train_windows, "stage_name"),
        split_distribution=_count_by(model_windows, "split"),
        cuda_status=cuda_status,
        run_dir=str(run_dir),
        model_file=str(model_file),
        metrics_file=str(metrics_file),
        config_file=str(config_file),
        feature_table_file=str(feature_table_file),
        report_file=str(report_file),
        log_file=str(log_file),
        code_snapshot_dir=str(code_snapshot_dir),
        sample_manifest_file=str(sample_manifest_file),
    )
    write_mini_train_report(result, config, report_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                "PHM2010 C1 窗口小范围训练运行日志",
                f"运行编号: {run_id}",
                f"选择方案: {selected_plan_id}",
                f"窗口 manifest: {window_manifest_file}",
                f"全量窗口样本数: {result.full_window_count}",
                f"本次抽样比例: {DEFAULT_SAMPLE_FRACTION:.0%}",
                f"本次样本数: {result.sample_count}",
                f"训练小样本/完整验证集: {result.train_count}/{result.validation_count}",
                "最终测试集: 未执行（方案尚未冻结）",
                f"split hash: {split_manifest.split_hash}",
                f"split lock: {split_lock_file}",
                f"sample manifest: {sample_manifest_file}",
                f"验证集 Macro-F1: {result.macro_f1:.4f}",
                f"验证集 Balanced Accuracy: {result.balanced_accuracy:.4f}",
                f"CUDA 状态: {cuda_status.note}",
                f"模型文件: {model_file}",
                f"指标文件: {metrics_file}",
                f"代码快照: {code_snapshot_dir}",
                f"报告文件: {report_file}",
            ]
        ),
        encoding="utf-8",
    )
    return result
