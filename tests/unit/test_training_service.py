"""统一训练服务、sklearn 后端和 PyTorch 1D-CNN 的行为测试。"""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from toolwear_agent.core.settings import Settings
from toolwear_agent.core.errors import PathBoundaryError
from toolwear_agent.data.leakage import assert_no_window_leakage, write_leakage_audit
from toolwear_agent.data.sampling import build_training_sample, write_sample_manifest
from toolwear_agent.data.splitting import (
    attach_split_hash,
    build_split_manifest,
    create_or_verify_split_lock,
    write_split_manifest,
)
from toolwear_agent.schemas import ModuleSpec, PipelineSpec, RunConfig, SplitSpec
from toolwear_agent.schemas.training import TrainingDataRef
from toolwear_agent.training.models import LightweightCNN1D
from toolwear_agent.training.backends import PytorchTrainingBackend
from toolwear_agent.training.service import TrainingService
from toolwear_agent.training.windows import CutLabel, WindowRecord, write_window_manifest


class SyntheticTrainingFixture:
    """在临时目录构造一个含 train/validation/test 的四阶段数据集。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw_root = root / "datasets" / "raw" / "phm2010"
        self.processed_root = root / "datasets" / "processed" / "phm2010"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.records: list[WindowRecord] = []
        self.cut_labels: list[CutLabel] = []
        self.split_by_cut: dict[int, str] = {}
        self._write_signals_and_windows()
        self.data_ref, self.split_hash = self._write_manifests()

    def _write_signal(self, file_path: Path, *, stage_id: int, cut: int) -> None:
        """写入可学习但很小的七通道时序信号。"""

        rng = np.random.default_rng(10_000 + cut)
        values = rng.normal(loc=float(stage_id) * 1.5, scale=0.2, size=(512, 7))
        with file_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerows(values.tolist())

    def _write_signals_and_windows(self) -> None:
        """每个阶段建立三个互不重叠的 cut，各自归入一个 split。"""

        cutter_dir = self.raw_root / "c1"
        cutter_dir.mkdir(parents=True, exist_ok=True)
        split_offsets = {"train": 1, "validation": 2, "test": 3}
        for stage_id in range(4):
            for split, offset in split_offsets.items():
                cut = stage_id * 10 + offset
                signal_file = cutter_dir / f"c_1_{cut:03d}.csv"
                self._write_signal(signal_file, stage_id=stage_id, cut=cut)
                self.cut_labels.append(
                    CutLabel(
                        cut=cut,
                        file_path=str(signal_file),
                        row_count=512,
                        vb_value=float(50 + stage_id * 50),
                        stage_id=stage_id,
                        stage_name=f"stage_{stage_id}",
                    )
                )
                self.split_by_cut[cut] = split
                for window_index, start_row in enumerate((0, 256)):
                    self.records.append(
                        WindowRecord(
                            window_id=f"c1_{cut:03d}_w{window_index:03d}",
                            cut=cut,
                            file_path=str(signal_file),
                            row_count=512,
                            start_row=start_row,
                            end_row=start_row + 256,
                            window_size=256,
                            stride=256,
                            overlap_ratio=0.0,
                            vb_value=float(50 + stage_id * 50),
                            stage_id=stage_id,
                            stage_name=f"stage_{stage_id}",
                            split=split,
                        )
                    )

    def _write_manifests(self) -> tuple[TrainingDataRef, str]:
        """写出和真实 C1 训练相同形态的四类输入证据。"""

        split_manifest = attach_split_hash(
            build_split_manifest(
                cut_labels=self.cut_labels,
                split_by_cut=self.split_by_cut,
                dataset_id="phm2010",
                cutter_id="c1",
                split_spec=SplitSpec(random_seed=42),
            )
        )
        assert split_manifest.split_hash is not None
        split_file = self.processed_root / "split_manifest.json"
        lock_file = self.root / "state" / "splits" / "synthetic-exp" / "r0001" / "split_lock.json"
        window_file = self.processed_root / "window_manifest.csv"
        sample_file = self.processed_root / "train_sample.json"
        leakage_file = self.processed_root / "leakage_audit.json"
        write_split_manifest(split_manifest, split_file)
        create_or_verify_split_lock(
            manifest=split_manifest,
            lock_file=lock_file,
            experiment_id="synthetic-exp",
            revision=1,
            manifest_file=split_file,
        )
        write_window_manifest(self.records, window_file)
        sample = build_training_sample(
            self.records,
            dataset_id="phm2010",
            cutter_id="c1",
            split_hash=split_manifest.split_hash,
            fraction=1.0,
            random_seed=42,
        )
        write_sample_manifest(sample.manifest, sample_file)
        write_leakage_audit(assert_no_window_leakage(self.records), leakage_file)
        return (
            TrainingDataRef(
                dataset_id="phm2010",
                cutter_id="c1",
                window_manifest_file=window_file,
                training_sample_manifest_file=sample_file,
                split_manifest_file=split_file,
                split_lock_file=lock_file,
                leakage_audit_file=leakage_file,
            ),
            split_manifest.split_hash,
        )


def _random_forest_pipeline() -> PipelineSpec:
    """构造由 Registry 可执行的最小 RandomForest Pipeline。"""

    return PipelineSpec(
        pipeline_id="rf-pipeline",
        display_name="统计特征 RandomForest",
        source="user",
        input_channels=("force_x", "force_y", "force_z"),
        modules=(
            ModuleSpec(
                module_id="sliding_window",
                kind="windowing",
                order=10,
                parameters={"window_length": 256, "overlap": 0.0},
            ),
            ModuleSpec(module_id="statistical_features", kind="feature", order=20),
            ModuleSpec(
                module_id="random_forest",
                kind="model",
                order=30,
                parameters={"n_estimators": 10, "max_depth": 4, "class_weight": "balanced"},
            ),
            ModuleSpec(module_id="sklearn", kind="trainer", order=40),
        ),
        rationale="验证统一传统模型训练入口。",
        risks=("仅为快速训练测试。",),
        expected_cost="low",
    )


def _cnn_pipeline(channel_count: int = 3) -> PipelineSpec:
    """构造最小 1D-CNN Pipeline，可覆盖 1/3/7 通道。"""

    all_channels = (
        "force_x",
        "force_y",
        "force_z",
        "vibration_x",
        "vibration_y",
        "vibration_z",
        "acoustic_emission_rms",
    )
    return PipelineSpec(
        pipeline_id="cnn-pipeline",
        display_name="轻量 1D-CNN",
        source="user",
        input_channels=all_channels[:channel_count],
        modules=(
            ModuleSpec(
                module_id="sliding_window",
                kind="windowing",
                order=10,
                parameters={"window_length": 256, "overlap": 0.0},
            ),
            ModuleSpec(module_id="zscore", kind="preprocess", order=20),
            ModuleSpec(module_id="raw_1d", kind="feature", order=30),
            ModuleSpec(
                module_id="cnn_1d",
                kind="model",
                order=40,
                parameters={"base_channels": 8, "dropout": 0.1},
            ),
            ModuleSpec(
                module_id="cross_entropy",
                kind="loss",
                order=50,
                parameters={"label_smoothing": 0.0},
            ),
            ModuleSpec(module_id="pytorch", kind="trainer", order=60),
        ),
        rationale="验证原始时序的真实前向、反向和损失记录。",
        risks=("小样本结果仅用于链路验证。",),
        expected_cost="medium",
    )


class LightweightCNN1DTests(unittest.TestCase):
    """验证模型结构不会绑定固定通道数或固定窗口长度。"""

    def test_forward_supports_one_three_and_seven_channels(self) -> None:
        """1/3/7 通道和不同长度都应输出四分类 logits。"""

        for channel_count, window_length in ((1, 256), (3, 384), (7, 512)):
            with self.subTest(channel_count=channel_count, window_length=window_length):
                model = LightweightCNN1D(
                    input_channels=channel_count,
                    base_channels=8,
                    class_count=4,
                    dropout=0.1,
                )
                output = model(torch.randn(2, channel_count, window_length))
                self.assertEqual(tuple(output.shape), (2, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "当前机器没有 CUDA")
    def test_cuda_alias_resolves_to_an_indexed_device(self) -> None:
        """PyTorch 2.11 的 set_device 要求 cuda:0，而不是无编号的 cuda。"""

        device, cuda_available = PytorchTrainingBackend._resolve_device("cuda")

        self.assertTrue(cuda_available)
        self.assertEqual(device.type, "cuda")
        self.assertIsNotNone(device.index)


class TrainingServiceTests(unittest.TestCase):
    """用真实 sklearn/PyTorch 训练验证统一服务的外部行为。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fixture = SyntheticTrainingFixture(self.root)
        repository_root = Path(__file__).resolve().parents[2]
        self.settings = Settings(
            project_root=repository_root.parent,
            app_root=repository_root,
            ai_infra_root=self.root,
            phm2010_raw_root=self.fixture.raw_root,
            train_device="cpu",
        )
        self.service = TrainingService(self.settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_config(self, *, run_id: str, pipeline_id: str, epochs: int = 1) -> RunConfig:
        """创建与锁定 split 一致且明确禁止 test 的运行配置。"""

        return RunConfig(
            run_id=run_id,
            experiment_id="synthetic-exp",
            revision=1,
            pipeline_id=pipeline_id,
            run_kind="smoke",
            split_hash=self.fixture.split_hash,
            sample_fraction=1.0,
            batch_size=4,
            epochs=epochs,
            learning_rate=0.001,
            device="cpu",
            random_seed=42,
            num_workers=0,
            evaluate_test=False,
        )

    def test_run_config_rejects_unsupported_device_syntax(self) -> None:
        """API 不能把任意字符串继续传给 torch.device。"""

        payload = self._run_config(run_id="bad-device", pipeline_id="cnn-pipeline").model_dump(
            mode="python"
        )
        payload["device"] = "gpu"
        with self.assertRaisesRegex(ValueError, "device"):
            RunConfig.model_validate(payload)

    def test_sklearn_backend_writes_real_metrics_model_and_evidence(self) -> None:
        """传统模型也必须通过 TrainingService，而不是旧页面分支。"""

        pipeline = _random_forest_pipeline()
        result = self.service.train(
            pipeline=pipeline,
            run_config=self._run_config(run_id="rf-smoke", pipeline_id=pipeline.pipeline_id),
            data_ref=self.fixture.data_ref,
        )

        self.assertEqual(result.runtime.backend, "sklearn")
        self.assertEqual(result.train_sample_count, 8)
        self.assertEqual(result.validation_sample_count, 8)
        self.assertEqual({item.split.value for item in result.evaluation.metrics}, {"train", "validation"})
        self.assertTrue(Path(result.artifacts.model_file).is_file())
        self.assertTrue(Path(result.artifacts.metrics_file).is_file())
        self.assertTrue(Path(result.artifacts.evidence_index_file).is_file())
        self.assertIsNotNone(result.artifacts.validation_tsne_file)
        self.assertTrue(Path(result.artifacts.validation_tsne_file).is_file())
        metrics_payload = json.loads(Path(result.artifacts.metrics_file).read_text(encoding="utf-8"))
        self.assertNotIn("test", metrics_payload["metrics_by_split"])
        evidence_payload = json.loads(
            Path(result.artifacts.evidence_index_file).read_text(encoding="utf-8")
        )
        log_evidence = next(
            item for item in evidence_payload["evidence"] if item["uri"] == str(result.artifacts.log_file)
        )
        actual_log_hash = hashlib.sha256(Path(result.artifacts.log_file).read_bytes()).hexdigest()
        self.assertEqual(log_evidence["sha256"], actual_log_hash)
        tsne_evidence = next(
            item
            for item in evidence_payload["evidence"]
            if item["description"] == "仅使用 validation 的 t-SNE 特征分布"
        )
        self.assertEqual(tsne_evidence["media_type"], "image/png")
        self.assertEqual(tsne_evidence["uri"], str(result.artifacts.validation_tsne_file))

    def test_pytorch_backend_records_real_epoch_losses_and_checkpoint(self) -> None:
        """CPU smoke 也要真实反向传播；生产 CUDA 运行使用同一后端。"""

        pipeline = _cnn_pipeline(channel_count=3)
        result = self.service.train(
            pipeline=pipeline,
            run_config=self._run_config(run_id="cnn-smoke", pipeline_id=pipeline.pipeline_id),
            data_ref=self.fixture.data_ref,
        )

        self.assertEqual(result.runtime.backend, "pytorch")
        self.assertEqual(result.runtime.resolved_device, "cpu")
        self.assertFalse(result.runtime.cuda_used)
        self.assertEqual(len(result.epoch_history), 1)
        self.assertGreaterEqual(result.epoch_history[0].train_loss, 0.0)
        self.assertGreaterEqual(result.epoch_history[0].validation_loss, 0.0)
        self.assertTrue(Path(result.artifacts.loss_curve_file).is_file())
        checkpoint = torch.load(result.artifacts.model_file, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["input_channels"], 3)

    def test_split_hash_mismatch_is_rejected_before_reading_signals(self) -> None:
        """运行配置与锁定 split 不一致时必须在训练前阻断。"""

        pipeline = _random_forest_pipeline()
        run_config = self._run_config(run_id="bad-split", pipeline_id=pipeline.pipeline_id).model_copy(
            update={"split_hash": "f" * 64}
        )

        with self.assertRaisesRegex(ValueError, "split_hash"):
            self.service.train(
                pipeline=pipeline,
                run_config=run_config,
                data_ref=self.fixture.data_ref,
            )

    def test_manifest_path_outside_ai_infra_root_is_rejected(self) -> None:
        """未来 API 传入的 Manifest 路径不能读取任意本地文件。"""

        pipeline = _random_forest_pipeline()
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "window_manifest.csv"
            outside_file.write_text("untrusted", encoding="utf-8")
            payload = self.fixture.data_ref.model_dump(mode="python")
            payload["window_manifest_file"] = outside_file
            untrusted_ref = TrainingDataRef.model_validate(payload)

            with self.assertRaises(PathBoundaryError):
                self.service.train(
                    pipeline=pipeline,
                    run_config=self._run_config(
                        run_id="unsafe-path",
                        pipeline_id=pipeline.pipeline_id,
                    ),
                    data_ref=untrusted_ref,
                )


if __name__ == "__main__":
    unittest.main()
