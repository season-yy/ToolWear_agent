"""真实 C1 统一训练入口的配置测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.core.settings import Settings
from toolwear_agent.registry import validate_pipeline_with_default_registries
from toolwear_agent.training.c1_runs import build_c1_pipeline, build_c1_training_data_ref


class C1TrainingConfigurationTests(unittest.TestCase):
    """验证 CLI/API 后续共用的 C1 配置不会退回方案 ID 分支。"""

    def test_build_c1_cnn_pipeline_applies_user_parameters(self) -> None:
        """通道、模型规模和损失函数都必须进入统一 PipelineSpec。"""

        pipeline = build_c1_pipeline(
            "multichannel_window_1d_cnn",
            input_channels=("acoustic_emission_rms",),
            base_channels=16,
            dropout=0.3,
            loss_id="weighted_cross_entropy",
        )

        self.assertTrue(pipeline.trainable)
        self.assertEqual(pipeline.input_channels, ("acoustic_emission_rms",))
        self.assertIn("weighted_cross_entropy", pipeline.module_ids)
        model_module = next(module for module in pipeline.modules if module.kind.value == "model")
        self.assertEqual(model_module.parameters["base_channels"], 16)
        self.assertEqual(model_module.parameters["dropout"], 0.3)
        validation = validate_pipeline_with_default_registries(pipeline)
        self.assertTrue(validation.valid, validation.issues)

    def test_build_c1_random_forest_pipeline_applies_tree_parameters(self) -> None:
        """传统模型参数也必须写入模块配置，而不是藏在训练函数常量中。"""

        pipeline = build_c1_pipeline(
            "statistical_features_random_forest",
            n_estimators=120,
            max_depth=12,
            class_weight="balanced_subsample",
        )

        model_module = next(module for module in pipeline.modules if module.kind.value == "model")
        self.assertEqual(
            model_module.parameters,
            {"n_estimators": 120, "max_depth": 12, "class_weight": "balanced_subsample"},
        )
        self.assertTrue(validate_pipeline_with_default_registries(pipeline).valid)

    def test_data_ref_uses_stable_processed_and_split_lock_paths(self) -> None:
        """真实训练必须引用既有锁定文件，不能每次重新切分。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(ai_infra_root=root, phm2010_raw_root=root / "raw")
            data_ref = build_c1_training_data_ref(settings)

        self.assertEqual(
            data_ref.window_manifest_file,
            root / "datasets" / "processed" / "phm2010" / "phm2010_c1_window_manifest.csv",
        )
        self.assertEqual(
            data_ref.split_lock_file,
            root / "state" / "splits" / "phm2010_c1_p0" / "r0001" / "split_lock.json",
        )


if __name__ == "__main__":
    unittest.main()
