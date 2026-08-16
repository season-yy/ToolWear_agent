"""Module Registry、Trainer Registry 与 Pipeline 兼容性测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from toolwear_agent.schemas import ModuleSpec, ParameterRule, PipelineSpec, RegistryCatalog
from toolwear_agent.schemas.converters import candidate_plan_to_pipeline
from toolwear_agent.registry import (
    build_default_registry_catalog,
    build_default_module_registry,
    build_default_trainer_registry,
    load_registry_catalog,
    validate_pipeline_against_registries,
    write_registry_catalog,
)
from toolwear_agent.training.candidates import build_default_candidate_set


def _random_forest_pipeline(**module_parameters: object) -> PipelineSpec:
    """返回使用默认 Registry ID 的最小 RandomForest Pipeline。"""

    return PipelineSpec(
        pipeline_id="rf-pipeline",
        display_name="统计特征 + RandomForest",
        source="user",
        input_channels=(
            "force_x",
            "force_y",
            "force_z",
            "vibration_x",
            "vibration_y",
            "vibration_z",
            "acoustic_emission_rms",
        ),
        modules=(
            ModuleSpec(module_id="sliding_window", kind="windowing", order=10),
            ModuleSpec(module_id="statistical_features", kind="feature", order=20),
            ModuleSpec(
                module_id="random_forest",
                kind="model",
                order=30,
                parameters=module_parameters,
            ),
            ModuleSpec(module_id="sklearn", kind="trainer", order=40),
        ),
        rationale="低成本可解释基线。",
        risks=("跨刀具泛化需要后续验证。",),
        expected_cost="low",
    )


def _cnn_pipeline(*, trainer_id: str = "pytorch", trainable: bool = False) -> PipelineSpec:
    """返回结构合法但当前实现状态可变的 1D-CNN Pipeline。"""

    return PipelineSpec(
        pipeline_id="cnn-pipeline",
        display_name="多通道 1D-CNN",
        source="user",
        input_channels=("force_x", "force_y", "force_z"),
        modules=(
            ModuleSpec(module_id="sliding_window", kind="windowing", order=10),
            ModuleSpec(module_id="zscore", kind="preprocess", order=20),
            ModuleSpec(module_id="raw_1d", kind="feature", order=30),
            ModuleSpec(module_id="cnn_1d", kind="model", order=40),
            ModuleSpec(module_id="cross_entropy", kind="loss", order=50),
            ModuleSpec(module_id=trainer_id, kind="trainer", order=60),
        ),
        rationale="保留原始时序结构。",
        risks=("当前 CUDA 训练服务尚未实现。",),
        expected_cost="medium",
        trainable=trainable,
    )


class DefaultRegistryTest(unittest.TestCase):
    """验证大纲要求的 P0 能力都被明确登记。"""

    def test_default_module_registry_contains_required_capabilities(self) -> None:
        """输入、窗口、特征、模型、损失和融合能力必须可查询。"""

        registry = build_default_module_registry()

        self.assertEqual(
            {preset.preset_id for preset in registry.list_input_presets()},
            {"force_xyz", "vibration_xyz", "all_7_channels"},
        )
        self.assertTrue(
            {
                "sliding_window",
                "stable_region",
                "zscore",
                "robust_scaler",
                "statistical_features",
                "raw_1d",
                "random_forest",
                "extra_trees",
                "cnn_1d",
                "cross_entropy",
                "weighted_cross_entropy",
                "early_concat",
            }.issubset({module.module_id for module in registry.list_modules()})
        )

    def test_default_trainer_registry_has_sklearn_and_pytorch(self) -> None:
        """两种训练后端必须有实现状态和资源要求。"""

        registry = build_default_trainer_registry()

        self.assertEqual({trainer.trainer_id for trainer in registry.list_trainers()}, {"sklearn", "pytorch"})
        self.assertTrue(registry.get("sklearn").implemented)
        self.assertTrue(registry.get("pytorch").implemented)
        self.assertTrue(registry.get("pytorch").requires_cuda)

    def test_registry_catalog_can_be_written_and_loaded_as_json(self) -> None:
        """Catalog 必须能直接供后续 API 和页面读取。"""

        catalog = build_default_registry_catalog()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "registry_catalog.json"
            write_registry_catalog(catalog, output_file)
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            restored = load_registry_catalog(output_file)

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(len(payload["input_presets"]), 3)
        self.assertGreaterEqual(len(payload["modules"]), 12)
        self.assertEqual(len(payload["trainers"]), 2)
        self.assertEqual(restored, catalog)

    def test_registry_catalog_rejects_tampered_file(self) -> None:
        """Catalog 落盘内容与 hash 不一致时必须拒绝加载。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "registry_catalog.json"
            write_registry_catalog(build_default_registry_catalog(), output_file)
            content = output_file.read_text(encoding="utf-8")
            output_file.write_text(
                content.replace("RandomForest 分类器", "被修改的分类器"),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_registry_catalog(output_file)

    def test_parameter_rule_rejects_invalid_default(self) -> None:
        """Registry 自己的默认参数也必须符合声明类型和范围。"""

        with self.assertRaises(ValidationError):
            ParameterRule(
                value_type="integer",
                description="非法默认值测试。",
                default="three",
                minimum=1,
                maximum=10,
            )

    def test_catalog_rejects_trainer_reference_to_unknown_model(self) -> None:
        """训练器不能声称支持 Catalog 中不存在的模型。"""

        payload = build_default_registry_catalog().model_dump(mode="python")
        payload["catalog_hash"] = None
        payload["trainers"][0]["supported_model_ids"] = ("unknown_model",)

        with self.assertRaises(ValidationError):
            RegistryCatalog.model_validate(payload)


class PipelineRegistryValidationTest(unittest.TestCase):
    """验证模块链不会把页面文字直接当成可执行配置。"""

    def setUp(self) -> None:
        self.modules = build_default_module_registry()
        self.trainers = build_default_trainer_registry()

    def validate(self, pipeline: PipelineSpec):
        """使用默认两个 Registry 校验 Pipeline。"""

        return validate_pipeline_against_registries(pipeline, self.modules, self.trainers)

    def test_random_forest_pipeline_is_trainable(self) -> None:
        """已实现的统计特征基线必须通过全部兼容性检查。"""

        result = self.validate(_random_forest_pipeline(n_estimators=300, class_weight="balanced"))

        self.assertTrue(result.valid, result.issues)
        self.assertFalse(any(issue.severity.value == "error" for issue in result.issues))

    def test_unknown_module_is_rejected(self) -> None:
        """LLM 自创模块名不能伪装成本地已实现能力。"""

        pipeline = _random_forest_pipeline()
        modules = list(pipeline.modules)
        modules[1] = ModuleSpec(module_id="llm_magic_attention", kind="feature", order=20)
        pipeline = pipeline.model_copy(update={"modules": tuple(modules)})

        result = self.validate(pipeline)

        self.assertFalse(result.valid)
        self.assertIn("MODULE_NOT_REGISTERED", {issue.code for issue in result.issues})

    def test_module_kind_mismatch_is_rejected(self) -> None:
        """同一个 ID 不能在 Pipeline 中冒充另一种模块类别。"""

        pipeline = _random_forest_pipeline()
        modules = list(pipeline.modules)
        modules[1] = ModuleSpec(module_id="statistical_features", kind="preprocess", order=20)
        pipeline = pipeline.model_copy(update={"modules": tuple(modules)})

        result = self.validate(pipeline)

        self.assertFalse(result.valid)
        self.assertIn("MODULE_KIND_MISMATCH", {issue.code for issue in result.issues})

    def test_parameter_outside_schema_is_rejected(self) -> None:
        """模型参数必须满足 Registry 的类型、范围和枚举约束。"""

        result = self.validate(_random_forest_pipeline(n_estimators=0, class_weight="unknown"))

        self.assertFalse(result.valid)
        codes = {issue.code for issue in result.issues}
        self.assertIn("PARAMETER_OUT_OF_RANGE", codes)
        self.assertIn("PARAMETER_NOT_ALLOWED", codes)

    def test_cnn_cannot_use_sklearn_trainer(self) -> None:
        """PyTorch 模型接 sklearn trainer 必须在训练前被拒绝。"""

        result = self.validate(_cnn_pipeline(trainer_id="sklearn"))

        self.assertFalse(result.valid)
        self.assertIn("TRAINER_BACKEND_MISMATCH", {issue.code for issue in result.issues})

    def test_cnn_rejects_channel_counts_other_than_one_three_or_seven(self) -> None:
        """离散通道约束必须在用户审批前由 Registry 拦截。"""

        pipeline = _cnn_pipeline(trainable=True).model_copy(
            update={"input_channels": ("force_x", "force_y")}
        )

        result = self.validate(pipeline)

        self.assertFalse(result.valid)
        self.assertIn("CHANNEL_COUNT_NOT_SUPPORTED", {issue.code for issue in result.issues})

    def test_feature_type_mismatch_is_rejected(self) -> None:
        """统计特征输出不能直接送给要求 raw_1d 的 CNN。"""

        pipeline = _cnn_pipeline()
        modules = list(pipeline.modules)
        modules[2] = ModuleSpec(module_id="statistical_features", kind="feature", order=30)
        pipeline = pipeline.model_copy(update={"modules": tuple(modules)})

        result = self.validate(pipeline)

        self.assertFalse(result.valid)
        self.assertIn("FEATURE_TYPE_MISMATCH", {issue.code for issue in result.issues})

    def test_implemented_cnn_is_trainable(self) -> None:
        """训练底座完成后，规范 1D-CNN Pipeline 必须可以执行。"""

        preview_result = self.validate(_cnn_pipeline(trainable=False))
        train_result = self.validate(_cnn_pipeline(trainable=True))

        self.assertTrue(preview_result.valid, preview_result.issues)
        self.assertNotIn("MODULE_NOT_IMPLEMENTED", {issue.code for issue in preview_result.issues})
        self.assertTrue(train_result.valid, train_result.issues)

    def test_fixed_candidates_all_follow_registry_contract(self) -> None:
        """现有三个固定候选转换后都必须使用 Registry 模块。"""

        candidates = build_default_candidate_set("phm2010", "C1").plans
        pipelines = [candidate_plan_to_pipeline(candidate) for candidate in candidates]
        results = [self.validate(pipeline) for pipeline in pipelines]

        self.assertTrue(all(result.valid for result in results), results)
        self.assertTrue(pipelines[0].trainable)
        self.assertTrue(pipelines[1].trainable)
        self.assertFalse(pipelines[2].trainable)
        self.assertTrue(pipelines[2].experimental_extensions)

    def test_experimental_extension_cannot_be_marked_trainable(self) -> None:
        """新能力必须显式作为 experimental extension，且不可假装已实现。"""

        payload = _random_forest_pipeline().model_dump(mode="python")
        payload["experimental_extensions"] = [
            {
                "extension_id": "domain_attention",
                "display_name": "域注意力",
                "kind": "fusion",
                "rationale": "仅用于后续研究。",
            }
        ]
        with self.assertRaises(ValidationError):
            PipelineSpec.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
