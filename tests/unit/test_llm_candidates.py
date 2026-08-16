from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.agentteams.llm_candidates import (
    TRAINABLE_PLAN_IDS,
    build_architect_prompt,
    generate_llm_candidate_set,
    normalize_plan_id,
    validate_candidate_payload,
    write_llm_candidate_outputs,
)
from toolwear_agent.common.config import Settings


def _settings(ai_root: Path) -> Settings:
    return Settings(
        llm_provider="qwen",
        llm_api_key="",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_model="",
        project_root=Path("F:/Toolwear_agent"),
        app_root=Path("F:/Toolwear_agent/toolwear_agent_app"),
        ai_infra_root=ai_root,
        dataset_manifest=ai_root / "datasets" / "manifests" / "phm2010.yaml",
        phm2010_raw_root=ai_root / "datasets" / "raw" / "phm2010",
        experiment_root=ai_root / "experiments" / "runs",
        artifact_root=ai_root / "artifacts",
        log_root=ai_root / "logs",
        train_device="cuda",
        random_seed=42,
        primary_task="four_stage_classification",
        enable_vb_regression=False,
        vb_aggregation="max",
        vb_stage_thresholds_um=(90.0, 130.0, 160.0),
        fastapi_host="127.0.0.1",
        fastapi_port=18100,
        streamlit_host="127.0.0.1",
        streamlit_port=18101,
    )


class LlmCandidateTests(unittest.TestCase):
    def test_validate_candidate_payload_applies_trainable_whitelist(self) -> None:
        raw = [
            {
                "plan_id": "statistical_features_random_forest",
                "display_name": "RF",
                "module_pipeline": ["features", "rf"],
                "reason": "fast",
                "risk": "limited",
                "expected_cost": "low",
                "trainable_now": True,
                "training_backend": "sklearn_random_forest",
            },
            {
                "plan_id": "multichannel_window_1d_cnn",
                "display_name": "CNN",
                "module_pipeline": ["window", "cnn"],
                "reason": "temporal",
                "risk": "not ready",
                "expected_cost": "mid",
                "trainable_now": True,
                "training_backend": "cnn",
            },
        ]

        plans = validate_candidate_payload(raw)

        self.assertTrue(plans[0].trainable_now)
        self.assertTrue(plans[1].trainable_now)
        self.assertLessEqual({plan.plan_id for plan in plans if plan.trainable_now}, TRAINABLE_PLAN_IDS)

    def test_build_architect_prompt_mentions_whitelist(self) -> None:
        messages = build_architect_prompt("快速比较传统模型")

        joined = "\n".join(message["content"] for message in messages)
        self.assertIn("statistical_features_random_forest", joined)
        self.assertIn("statistical_features_extra_trees", joined)
        self.assertIn("multichannel_window_1d_cnn", joined)

    def test_normalize_plan_id_maps_llm_alias_to_trainable_id(self) -> None:
        plan_id = normalize_plan_id(
            "plan_stat_rf_001",
            ["statistical_feature_extraction", "random_forest_classifier"],
            "sklearn",
        )

        self.assertEqual(plan_id, "statistical_features_random_forest")
        self.assertEqual(
            normalize_plan_id("plan_stat_rf_baseline", ["统计特征", "随机森林"], "sklearn"),
            "statistical_features_random_forest",
        )
        self.assertEqual(
            normalize_plan_id("plan_stat_et_baseline", ["统计特征", "极端随机树"], "sklearn"),
            "statistical_features_extra_trees",
        )
        self.assertEqual(
            normalize_plan_id("plan_raw_1dcnn", ["原始时序", "轻量卷积"], "pytorch"),
            "multichannel_window_1d_cnn",
        )

    def test_generate_llm_candidate_set_falls_back_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate_set = generate_llm_candidate_set(_settings(Path(temp_dir)), "快速验证")

        self.assertTrue(candidate_set.used_fallback)
        self.assertGreaterEqual(len(candidate_set.plans), 2)

    def test_llm_output_contains_canonical_pipeline_specs(self) -> None:
        """落盘结果必须同时提供页面和训练共用的 PipelineSpec。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            candidate_set = generate_llm_candidate_set(settings, "快速验证")
            candidate_file, _, _ = write_llm_candidate_outputs(candidate_set, settings)
            payload = json.loads(candidate_file.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["pipeline_specs"]), len(candidate_set.plans))
        self.assertEqual(
            payload["pipeline_specs"][0]["pipeline_id"],
            candidate_set.plans[0].plan_id,
        )
        self.assertEqual(len(payload["registry_validations"]), len(candidate_set.plans))
        self.assertTrue(all(item["valid"] for item in payload["registry_validations"]))

    def test_experimental_extension_forces_non_trainable_pipeline(self) -> None:
        """LLM 提出的未实现能力必须进入扩展区，不能伪装成可训练模块。"""

        plans = validate_candidate_payload(
            [
                {
                    "plan_id": "statistical_features_random_forest",
                    "display_name": "RF + 域注意力设想",
                    "module_pipeline": ["statistical_features", "random_forest"],
                    "reason": "探索跨刀具能力",
                    "risk": "尚未实现",
                    "expected_cost": "medium",
                    "trainable_now": True,
                    "training_backend": "sklearn",
                    "experimental_extension": {
                        "extension_id": "domain_attention",
                        "display_name": "域注意力",
                        "kind": "fusion",
                        "rationale": "后续跨刀具实验候选。",
                    },
                }
            ]
        )

        self.assertFalse(plans[0].trainable_now)
        self.assertEqual(plans[0].experimental_extension["extension_id"], "domain_attention")


if __name__ == "__main__":
    unittest.main()
