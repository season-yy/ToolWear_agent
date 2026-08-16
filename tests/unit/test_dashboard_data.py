from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.agentteams.llm_candidates import LlmCandidatePlan
from toolwear_agent.common.config import Settings
from toolwear_agent.frontend.dashboard_data import (
    build_candidate_choices,
    build_dashboard_paths,
    build_result_snapshot,
    load_candidate_cards,
    load_json_file,
)
from toolwear_agent.schemas.converters import llm_candidate_plan_to_pipeline
from toolwear_agent.training.candidates import build_default_candidate_set, write_candidate_json


def _settings(ai_root: Path) -> Settings:
    """构造测试用配置。"""

    return Settings(
        llm_provider="qwen",
        llm_api_key="",
        llm_base_url="",
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


class DashboardDataTests(unittest.TestCase):
    def test_load_json_file_returns_empty_dict_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_json_file(Path(temp_dir) / "missing.json")

        self.assertEqual(result, {})

    def test_load_candidate_cards_reads_candidate_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate_file = Path(temp_dir) / "candidate.json"
            write_candidate_json(build_default_candidate_set("phm2010", "c1"), candidate_file)

            cards = load_candidate_cards(candidate_file)

        self.assertEqual(len(cards), 3)
        self.assertEqual(cards[0]["plan_id"], "statistical_features_random_forest")
        self.assertEqual(cards[0]["source"], "static")

    def test_build_candidate_choices_prefers_llm_candidates(self) -> None:
        snapshot = {
            "candidate_cards": [
                {
                    "plan_id": "multichannel_window_1d_cnn",
                    "display_name": "固定模板 CNN",
                    "source": "static",
                }
            ],
            "llm_candidates": {
                "plans": [
                    {
                        "plan_id": "statistical_features_extra_trees",
                        "display_name": "统计特征 + ExtraTrees",
                        "module_pipeline": ["signal_windowing", "extra_trees_classifier"],
                        "reason": "用于和 RandomForest 做低成本对照。",
                        "risk": "可能和 RandomForest 指标接近。",
                        "expected_cost": "低",
                        "trainable_now": True,
                        "training_backend": "sklearn_extra_trees",
                    }
                ]
            },
        }

        choices = build_candidate_choices(snapshot)

        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0]["plan_id"], "statistical_features_extra_trees")
        self.assertEqual(choices[0]["source"], "llm")
        self.assertIn("extra_trees", choices[0]["model_structure"])
        self.assertNotIn("extra_trees_classifier", choices[0]["model_structure"])
        self.assertTrue(choices[0]["registry_validated"])
        self.assertFalse(choices[0]["single_train_supported"])
        self.assertTrue(choices[0]["compare_train_supported"])

    def test_build_candidate_choices_prefers_canonical_pipeline_specs(self) -> None:
        """新文件存在 PipelineSpec 时，卡片和选择必须从同一 Schema 生成。"""

        pipeline = llm_candidate_plan_to_pipeline(
            LlmCandidatePlan(
                plan_id="statistical_features_extra_trees",
                display_name="统计特征 + ExtraTrees",
                module_pipeline=["statistical_features", "extra_trees"],
                reason="低成本对照。",
                risk="与 RF 可能接近。",
                expected_cost="low",
                trainable_now=True,
                training_backend="sklearn",
            )
        )
        snapshot = {
            "candidate_cards": [],
            "llm_candidates": {
                "plans": [{"plan_id": "legacy-plan", "display_name": "旧候选"}],
                "pipeline_specs": [pipeline.model_dump(mode="json")],
            },
        }

        choices = build_candidate_choices(snapshot)

        self.assertEqual([item["plan_id"] for item in choices], ["statistical_features_extra_trees"])
        self.assertEqual(choices[0]["source"], "llm")
        self.assertIn("extra_trees", choices[0]["model_structure"])
        self.assertTrue(choices[0]["registry_validated"])

    def test_unregistered_canonical_pipeline_falls_back_to_static(self) -> None:
        """陈旧或伪造模块链不能进入页面单选项。"""

        static_card = {
            "plan_id": "statistical_features_random_forest",
            "display_name": "统计特征 + RandomForest",
            "source": "static",
        }
        snapshot = {
            "candidate_cards": [static_card],
            "llm_candidates": {
                "pipeline_specs": [
                    {
                        "pipeline_id": "invalid",
                        "display_name": "非法候选",
                        "source": "llm",
                        "input_channels": ["force_x"],
                        "modules": [
                            {"module_id": "magic", "kind": "windowing", "order": 10},
                            {"module_id": "magic_model", "kind": "model", "order": 20},
                            {"module_id": "magic_trainer", "kind": "trainer", "order": 30},
                        ],
                        "rationale": "非法模块测试。",
                        "risks": ["不可执行。"],
                        "expected_cost": "low",
                    }
                ]
            },
        }

        choices = build_candidate_choices(snapshot)

        self.assertEqual(choices, [static_card])

    def test_build_candidate_choices_falls_back_to_static_candidates(self) -> None:
        static_card = {
            "plan_id": "statistical_features_random_forest",
            "display_name": "统计特征 + RandomForest",
            "source": "static",
        }

        choices = build_candidate_choices({"candidate_cards": [static_card], "llm_candidates": {}})

        self.assertEqual(choices, [static_card])

    def test_build_dashboard_paths_points_to_expected_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = build_dashboard_paths(_settings(Path(temp_dir)))

        self.assertEqual(paths.p0_report.name, "phm2010_c1_p0_experiment_report.md")
        self.assertEqual(paths.trace_report.name, "phm2010_c1_agentteams_trace.md")
        self.assertEqual(paths.skill_manifest.name, "phm2010_c1_skill_manifest.md")

    def test_build_result_snapshot_has_core_agents_and_empty_results_without_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = build_result_snapshot(_settings(Path(temp_dir)))

        self.assertEqual(len(snapshot["core_agents"]), 6)
        self.assertEqual(snapshot["metrics"], {})
        self.assertEqual(snapshot["run_dir"], "")
        self.assertEqual(snapshot["candidate_choices"], [])


if __name__ == "__main__":
    unittest.main()
