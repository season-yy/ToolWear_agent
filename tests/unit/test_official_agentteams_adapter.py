from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.agentteams.official_adapter import run_c1_official_agentteams_minimal
from toolwear_agent.common.config import Settings


def _settings(root: Path) -> Settings:
    """构造测试用配置。"""

    return Settings(
        llm_provider="qwen",
        llm_api_key="",
        llm_base_url="",
        llm_model="qwen-test",
        project_root=root / "project",
        app_root=root / "project" / "toolwear_agent_app",
        ai_infra_root=root / "ai_infra",
        dataset_manifest=root / "ai_infra" / "datasets" / "manifests" / "phm2010.yaml",
        phm2010_raw_root=root / "ai_infra" / "datasets" / "raw" / "phm2010",
        experiment_root=root / "ai_infra" / "experiments" / "runs",
        artifact_root=root / "ai_infra" / "artifacts",
        log_root=root / "ai_infra" / "logs",
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


class OfficialAgentTeamsAdapterTests(unittest.TestCase):
    def test_run_c1_official_agentteams_minimal_writes_package_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))

            package = run_c1_official_agentteams_minimal(settings)

            self.assertEqual(package.team_name, "toolwear-phm2010-c1-team")
            self.assertEqual(package.leader_name, "toolwear-experiment-manager")
            self.assertEqual(len(package.worker_specs), 6)
            self.assertEqual(len(package.worker_names), 5)
            self.assertEqual(len(package.skill_specs), 10)
            self.assertIn("toolwear-mini-train", package.worker_specs[3].skills)
            self.assertTrue(Path(package.output_files["package_json"]).exists())
            self.assertTrue(Path(package.output_files["element_message"]).exists())
            self.assertTrue(Path(package.output_files["report"]).exists())
            self.assertTrue(Path(package.output_files["worker_skills_dir"]).exists())
            for skill in package.skill_specs:
                skill_root = Path(skill.skill_file).parent
                self.assertTrue((skill_root / "scripts" / "client.py").exists())
                self.assertTrue((skill_root / "schema" / "input.schema.json").exists())
                self.assertTrue((skill_root / "schema" / "output.schema.json").exists())
            self.assertIn("https://github.com/agentscope-ai/AgentTeams", package.official_sources[0])


if __name__ == "__main__":
    unittest.main()
