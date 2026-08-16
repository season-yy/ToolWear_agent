from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.agentteams.reporting import (
    build_report_sources,
    find_latest_decided_run,
    render_p0_experiment_report,
)


class ReportingTests(unittest.TestCase):
    def test_find_latest_decided_run_requires_core_evidence_files(self) -> None:
        """只把已经完成训练、可视化、诊断和决策的运行目录当成报告输入。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            incomplete = root / "phm2010_c1_window_mini_train_20260814_100000"
            complete = root / "phm2010_c1_window_mini_train_20260814_110000"
            incomplete.mkdir()
            complete.mkdir()
            (incomplete / "metrics_summary.json").write_text("{}", encoding="utf-8")
            for file_name in [
                "metrics_summary.json",
                "train_config.json",
                "visual_report_manifest.json",
                "agent_diagnosis.json",
                "agent_decision.json",
            ]:
                (complete / file_name).write_text("{}", encoding="utf-8")

            self.assertEqual(find_latest_decided_run(root), complete)

    def test_build_report_sources_points_to_expected_p0_reports(self) -> None:
        """总报告要能追溯前面 P0 步骤已经生成的分报告。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sources = build_report_sources(root, root / "runs" / "run_001")

            self.assertIn("data_profile", sources)
            self.assertIn("agent_decision", sources)
            self.assertTrue(str(sources["visual_report"]).endswith("phm2010_c1_visual_report.md"))

    def test_render_p0_experiment_report_keeps_boundary_clear(self) -> None:
        """报告正文必须说明当前只是 C1 内部 PoC，不能写成已经完成泛化。"""

        report = render_p0_experiment_report(
            run_id="run_001",
            metrics_summary={
                "full_window_count": 10080,
                "sample_count": 2021,
                "sample_fraction": 0.2,
                "validation_macro_f1": 1.0,
                "validation_balanced_accuracy": 1.0,
                "test_macro_f1": 1.0,
                "test_balanced_accuracy": 1.0,
            },
            train_config={
                "model_family": "RandomForest",
                "window_size": 4096,
                "overlap_ratio": 0.5,
                "max_windows_per_cut": 32,
                "window_manifest_file": "D:/AI_infra/datasets/processed/phm2010/phm2010_c1_window_manifest.csv",
            },
            diagnosis={
                "overall_conclusion": "当前基线可用，但尚未证明跨刀具泛化。",
                "recommendations": ["保留基线。"],
            },
            decision={
                "overall_decision": "继续当前方案。",
                "stop_conditions": ["跨刀具验证连续失败时停止。"],
            },
            visual_manifest={
                "tsne_png": "D:/AI_infra/figures/tsne.png",
                "test_confusion_matrix_png": "D:/AI_infra/figures/cm.png",
            },
            source_files={"data_profile": Path("D:/AI_infra/reports/phm2010_c1_data_profile.md")},
        )

        self.assertIn("作品简介", report)
        self.assertIn("C1 内部", report)
        self.assertIn("不等价于跨刀具泛化", report)
        self.assertIn("RandomForest", report)
        self.assertIn("证据索引", report)


if __name__ == "__main__":
    unittest.main()
