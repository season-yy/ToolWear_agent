from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from toolwear_agent.training.visualization import classification_report_to_rows, write_metrics_summary


class VisualizationTests(unittest.TestCase):
    def test_classification_report_to_rows_maps_stage_names(self) -> None:
        report = {
            "0": {"precision": 1.0, "recall": 0.5, "f1-score": 0.66, "support": 2.0},
            "accuracy": 0.75,
        }

        rows = classification_report_to_rows(report)

        self.assertEqual(rows[0]["stage_name"], "初期磨损")
        self.assertEqual(rows[0]["f1_score"], 0.66)
        self.assertEqual(rows[1]["label"], "accuracy")

    def test_metrics_summary_supports_validation_only_run(self) -> None:
        metrics = {
            "run_id": "run-1",
            "full_window_count": 100,
            "sample_count": 12,
            "sample_fraction": 0.2,
            "train_count": 12,
            "validation": {
                "count": 20,
                "macro_f1": 0.8,
                "balanced_accuracy": 0.75,
                "accuracy": 0.8,
            },
            "stage_distribution": {"0": 3, "1": 3, "2": 3, "3": 3},
            "split_distribution": {"train": 12, "validation": 20},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "metrics_summary.json"
            write_metrics_summary(metrics, output_file)
            summary = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(summary["final_test_status"], "not_run_pipeline_not_frozen")
        self.assertNotIn("test_macro_f1", summary)
        self.assertNotIn("test_balanced_accuracy", summary)


if __name__ == "__main__":
    unittest.main()
