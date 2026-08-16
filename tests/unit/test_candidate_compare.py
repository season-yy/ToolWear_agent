from __future__ import annotations

import csv
from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.training.candidate_compare import (
    CandidateCompareMetric,
    build_classifier,
    load_feature_table,
    select_best_candidate,
    train_compare_candidate,
)


class CandidateCompareTests(unittest.TestCase):
    def test_build_classifier_supports_random_forest_and_extra_trees(self) -> None:
        rf_name, _rf = build_classifier("statistical_features_random_forest", 42)
        et_name, _et = build_classifier("statistical_features_extra_trees", 42)

        self.assertEqual(rf_name, "RandomForestClassifier")
        self.assertEqual(et_name, "ExtraTreesClassifier")

    def test_load_feature_table_reads_features_labels_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_file = Path(temp_dir) / "feature_table.csv"
            with feature_file.open("w", encoding="utf-8-sig", newline="") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(["stage_id", "split", "force_x_mean", "force_x_std"])
                writer.writerow([0, "train", 0.1, 0.2])
                writer.writerow([1, "validation", 1.1, 1.2])

            x_values, y_values, splits = load_feature_table(feature_file)

        self.assertEqual(x_values, [[0.1, 0.2], [1.1, 1.2]])
        self.assertEqual(y_values, [0, 1])
        self.assertEqual(splits, ["train", "validation"])

    def test_train_compare_candidate_returns_metrics(self) -> None:
        x_values = [[0.0], [0.1], [1.0], [1.1], [0.2], [1.2], [999.0]]
        y_values = [0, 0, 1, 1, 0, 1, 1]
        splits = ["train", "train", "train", "train", "validation", "validation", "test"]

        metric = train_compare_candidate("statistical_features_extra_trees", x_values, y_values, splits, 42)

        self.assertEqual(metric.plan_id, "statistical_features_extra_trees")
        self.assertGreaterEqual(metric.validation_macro_f1, 0.0)

    def test_candidate_selection_never_uses_test_metric(self) -> None:
        low_validation = CandidateCompareMetric(
            plan_id="low_validation",
            display_name="低验证分候选",
            classifier_name="A",
            train_count=10,
            validation_macro_f1=0.4,
            validation_balanced_accuracy=0.4,
            recommendation="",
        )
        high_validation = CandidateCompareMetric(
            plan_id="high_validation",
            display_name="高验证分候选",
            classifier_name="B",
            train_count=10,
            validation_macro_f1=0.8,
            validation_balanced_accuracy=0.7,
            recommendation="",
        )

        best = select_best_candidate([low_validation, high_validation])

        self.assertEqual(best.plan_id, "high_validation")
        self.assertNotIn("test_macro_f1", asdict(best))
        self.assertNotIn("test_balanced_accuracy", asdict(best))


if __name__ == "__main__":
    unittest.main()
