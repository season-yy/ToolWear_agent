from __future__ import annotations

import unittest

from toolwear_agent.training.features import SignalFeatureRow
from toolwear_agent.training.mini_train import detect_cuda_status, train_random_forest_baseline
from toolwear_agent.training.windows import WindowRecord


class MiniTrainTests(unittest.TestCase):
    def test_detect_cuda_status_returns_structured_result(self) -> None:
        status = detect_cuda_status("cuda")

        self.assertEqual(status.requested_device, "cuda")
        self.assertIsInstance(status.torch_available, bool)
        self.assertIsInstance(status.cuda_available, bool)
        self.assertIsInstance(status.note, str)

    def test_mini_train_returns_validation_only_and_does_not_evaluate_test(self) -> None:
        feature_rows = [
            SignalFeatureRow(cut=index, feature_names=["feature"], features=[value], sampled_rows=10)
            for index, value in enumerate([0.0, 0.1, 1.0, 1.1, 0.2, 1.2, 999.0], start=1)
        ]
        splits = ["train", "train", "train", "train", "validation", "validation", "test"]
        labels = [0, 0, 1, 1, 0, 1, 1]
        records = [
            WindowRecord(
                window_id=f"window_{index}",
                cut=index,
                file_path=f"{index}.csv",
                row_count=100,
                start_row=0,
                end_row=10,
                window_size=10,
                stride=5,
                overlap_ratio=0.5,
                vb_value=0.0,
                stage_id=labels[index - 1],
                stage_name=str(labels[index - 1]),
                split=split,
            )
            for index, split in enumerate(splits, start=1)
        ]

        _classifier, metrics = train_random_forest_baseline(feature_rows, records, labels, random_seed=42)

        self.assertEqual(metrics["train_count"], 4)
        self.assertEqual(metrics["validation"]["count"], 2)
        self.assertNotIn("test", metrics)


if __name__ == "__main__":
    unittest.main()
