from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.training.features import build_feature_names, extract_signal_statistics


class FeatureExtractionTests(unittest.TestCase):
    def test_build_feature_names_uses_channel_and_statistic_names(self) -> None:
        feature_names = build_feature_names(["a", "b"])

        self.assertEqual(
            feature_names,
            [
                "a_mean",
                "a_std",
                "a_min",
                "a_max",
                "a_ptp",
                "a_rms",
                "b_mean",
                "b_std",
                "b_min",
                "b_max",
                "b_ptp",
                "b_rms",
            ],
        )

    def test_extract_signal_statistics_reads_limited_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            signal_file = Path(temp_dir) / "signal.csv"
            signal_file.write_text("1,2\n3,4\n100,100\n", encoding="utf-8")

            row = extract_signal_statistics(
                signal_file=signal_file,
                cut=1,
                channel_names=["a", "b"],
                max_rows=2,
            )

        self.assertEqual(row.cut, 1)
        self.assertEqual(row.sampled_rows, 2)
        self.assertEqual(row.feature_names[0], "a_mean")
        self.assertAlmostEqual(row.features[0], 2.0)
        self.assertAlmostEqual(row.features[1], 1.0)
        self.assertAlmostEqual(row.features[2], 1.0)
        self.assertAlmostEqual(row.features[3], 3.0)
        self.assertAlmostEqual(row.features[4], 2.0)

    def test_extract_signal_statistics_can_start_from_window_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            signal_file = Path(temp_dir) / "signal.csv"
            signal_file.write_text("1,2\n3,4\n5,6\n", encoding="utf-8")

            row = extract_signal_statistics(
                signal_file=signal_file,
                cut=1,
                channel_names=["a", "b"],
                max_rows=2,
                start_row=1,
            )

        self.assertEqual(row.sampled_rows, 2)
        self.assertAlmostEqual(row.features[0], 4.0)
        self.assertAlmostEqual(row.features[6], 5.0)


if __name__ == "__main__":
    unittest.main()
