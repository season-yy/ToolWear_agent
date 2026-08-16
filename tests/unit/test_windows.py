from __future__ import annotations

import unittest

from toolwear_agent.training.windows import (
    CutLabel,
    all_window_starts,
    assign_cut_splits,
    overlap_to_stride,
    stratified_fraction_sample,
    uniformly_pick_starts,
    validate_no_cut_leakage,
    WindowRecord,
)


class WindowBuildTests(unittest.TestCase):
    def test_overlap_to_stride_converts_half_overlap(self) -> None:
        self.assertEqual(overlap_to_stride(4096, 0.5), 2048)

    def test_uniformly_pick_starts_covers_beginning_and_end(self) -> None:
        starts = list(range(0, 100, 10))

        picked = uniformly_pick_starts(starts, limit=4)

        self.assertEqual(picked[0], 0)
        self.assertEqual(picked[-1], 90)
        self.assertEqual(len(picked), 4)

    def test_all_window_starts_uses_stride(self) -> None:
        self.assertEqual(all_window_starts(row_count=10, window_size=4, stride=3), [0, 3, 6])

    def test_assign_cut_splits_keeps_each_cut_in_one_split(self) -> None:
        labels = [
            CutLabel(cut=index, file_path=f"{index}.csv", row_count=100, vb_value=1.0, stage_id=0, stage_name="初期")
            for index in range(1, 11)
        ]

        split_by_cut = assign_cut_splits(labels)

        self.assertEqual(len(split_by_cut), 10)
        self.assertEqual(set(split_by_cut.values()), {"train", "validation", "test"})

    def test_stratified_fraction_sample_samples_each_split_stage_group(self) -> None:
        records = []
        for split in ["train", "validation"]:
            for stage_id in [0, 1]:
                for index in range(10):
                    records.append(
                        WindowRecord(
                            window_id=f"{split}_{stage_id}_{index}",
                            cut=index,
                            file_path="a.csv",
                            row_count=100,
                            start_row=index,
                            end_row=index + 10,
                            window_size=10,
                            stride=5,
                            overlap_ratio=0.5,
                            vb_value=1.0,
                            stage_id=stage_id,
                            stage_name=str(stage_id),
                            split=split,
                        )
                    )

        sampled = stratified_fraction_sample(records, fraction=0.2)

        self.assertEqual(len(sampled), 4)
        self.assertEqual({record.split for record in sampled}, {"train"})

    def test_validate_no_cut_leakage_raises_when_cut_crosses_split(self) -> None:
        records = [
            WindowRecord("a", 1, "a.csv", 100, 0, 10, 10, 5, 0.5, 1.0, 0, "初期", "train"),
            WindowRecord("b", 1, "a.csv", 100, 10, 20, 10, 5, 0.5, 1.0, 0, "初期", "val"),
        ]

        with self.assertRaises(ValueError):
            validate_no_cut_leakage(records)


if __name__ == "__main__":
    unittest.main()
