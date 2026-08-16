from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.data.leakage import (
    SplitLeakageError,
    assert_no_window_leakage,
    assert_windows_match_split_manifest,
    audit_window_splits,
)
from toolwear_agent.data.sampling import build_training_sample, load_sample_manifest, write_sample_manifest
from toolwear_agent.data.splitting import (
    SplitLockConflictError,
    attach_split_hash,
    build_split_manifest,
    create_or_verify_split_lock,
    load_split_lock,
    load_split_manifest,
    write_split_manifest,
)
from toolwear_agent.schemas import SplitSpec
from toolwear_agent.training.windows import CutLabel, WindowRecord, assign_cut_splits, build_window_records


def _window(
    *,
    window_id: str,
    cut: int,
    stage_id: int,
    split: str,
    start_row: int,
    file_path: str | None = None,
) -> WindowRecord:
    """构造一个轻量窗口记录，避免单元测试读取真实 PHM2010 文件。"""

    source_file = file_path or f"C:/data/c1/c_1_{cut:03d}.csv"
    return WindowRecord(
        window_id=window_id,
        cut=cut,
        file_path=source_file,
        row_count=1_000,
        start_row=start_row,
        end_row=start_row + 100,
        window_size=100,
        stride=50,
        overlap_ratio=0.5,
        vb_value=float(stage_id * 50),
        stage_id=stage_id,
        stage_name=f"stage_{stage_id}",
        split=split,
    )


class SplitCorrectnessTests(unittest.TestCase):
    def _build_manifest(self):
        labels = [
            CutLabel(
                cut=cut,
                file_path=f"C:/data/c1/c_1_{cut:03d}.csv",
                row_count=1_000,
                vb_value=float(stage_id * 50),
                stage_id=stage_id,
                stage_name=f"stage_{stage_id}",
            )
            for stage_id in range(4)
            for cut in range(stage_id * 10 + 1, stage_id * 10 + 9)
        ]
        split_spec = SplitSpec(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, random_seed=42)
        assignments = assign_cut_splits(
            labels,
            train_ratio=split_spec.train_ratio,
            val_ratio=split_spec.validation_ratio,
            random_seed=split_spec.random_seed,
        )
        manifest = build_split_manifest(
            cut_labels=labels,
            split_by_cut=assignments,
            dataset_id="phm2010",
            cutter_id="c1",
            split_spec=split_spec,
        )
        return labels, assignments, attach_split_hash(manifest)

    def test_split_hash_is_stable_and_manifest_tampering_is_rejected(self) -> None:
        _labels, _assignments, manifest = self._build_manifest()

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_file = Path(temp_dir) / "split_manifest.json"
            write_split_manifest(manifest, manifest_file)
            loaded = load_split_manifest(manifest_file)
            self.assertEqual(loaded.split_hash, manifest.split_hash)

            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
            payload["assignments"][0]["split"] = "test"
            manifest_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_split_manifest(manifest_file)

    def test_same_experiment_revision_cannot_replace_locked_split(self) -> None:
        labels, assignments, manifest = self._build_manifest()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_file = root / "split_manifest.json"
            lock_file = root / "split_lock.json"
            write_split_manifest(manifest, manifest_file)
            first_lock = create_or_verify_split_lock(
                manifest=manifest,
                lock_file=lock_file,
                experiment_id="phm2010_c1_p0",
                revision=1,
                manifest_file=manifest_file,
            )
            second_lock = create_or_verify_split_lock(
                manifest=manifest,
                lock_file=lock_file,
                experiment_id="phm2010_c1_p0",
                revision=1,
                manifest_file=manifest_file,
            )
            self.assertEqual(first_lock.split_hash, second_lock.split_hash)
            self.assertEqual(load_split_lock(lock_file).split_hash, manifest.split_hash)

            changed_assignments = dict(assignments)
            changed_cut = labels[0].cut
            changed_assignments[changed_cut] = "test" if assignments[changed_cut] != "test" else "train"
            changed_manifest = attach_split_hash(
                build_split_manifest(
                    cut_labels=labels,
                    split_by_cut=changed_assignments,
                    dataset_id="phm2010",
                    cutter_id="c1",
                    split_spec=manifest.split_spec,
                )
            )
            with self.assertRaises(SplitLockConflictError):
                create_or_verify_split_lock(
                    manifest=changed_manifest,
                    lock_file=lock_file,
                    experiment_id="phm2010_c1_p0",
                    revision=1,
                    manifest_file=manifest_file,
                )

    def test_leakage_audit_rejects_same_source_file_across_splits(self) -> None:
        records = [
            _window(window_id="train_window", cut=1, stage_id=0, split="train", start_row=0),
            _window(window_id="test_window", cut=1, stage_id=0, split="test", start_row=100),
        ]

        audit = audit_window_splits(records)

        self.assertFalse(audit.valid)
        self.assertIn("SOURCE_FILE_CROSS_SPLIT", {issue.code for issue in audit.issues})
        with self.assertRaises(SplitLeakageError):
            assert_no_window_leakage(records)

    def test_window_assignments_must_match_locked_split_manifest(self) -> None:
        labels, assignments, manifest = self._build_manifest()
        label = labels[0]
        wrong_split = "test" if assignments[label.cut] != "test" else "train"
        records = [
            _window(
                window_id="moved_window",
                cut=label.cut,
                stage_id=label.stage_id,
                split=wrong_split,
                start_row=0,
                file_path=label.file_path,
            )
        ]

        with self.assertRaises(SplitLeakageError):
            assert_windows_match_split_manifest(records, manifest)


class TrainingSamplingTests(unittest.TestCase):
    def _records(self) -> list[WindowRecord]:
        records: list[WindowRecord] = []
        for stage_id in range(4):
            for cut_offset in range(5):
                cut = stage_id * 10 + cut_offset + 1
                for window_index in range(10):
                    records.append(
                        _window(
                            window_id=f"train_s{stage_id}_c{cut}_w{window_index}",
                            cut=cut,
                            stage_id=stage_id,
                            split="train",
                            start_row=window_index * 100,
                        )
                    )
            records.append(
                _window(
                    window_id=f"validation_s{stage_id}",
                    cut=100 + stage_id,
                    stage_id=stage_id,
                    split="validation",
                    start_row=0,
                )
            )
            records.append(
                _window(
                    window_id=f"test_s{stage_id}",
                    cut=200 + stage_id,
                    stage_id=stage_id,
                    split="test",
                    start_row=0,
                )
            )
        return records

    def test_training_sample_is_reproducible_stage_balanced_and_train_only(self) -> None:
        records = self._records()

        first = build_training_sample(
            records,
            dataset_id="phm2010",
            cutter_id="c1",
            split_hash="a" * 64,
            fraction=0.2,
            random_seed=42,
        )
        second = build_training_sample(
            list(reversed(records)),
            dataset_id="phm2010",
            cutter_id="c1",
            split_hash="a" * 64,
            fraction=0.2,
            random_seed=42,
        )

        self.assertEqual([item.window_id for item in first.records], [item.window_id for item in second.records])
        self.assertEqual(first.manifest.sample_hash, second.manifest.sample_hash)
        self.assertEqual(first.manifest.selected_count, 40)
        self.assertEqual(set(first.manifest.stage_distribution.values()), {10})
        self.assertEqual({item.split for item in first.records}, {"train"})
        self.assertFalse(any(item.window_id.startswith(("validation", "test")) for item in first.records))

    def test_training_sample_covers_every_cut_and_each_cut_timeline(self) -> None:
        sample = build_training_sample(
            self._records(),
            dataset_id="phm2010",
            cutter_id="c1",
            split_hash="b" * 64,
            fraction=0.2,
            random_seed=7,
        )

        starts_by_cut: dict[int, list[int]] = {}
        for record in sample.records:
            starts_by_cut.setdefault(record.cut, []).append(record.start_row)

        self.assertEqual(len(starts_by_cut), 20)
        self.assertTrue(all(sorted(starts) == [0, 900] for starts in starts_by_cut.values()))

    def test_sample_manifest_round_trip_verifies_hash(self) -> None:
        sample = build_training_sample(
            self._records(),
            dataset_id="phm2010",
            cutter_id="c1",
            split_hash="c" * 64,
            fraction=0.2,
            random_seed=42,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "sample_manifest.json"
            write_sample_manifest(sample.manifest, output_file)
            loaded = load_sample_manifest(output_file)

        self.assertEqual(loaded.sample_hash, sample.manifest.sample_hash)
        self.assertEqual(loaded.selected_count, len(sample.records))


if __name__ == "__main__":
    unittest.main()
