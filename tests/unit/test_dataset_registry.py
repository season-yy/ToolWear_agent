"""Dataset Registry 与 PHM2010 Adapter 的单元契约测试。"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.data.adapters.phm2010 import PHM2010Adapter
from toolwear_agent.data.registry import DatasetRegistry


def _write_signal(path: Path, *, rows: int = 3, channels: int = 7) -> None:
    """写入很小的无表头信号文件，避免单元测试依赖真实大数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        for row_index in range(rows):
            writer.writerow([float(row_index + channel) for channel in range(channels)])


def _write_wear(path: Path, *, rows: int = 2) -> None:
    """写入与信号刀次对应的三刃 VB 标签文件。"""

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["cut", "flute_1", "flute_2", "flute_3"])
        for cut in range(1, rows + 1):
            writer.writerow([cut, 10.0 + cut, 20.0 + cut, 30.0 + cut])


class PHM2010AdapterTest(unittest.TestCase):
    """验证 Adapter 不依赖某一把固定刀具。"""

    def test_discovers_all_expected_cutters_and_label_roles(self) -> None:
        """存在与否和有无标签必须是两套独立、明确的状态。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for cutter_number in (1, 2, 4, 6):
                cutter_dir = root / f"c{cutter_number}"
                for cut in (1, 2):
                    _write_signal(cutter_dir / f"c_{cutter_number}_{cut:03d}.csv")
                if cutter_number in (1, 4, 6):
                    _write_wear(cutter_dir / f"c{cutter_number}_wear.csv")

            adapter = PHM2010Adapter(expected_signal_count=2)
            inspection = adapter.inspect(root)
            manifest = inspection.manifest

            self.assertEqual(tuple(manifest.cutters), ("C1", "C2", "C3", "C4", "C5", "C6"))
            self.assertTrue(manifest.cutters["C1"].available)
            self.assertTrue(manifest.cutters["C1"].labeled)
            self.assertTrue(manifest.cutters["C2"].available)
            self.assertFalse(manifest.cutters["C2"].labeled)
            self.assertFalse(manifest.cutters["C3"].available)
            self.assertEqual(manifest.cutters["C4"].signal_file_count, 2)
            self.assertEqual(manifest.cutters["C6"].wear_row_count, 2)
            self.assertEqual(manifest.cutters["C1"].detected_channel_count, 7)
            self.assertEqual(manifest.cutters["C1"].sampled_signal_lengths, (3, 3))
            self.assertTrue(inspection.validation.valid)
            self.assertEqual(len(manifest.manifest_hash or ""), 64)

    def test_reports_missing_wear_file_for_labeled_cutter(self) -> None:
        """有标签刀具缺少 wear 文件时必须报告错误，不能静默改成无标签。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cutter_dir = root / "c1"
            for cut in (1, 2):
                _write_signal(cutter_dir / f"c_1_{cut:03d}.csv")

            inspection = PHM2010Adapter(expected_signal_count=2).inspect(root)

            self.assertTrue(inspection.manifest.cutters["C1"].labeled)
            self.assertFalse(inspection.validation.valid)
            self.assertIn("WEAR_FILE_MISSING", {issue.code for issue in inspection.validation.issues})

    def test_records_logical_and_resolved_cutter_paths(self) -> None:
        """Adapter 要同时保留入口路径和解析后的真实路径。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cutter_dir = root / "c1"
            _write_signal(cutter_dir / "c_1_001.csv")
            _write_wear(cutter_dir / "c1_wear.csv", rows=1)

            manifest = PHM2010Adapter(expected_signal_count=1).inspect(root).manifest
            cutter = manifest.cutters["C1"]

            self.assertEqual(cutter.relative_path, "c1")
            self.assertEqual(cutter.resolved_path, cutter_dir.resolve())

    def test_rejects_wrong_cut_ids_even_when_file_count_matches(self) -> None:
        """文件数量正确但 cut 集合错误时仍必须判为无效。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cutter_dir = root / "c1"
            _write_signal(cutter_dir / "c_1_001.csv")
            _write_signal(cutter_dir / "c_1_003.csv")
            _write_wear(cutter_dir / "c1_wear.csv", rows=2)

            inspection = PHM2010Adapter(expected_signal_count=2).inspect(root)
            codes = {issue.code for issue in inspection.validation.issues}

            self.assertFalse(inspection.validation.valid)
            self.assertIn("SIGNAL_CUT_SET_MISMATCH", codes)
            self.assertIn("WEAR_CUT_SET_MISMATCH", codes)

    def test_rejects_non_numeric_wear_values(self) -> None:
        """VB 标签包含非有限数值时不得进入后续标签与训练步骤。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cutter_dir = root / "c1"
            _write_signal(cutter_dir / "c_1_001.csv")
            wear_path = cutter_dir / "c1_wear.csv"
            with wear_path.open("w", encoding="utf-8", newline="") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(["cut", "flute_1", "flute_2", "flute_3"])
                writer.writerow([1, "not-a-number", 20.0, 30.0])

            inspection = PHM2010Adapter(expected_signal_count=1).inspect(root)

            self.assertFalse(inspection.validation.valid)
            self.assertIn(
                "WEAR_VALUES_INVALID",
                {issue.code for issue in inspection.validation.issues},
            )


class DatasetRegistryTest(unittest.TestCase):
    """验证注册表持久化和清单完整性。"""

    def test_registry_round_trip_preserves_manifest_and_hash(self) -> None:
        """保存再加载后必须得到同一份强类型 DatasetManifest。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            cutter_dir = root / "c1"
            _write_signal(cutter_dir / "c_1_001.csv")
            _write_wear(cutter_dir / "c1_wear.csv", rows=1)
            manifest = PHM2010Adapter(expected_signal_count=1).inspect(root).manifest
            registry_path = Path(temp_dir) / "manifests" / "datasets.yaml"

            registry = DatasetRegistry(registry_path)
            registry.register(manifest)
            restored = DatasetRegistry(registry_path).get("phm2010")

            self.assertEqual(restored, manifest)
            self.assertEqual(restored.manifest_hash, manifest.manifest_hash)
            self.assertIn("datasets:", registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry.allowed_resolved_roots("phm2010"), (cutter_dir.resolve(),))

    def test_registry_rejects_manifest_with_invalid_hash(self) -> None:
        """清单内容和保存的 hash 不一致时必须拒绝读取。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "raw"
            cutter_dir = root / "c1"
            _write_signal(cutter_dir / "c_1_001.csv")
            _write_wear(cutter_dir / "c1_wear.csv", rows=1)
            manifest = PHM2010Adapter(expected_signal_count=1).inspect(root).manifest
            registry_path = Path(temp_dir) / "datasets.yaml"
            registry = DatasetRegistry(registry_path)
            registry.register(manifest)

            content = registry_path.read_text(encoding="utf-8")
            registry_path.write_text(content.replace("PHM 2010", "PHM 2010 changed"), encoding="utf-8")

            with self.assertRaises(ValueError):
                DatasetRegistry(registry_path)


if __name__ == "__main__":
    unittest.main()
