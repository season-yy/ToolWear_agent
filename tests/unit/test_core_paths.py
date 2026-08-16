"""实体路径解析和路径越界防护测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.core.errors import InvalidIdentifierError, PathBoundaryError
from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.settings import Settings


class PathResolverTest(unittest.TestCase):
    """验证路径由实体 ID 稳定生成，而不是依赖最新修改时间。"""

    def _resolver(self, temp_dir: str) -> PathResolver:
        return PathResolver(Settings(ai_infra_root=Path(temp_dir) / "infra"))

    def test_entity_paths_are_stable_and_nested_under_experiment(self) -> None:
        """Experiment、Revision、Run、Evidence 路径应具有固定层级。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = self._resolver(temp_dir)

            experiment = resolver.experiment_path("exp-demo")
            revision = resolver.revision_path("exp-demo", 2)
            run = resolver.run_path("exp-demo", 2, "run-smoke")
            report = resolver.report_path("exp-demo", 2, "run-smoke")
            evidence = resolver.evidence_path("exp-demo", 2, "run-smoke")

        self.assertEqual(revision, experiment / "revisions" / "r0002")
        self.assertEqual(run, revision / "runs" / "run-smoke")
        self.assertEqual(report, run / "report")
        self.assertEqual(evidence, run / "evidence")

    def test_entity_identifier_rejects_path_traversal(self) -> None:
        """用户提供的实体 ID 不能包含路径分隔符或上级目录。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = self._resolver(temp_dir)

            for invalid_id in ("../escape", "a/b", "a\\b", "", ".", ".."):
                with self.subTest(invalid_id=invalid_id):
                    with self.assertRaises(InvalidIdentifierError):
                        resolver.experiment_path(invalid_id)

    def test_assert_within_rejects_unrelated_path(self) -> None:
        """解析后的路径不在允许根目录时必须拒绝。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            resolver = self._resolver(temp_dir)
            allowed = temp_root / "allowed"
            allowed.mkdir()

            accepted = resolver.assert_within(allowed / "child.csv", [allowed])
            self.assertEqual(accepted, (allowed / "child.csv").resolve())

            with self.assertRaises(PathBoundaryError):
                resolver.assert_within(temp_root / "outside.csv", [allowed])

    def test_create_runtime_directories_does_not_create_raw_dataset(self) -> None:
        """初始化运行目录时只能创建可写区，不能创建或伪造原始数据区。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = self._resolver(temp_dir)

            created = resolver.ensure_runtime_directories()

            self.assertTrue(all(path.is_dir() for path in created))
            self.assertNotIn(resolver.settings.phm2010_raw_root, created)
            self.assertFalse(resolver.settings.phm2010_raw_root.exists())


if __name__ == "__main__":
    unittest.main()
