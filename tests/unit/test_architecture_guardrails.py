"""防止迁移期间重新引入绝对路径和 C1 结构耦合。"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "toolwear_agent"

# 旧 training/agentteams/frontend 正在渐进迁移；新架构目录从创建起就不允许
# 把 Dataset/Cutter 写进函数名或文件名。
MODERN_ARCHITECTURE_DIRS = (
    "core",
    "schemas",
    "data",
    "registry",
    "ml",
    "state",
    "services",
    "backend",
    "agents",
)


def _python_files(root: Path) -> list[Path]:
    """稳定返回目录中的业务 Python 文件。"""

    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


class ArchitectureGuardrailsTest(unittest.TestCase):
    """把关键架构约束变成自动化回归测试。"""

    def test_business_python_has_no_machine_specific_absolute_root(self) -> None:
        """所有业务 Python 都不能写死当前电脑的 F/D 盘项目路径。"""

        forbidden = re.compile(r"(?:F:/Toolwear_agent|D:/AI_infra)", re.IGNORECASE)
        violations: list[str] = []
        for source_file in _python_files(PACKAGE_ROOT):
            for line_number, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), start=1):
                if forbidden.search(line):
                    relative = source_file.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number}")

        self.assertEqual(violations, [], f"发现本机绝对路径：{violations}")

    def test_new_architecture_does_not_encode_c1_in_business_names(self) -> None:
        """新框架以 DatasetRef 表达 C1，不能再生成 C1 专用业务结构。"""

        violations: list[str] = []
        for directory_name in MODERN_ARCHITECTURE_DIRS:
            root = PACKAGE_ROOT / directory_name
            for source_file in _python_files(root):
                text = source_file.read_text(encoding="utf-8")
                if "phm2010_c1_" in text.lower() or "phm2010_c1_" in source_file.name.lower():
                    violations.append(str(source_file.relative_to(REPOSITORY_ROOT)))

        self.assertEqual(violations, [], f"新架构仍含 C1 结构命名：{violations}")


if __name__ == "__main__":
    unittest.main()
