"""不使用临时目录的初赛白名单 ZIP 构建器。"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ALLOWLIST_FILES = (
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "项目进展说明.md",
)
ALLOWLIST_DIRECTORIES = (
    "deploy",
    "docs",
    "examples",
    "scripts",
    "tasks",
    "tests",
    "toolwear_agent",
)
BLOCKED_FILE_NAMES = {
    "frontend-evaluation-desktop.png",
}
BLOCKED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".vscode",
    "__pycache__",
    "artifacts",
    "cache",
    "experiments",
    "generated_code",
    "logs",
    "reports",
    "secrets",
}
BLOCKED_SUFFIXES = {
    ".7z",
    ".bin",
    ".dll",
    ".exe",
    ".joblib",
    ".npy",
    ".ndjson",
    ".npz",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".rar",
    ".zip",
}
TEXT_SUFFIXES = {
    "",
    ".css",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r'(?i)"initialPassword"\s*:\s*"[^\"]+"'),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._-]{16,}"),
)
GENERIC_ASSIGNMENT_PATTERNS = (
    re.compile(r"(?m)^[ \t]*LLM_API_KEY[ \t]*=[ \t]*[^ \t\r\n#]+"),
    re.compile(r"(?m)^[ \t]*TOOL_API_TOKEN[ \t]*=[ \t]*[^ \t\r\n#]+"),
    re.compile(r'(?i)(llm_api_key|tool_api_token)\s*=\s*["\'][^"\']{20,}["\']'),
)
MAX_FILE_BYTES = 5 * 1024 * 1024


class SubmissionSecurityError(RuntimeError):
    """白名单、密钥或文件大小检查失败。"""


@dataclass(frozen=True)
class SubmissionPackageResult:
    output_file: str
    file_count: int
    size_bytes: int
    sha256: str
    git_commit: str
    git_dirty: bool | None


def _is_allowed_file(root: Path, path: Path) -> bool:
    """判断文件是否位于白名单且不属于运行产物。"""

    relative = path.relative_to(root)
    if path.is_symlink() or path.name.lower() in BLOCKED_FILE_NAMES:
        return False
    if any(part.lower() in BLOCKED_PARTS for part in relative.parts):
        return False
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    return True


def _collect_files(root: Path) -> tuple[Path, ...]:
    """仅从明确列出的文件和目录收集普通文件。"""

    selected: set[Path] = set()
    for relative_name in ALLOWLIST_FILES:
        path = root / relative_name
        if path.is_file() and _is_allowed_file(root, path):
            selected.add(path)
    for relative_name in ALLOWLIST_DIRECTORIES:
        directory = root / relative_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and _is_allowed_file(root, path):
                selected.add(path)
    return tuple(sorted(selected, key=lambda item: item.relative_to(root).as_posix()))


def _scan_file(path: Path) -> None:
    """拒绝大文件和常见明文凭据；不输出命中的具体内容。"""

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise SubmissionSecurityError(f"白名单文件超过 5 MiB：{path}")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionSecurityError(f"文本白名单文件不是 UTF-8：{path}") from exc
    high_confidence_hit = any(pattern.search(text) for pattern in HIGH_CONFIDENCE_SECRET_PATTERNS)
    # 测试允许使用 ``llm_api_key="x"`` 这类短夹具；真实厂商格式仍在所有目录拦截。
    assignment_hit = "tests" not in path.parts and any(
        pattern.search(text) for pattern in GENERIC_ASSIGNMENT_PATTERNS
    )
    if high_confidence_hit or assignment_hit:
        raise SubmissionSecurityError(f"白名单文件命中密钥模式：{path}")


def _git_state(root: Path) -> tuple[str, bool | None]:
    """记录源码版本和工作区状态；非 Git 测试目录返回 unknown。"""

    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        return "unknown", None
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return process.stdout.strip(), dirty


def build_submission_package(repository_root: Path, output_file: Path) -> SubmissionPackageResult:
    """扫描白名单并直接写 ZIP，不复制数据、不创建待清理的 staging 目录。"""

    root = repository_root.resolve(strict=True)
    output = output_file.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"提交包已存在，拒绝覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _collect_files(root)
    if not files:
        raise SubmissionSecurityError("白名单中没有可打包文件。")
    for path in files:
        _scan_file(path)

    commit, git_dirty = _git_state(root)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": git_dirty,
        "policy": "allowlist",
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(
            "submission_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    package_bytes = output.read_bytes()
    return SubmissionPackageResult(
        output_file=str(output),
        file_count=len(files),
        size_bytes=len(package_bytes),
        sha256=hashlib.sha256(package_bytes).hexdigest(),
        git_commit=commit,
        git_dirty=git_dirty,
    )


__all__ = [
    "SubmissionPackageResult",
    "SubmissionSecurityError",
    "build_submission_package",
]
