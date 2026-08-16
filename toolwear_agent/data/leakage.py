"""窗口级数据泄漏与索引完整性审计。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from toolwear_agent.core.errors import ToolWearError
from toolwear_agent.data.splitting import normalize_split_name
from toolwear_agent.schemas import SplitManifest


class WindowLike(Protocol):
    """泄漏审计所需的最小窗口字段。"""

    window_id: str
    cut: int
    file_path: str
    row_count: int
    start_row: int
    end_row: int
    split: str


class SplitLeakageError(ToolWearError, ValueError):
    """检测到跨 split 泄漏或窗口索引损坏。"""

    error_code = "SPLIT_LEAKAGE_DETECTED"


@dataclass(frozen=True)
class LeakageIssue:
    """一条可机器识别的泄漏审计问题。"""

    code: str
    detail: str


@dataclass(frozen=True)
class LeakageAudit:
    """窗口索引的泄漏审计结果。"""

    valid: bool
    record_count: int
    source_file_count: int
    cut_group_count: int
    issues: tuple[LeakageIssue, ...]


def _normalized_source_file(value: str) -> str:
    """统一路径分隔符和大小写，避免同一路径被当成两个文件。"""

    return str(value).replace("\\", "/").casefold()


def audit_window_splits(records: Iterable[WindowLike]) -> LeakageAudit:
    """检查源文件、cut、窗口 ID 和窗口区间是否存在泄漏或冲突。"""

    materialized = list(records)
    issues: list[LeakageIssue] = []
    splits_by_source: dict[str, set[str]] = {}
    splits_by_cut_group: dict[tuple[str, int], set[str]] = {}
    seen_window_ids: set[str] = set()
    seen_spans: set[tuple[str, int, int]] = set()

    for record in materialized:
        source_file = _normalized_source_file(record.file_path)
        try:
            split = normalize_split_name(record.split)
        except ValueError:
            issues.append(LeakageIssue("UNKNOWN_SPLIT", f"{record.window_id}: {record.split}"))
            continue

        splits_by_source.setdefault(source_file, set()).add(split)
        splits_by_cut_group.setdefault((source_file, record.cut), set()).add(split)

        if record.window_id in seen_window_ids:
            issues.append(LeakageIssue("DUPLICATE_WINDOW_ID", record.window_id))
        seen_window_ids.add(record.window_id)

        span = (source_file, record.start_row, record.end_row)
        if span in seen_spans:
            issues.append(
                LeakageIssue(
                    "DUPLICATE_WINDOW_SPAN",
                    f"{record.file_path}:{record.start_row}-{record.end_row}",
                )
            )
        seen_spans.add(span)

        if record.start_row < 0 or record.end_row <= record.start_row or record.end_row > record.row_count:
            issues.append(
                LeakageIssue(
                    "INVALID_WINDOW_BOUNDS",
                    f"{record.window_id}: {record.start_row}-{record.end_row}/{record.row_count}",
                )
            )

    for source_file, splits in sorted(splits_by_source.items()):
        if len(splits) > 1:
            issues.append(
                LeakageIssue(
                    "SOURCE_FILE_CROSS_SPLIT",
                    f"{source_file}: {', '.join(sorted(splits))}",
                )
            )
    for (source_file, cut), splits in sorted(splits_by_cut_group.items()):
        if len(splits) > 1:
            issues.append(
                LeakageIssue(
                    "CUT_CROSS_SPLIT",
                    f"{source_file} cut={cut}: {', '.join(sorted(splits))}",
                )
            )

    return LeakageAudit(
        valid=not issues,
        record_count=len(materialized),
        source_file_count=len(splits_by_source),
        cut_group_count=len(splits_by_cut_group),
        issues=tuple(issues),
    )


def assert_no_window_leakage(records: Iterable[WindowLike]) -> LeakageAudit:
    """执行审计并在发现问题时阻断后续训练。"""

    audit = audit_window_splits(records)
    if not audit.valid:
        details = "; ".join(f"{issue.code}: {issue.detail}" for issue in audit.issues)
        raise SplitLeakageError(f"窗口数据泄漏审计失败: {details}")
    return audit


def assert_windows_match_split_manifest(
    records: Iterable[WindowLike],
    manifest: SplitManifest,
) -> None:
    """阻止一个 cut 的全部窗口被整体搬到锁定清单之外的 split。"""

    expected_by_group = {
        (_normalized_source_file(item.source_file), item.cut_id): item.split
        for item in manifest.assignments
    }
    mismatches: list[str] = []
    for record in records:
        group_key = (_normalized_source_file(record.file_path), record.cut)
        expected = expected_by_group.get(group_key)
        actual = normalize_split_name(record.split)
        if expected is None:
            mismatches.append(f"{record.window_id}: 未出现在 SplitManifest")
        elif actual != expected:
            mismatches.append(f"{record.window_id}: actual={actual}, expected={expected}")
    if mismatches:
        raise SplitLeakageError("窗口归属与锁定 SplitManifest 不一致: " + "; ".join(mismatches[:20]))


def write_leakage_audit(audit: LeakageAudit, output_file: Path) -> Path:
    """原子写出泄漏审计证据。"""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(asdict(audit), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_file.replace(output_file)
    return output_file
