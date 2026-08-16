"""按 Dataset、Experiment、Revision 和 Run 实体计算稳定路径。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from toolwear_agent.core.errors import InvalidIdentifierError
from toolwear_agent.core.security import ensure_path_within, validate_entity_id
from toolwear_agent.core.settings import Settings


class PathResolver:
    """集中构造项目路径，并对用户可控 ID 做路径穿越防护。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def dataset_manifest_path(self, dataset_id: str) -> Path:
        """返回指定数据集的 YAML manifest 路径。"""

        safe_id = validate_entity_id(dataset_id, field_name="dataset_id")
        return self.settings.dataset_manifest.parent / f"{safe_id}.yaml"

    def experiment_path(self, experiment_id: str) -> Path:
        """返回实验根目录。"""

        safe_id = validate_entity_id(experiment_id, field_name="experiment_id")
        return self.settings.experiment_root / safe_id

    def revision_path(self, experiment_id: str, revision: int) -> Path:
        """返回实验某个不可变修订版本的目录。"""

        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise InvalidIdentifierError("revision 必须是大于等于 1 的整数。")
        return self.experiment_path(experiment_id) / "revisions" / f"r{revision:04d}"

    def run_path(self, experiment_id: str, revision: int, run_id: str) -> Path:
        """返回一次训练或评估 Run 的目录。"""

        safe_run_id = validate_entity_id(run_id, field_name="run_id")
        return self.revision_path(experiment_id, revision) / "runs" / safe_run_id

    def report_path(self, experiment_id: str, revision: int, run_id: str) -> Path:
        """返回 Run 的报告目录。"""

        return self.run_path(experiment_id, revision, run_id) / "report"

    def evidence_path(self, experiment_id: str, revision: int, run_id: str) -> Path:
        """返回 Run 的证据目录。"""

        return self.run_path(experiment_id, revision, run_id) / "evidence"

    def generated_code_sandbox(self, experiment_id: str, revision: int, run_id: str) -> Path:
        """返回只属于本 Run 的生成代码沙箱。"""

        return self.run_path(experiment_id, revision, run_id) / "generated_code"

    def agent_trace_path(self, experiment_id: str, revision: int, task_id: str) -> Path:
        """返回一次 Agent 调用的独立证据目录。"""

        safe_task_id = validate_entity_id(task_id, field_name="task_id")
        return self.revision_path(experiment_id, revision) / "agents" / safe_task_id

    def runtime_directories(self) -> tuple[Path, ...]:
        """列出允许应用创建的目录，不包含任何原始数据目录。"""

        directories = (
            self.settings.dataset_manifest.parent,
            self.settings.experiment_root,
            self.settings.artifact_root,
            self.settings.log_root,
            self.settings.state_root,
            self.settings.report_root,
            self.settings.evidence_root,
            self.settings.generated_code_root,
            self.settings.cache_root,
        )
        # dict 保序去重，避免用户把多个逻辑目录配置为同一路径时重复创建。
        return tuple(dict.fromkeys(directories))

    def ensure_runtime_directories(self) -> tuple[Path, ...]:
        """创建明确的可写运行目录，不扫描、清理或创建原始数据区。"""

        directories = self.runtime_directories()
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def assert_within(self, path: str | Path, allowed_roots: Iterable[str | Path]) -> Path:
        """解析并校验任意读取路径是否位于显式允许范围。"""

        return ensure_path_within(path, allowed_roots)

    def assert_dataset_read_path(
        self,
        path: str | Path,
        *,
        additional_roots: Iterable[str | Path] = (),
    ) -> Path:
        """校验数据读取路径，包括 manifest 明确声明的 Junction 真实目标。"""

        roots = [self.settings.phm2010_raw_root, *additional_roots]
        return self.assert_within(path, roots)
