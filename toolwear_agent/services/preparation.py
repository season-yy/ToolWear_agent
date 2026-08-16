"""数据体检、标签、无泄漏切分和小样本清单服务。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from toolwear_agent.core.paths import PathResolver
from toolwear_agent.core.settings import Settings
from toolwear_agent.data.leakage import (
    assert_no_window_leakage,
    assert_windows_match_split_manifest,
    write_leakage_audit,
)
from toolwear_agent.data.registry import DatasetRegistry
from toolwear_agent.data.sampling import build_training_sample, write_sample_manifest
from toolwear_agent.data.splitting import (
    attach_split_hash,
    build_split_manifest,
    create_or_verify_split_lock,
    load_split_manifest,
    write_split_manifest,
)
from toolwear_agent.schemas import EvidenceRef, ExperimentState, TrainingDataRef
from toolwear_agent.schemas.api import ExperimentActionResponse
from toolwear_agent.schemas.experiment import ExperimentStatus
from toolwear_agent.services.errors import InvalidWorkflowStateError
from toolwear_agent.state import EntityNotFoundError, SQLiteExperimentRepository
from toolwear_agent.training.labels import (
    build_label_dataset,
    write_label_csv,
    write_label_json,
    write_label_report,
)
from toolwear_agent.training.windows import (
    assign_cut_splits,
    build_window_records,
    load_cut_labels,
    write_split_manifest as write_split_csv,
    write_window_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


class DataPreparationService:
    """只处理数据证据，不负责候选生成或模型训练。"""

    def __init__(
        self,
        settings: Settings,
        repository: SQLiteExperimentRepository,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.path_resolver = PathResolver(settings)
        self.dataset_registry = DatasetRegistry(settings.dataset_manifest)

    @staticmethod
    def _require_state(state: ExperimentState, *allowed: ExperimentStatus) -> None:
        if state.state not in allowed:
            values = ", ".join(item.value for item in allowed)
            raise InvalidWorkflowStateError(
                f"{state.state.value} 状态不能执行该数据动作；允许状态：{values}。"
            )

    @staticmethod
    def _single_cutter(state: ExperimentState) -> str:
        if len(state.dataset_ref.cutter_ids) != 1:
            raise ValueError("P0 数据准备当前一次只支持一把有标签刀具。")
        return state.dataset_ref.cutter_ids[0]

    def _preparation_dir(self, state: ExperimentState) -> Path:
        return self.path_resolver.revision_path(state.experiment_id, state.revision) / "preparation"

    def _existing_evidence(self, evidence_id: str) -> EvidenceRef | None:
        try:
            return self.repository.get_evidence(evidence_id)
        except EntityNotFoundError:
            return None

    def _register_file(
        self,
        *,
        state: ExperimentState,
        evidence_id: str,
        kind: str,
        path: Path,
        description: str,
        idempotency_key: str | None,
    ) -> EvidenceRef:
        evidence = EvidenceRef(
            evidence_id=evidence_id,
            experiment_id=state.experiment_id,
            kind=kind,
            uri=str(path),
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
            media_type="application/json" if path.suffix.lower() == ".json" else "text/csv",
            description=description,
            created_by="DataStewardAgent",
        )
        return self.repository.register_evidence(
            evidence,
            idempotency_key=idempotency_key,
        )

    def profile(
        self,
        experiment_id: str,
        *,
        rationale: str,
        idempotency_key: str | None,
    ) -> ExperimentActionResponse:
        state = self.repository.get_experiment(experiment_id)
        evidence_id = f"{experiment_id}-profile-r{state.revision}"
        existing = self._existing_evidence(evidence_id)
        if existing is not None:
            return ExperimentActionResponse(
                operation="profile",
                summary="数据体检结果已存在，已按证据 ID 恢复。",
                state=state,
                evidence=(existing,),
            )
        self._require_state(
            state,
            ExperimentStatus.DRAFT,
            ExperimentStatus.BLOCKED_DATA,
            ExperimentStatus.DATA_VALIDATING,
        )
        if state.state in {ExperimentStatus.DRAFT, ExperimentStatus.BLOCKED_DATA}:
            state = self.repository.transition_state(
                experiment_id,
                ExperimentStatus.DATA_VALIDATING,
                actor="DataStewardAgent",
                reason=rationale,
                idempotency_key=(f"{idempotency_key}:state" if idempotency_key else None),
            )
        manifest = self.dataset_registry.get(state.dataset_ref.dataset_id)
        cutter_ids = state.dataset_ref.cutter_ids
        profile_payload = {
            "dataset_id": manifest.dataset_id,
            "manifest_hash": manifest.manifest_hash,
            "channels": manifest.channels,
            "sampling_rate_hz": manifest.sampling_rate_hz,
            "cutters": {
                cutter_id: manifest.cutters[cutter_id].model_dump(mode="json")
                for cutter_id in cutter_ids
            },
            "validation": {
                "available": all(manifest.cutters[item].available for item in cutter_ids),
                "labeled": all(manifest.cutters[item].labeled for item in cutter_ids),
                "requested_channels_valid": set(state.preferences.input_channels)
                <= set(manifest.channels),
            },
        }
        output_file = _write_json(self._preparation_dir(state) / "profile.json", profile_payload)
        evidence = self._register_file(
            state=state,
            evidence_id=evidence_id,
            kind="report",
            path=output_file,
            description="Dataset Registry 数据体检快照",
            idempotency_key=idempotency_key,
        )
        return ExperimentActionResponse(
            operation="profile",
            summary="数据集、刀具、通道和标签可用性体检完成。",
            state=self.repository.get_experiment(experiment_id),
            payload={"profile_file": str(output_file)},
            evidence=(evidence,),
        )

    def labels(
        self,
        experiment_id: str,
        *,
        rationale: str,
        idempotency_key: str | None,
    ) -> ExperimentActionResponse:
        del rationale
        state = self.repository.get_experiment(experiment_id)
        evidence_id = f"{experiment_id}-labels-r{state.revision}"
        existing = self._existing_evidence(evidence_id)
        if existing is not None:
            return ExperimentActionResponse(
                operation="labels",
                summary="标签结果已存在，已按证据 ID 恢复。",
                state=state,
                evidence=(existing,),
            )
        self._require_state(state, ExperimentStatus.DATA_VALIDATING)
        cutter_id = self._single_cutter(state)
        manifest = self.dataset_registry.get(state.dataset_ref.dataset_id)
        cutter = manifest.cutters[cutter_id]
        if cutter.resolved_path is None or cutter.wear_file is None:
            raise ValueError(f"刀具 {cutter_id} 缺少可读磨损标签路径。")
        wear_file = cutter.resolved_path / cutter.wear_file
        output_dir = self._preparation_dir(state)
        label_dataset = build_label_dataset(
            wear_file=wear_file,
            cutter=cutter_id.lower(),
            aggregation=state.label_policy.aggregation.value,
            thresholds=state.label_policy.stage_thresholds_um,
            stage_names=state.label_policy.stage_names,
            specified_flute=state.label_policy.specified_flute,
            dataset_id=state.dataset_ref.dataset_id,
        )
        json_file = write_label_json(label_dataset, output_dir / "labels.json")
        csv_file = write_label_csv(label_dataset, output_dir / "labels.csv")
        report_file = write_label_report(label_dataset, output_dir / "labels.md")
        evidence = self._register_file(
            state=state,
            evidence_id=evidence_id,
            kind="config",
            path=json_file,
            description="用户标签策略生成的四阶段标签",
            idempotency_key=idempotency_key,
        )
        return ExperimentActionResponse(
            operation="labels",
            summary="VB 聚合和四阶段标签生成完成。",
            state=state,
            payload={
                "label_json": str(json_file),
                "label_csv": str(csv_file),
                "label_report": str(report_file),
                "record_count": label_dataset.record_count,
                "stage_distribution": label_dataset.stage_distribution,
            },
            evidence=(evidence,),
        )

    def _can_reuse_prebuilt_cache(self, state: ExperimentState, cutter_id: str) -> bool:
        """判断当前参数是否与已登记的预处理缓存完全一致。"""

        return (
            state.dataset_ref.dataset_id == "phm2010"
            and cutter_id.casefold() == "c1"
            and state.label_policy.aggregation.value == "max"
            and state.label_policy.stage_thresholds_um == (90.0, 130.0, 160.0)
            and state.split_spec.train_ratio == 0.6
            and state.split_spec.validation_ratio == 0.2
            and state.split_spec.test_ratio == 0.2
            and state.split_spec.random_seed == 42
            and state.preferences.window_length == 4096
            and state.preferences.overlap == 0.5
            and state.preferences.max_windows_per_cut == 32
            and state.preferences.sample_fraction == 0.2
        )

    def _reuse_prebuilt_data_ref(self, state: ExperimentState, cutter_id: str) -> TrainingDataRef:
        """按数据集和刀具标识定位缓存，不把具体刀具固化到业务结构中。"""

        dataset_id = state.dataset_ref.dataset_id
        cutter_token = cutter_id.lower()
        cache_prefix = f"{dataset_id}_{cutter_token}"
        root = self.settings.ai_infra_root / "datasets" / "processed" / dataset_id
        required = {
            "window_manifest_file": root / f"{cache_prefix}_window_manifest.csv",
            "training_sample_manifest_file": root / f"{cache_prefix}_train_sample_20pct.json",
            "split_manifest_file": root / f"{cache_prefix}_split_manifest.json",
            "leakage_audit_file": root / f"{cache_prefix}_leakage_audit.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("预处理缓存不完整：" + "；".join(missing))
        split_manifest = load_split_manifest(required["split_manifest_file"])
        lock_file = (
            self.settings.state_root
            / "splits"
            / state.experiment_id
            / f"r{state.revision:04d}"
            / "split_lock.json"
        )
        create_or_verify_split_lock(
            manifest=split_manifest,
            lock_file=lock_file,
            experiment_id=state.experiment_id,
            revision=state.revision,
            manifest_file=required["split_manifest_file"],
        )
        return TrainingDataRef(
            dataset_id=state.dataset_ref.dataset_id,
            cutter_id=cutter_id.lower(),
            split_lock_file=lock_file,
            **required,
        )

    def _build_custom_data_ref(self, state: ExperimentState, cutter_id: str) -> TrainingDataRef:
        manifest = self.dataset_registry.get(state.dataset_ref.dataset_id)
        if manifest.adapter != "phm2010":
            raise ValueError(f"当前窗口构建器不支持 Adapter：{manifest.adapter}")
        cutter = manifest.cutters[cutter_id]
        if cutter.resolved_path is None:
            raise ValueError(f"刀具 {cutter_id} 缺少已体检 resolved_path。")
        output_dir = self._preparation_dir(state) / "split"
        label_file = self._preparation_dir(state) / "labels.csv"
        if not label_file.is_file():
            raise FileNotFoundError("请先执行 labels，再执行 split。")
        cut_labels = load_cut_labels(label_file, cutter.resolved_path, cutter_id.lower())
        split_by_cut = assign_cut_splits(
            cut_labels,
            train_ratio=state.split_spec.train_ratio,
            val_ratio=state.split_spec.validation_ratio,
            random_seed=state.split_spec.random_seed,
        )
        records = build_window_records(
            cut_labels,
            split_by_cut,
            cutter_id.lower(),
            window_size=state.preferences.window_length,
            overlap_ratio=state.preferences.overlap,
            max_windows_per_cut=state.preferences.max_windows_per_cut,
        )
        audit = assert_no_window_leakage(records)
        split_manifest = attach_split_hash(
            build_split_manifest(
                cut_labels=cut_labels,
                split_by_cut=split_by_cut,
                dataset_id=state.dataset_ref.dataset_id,
                cutter_id=cutter_id.lower(),
                split_spec=state.split_spec,
            )
        )
        split_manifest_file = output_dir / "split_manifest.json"
        window_manifest_file = output_dir / "window_manifest.csv"
        leakage_audit_file = output_dir / "leakage_audit.json"
        sample_manifest_file = output_dir / "training_sample_manifest.json"
        split_csv_file = output_dir / "split_manifest.csv"
        write_split_manifest(split_manifest, split_manifest_file)
        write_split_csv(cut_labels, split_by_cut, split_csv_file)
        write_window_manifest(records, window_manifest_file)
        write_leakage_audit(audit, leakage_audit_file)
        assert_windows_match_split_manifest(records, split_manifest)
        if split_manifest.split_hash is None:  # pragma: no cover - attach 已保证
            raise ValueError("split_hash 不能为空。")
        sample = build_training_sample(
            records,
            dataset_id=state.dataset_ref.dataset_id,
            cutter_id=cutter_id.lower(),
            split_hash=split_manifest.split_hash,
            fraction=state.preferences.sample_fraction,
            random_seed=state.split_spec.random_seed,
        )
        write_sample_manifest(sample.manifest, sample_manifest_file)
        lock_file = (
            self.settings.state_root
            / "splits"
            / state.experiment_id
            / f"r{state.revision:04d}"
            / "split_lock.json"
        )
        create_or_verify_split_lock(
            manifest=split_manifest,
            lock_file=lock_file,
            experiment_id=state.experiment_id,
            revision=state.revision,
            manifest_file=split_manifest_file,
        )
        return TrainingDataRef(
            dataset_id=state.dataset_ref.dataset_id,
            cutter_id=cutter_id.lower(),
            window_manifest_file=window_manifest_file,
            training_sample_manifest_file=sample_manifest_file,
            split_manifest_file=split_manifest_file,
            split_lock_file=lock_file,
            leakage_audit_file=leakage_audit_file,
        )

    def split(
        self,
        experiment_id: str,
        *,
        rationale: str,
        idempotency_key: str | None,
    ) -> ExperimentActionResponse:
        del rationale
        state = self.repository.get_experiment(experiment_id)
        evidence_id = f"{experiment_id}-split-r{state.revision}"
        existing = self._existing_evidence(evidence_id)
        data_ref_file = self._preparation_dir(state) / "training_data_ref.json"
        if existing is not None and data_ref_file.is_file():
            data_ref = TrainingDataRef.model_validate_json(data_ref_file.read_text(encoding="utf-8"))
            return ExperimentActionResponse(
                operation="split",
                summary="切分与小样本结果已存在，已按证据 ID 恢复。",
                state=state,
                payload=data_ref.model_dump(mode="json"),
                evidence=(existing,),
            )
        self._require_state(state, ExperimentStatus.DATA_VALIDATING)
        cutter_id = self._single_cutter(state)
        if self._can_reuse_prebuilt_cache(state, cutter_id):
            data_ref = self._reuse_prebuilt_data_ref(state, cutter_id)
            mode = "reused_default_cache"
        else:
            data_ref = self._build_custom_data_ref(state, cutter_id)
            mode = "built_revision_specific"
        _write_json(data_ref_file, data_ref.model_dump(mode="json"))
        evidence = self._register_file(
            state=state,
            evidence_id=evidence_id,
            kind="split",
            path=data_ref_file,
            description="训练数据引用、split lock 与小样本清单",
            idempotency_key=idempotency_key,
        )
        return ExperimentActionResponse(
            operation="split",
            summary="cut 级切分、泄漏审计和训练小样本清单已锁定。",
            state=state,
            payload={"mode": mode, **data_ref.model_dump(mode="json")},
            evidence=(evidence,),
        )

    def load_training_data_ref(self, state: ExperimentState) -> TrainingDataRef:
        path = self._preparation_dir(state) / "training_data_ref.json"
        if not path.is_file():
            raise FileNotFoundError("实验缺少 training_data_ref.json，请先执行 split。")
        return TrainingDataRef.model_validate_json(path.read_text(encoding="utf-8"))

    def bind_data_ref_to_revision(
        self,
        state: ExperimentState,
        target_revision: int,
    ) -> TrainingDataRef:
        """让后续 revision 复用同一切分，同时生成自己的不可变 split lock。"""

        source = self.load_training_data_ref(state)
        if target_revision == state.revision:
            return source
        split_manifest = load_split_manifest(source.split_manifest_file)
        lock_file = (
            self.settings.state_root
            / "splits"
            / state.experiment_id
            / f"r{target_revision:04d}"
            / "split_lock.json"
        )
        create_or_verify_split_lock(
            manifest=split_manifest,
            lock_file=lock_file,
            experiment_id=state.experiment_id,
            revision=target_revision,
            manifest_file=source.split_manifest_file,
        )
        rebound = source.model_copy(update={"split_lock_file": lock_file})
        output_file = (
            self.path_resolver.revision_path(state.experiment_id, target_revision)
            / "preparation"
            / "training_data_ref.json"
        )
        _write_json(output_file, rebound.model_dump(mode="json"))
        return rebound
