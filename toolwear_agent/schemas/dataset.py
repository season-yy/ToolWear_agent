"""数据集、标签策略和无泄漏切分的统一 Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from toolwear_agent.schemas.base import EntityId, NonEmptyText, SchemaModel, Sha256Hex, utc_now
from toolwear_agent.schemas.evaluation import ValidationResult


def _validate_relative_path(value: str, *, field_name: str) -> str:
    """校验 manifest 中的路径只能相对数据集根目录。"""

    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or normalized in {"", "."}:
        raise ValueError(f"{field_name} 必须是数据集根目录内的相对路径。")
    return normalized


class CutterManifest(SchemaModel):
    """一个刀具/工况在数据集中的文件布局。"""

    cutter_id: EntityId
    relative_path: str
    labeled: bool
    signal_glob: NonEmptyText = "c_*_*.csv"
    wear_file: str | None = None
    available: bool = True
    resolved_path: Path | None = None
    signal_file_count: int = Field(default=0, ge=0)
    wear_row_count: int | None = Field(default=None, ge=0)
    detected_channel_count: int | None = Field(default=None, ge=1)
    sampled_signal_lengths: tuple[int, ...] = ()
    inventory_hash: Sha256Hex | None = None
    wear_sha256: Sha256Hex | None = None

    @field_validator("relative_path")
    @classmethod
    def _relative_path_is_safe(cls, value: str) -> str:
        return _validate_relative_path(value, field_name="relative_path")

    @field_validator("wear_file")
    @classmethod
    def _wear_file_is_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_relative_path(value, field_name="wear_file")

    @model_validator(mode="after")
    def _labeled_cutter_has_wear_file(self) -> "CutterManifest":
        if self.labeled and not self.wear_file:
            raise ValueError("labeled=true 时必须配置 wear_file。")
        if not self.available and self.signal_file_count:
            raise ValueError("available=false 时 signal_file_count 必须为 0。")
        if any(length < 1 for length in self.sampled_signal_lengths):
            raise ValueError("sampled_signal_lengths 只能包含正整数。")
        return self


class DatasetManifest(SchemaModel):
    """可由 Dataset Registry 发现和验证的数据集描述。"""

    dataset_id: EntityId
    display_name: NonEmptyText
    adapter: EntityId
    root: Path
    channels: tuple[EntityId, ...] = Field(min_length=1)
    cutters: dict[str, CutterManifest] = Field(min_length=1)
    sampling_rate_hz: float | None = Field(default=None, gt=0)
    description: str = ""
    manifest_hash: Sha256Hex | None = None

    @field_validator("channels")
    @classmethod
    def _channels_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("channels 不能重复。")
        return value

    @model_validator(mode="after")
    def _cutter_keys_match_ids(self) -> "DatasetManifest":
        for key, cutter in self.cutters.items():
            if key != cutter.cutter_id:
                raise ValueError(f"cutters 键 {key} 与 cutter_id {cutter.cutter_id} 不一致。")
        return self

    @property
    def available_cutter_ids(self) -> tuple[str, ...]:
        """返回当前机器上实际可读的刀具 ID。"""

        return tuple(cutter_id for cutter_id, cutter in self.cutters.items() if cutter.available)

    @property
    def labeled_cutter_ids(self) -> tuple[str, ...]:
        """返回实际可用且带 VB 标签的刀具 ID。"""

        return tuple(
            cutter_id
            for cutter_id, cutter in self.cutters.items()
            if cutter.available and cutter.labeled
        )

    @property
    def unlabeled_cutter_ids(self) -> tuple[str, ...]:
        """返回实际可用但没有 VB 标签的刀具 ID。"""

        return tuple(
            cutter_id
            for cutter_id, cutter in self.cutters.items()
            if cutter.available and not cutter.labeled
        )


class DatasetInspection(SchemaModel):
    """Adapter 对一个数据集执行只读发现和体检后的结果。"""

    manifest: DatasetManifest
    validation: ValidationResult


class DatasetRef(SchemaModel):
    """实验对某个数据集和刀具集合的不可变引用。"""

    dataset_id: EntityId
    cutter_ids: tuple[EntityId, ...] = Field(min_length=1)
    manifest_hash: Sha256Hex | None = None
    source_revision: str | None = None

    @field_validator("cutter_ids")
    @classmethod
    def _cutters_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("cutter_ids 不能重复。")
        return value


class VbAggregation(str, Enum):
    """三刃 VB 值的聚合方式。"""

    MAX = "max"
    MEAN = "mean"
    SPECIFIED_FLUTE = "specified_flute"


class LabelPolicy(SchemaModel):
    """VB 连续值到四阶段标签的可追溯规则。"""

    aggregation: VbAggregation = VbAggregation.MAX
    stage_thresholds_um: tuple[float, float, float] = (90.0, 130.0, 160.0)
    stage_names: tuple[str, str, str, str] = ("initial", "normal", "severe", "failure")
    enable_regression: bool = False
    specified_flute: int | None = Field(default=None, ge=1, le=3)
    vb_unit: str = "um"

    @field_validator("stage_thresholds_um", mode="before")
    @classmethod
    def _parse_thresholds(cls, value: object) -> tuple[float, float, float]:
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            raise ValueError("stage_thresholds_um 必须包含三个数值。")
        if len(parts) != 3:
            raise ValueError("四阶段分类必须有三个边界。")
        thresholds = tuple(float(item) for item in parts)
        if not thresholds[0] < thresholds[1] < thresholds[2]:
            raise ValueError("stage_thresholds_um 必须严格递增。")
        return thresholds  # type: ignore[return-value]

    @model_validator(mode="after")
    def _specified_flute_matches_mode(self) -> "LabelPolicy":
        if self.aggregation is VbAggregation.SPECIFIED_FLUTE and self.specified_flute is None:
            raise ValueError("specified_flute 聚合必须指定 1-3 号刀刃。")
        if self.aggregation is not VbAggregation.SPECIFIED_FLUTE and self.specified_flute is not None:
            raise ValueError("只有 specified_flute 聚合可以设置 specified_flute。")
        return self


class SplitStrategy(str, Enum):
    """P0 支持的无泄漏切分策略。"""

    GROUP_BY_CUT = "group_by_cut"
    CROSS_CUTTER = "cross_cutter"


class SplitSpec(SchemaModel):
    """按 cut 分组且可锁定的 train/validation/test 切分配置。"""

    strategy: SplitStrategy = SplitStrategy.GROUP_BY_CUT
    train_ratio: float = Field(default=0.6, gt=0, lt=1)
    validation_ratio: float = Field(default=0.2, gt=0, lt=1)
    test_ratio: float = Field(default=0.2, gt=0, lt=1)
    group_key: str = "cut_id"
    random_seed: int = Field(default=42, ge=0)
    time_aware: bool = True
    train_cutters: tuple[EntityId, ...] = ()
    validation_cutters: tuple[EntityId, ...] = ()
    test_cutters: tuple[EntityId, ...] = ()
    locked: bool = False
    split_hash: Sha256Hex | None = None

    @model_validator(mode="after")
    def _validate_split(self) -> "SplitSpec":
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("train/validation/test 比例之和必须为 1。")

        cutter_groups = [set(self.train_cutters), set(self.validation_cutters), set(self.test_cutters)]
        if any(len(group) != len(values) for group, values in zip(cutter_groups, (
            self.train_cutters,
            self.validation_cutters,
            self.test_cutters,
        ))):
            raise ValueError("每个 split 内的 cutter 不能重复。")
        if cutter_groups[0] & cutter_groups[1] or cutter_groups[0] & cutter_groups[2] or cutter_groups[1] & cutter_groups[2]:
            raise ValueError("train、validation、test 的 cutter 不能重叠。")

        if self.strategy is SplitStrategy.CROSS_CUTTER:
            if not self.train_cutters or not self.test_cutters:
                raise ValueError("cross_cutter 至少需要 train_cutters 和 test_cutters。")
        elif any(cutter_groups):
            raise ValueError("group_by_cut 不应固定跨刀具集合。")

        if self.locked and self.split_hash is None:
            raise ValueError("locked=true 时必须保存 split_hash。")
        return self


class SplitAssignment(SchemaModel):
    """一个刀次在固定数据切分中的唯一归属。"""

    cutter_id: EntityId
    cut_id: int = Field(ge=1)
    source_file: NonEmptyText
    row_count: int = Field(ge=1)
    vb_value: float
    stage_id: int = Field(ge=0)
    stage_name: NonEmptyText
    split: Literal["train", "validation", "test"]


class SplitManifest(SchemaModel):
    """可计算稳定哈希、可被实验修订锁定的 cut 级切分清单。"""

    dataset_id: EntityId
    cutter_id: EntityId
    split_spec: SplitSpec
    assignments: tuple[SplitAssignment, ...] = Field(min_length=1)
    split_hash: Sha256Hex | None = None

    @model_validator(mode="after")
    def _assignments_are_unique_and_consistent(self) -> "SplitManifest":
        cut_keys = [(item.cutter_id, item.cut_id) for item in self.assignments]
        source_files = [item.source_file.casefold() for item in self.assignments]
        if len(cut_keys) != len(set(cut_keys)):
            raise ValueError("SplitManifest 中同一 cutter/cut 只能出现一次。")
        if len(source_files) != len(set(source_files)):
            raise ValueError("SplitManifest 中同一个源文件只能出现一次。")
        if any(item.cutter_id != self.cutter_id for item in self.assignments):
            raise ValueError("SplitAssignment.cutter_id 必须与 Manifest 一致。")
        return self


class SplitLock(SchemaModel):
    """把一个 split_hash 固定到 experiment revision，防止事后重划测试集。"""

    experiment_id: EntityId
    revision: int = Field(ge=1)
    dataset_id: EntityId
    cutter_id: EntityId
    split_hash: Sha256Hex
    manifest_file: NonEmptyText
    locked_at: datetime = Field(default_factory=utc_now)


class SampledWindowRef(SchemaModel):
    """小样本清单中的窗口引用，不复制原始信号。"""

    window_id: EntityId
    cutter_id: EntityId
    cut_id: int = Field(ge=1)
    source_file: NonEmptyText
    stage_id: int = Field(ge=0)
    stage_name: NonEmptyText
    start_row: int = Field(ge=0)
    end_row: int = Field(gt=0)
    split: Literal["train"] = "train"

    @model_validator(mode="after")
    def _window_bounds_are_valid(self) -> "SampledWindowRef":
        if self.end_row <= self.start_row:
            raise ValueError("采样窗口 end_row 必须大于 start_row。")
        return self


class TrainingSampleManifest(SchemaModel):
    """训练集内小样本的可复现清单和完整性哈希。"""

    sample_id: EntityId
    dataset_id: EntityId
    cutter_id: EntityId
    source_split_hash: Sha256Hex
    fraction: float = Field(gt=0, le=1)
    random_seed: int = Field(ge=0)
    full_train_count: int = Field(ge=1)
    selected_count: int = Field(ge=1)
    stage_distribution: dict[str, int] = Field(min_length=1)
    cut_distribution: dict[str, int] = Field(min_length=1)
    windows: tuple[SampledWindowRef, ...] = Field(min_length=1)
    sample_hash: Sha256Hex | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _counts_match_windows(self) -> "TrainingSampleManifest":
        if self.selected_count != len(self.windows):
            raise ValueError("selected_count 必须等于 windows 数量。")
        if self.selected_count > self.full_train_count:
            raise ValueError("selected_count 不能大于 full_train_count。")
        if sum(self.stage_distribution.values()) != self.selected_count:
            raise ValueError("stage_distribution 合计必须等于 selected_count。")
        if sum(self.cut_distribution.values()) != self.selected_count:
            raise ValueError("cut_distribution 合计必须等于 selected_count。")
        window_ids = [item.window_id for item in self.windows]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("TrainingSampleManifest 中 window_id 不能重复。")
        return self
