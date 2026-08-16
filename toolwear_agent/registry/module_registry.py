"""P0 Module Registry 与输入通道预设。"""

from __future__ import annotations

from toolwear_agent.schemas import (
    FeatureType,
    InputPresetDefinition,
    ModuleDefinition,
    ParameterRule,
)


PHM2010_CHANNEL_IDS = (
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "acoustic_emission_rms",
)


class ModuleRegistry:
    """保存可由页面、LLM 和训练服务共同引用的模块定义。"""

    def __init__(
        self,
        modules: tuple[ModuleDefinition, ...] = (),
        input_presets: tuple[InputPresetDefinition, ...] = (),
    ) -> None:
        self._modules: dict[str, ModuleDefinition] = {}
        self._input_presets: dict[str, InputPresetDefinition] = {}
        for module in modules:
            self.register(module)
        for preset in input_presets:
            self.register_input_preset(preset)

    def register(self, module: ModuleDefinition) -> None:
        """登记模块；重复 ID 必须显式报错，不能静默覆盖。"""

        if module.module_id in self._modules:
            raise ValueError(f"Module Registry 已存在模块: {module.module_id}")
        self._modules[module.module_id] = module

    def register_input_preset(self, preset: InputPresetDefinition) -> None:
        """登记页面可选择的输入通道组合。"""

        if preset.preset_id in self._input_presets:
            raise ValueError(f"Module Registry 已存在输入预设: {preset.preset_id}")
        unknown = set(preset.channel_ids) - set(PHM2010_CHANNEL_IDS)
        if unknown:
            raise ValueError(f"输入预设包含未知通道: {sorted(unknown)}")
        self._input_presets[preset.preset_id] = preset

    def get(self, module_id: str) -> ModuleDefinition:
        """按稳定 ID 返回模块定义。"""

        try:
            return self._modules[module_id]
        except KeyError as exc:
            raise KeyError(f"Module Registry 中不存在模块: {module_id}") from exc

    def get_input_preset(self, preset_id: str) -> InputPresetDefinition:
        """按稳定 ID 返回输入通道预设。"""

        try:
            return self._input_presets[preset_id]
        except KeyError as exc:
            raise KeyError(f"Module Registry 中不存在输入预设: {preset_id}") from exc

    def list_modules(self) -> tuple[ModuleDefinition, ...]:
        """按类别和 ID 稳定返回所有模块。"""

        return tuple(sorted(self._modules.values(), key=lambda item: (item.kind.value, item.module_id)))

    def list_input_presets(self) -> tuple[InputPresetDefinition, ...]:
        """按 ID 稳定返回输入预设。"""

        return tuple(self._input_presets[key] for key in sorted(self._input_presets))


def _integer(
    description: str,
    default: int,
    minimum: int,
    maximum: int,
) -> ParameterRule:
    """构造整数参数约束。"""

    return ParameterRule(
        value_type="integer",
        description=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _number(
    description: str,
    default: float,
    minimum: float,
    maximum: float,
) -> ParameterRule:
    """构造浮点参数约束。"""

    return ParameterRule(
        value_type="number",
        description=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def _boolean(description: str, default: bool) -> ParameterRule:
    """构造布尔参数约束。"""

    return ParameterRule(value_type="boolean", description=description, default=default)


def _choice(description: str, default: str, choices: tuple[str, ...]) -> ParameterRule:
    """构造字符串枚举参数约束。"""

    return ParameterRule(
        value_type="string",
        description=description,
        default=default,
        choices=choices,
    )


def _default_input_presets() -> tuple[InputPresetDefinition, ...]:
    """返回比赛 P0 页面允许直接选择的三组输入。"""

    return (
        InputPresetDefinition(
            preset_id="force_xyz",
            display_name="三向切削力",
            channel_ids=("force_x", "force_y", "force_z"),
            description="仅使用 X/Y/Z 三向切削力信号。",
        ),
        InputPresetDefinition(
            preset_id="vibration_xyz",
            display_name="三向振动",
            channel_ids=("vibration_x", "vibration_y", "vibration_z"),
            description="仅使用 X/Y/Z 三向振动信号。",
        ),
        InputPresetDefinition(
            preset_id="all_7_channels",
            display_name="全部七通道",
            channel_ids=PHM2010_CHANNEL_IDS,
            description="使用三向力、三向振动和声发射 RMS。",
        ),
    )


def _default_modules() -> tuple[ModuleDefinition, ...]:
    """返回 P0 已实现能力和明确标记的后续能力。"""

    return (
        ModuleDefinition(
            module_id="stable_region",
            display_name="稳定切削区截取",
            kind="preprocess",
            required_feature_type=FeatureType.RAW_SIGNAL,
            output_feature_type=FeatureType.RAW_SIGNAL,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "trim_fraction": _number("从信号首尾各裁去的比例。", 0.1, 0.0, 0.4),
            },
            resource_class="low",
            implemented=False,
            experimental=True,
        ),
        ModuleDefinition(
            module_id="sliding_window",
            display_name="重叠滑动窗口",
            kind="windowing",
            required_feature_type=FeatureType.RAW_SIGNAL,
            output_feature_type=FeatureType.WINDOWED_SIGNAL,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "window_length": _integer("每个窗口的采样点数。", 4096, 256, 65536),
                "overlap": _number("相邻窗口重叠比例。", 0.5, 0.0, 0.95),
            },
            resource_class="low",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="zscore",
            display_name="按训练集 Z-score 标准化",
            kind="preprocess",
            required_feature_type=FeatureType.WINDOWED_SIGNAL,
            output_feature_type=FeatureType.WINDOWED_SIGNAL,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "epsilon": _number("标准差下限，避免除零。", 1e-8, 1e-12, 1e-3),
            },
            resource_class="low",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="robust_scaler",
            display_name="稳健特征缩放",
            kind="preprocess",
            required_feature_type=FeatureType.TABULAR_FEATURES,
            output_feature_type=FeatureType.TABULAR_FEATURES,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "lower_quantile": _number("下分位点。", 25.0, 0.0, 49.0),
                "upper_quantile": _number("上分位点。", 75.0, 51.0, 100.0),
            },
            resource_class="low",
            implemented=False,
            experimental=True,
        ),
        ModuleDefinition(
            module_id="statistical_features",
            display_name="时域统计特征",
            kind="feature",
            required_feature_type=FeatureType.WINDOWED_SIGNAL,
            output_feature_type=FeatureType.TABULAR_FEATURES,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "include_frequency_features": _boolean("是否增加少量频域能量特征。", False),
            },
            resource_class="low",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="raw_1d",
            display_name="原始一维时序",
            kind="feature",
            required_feature_type=FeatureType.WINDOWED_SIGNAL,
            output_feature_type=FeatureType.RAW_1D,
            min_channels=1,
            max_channels=7,
            resource_class="low",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="early_concat",
            display_name="通道早期拼接",
            kind="fusion",
            required_feature_type=FeatureType.RAW_1D,
            output_feature_type=FeatureType.RAW_1D,
            min_channels=2,
            max_channels=7,
            resource_class="low",
            implemented=False,
        ),
        ModuleDefinition(
            module_id="random_forest",
            display_name="RandomForest 分类器",
            kind="model",
            required_feature_type=FeatureType.TABULAR_FEATURES,
            output_feature_type=FeatureType.LOGITS,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "n_estimators": _integer("决策树数量。", 300, 10, 2000),
                "max_depth": _integer("最大深度；0 表示不限制。", 0, 0, 100),
                "class_weight": _choice(
                    "类别权重策略。",
                    "balanced",
                    ("none", "balanced", "balanced_subsample"),
                ),
            },
            trainer_backend="sklearn",
            resource_class="low",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="extra_trees",
            display_name="ExtraTrees 分类器",
            kind="model",
            required_feature_type=FeatureType.TABULAR_FEATURES,
            output_feature_type=FeatureType.LOGITS,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "n_estimators": _integer("极端随机树数量。", 300, 10, 2000),
                "max_depth": _integer("最大深度；0 表示不限制。", 0, 0, 100),
                "class_weight": _choice(
                    "类别权重策略。",
                    "balanced",
                    ("none", "balanced", "balanced_subsample"),
                ),
            },
            trainer_backend="sklearn",
            resource_class="low",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="cnn_1d",
            display_name="轻量 1D-CNN",
            kind="model",
            required_feature_type=FeatureType.RAW_1D,
            output_feature_type=FeatureType.LOGITS,
            min_channels=1,
            max_channels=7,
            supported_channel_counts=(1, 3, 7),
            parameters_schema={
                "base_channels": _integer("首层卷积通道数。", 32, 8, 256),
                "dropout": _number("分类头 Dropout 比例。", 0.2, 0.0, 0.8),
            },
            trainer_backend="pytorch",
            resource_class="medium",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="cross_entropy",
            display_name="交叉熵损失",
            kind="loss",
            required_feature_type=FeatureType.LOGITS,
            output_feature_type=FeatureType.LOGITS,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "label_smoothing": _number("标签平滑比例。", 0.0, 0.0, 0.3),
            },
            trainer_backend="pytorch",
            resource_class="low",
            implemented=True,
        ),
        ModuleDefinition(
            module_id="weighted_cross_entropy",
            display_name="加权交叉熵损失",
            kind="loss",
            required_feature_type=FeatureType.LOGITS,
            output_feature_type=FeatureType.LOGITS,
            min_channels=1,
            max_channels=7,
            parameters_schema={
                "weight_source": _choice(
                    "类别权重来源。",
                    "train_distribution",
                    ("train_distribution", "manual"),
                ),
            },
            trainer_backend="pytorch",
            resource_class="low",
            implemented=True,
        ),
    )


def build_default_module_registry() -> ModuleRegistry:
    """构建项目默认且不可被 LLM 隐式扩展的 Module Registry。"""

    return ModuleRegistry(modules=_default_modules(), input_presets=_default_input_presets())
