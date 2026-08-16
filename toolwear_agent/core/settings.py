"""从环境变量和 `.env` 加载 ToolWear 的强类型集中配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def repository_root() -> Path:
    """根据本文件位置计算代码仓库根目录。

    该结果与终端当前目录无关，所以 API、测试和 Streamlit 从不同位置启动时
    仍会得到同一个仓库根目录。
    """

    return Path(__file__).resolve().parents[2]


def _default_project_root() -> Path:
    """返回包含代码仓库、baseline 和项目文档的外层目录。"""

    return repository_root().parent


def _default_runtime_root() -> Path:
    """返回无 `.env` 时的安全本地运行目录。"""

    return repository_root() / ".runtime"


def candidate_env_files() -> tuple[Path, ...]:
    """按优先级返回允许自动读取的 `.env` 位置。"""

    app_root = repository_root()
    return app_root / ".env", app_root.parent / ".env"


def find_env_file() -> Path | None:
    """寻找仓库内或外层项目目录中的 `.env`。"""

    for env_file in candidate_env_files():
        if env_file.is_file():
            return env_file
    return None


class Settings(BaseSettings):
    """ToolWear 的唯一强类型配置对象。

    只有基础根目录允许直接配置，其他路径会从根目录派生。保留旧字段名是为了
    让当前 C1 PoC 在迁移期继续运行，后续业务代码应优先使用 `PathResolver`。
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        validate_default=True,
        frozen=True,
    )

    project_root: Path = Field(default_factory=_default_project_root)
    app_root: Path = Field(default_factory=repository_root)
    ai_infra_root: Path = Field(default_factory=_default_runtime_root)

    # 下列路径允许单独覆盖；未配置时统一从 AI_INFRA_ROOT 派生。
    dataset_manifest: Path | None = None
    phm2010_raw_root: Path | None = None
    experiment_root: Path | None = None
    artifact_root: Path | None = None
    log_root: Path | None = None
    state_root: Path | None = None
    state_db_path: Path | None = None
    report_root: Path | None = None
    evidence_root: Path | None = None
    generated_code_root: Path | None = None
    cache_root: Path | None = None

    llm_provider: str = "qwen"
    llm_api_key: str = Field(default="", repr=False, exclude=True)
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=90.0, gt=0, le=600)

    tool_api_token: str = Field(default="", repr=False, exclude=True)
    tool_api_token_file: Path | None = Field(default=None, repr=False, exclude=True)
    train_device: Literal["auto", "cpu", "cuda"] = "cuda"
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)
    primary_task: Literal["four_stage_classification"] = "four_stage_classification"
    enable_vb_regression: bool = False
    vb_aggregation: Literal["max", "mean", "specified_flute"] = "max"
    vb_stage_thresholds_um: Annotated[tuple[float, float, float], NoDecode] = (90.0, 130.0, 160.0)

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = Field(default=18100, ge=1, le=65535)
    streamlit_host: str = "127.0.0.1"
    streamlit_port: int = Field(default=18101, ge=1, le=65535)

    @field_validator("project_root", "app_root", "ai_infra_root", mode="after")
    @classmethod
    def _normalize_root(cls, value: Path) -> Path:
        """把相对根目录稳定地解释为相对仓库的位置。"""

        expanded = value.expanduser()
        if not expanded.is_absolute():
            expanded = repository_root() / expanded
        return expanded.resolve(strict=False)

    @field_validator("vb_stage_thresholds_um", mode="before")
    @classmethod
    def _parse_stage_thresholds(cls, value: object) -> tuple[float, float, float]:
        """解析三个四阶段边界，并拒绝重复或倒序阈值。"""

        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            raise ValueError("VB_STAGE_THRESHOLDS_UM 必须是三个逗号分隔的数值。")

        if len(parts) != 3:
            raise ValueError("四阶段分类必须配置三个 VB 边界。")
        thresholds = tuple(float(part) for part in parts)
        if not thresholds[0] < thresholds[1] < thresholds[2]:
            raise ValueError("三个 VB 边界必须严格递增。")
        return thresholds  # type: ignore[return-value]

    @model_validator(mode="after")
    def _derive_runtime_paths(self) -> "Settings":
        """从 AI 基础目录派生所有可写区和数据配置路径。"""

        root = self.ai_infra_root
        defaults = {
            "dataset_manifest": root / "datasets" / "manifests" / "phm2010.yaml",
            "phm2010_raw_root": root / "datasets" / "raw" / "phm2010",
            "experiment_root": root / "experiments" / "runs",
            "artifact_root": root / "artifacts",
            "log_root": root / "logs",
            "state_root": root / "state",
            "state_db_path": root / "state" / "toolwear.db",
            "report_root": root / "reports",
            "evidence_root": root / "evidence",
            "generated_code_root": root / "generated_code",
            "cache_root": root / "cache",
            "tool_api_token_file": root / "secrets" / "toolwear_api_token",
        }
        for field_name, default_path in defaults.items():
            configured = getattr(self, field_name)
            selected = configured if configured is not None else default_path
            selected = selected.expanduser()
            if not selected.is_absolute():
                selected = root / selected
            object.__setattr__(self, field_name, selected.resolve(strict=False))
        if not self.tool_api_token and self.tool_api_token_file.is_file():
            token = self.tool_api_token_file.read_text(encoding="utf-8").strip()
            if not token or "\n" in token or "\r" in token or len(token) > 4096:
                raise ValueError("TOOL_API_TOKEN_FILE 内容为空或格式不安全。")
            object.__setattr__(self, "tool_api_token", token)
        return self

    @property
    def repo_root(self) -> Path:
        """提供语义明确且不可从环境覆盖的仓库根目录。"""

        return repository_root()

    @property
    def tool_api_base_url(self) -> str:
        """返回供页面和 Skill 调用的 ToolWear API 地址。"""

        return f"http://{self.fastapi_host}:{self.fastapi_port}"


def load_settings(env_file: str | Path | None = None) -> Settings:
    """加载 Settings，系统环境变量优先于选中的 `.env` 文件。"""

    selected_env_file = Path(env_file) if env_file is not None else find_env_file()
    return Settings(_env_file=selected_env_file)
