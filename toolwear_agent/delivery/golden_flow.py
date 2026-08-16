"""通过 Tool API 和 EvidenceRef 验收一条已经真实执行的 Golden Flow。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


EXPECTED_AGENTS = {
    "ExperimentManagerAgent",
    "DataStewardAgent",
    "AlgorithmArchitectAgent",
    "CodeTrainingEngineerAgent",
    "EvaluationGovernorAgent",
    "ReportMemoryCuratorAgent",
}
ACCEPTED_FINAL_STATES = {
    "WAITING_PLAN_SELECTION",
    "WAITING_FULL_APPROVAL",
    "COMPLETED_MINI",
}
REQUIRED_ARTIFACT_KINDS = {"report", "metrics", "model", "config", "trace"}


class GoldenFlowClient(Protocol):
    """验收器只依赖前端已经使用的只读 API。"""

    def health(self) -> dict[str, Any]: ...
    def get_experiment(self, experiment_id: str) -> dict[str, Any]: ...
    def latest_recommendations(self, experiment_id: str) -> dict[str, Any]: ...
    def runs(self, experiment_id: str) -> list[dict[str, Any]]: ...
    def agent_runs(self, experiment_id: str) -> list[dict[str, Any]]: ...
    def artifacts(self, experiment_id: str) -> list[dict[str, Any]]: ...
    def events(self, experiment_id: str) -> list[dict[str, Any]]: ...


class GoldenFlowVerificationError(RuntimeError):
    """真实业务链路或证据不完整。"""


@dataclass(frozen=True)
class GoldenFlowVerification:
    status: str
    experiment_id: str
    trace_id: str
    run_id: str
    pipeline_count: int
    agent_count: int
    verified_artifact_count: int
    validation_macro_f1: float
    validation_balanced_accuracy: float
    agentteams_status: str
    higress_status: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoldenFlowVerificationError(message)


def _verify_artifacts(
    artifacts: list[dict[str, Any]],
    allowed_root: Path,
) -> int:
    """校验登记产物位于运行根目录内，且当前内容与 SHA-256 一致。"""

    root = allowed_root.resolve(strict=True)
    kinds = {str(item.get("kind", "")) for item in artifacts}
    _require(REQUIRED_ARTIFACT_KINDS <= kinds, "Golden Flow 缺少报告、指标、模型、配置或 Trace。")
    verified = 0
    for item in artifacts:
        path = Path(str(item.get("uri", ""))).resolve(strict=False)
        _require(path.is_relative_to(root), f"EvidenceRef 越过允许目录：{path}")
        _require(path.is_file(), f"EvidenceRef 文件不存在：{path}")
        expected_hash = str(item.get("sha256", ""))
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(bool(expected_hash) and actual_hash == expected_hash, f"EvidenceRef 哈希不匹配：{path}")
        verified += 1
    return verified


def verify_golden_flow(
    client: GoldenFlowClient,
    *,
    experiment_id: str,
    allowed_artifact_root: Path,
) -> GoldenFlowVerification:
    """把业务、Agent、框架和文件证据串成一条不可伪装的验收链。"""

    health = client.health()
    components = health.get("components", {})
    agentteams = components.get("agentteams", {})
    higress = components.get("higress", {})
    cuda = components.get("cuda", {})
    _require(agentteams.get("status") == "verified", "AgentTeams 尚未通过真实部署验证。")
    _require(agentteams.get("worker_count") == 6, "AgentTeams 不是固定六角色。")
    _require(higress.get("status") == "verified", "Higress 尚未通过验证。")
    _require(cuda.get("status") == "available", "当前 CUDA 健康状态不可用。")

    experiment = client.get_experiment(experiment_id)
    _require(experiment.get("state") in ACCEPTED_FINAL_STATES, "实验尚未完成一轮小样本闭环。")
    dataset_ref = experiment.get("dataset_ref", {})
    _require(dataset_ref.get("dataset_id") == "phm2010", "Golden Flow 数据集不是 PHM2010。")
    _require("C1" in dataset_ref.get("cutter_ids", []), "Golden Flow 未使用 C1。")

    recommendations = client.latest_recommendations(experiment_id)
    pipelines = recommendations.get("pipelines", [])
    _require(2 <= len(pipelines) <= 3, "候选方案必须为 2 至 3 个。")
    _require(
        recommendations.get("provider") == "qwen" and not recommendations.get("used_fallback"),
        "候选方案缺少真实千问成功记录。",
    )

    runs = client.runs(experiment_id)
    successful_runs = [item for item in runs if item.get("status") == "succeeded"]
    _require(bool(successful_runs), "没有成功的真实训练 Run。")
    best_run_id = str(experiment.get("best_run_id") or successful_runs[-1].get("run_id", ""))
    run = next((item for item in successful_runs if item.get("run_id") == best_run_id), successful_runs[-1])
    summary = run.get("result_summary") or {}
    macro_f1 = summary.get("validation_macro_f1")
    balanced_accuracy = summary.get("validation_balanced_accuracy")
    _require(isinstance(macro_f1, (float, int)), "Run 缺少 validation Macro-F1。")
    _require(isinstance(balanced_accuracy, (float, int)), "Run 缺少 validation Balanced Accuracy。")

    agent_runs = client.agent_runs(experiment_id)
    successful_agent_names = {
        str(item.get("result", {}).get("agent_name", ""))
        for item in agent_runs
        if item.get("result", {}).get("llm_call", {}).get("status") == "success"
    }
    _require(EXPECTED_AGENTS <= successful_agent_names, "六个业务 Agent 没有全部留下真实 LLM 成功记录。")

    events = client.events(experiment_id)
    transitions = {
        (str(item.get("before_state")), str(item.get("after_state"))) for item in events
    }
    _require(
        ("WAITING_PLAN_SELECTION", "PIPELINE_VALIDATING") in transitions,
        "缺少用户批准候选方案的状态证据。",
    )
    _require(("EVALUATING", "DECIDING") in transitions, "缺少评估进入人工决策的状态证据。")

    verified_artifacts = _verify_artifacts(client.artifacts(experiment_id), allowed_artifact_root)
    return GoldenFlowVerification(
        status="passed",
        experiment_id=experiment_id,
        trace_id=str(experiment.get("trace_id", "")),
        run_id=str(run.get("run_id", "")),
        pipeline_count=len(pipelines),
        agent_count=len(EXPECTED_AGENTS),
        verified_artifact_count=verified_artifacts,
        validation_macro_f1=float(macro_f1),
        validation_balanced_accuracy=float(balanced_accuracy),
        agentteams_status=str(agentteams.get("status")),
        higress_status=str(higress.get("status")),
    )


__all__ = [
    "GoldenFlowVerification",
    "GoldenFlowVerificationError",
    "verify_golden_flow",
]
