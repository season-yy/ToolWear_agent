"""AgentTeams 真实部署状态的校验、脱敏归档与恢复。

本模块不在健康检查时调用 Docker。部署验证由显式命令执行一次，验证结果写入
``AI_INFRA_ROOT/agentteams/status.json``；API 只读取这份已校验快照，因此不会
因为刷新页面而阻塞，也不会把临时可用误报成长期健康。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from toolwear_agent.core.settings import Settings


EXPECTED_WORKERS = (
    "toolwear-experiment-manager",
    "toolwear-data-steward",
    "toolwear-algorithm-architect",
    "toolwear-code-training-engineer",
    "toolwear-evaluation-governor",
    "toolwear-report-memory-curator",
)

EXPECTED_SKILL_AGENTS = {
    "DataStewardAgent",
    "AlgorithmArchitectAgent",
    "CodeTrainingEngineerAgent",
    "EvaluationGovernorAgent",
    "ReportMemoryCuratorAgent",
}


class _StrictModel(BaseModel):
    """证据文件拒绝未知字段，防止格式悄悄漂移。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameworkStatus(_StrictModel):
    name: Literal["AgentTeams"] = "AgentTeams"
    version: str


class TeamStatus(_StrictModel):
    resource_name: str
    runtime_name: str
    phase: str
    room_id: str
    leader_ready: bool
    ready_workers: int = Field(ge=0)
    total_workers: int = Field(ge=0)


class WorkerStatus(_StrictModel):
    name: str
    phase: str
    model: str
    runtime: str
    image: str


class HigressStatus(_StrictModel):
    status: Literal["verified", "pending_verification"]
    provider: str = ""
    route: str = ""


class MatrixStatus(_StrictModel):
    status: Literal["verified", "pending_verification"]
    room_id: str = ""
    human_event_id: str = ""
    assignment_event_ids: dict[str, str] = Field(default_factory=dict)
    leader_summary_event_id: str = ""


class ToolWearTraceStatus(_StrictModel):
    correlation_id: str = ""
    experiment_id: str = ""
    trace_id: str = ""
    skill_invocations: int = Field(default=0, ge=0)
    agents: tuple[str, ...] = ()


class EvidenceStatus(_StrictModel):
    directory: str = ""
    manifest: str = ""
    report: str = ""


class AgentTeamsDeploymentStatus(_StrictModel):
    """供 API 和前端使用的脱敏部署快照。"""

    schema_version: Literal[1] = 1
    verification_id: str = ""
    verified_at: str = ""
    status: Literal["verified", "pending_verification"]
    framework: FrameworkStatus | None = None
    team: TeamStatus | None = None
    workers: tuple[WorkerStatus, ...] = ()
    higress: HigressStatus = HigressStatus(status="pending_verification")
    matrix: MatrixStatus = MatrixStatus(status="pending_verification")
    toolwear_trace: ToolWearTraceStatus = ToolWearTraceStatus()
    evidence: EvidenceStatus = EvidenceStatus()

    @model_validator(mode="after")
    def _verified_snapshot_must_be_complete(self) -> "AgentTeamsDeploymentStatus":
        """只有六个角色与全部协作边界齐全时才允许标记 verified。"""

        if self.status != "verified":
            return self
        if self.framework is None or self.team is None:
            raise ValueError("已验证状态必须包含 framework 和 team。")
        if self.team.phase != "Active" or not self.team.leader_ready:
            raise ValueError("Team 必须为 Active 且 Leader Ready。")
        if self.team.ready_workers != 5 or self.team.total_workers != 5:
            raise ValueError("Team 必须有 5/5 个普通 Worker Ready。")
        if {item.name for item in self.workers} != set(EXPECTED_WORKERS):
            raise ValueError("部署状态必须精确包含固定六个 Worker。")
        if any(item.phase != "Running" for item in self.workers):
            raise ValueError("六个 Worker 必须全部处于 Running。")
        if self.higress.status != "verified" or self.matrix.status != "verified":
            raise ValueError("Higress 和 Matrix 必须同时完成验证。")
        if not EXPECTED_SKILL_AGENTS <= set(self.toolwear_trace.agents):
            raise ValueError("五个专业 Worker 必须都留下 Tool API 审计。")
        return self


class VerificationManifest(_StrictModel):
    """人工确认过的 Matrix/Higress 事实，不包含密码或 Token。"""

    verification_id: str
    verified_at: str
    framework_version: str
    team_resource_name: str
    team_runtime_name: str
    human_event_id: str
    assignment_event_ids: dict[str, str]
    leader_summary_event_id: str
    correlation_id: str
    experiment_id: str
    trace_id: str
    higress_provider: str
    higress_route: str


CommandRunner = Callable[..., object]


def _pending_status() -> AgentTeamsDeploymentStatus:
    """返回保守默认值；没有证据时绝不猜测已接入。"""

    return AgentTeamsDeploymentStatus(status="pending_verification")


def deployment_status_file(settings: Settings) -> Path:
    """返回稳定的部署状态快照位置。"""

    return settings.ai_infra_root / "agentteams" / "status.json"


def load_deployment_status(settings: Settings) -> AgentTeamsDeploymentStatus:
    """读取已验证快照；缺失、损坏或过期格式都降级为待验证。"""

    status_file = deployment_status_file(settings)
    if not status_file.is_file():
        return _pending_status()
    try:
        return AgentTeamsDeploymentStatus.model_validate_json(
            status_file.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError, json.JSONDecodeError):
        return _pending_status()


def _run_checked(command_runner: CommandRunner, command: list[str]) -> str:
    """运行只读检查命令，并把失败变成明确的验证错误。"""

    result = command_runner(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    returncode = int(getattr(result, "returncode", 1))
    if returncode != 0:
        stderr = str(getattr(result, "stderr", "")).strip()
        raise RuntimeError(f"只读检查命令失败：{' '.join(command)}；{stderr or '无错误输出'}")
    return str(getattr(result, "stdout", ""))


def _parse_workers(table: str, docker_table: str) -> tuple[WorkerStatus, ...]:
    """解析官方 agt Worker 表和 Docker 表，并按固定角色顺序返回。"""

    worker_rows: dict[str, tuple[str, str, str]] = {}
    for raw_line in table.splitlines()[1:]:
        columns = raw_line.split()
        if len(columns) < 5:
            continue
        name, phase, model, _, runtime = columns[:5]
        worker_rows[name] = (phase, model, runtime)

    images: dict[str, str] = {}
    for raw_line in docker_table.splitlines():
        columns = raw_line.split("\t")
        if len(columns) < 3:
            continue
        container_name, image, container_status = columns[:3]
        prefix = "agentteams-worker-"
        if container_name.startswith(prefix) and container_status.lower().startswith("up"):
            images[container_name.removeprefix(prefix)] = image

    missing = set(EXPECTED_WORKERS) - set(worker_rows)
    if missing:
        raise ValueError(f"AgentTeams 缺少固定 Worker：{sorted(missing)}")
    if set(EXPECTED_WORKERS) - set(images):
        raise ValueError("存在未运行或无法确认镜像的 AgentTeams Worker 容器。")

    return tuple(
        WorkerStatus(
            name=name,
            phase=worker_rows[name][0],
            model=worker_rows[name][1],
            runtime=worker_rows[name][2],
            image=images[name],
        )
        for name in EXPECTED_WORKERS
    )


def _load_skill_audit(settings: Settings, correlation_id: str) -> list[dict[str, object]]:
    """只提取本次协作对应的脱敏 Tool API 审计行。"""

    audit_file = settings.log_root / "agentteams" / "skill_invocations.jsonl"
    if not audit_file.is_file():
        raise ValueError("未找到 AgentTeams Skill 调用审计日志。")
    events: list[dict[str, object]] = []
    for raw_line in audit_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        event = json.loads(raw_line)
        if event.get("correlation_id") == correlation_id:
            events.append(event)
    successful_agents = {
        str(event.get("agent_name", ""))
        for event in events
        if int(event.get("status_code", 0)) == 200
    }
    if not EXPECTED_SKILL_AGENTS <= successful_agents:
        missing = EXPECTED_SKILL_AGENTS - successful_agents
        raise ValueError(f"缺少成功的专业 Worker Skill 审计：{sorted(missing)}")
    return events


def _render_report(status: AgentTeamsDeploymentStatus) -> str:
    """生成适合初赛审阅的简明 E2E 证据报告。"""

    assert status.team is not None
    lines = [
        "# ToolWear AgentTeams 六 Agent 端到端验证",
        "",
        f"- 验证编号：`{status.verification_id}`",
        f"- 验证时间：`{status.verified_at}`",
        f"- AgentTeams：`{status.framework.version if status.framework else ''}`",
        f"- Team：`{status.team.resource_name}` / `{status.team.runtime_name}`",
        f"- Matrix Room：`{status.matrix.room_id}`",
        f"- Higress：`{status.higress.provider}` / `{status.higress.route}`",
        f"- ToolWear Trace：`{status.toolwear_trace.trace_id}`",
        f"- Skill 审计次数：`{status.toolwear_trace.skill_invocations}`",
        "",
        "## 六 Agent 运行状态",
        "",
        "| Worker | Phase | Runtime | Model | Image |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| `{worker.name}` | `{worker.phase}` | `{worker.runtime}` | `{worker.model}` | `{worker.image}` |"
        for worker in status.workers
    )
    lines.extend(
        [
            "",
            "## 协作证据",
            "",
            f"- 人类发起事件：`{status.matrix.human_event_id}`",
            f"- Leader 汇总事件：`{status.matrix.leader_summary_event_id}`",
        ]
    )
    lines.extend(
        f"- {agent_name} 分派事件：`{event_id}`"
        for agent_name, event_id in status.matrix.assignment_event_ids.items()
    )
    lines.extend(
        [
            "",
            "本报告只保存资源名、状态、模型名、事件 ID 与 ToolWear Trace 关联，不保存 API 密钥、密码或鉴权令牌。",
            "",
        ]
    )
    return "\n".join(lines)


def verify_and_record_deployment(
    settings: Settings,
    manifest_file: Path,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> AgentTeamsDeploymentStatus:
    """核对真实 Team、Worker 和 Skill 审计，并生成脱敏状态与报告。"""

    manifest = VerificationManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
    team_raw = _run_checked(
        command_runner,
        [
            "docker",
            "exec",
            "agentteams-manager",
            "agt",
            "get",
            "teams",
            manifest.team_resource_name,
            "-o",
            "json",
        ],
    )
    team_data = json.loads(team_raw)
    worker_table = _run_checked(
        command_runner,
        [
            "docker",
            "exec",
            "agentteams-manager",
            "agt",
            "get",
            "workers",
            "--team",
            manifest.team_resource_name,
        ],
    )
    docker_table = _run_checked(
        command_runner,
        ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
    )
    workers = _parse_workers(worker_table, docker_table)
    audit_events = _load_skill_audit(settings, manifest.correlation_id)
    audit_agents = tuple(sorted({str(item["agent_name"]) for item in audit_events}))

    team = TeamStatus(
        resource_name=str(team_data.get("name", "")),
        runtime_name=str(team_data.get("teamName", "")),
        phase=str(team_data.get("phase", "")),
        room_id=str(team_data.get("teamRoomID", "")),
        leader_ready=bool(team_data.get("leaderReady", False)),
        ready_workers=int(team_data.get("readyWorkers", 0)),
        total_workers=int(team_data.get("totalWorkers", 0)),
    )
    if team.resource_name != manifest.team_resource_name or team.runtime_name != manifest.team_runtime_name:
        raise ValueError("Team 实际名称与验证清单不一致。")
    if set(manifest.assignment_event_ids) != EXPECTED_SKILL_AGENTS:
        raise ValueError("Matrix 分派事件必须精确覆盖五个专业 Worker。")

    evidence_dir = settings.ai_infra_root / "agentteams" / "evidence" / manifest.verification_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_manifest = evidence_dir / "verification_manifest.json"
    report_file = evidence_dir / "agentteams_e2e_report.md"
    status = AgentTeamsDeploymentStatus(
        verification_id=manifest.verification_id,
        verified_at=manifest.verified_at,
        status="verified",
        framework=FrameworkStatus(version=manifest.framework_version),
        team=team,
        workers=workers,
        higress=HigressStatus(
            status="verified",
            provider=manifest.higress_provider,
            route=manifest.higress_route,
        ),
        matrix=MatrixStatus(
            status="verified",
            room_id=team.room_id,
            human_event_id=manifest.human_event_id,
            assignment_event_ids=manifest.assignment_event_ids,
            leader_summary_event_id=manifest.leader_summary_event_id,
        ),
        toolwear_trace=ToolWearTraceStatus(
            correlation_id=manifest.correlation_id,
            experiment_id=manifest.experiment_id,
            trace_id=manifest.trace_id,
            skill_invocations=len(audit_events),
            agents=audit_agents,
        ),
        evidence=EvidenceStatus(
            directory=str(evidence_dir),
            manifest=str(evidence_manifest),
            report=str(report_file),
        ),
    )
    evidence_manifest.write_text(
        json.dumps(status.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_file.write_text(_render_report(status), encoding="utf-8")
    status_file = deployment_status_file(settings)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(
        json.dumps(status.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return status


__all__ = [
    "AgentTeamsDeploymentStatus",
    "EXPECTED_WORKERS",
    "deployment_status_file",
    "load_deployment_status",
    "verify_and_record_deployment",
]
