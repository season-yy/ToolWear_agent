"""Agent 调用证据的原子写入和哈希计算。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from toolwear_agent.core.paths import PathResolver
from toolwear_agent.schemas import AgentLlmCallAudit, AgentResult, AgentTask, EvidenceRef
from toolwear_agent.schemas.agent import AgentResultStatus
from toolwear_agent.state import EntityNotFoundError, SQLiteExperimentRepository


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_agent_call_evidence(
    path_resolver: PathResolver,
    *,
    task: AgentTask,
    status: str,
    output_payload: dict[str, object],
    next_actions: tuple[str, ...],
    audit: AgentLlmCallAudit,
    response_sha256: str,
    error_code: str | None,
    error_message: str,
) -> EvidenceRef:
    """写入不含密钥、完整 Prompt 和原始模型正文的调用证据。"""

    output_dir = path_resolver.agent_trace_path(
        task.experiment_id,
        task.revision,
        task.task_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "agent_call.json"
    payload = {
        "schema_version": "1.0",
        "task": task.model_dump(mode="json"),
        "status": status,
        "output_payload": output_payload,
        "next_actions": next_actions,
        "llm_call": audit.model_dump(mode="json"),
        "response_sha256": response_sha256,
        "error_code": error_code,
        "error_message": error_message,
    }
    temporary = output_file.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_file)
    return EvidenceRef(
        evidence_id=f"{task.task_id}-call",
        experiment_id=task.experiment_id,
        kind="trace",
        uri=str(output_file),
        sha256=_sha256_file(output_file),
        size_bytes=output_file.stat().st_size,
        media_type="application/json",
        description=f"{task.assigned_to} 结构化 LLM 调用证据",
        created_by=task.assigned_to,
    )


def persist_agent_result(
    path_resolver: PathResolver,
    repository: SQLiteExperimentRepository,
    task: AgentTask,
    *,
    status: AgentResultStatus,
    summary: str,
    output_payload: dict[str, object],
    next_actions: tuple[str, ...],
    audit: AgentLlmCallAudit,
    response_content: str,
    error_code: str | None,
    error_message: str,
    idempotency_key: str | None,
) -> AgentResult:
    """写入 Trace、登记 EvidenceRef，并在同一入口保存 AgentResult。"""

    evidence = write_agent_call_evidence(
        path_resolver,
        task=task,
        status=status.value,
        output_payload=output_payload,
        next_actions=next_actions,
        audit=audit,
        response_sha256=hashlib.sha256(response_content.encode("utf-8")).hexdigest(),
        error_code=error_code,
        error_message=error_message,
    )
    try:
        evidence = repository.get_evidence(evidence.evidence_id)
    except EntityNotFoundError:
        evidence = repository.register_evidence(
            evidence,
            idempotency_key=(
                f"{idempotency_key}:evidence" if idempotency_key else None
            ),
        )
    result = AgentResult(
        task_id=task.task_id,
        agent_name=task.assigned_to,
        trace_id=task.trace_id,
        status=status,
        summary=summary,
        output_schema=task.output_schema,
        output_payload=output_payload,
        evidence=(evidence,),
        next_actions=next_actions,
        llm_call=audit,
        error_code=error_code,
        error_message=error_message or None,
    )
    return repository.save_agent_result(
        result,
        idempotency_key=(f"{idempotency_key}:result" if idempotency_key else None),
    )


__all__ = ["persist_agent_result", "write_agent_call_evidence"]
