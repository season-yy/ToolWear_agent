"""核验已真实执行的 C1 Golden Flow，并输出 JSON/Markdown 证据。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from toolwear_agent.core.settings import load_settings
from toolwear_agent.delivery.golden_flow import verify_golden_flow
from toolwear_agent.frontend.api_client import ToolWearApiClient


DEFAULT_EXPERIMENT_ID = "p0-diagnosis-smoke-20260815"


def _render_report(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# ToolWear 初赛 Golden Flow 验收",
            "",
            f"- 状态：`{payload['status']}`",
            f"- 实验：`{payload['experiment_id']}`",
            f"- ToolWear Trace：`{payload['trace_id']}`",
            f"- 真实 Run：`{payload['run_id']}`",
            f"- 候选方案：`{payload['pipeline_count']}`",
            f"- 真实 LLM Agent：`{payload['agent_count']}`",
            f"- 已复算 Evidence SHA-256：`{payload['verified_artifact_count']}`",
            f"- Validation Macro-F1：`{payload['validation_macro_f1']:.6f}`",
            f"- Validation Balanced Accuracy：`{payload['validation_balanced_accuracy']:.6f}`",
            f"- AgentTeams：`{payload['agentteams_status']}`",
            f"- Higress：`{payload['higress_status']}`",
            "",
            "验收器通过 FastAPI 读取实验事实，并直接复算已登记文件哈希；没有重新调用 LLM 或重复训练。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="核验 ToolWear C1 Golden Flow。")
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    args = parser.parse_args()
    settings = load_settings()
    client = ToolWearApiClient(
        settings.tool_api_base_url,
        token=settings.tool_api_token,
        timeout_seconds=30.0,
    )
    result = verify_golden_flow(
        client,
        experiment_id=args.experiment_id,
        allowed_artifact_root=settings.ai_infra_root,
    )
    payload = asdict(result)
    output_dir = settings.evidence_root / "golden_flow" / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    json_file = output_dir / "golden_flow_verification.json"
    report_file = output_dir / "golden_flow_verification.md"
    json_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_file.write_text(_render_report(payload), encoding="utf-8")
    print(f"status={result.status}")
    print(f"experiment_id={result.experiment_id}")
    print(f"run_id={result.run_id}")
    print(f"agents={result.agent_count}")
    print(f"artifacts={result.verified_artifact_count}")
    print(f"report={report_file}")


if __name__ == "__main__":
    main()
