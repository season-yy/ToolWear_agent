"""核验本机 AgentTeams 六 Agent 部署并生成脱敏证据。

运行方式：
    python scripts/verify_agentteams_deployment.py

脚本只执行 Docker/AgentTeams 只读查询，不创建、修改或删除容器。Matrix 事件 ID
来自仓库内经过人工核对的清单，Tool API 调用则必须在本机审计日志中真实存在。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从仓库根目录直接执行 ``python scripts/verify_agentteams_deployment.py``。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from toolwear_agent.agentteams.deployment_status import verify_and_record_deployment
from toolwear_agent.core.settings import load_settings


DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "docs"
    / "evidence"
    / "agentteams_e2e_final_manifest.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="核验 ToolWear AgentTeams 六 Agent 部署。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="不含密钥的 Matrix/Higress 验证清单。",
    )
    args = parser.parse_args()
    status = verify_and_record_deployment(load_settings(), args.manifest.resolve())
    assert status.team is not None
    print(f"verification_id={status.verification_id}")
    print(f"status={status.status}")
    print(f"team={status.team.runtime_name}")
    print(f"workers={len(status.workers)}")
    print(f"skill_invocations={status.toolwear_trace.skill_invocations}")
    print(f"report={status.evidence.report}")


if __name__ == "__main__":
    main()
