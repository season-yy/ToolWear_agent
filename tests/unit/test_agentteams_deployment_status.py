"""官方 AgentTeams 部署状态与证据归档测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.agentteams.deployment_status import (
    EXPECTED_WORKERS,
    load_deployment_status,
    verify_and_record_deployment,
)
from toolwear_agent.core.settings import Settings


class _CompletedProcess:
    """测试用的最小 subprocess 结果对象。"""

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class AgentTeamsDeploymentStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.settings = Settings(ai_infra_root=self.root / "ai_infra")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _manifest_file(self) -> Path:
        manifest = {
            "verification_id": "agt-e2e-test",
            "verified_at": "2026-08-15T18:00:00+08:00",
            "framework_version": "v1.2.2",
            "team_resource_name": "toolwear-phm2010-c1-team",
            "team_runtime_name": "ToolWear_agent",
            "human_event_id": "$human",
            "assignment_event_ids": {
                "DataStewardAgent": "$data",
                "AlgorithmArchitectAgent": "$algorithm",
                "CodeTrainingEngineerAgent": "$training",
                "EvaluationGovernorAgent": "$evaluation",
                "ReportMemoryCuratorAgent": "$report",
            },
            "leader_summary_event_id": "$leader",
            "correlation_id": "agt-e2e-test",
            "experiment_id": "exp-test",
            "trace_id": "trace-test",
            "higress_provider": "qwen-toolwear",
            "higress_route": "default-ai-route",
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def _write_audit_log(self) -> None:
        audit_file = self.settings.log_root / "agentteams" / "skill_invocations.jsonl"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        events = [
            ("DataStewardAgent", "toolwear-data-profile"),
            ("AlgorithmArchitectAgent", "toolwear-pipeline-recommend"),
            ("CodeTrainingEngineerAgent", "toolwear-mini-train"),
            ("EvaluationGovernorAgent", "toolwear-diagnosis"),
            ("ReportMemoryCuratorAgent", "toolwear-report-trace"),
        ]
        audit_file.write_text(
            "\n".join(
                json.dumps(
                    {
                        "event_id": f"audit-{index}",
                        "correlation_id": "agt-e2e-test",
                        "agent_name": agent_name,
                        "skill_name": skill_name,
                        "experiment_id": "exp-test",
                        "trace_id": "trace-test",
                        "status_code": 200,
                    }
                )
                for index, (agent_name, skill_name) in enumerate(events, start=1)
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _runner(command: list[str], **_: object) -> _CompletedProcess:
        if command[-2:] == ["-o", "json"]:
            return _CompletedProcess(
                json.dumps(
                    {
                        "name": "toolwear-phm2010-c1-team",
                        "teamName": "ToolWear_agent",
                        "phase": "Active",
                        "teamRoomID": "!room:matrix-local.agentteams.io:18080",
                        "leaderReady": True,
                        "readyWorkers": 5,
                        "totalWorkers": 5,
                    }
                )
            )
        if command[:2] == ["docker", "ps"]:
            return _CompletedProcess(
                "\n".join(
                    f"agentteams-worker-{name}\ttoolwear_agent/agentteams-copaw-worker:v1.2.2-teamfix\tUp 1 hour"
                    for name in EXPECTED_WORKERS
                )
            )
        worker_lines = ["NAME PHASE MODEL TEAM RUNTIME"]
        worker_lines.extend(
            f"{name} Running qwen3.7-flash-2026-07-15 toolwear-phm2010-c1-team copaw"
            for name in EXPECTED_WORKERS
        )
        return _CompletedProcess("\n".join(worker_lines))

    def test_missing_status_is_pending_and_does_not_guess(self) -> None:
        status = load_deployment_status(self.settings)

        self.assertEqual(status.status, "pending_verification")
        self.assertEqual(status.workers, ())

    def test_verified_deployment_writes_redacted_status_and_report(self) -> None:
        self._write_audit_log()

        status = verify_and_record_deployment(
            self.settings,
            self._manifest_file(),
            command_runner=self._runner,
        )

        self.assertEqual(status.status, "verified")
        self.assertEqual(status.team.phase, "Active")
        self.assertEqual(len(status.workers), 6)
        self.assertEqual(status.toolwear_trace.skill_invocations, 5)
        self.assertEqual(status.higress.provider, "qwen-toolwear")
        self.assertTrue((self.settings.ai_infra_root / "agentteams" / "status.json").is_file())
        report = Path(status.evidence.report).read_text(encoding="utf-8")
        self.assertIn("六 Agent", report)
        self.assertNotIn("api_key", report.lower())
        self.assertNotIn("bearer ", report.lower())

        restored = load_deployment_status(self.settings)
        self.assertEqual(restored.verification_id, "agt-e2e-test")


if __name__ == "__main__":
    unittest.main()
