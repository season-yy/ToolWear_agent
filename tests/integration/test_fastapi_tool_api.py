"""FastAPI Tool API 的真实 SQLite 集成测试。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from toolwear_agent.agentteams.deployment_status import EXPECTED_WORKERS
from toolwear_agent.backend.main import create_app
from toolwear_agent.core.settings import Settings
from toolwear_agent.data.registry import DatasetRegistry
from toolwear_agent.schemas import CutterManifest, DatasetManifest, EvidenceRef


REQUIRED_PATHS = {
    "/api/v1/health",
    "/api/v1/agents",
    "/api/v1/capabilities",
    "/api/v1/datasets",
    "/api/v1/artifacts/{evidence_id}/content",
    "/api/v1/experiments",
    "/api/v1/experiments/{experiment_id}",
    "/api/v1/experiments/{experiment_id}/events",
    "/api/v1/experiments/{experiment_id}/recommendations",
    "/api/v1/experiments/{experiment_id}/revisions/{revision}",
    "/api/v1/experiments/{experiment_id}/runs",
    "/api/v1/experiments/{experiment_id}/profile",
    "/api/v1/experiments/{experiment_id}/labels",
    "/api/v1/experiments/{experiment_id}/split",
    "/api/v1/experiments/{experiment_id}/recommendations",
    "/api/v1/experiments/{experiment_id}/approve-pipeline",
    "/api/v1/experiments/{experiment_id}/validate",
    "/api/v1/experiments/{experiment_id}/runs/mini",
    "/api/v1/experiments/{experiment_id}/runs/{run_id}",
    "/api/v1/experiments/{experiment_id}/runs/{run_id}/logs",
    "/api/v1/experiments/{experiment_id}/evaluate",
    "/api/v1/experiments/{experiment_id}/decision",
    "/api/v1/experiments/{experiment_id}/report",
    "/api/v1/experiments/{experiment_id}/cancel",
    "/api/v1/experiments/{experiment_id}/artifacts",
    "/api/v1/experiments/{experiment_id}/agent-runs",
    "/api/v1/experiments/{experiment_id}/agents/{agent_name}/invoke",
}


class FastApiToolApiTest(unittest.TestCase):
    """验证 API 契约、幂等和进程重启恢复。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        runtime_root = Path(self.temp_dir.name) / "runtime"
        raw_cutter = runtime_root / "datasets" / "raw" / "phm2010" / "c1"
        raw_cutter.mkdir(parents=True)
        self.settings = Settings(
            ai_infra_root=runtime_root,
            train_device="cpu",
            llm_api_key="",
            llm_model="",
        )
        DatasetRegistry(self.settings.dataset_manifest).register(
            DatasetManifest(
                dataset_id="phm2010",
                display_name="PHM 2010 Test",
                adapter="phm2010",
                root=self.settings.phm2010_raw_root,
                channels=("force_x", "vibration_x"),
                cutters={
                    "C1": CutterManifest(
                        cutter_id="C1",
                        relative_path="c1",
                        labeled=True,
                        wear_file="c1_wear.csv",
                        resolved_path=raw_cutter,
                        signal_file_count=315,
                        wear_row_count=315,
                        detected_channel_count=7,
                    )
                },
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _request_body(self) -> dict[str, object]:
        return {
            "experiment_id": "exp-api-test",
            "title": "C1 四阶段 API 实验",
            "user_request": "使用 C1 数据比较随机森林和 1D-CNN。",
            "dataset_id": "phm2010",
            "cutter_ids": ["C1"],
            "input_channels": ["force_x", "vibration_x"],
            "window_length": 4096,
            "overlap": 0.5,
            "sample_fraction": 0.2,
            "mode": "quick",
        }

    def test_openapi_contains_complete_initial_round_contract(self) -> None:
        """初赛清单中的业务端点不能只存在于计划文档。"""

        with TestClient(create_app(settings=self.settings)) as client:
            paths = set(client.get("/openapi.json").json()["paths"])

        self.assertTrue(REQUIRED_PATHS <= paths)

    def test_health_capabilities_and_datasets_use_real_registries(self) -> None:
        """系统只读接口必须反映真实 SQLite、Dataset 和 Module Registry。"""

        with TestClient(create_app(settings=self.settings)) as client:
            health = client.get("/api/v1/health")
            capabilities = client.get("/api/v1/capabilities")
            datasets = client.get("/api/v1/datasets")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["database"]["user_version"], 1)
        self.assertEqual(
            health.json()["components"]["agentteams"]["status"],
            "pending_verification",
        )
        self.assertIn("cnn_1d", {item["module_id"] for item in capabilities.json()["modules"]})
        self.assertEqual(datasets.json()[0]["dataset_id"], "phm2010")

    def test_health_reports_verified_agentteams_evidence(self) -> None:
        """只有存在经过校验的状态文件，页面才能宣称 AgentTeams 已接入。"""

        status_file = self.settings.ai_infra_root / "agentteams" / "status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verification_id": "agt-health-test",
                    "verified_at": "2026-08-15T18:00:00+08:00",
                    "status": "verified",
                    "framework": {"name": "AgentTeams", "version": "v1.2.2"},
                    "team": {
                        "resource_name": "toolwear-phm2010-c1-team",
                        "runtime_name": "ToolWear_agent",
                        "phase": "Active",
                        "room_id": "!room:matrix",
                        "leader_ready": True,
                        "ready_workers": 5,
                        "total_workers": 5,
                    },
                    "workers": [
                        {
                            "name": worker_name,
                            "phase": "Running",
                            "model": "qwen-test",
                            "runtime": "copaw",
                            "image": "worker:test",
                        }
                        for worker_name in EXPECTED_WORKERS
                    ],
                    "higress": {
                        "status": "verified",
                        "provider": "qwen-toolwear",
                        "route": "default-ai-route",
                    },
                    "matrix": {
                        "status": "verified",
                        "room_id": "!room:matrix",
                        "human_event_id": "$human",
                        "assignment_event_ids": {
                            "DataStewardAgent": "$data",
                            "AlgorithmArchitectAgent": "$algorithm",
                            "CodeTrainingEngineerAgent": "$training",
                            "EvaluationGovernorAgent": "$evaluation",
                            "ReportMemoryCuratorAgent": "$report",
                        },
                        "leader_summary_event_id": "$leader",
                    },
                    "toolwear_trace": {
                        "correlation_id": "agt-health-test",
                        "experiment_id": "exp-test",
                        "trace_id": "trace-test",
                        "skill_invocations": 5,
                        "agents": [
                            "DataStewardAgent",
                            "AlgorithmArchitectAgent",
                            "CodeTrainingEngineerAgent",
                            "EvaluationGovernorAgent",
                            "ReportMemoryCuratorAgent",
                        ],
                    },
                    "evidence": {
                        "directory": "D:/evidence",
                        "manifest": "D:/evidence/manifest.json",
                        "report": "D:/evidence/report.md",
                    },
                }
            ),
            encoding="utf-8",
        )

        with TestClient(create_app(settings=self.settings)) as client:
            health = client.get("/api/v1/health").json()

        self.assertEqual(health["components"]["agentteams"]["status"], "verified")
        self.assertEqual(health["components"]["agentteams"]["worker_count"], 6)
        self.assertEqual(health["components"]["higress"]["provider"], "qwen-toolwear")

    def test_create_experiment_is_idempotent_and_survives_app_restart(self) -> None:
        """重复点击创建不会产生两个实验，API 重启后仍可按 ID 找回。"""

        headers = {"Idempotency-Key": "create-exp-api-test"}
        with TestClient(create_app(settings=self.settings)) as client:
            first = client.post("/api/v1/experiments", json=self._request_body(), headers=headers)
            replayed = client.post("/api/v1/experiments", json=self._request_body(), headers=headers)
            listed = client.get("/api/v1/experiments")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(replayed.status_code, 201)
        self.assertEqual(replayed.json(), first.json())
        self.assertEqual(len(listed.json()), 1)
        self.assertEqual(first.json()["state"], "DRAFT")
        self.assertEqual(first.json()["preferences"]["mode"], "quick")

        with TestClient(create_app(settings=self.settings)) as restarted_client:
            restored = restarted_client.get("/api/v1/experiments/exp-api-test")
            events = restarted_client.get("/api/v1/experiments/exp-api-test/events")

        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.json()["trace_id"], first.json()["trace_id"])
        self.assertEqual(len(events.json()), 1)

    def test_invalid_workflow_action_returns_stable_error_code(self) -> None:
        """绕过数据准备直接生成候选时，页面能收到机器可读错误。"""

        with TestClient(create_app(settings=self.settings)) as client:
            client.post(
                "/api/v1/experiments",
                json=self._request_body(),
                headers={"Idempotency-Key": "create-before-invalid-action"},
            )
            response = client.post(
                "/api/v1/experiments/exp-api-test/recommendations",
                json={"user_request": "直接生成候选。", "force_refresh": False},
                headers={"Idempotency-Key": "invalid-recommendation-state"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["error_code"], "INVALID_WORKFLOW_STATE")

    def test_registered_json_evidence_can_be_read_without_accepting_a_path(self) -> None:
        """前端只能使用 Evidence ID 读取小型证据，不能提交任意本地路径。"""

        payload = {"metric": "macro_f1", "value": 0.91}
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        evidence_file = self.settings.evidence_root / "api-content-test.json"
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_bytes(content)
        evidence = EvidenceRef(
            evidence_id="api-content-test",
            experiment_id="exp-api-test",
            kind="metrics",
            uri=str(evidence_file),
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            media_type="application/json",
            description="API 内容测试",
        )

        with TestClient(create_app(settings=self.settings)) as client:
            created = client.post(
                "/api/v1/experiments",
                json=self._request_body(),
                headers={"Idempotency-Key": "create-for-evidence-content"},
            )
            self.assertEqual(created.status_code, 201)
            client.app.state.container.repository.register_evidence(evidence)
            response = client.get("/api/v1/artifacts/api-content-test/content")
            missing = client.get("/api/v1/artifacts/not-registered/content")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
