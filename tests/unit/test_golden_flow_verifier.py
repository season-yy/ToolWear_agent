"""Golden Flow 现有真实证据验收测试。"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from toolwear_agent.delivery.golden_flow import GoldenFlowVerificationError, verify_golden_flow


SIX_AGENTS = (
    "ExperimentManagerAgent",
    "DataStewardAgent",
    "AlgorithmArchitectAgent",
    "CodeTrainingEngineerAgent",
    "EvaluationGovernorAgent",
    "ReportMemoryCuratorAgent",
)


class _FakeClient:
    def __init__(self, artifact_file: Path) -> None:
        digest = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
        self._artifacts = [
            {
                "evidence_id": f"evidence-{kind}",
                "kind": kind,
                "uri": str(artifact_file),
                "sha256": digest,
            }
            for kind in ("report", "metrics", "model", "config", "trace")
        ]

    def health(self) -> dict[str, object]:
        return {
            "components": {
                "agentteams": {"status": "verified", "worker_count": 6},
                "higress": {"status": "verified"},
                "cuda": {"status": "available"},
            }
        }

    def get_experiment(self, _: str) -> dict[str, object]:
        return {
            "state": "WAITING_PLAN_SELECTION",
            "trace_id": "trace-test",
            "best_run_id": "run-test",
            "dataset_ref": {"dataset_id": "phm2010", "cutter_ids": ["C1"]},
        }

    def latest_recommendations(self, _: str) -> dict[str, object]:
        return {
            "provider": "qwen",
            "used_fallback": False,
            "pipelines": [{"pipeline_id": "rf"}, {"pipeline_id": "cnn"}],
        }

    def runs(self, _: str) -> list[dict[str, object]]:
        return [
            {
                "run_id": "run-test",
                "status": "succeeded",
                "result_summary": {
                    "validation_macro_f1": 0.9,
                    "validation_balanced_accuracy": 0.88,
                    "resolved_device": "cuda",
                },
            }
        ]

    def agent_runs(self, _: str) -> list[dict[str, object]]:
        return [
            {
                "result": {
                    "agent_name": name,
                    "status": "success",
                    "llm_call": {"status": "success", "provider": "qwen", "total_tokens": 10},
                }
            }
            for name in SIX_AGENTS
        ]

    def artifacts(self, _: str) -> list[dict[str, object]]:
        return self._artifacts

    def events(self, _: str) -> list[dict[str, object]]:
        return [
            {"before_state": "WAITING_PLAN_SELECTION", "after_state": "PIPELINE_VALIDATING"},
            {"before_state": "EVALUATING", "after_state": "DECIDING"},
        ]


class GoldenFlowVerifierTests(unittest.TestCase):
    def test_accepts_complete_real_evidence_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = root / "report.md"
            report.write_text("# report\n", encoding="utf-8")

            result = verify_golden_flow(
                _FakeClient(report),
                experiment_id="exp-test",
                allowed_artifact_root=root,
            )

            self.assertEqual(result.status, "passed")
            self.assertEqual(result.agent_count, 6)
            self.assertEqual(result.pipeline_count, 2)
            self.assertEqual(result.verified_artifact_count, 5)

    def test_rejects_artifact_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = root / "report.md"
            report.write_text("# report\n", encoding="utf-8")
            client = _FakeClient(report)
            report.write_text("tampered\n", encoding="utf-8")

            with self.assertRaises(GoldenFlowVerificationError):
                verify_golden_flow(client, experiment_id="exp-test", allowed_artifact_root=root)


if __name__ == "__main__":
    unittest.main()
