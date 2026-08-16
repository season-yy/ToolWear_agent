"""validation 事实提取与 LLM 诊断边界测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import EpochLoss, MetricBundle
from toolwear_agent.schemas.diagnosis import EvaluationFacts
from toolwear_agent.services.evaluation_diagnosis import (
    DefaultDiagnosisProvider,
    build_evaluation_facts,
)
from toolwear_agent.services.llm_chat import ChatCompletion


def _metric_bundles() -> tuple[MetricBundle, MetricBundle]:
    train = MetricBundle(
        split="train",
        sample_count=100,
        macro_f1=0.96,
        balanced_accuracy=0.95,
        per_class={
            "初期磨损": {"f1-score": 0.95, "recall": 0.94, "support": 25},
            "正常磨损": {"f1-score": 0.97, "recall": 0.96, "support": 25},
            "剧烈磨损": {"f1-score": 0.96, "recall": 0.95, "support": 25},
            "失效磨损": {"f1-score": 0.96, "recall": 0.95, "support": 25},
        },
    )
    validation = MetricBundle(
        split="validation",
        sample_count=80,
        macro_f1=0.72,
        balanced_accuracy=0.70,
        per_class={
            "初期磨损": {"f1-score": 0.55, "recall": 0.50, "support": 10},
            "正常磨损": {"f1-score": 0.80, "recall": 0.82, "support": 40},
            "剧烈磨损": {"f1-score": 0.77, "recall": 0.75, "support": 20},
            "失效磨损": {"f1-score": 0.76, "recall": 0.73, "support": 10},
        },
        confusion_matrix=((5, 5, 0, 0), (2, 33, 5, 0), (0, 3, 15, 2), (0, 0, 3, 7)),
    )
    return train, validation


class _StubChatClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict[str, str]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        json_mode: bool,
    ) -> ChatCompletion:
        self.messages = messages
        return ChatCompletion(
            content=self.content,
            provider="qwen",
            model="qwen-test",
            latency_ms=25,
            prompt_tokens=120,
            completion_tokens=80,
            total_tokens=200,
        )


class EvaluationDiagnosisTest(unittest.TestCase):
    """验证事实层不接触 test，且外部 LLM 返回必须经过 Schema。"""

    def _facts(self) -> EvaluationFacts:
        train, validation = _metric_bundles()
        return build_evaluation_facts(
            experiment_id="exp-diagnosis",
            run_id="run-diagnosis",
            pipeline_id="multichannel_window_1d_cnn",
            train=train,
            validation=validation,
            class_labels=("初期磨损", "正常磨损", "剧烈磨损", "失效磨损"),
            epoch_history=(
                EpochLoss(epoch=1, train_loss=1.1, validation_loss=1.2, learning_rate=0.001),
                EpochLoss(epoch=2, train_loss=0.7, validation_loss=0.9, learning_rate=0.001),
            ),
            module_ids=("signal_windowing", "cnn_1d", "pytorch_trainer"),
            completed_mini_runs=1,
            max_mini_runs=3,
            source_evidence_ids=("metrics-evidence",),
        )

    def test_fact_builder_extracts_gap_weak_class_and_top_confusion(self) -> None:
        facts = self._facts()

        self.assertEqual(facts.basis_split, "validation")
        self.assertFalse(facts.final_test_used)
        self.assertAlmostEqual(facts.generalization_gap_macro_f1, 0.24)
        self.assertEqual(facts.weakest_class, "初期磨损")
        self.assertEqual(facts.top_confusions[0].actual_label, "初期磨损")
        self.assertEqual(facts.top_confusions[0].predicted_label, "正常磨损")
        self.assertEqual(facts.training_trend, "improving")

    def test_fact_schema_rejects_any_extra_test_metric(self) -> None:
        payload = self._facts().model_dump(mode="json")
        payload["test_macro_f1"] = 1.0

        with self.assertRaises(ValidationError):
            EvaluationFacts.model_validate(payload)

    def test_valid_llm_json_becomes_audited_diagnosis(self) -> None:
        response = {
            "overall_conclusion": "验证集结果尚可，但存在明显泛化差距。",
            "risk_level": "medium",
            "findings": [
                {
                    "finding_id": "generalization-gap",
                    "severity": "warning",
                    "category": "overfitting",
                    "title": "训练与验证差距偏大",
                    "detail": "Macro-F1 差距为 0.24。",
                    "evidence": "train=0.96, validation=0.72",
                }
            ],
            "recommendations": [
                {
                    "recommendation_id": "reduce-capacity",
                    "action_type": "adjust_parameter",
                    "target": "cnn_1d",
                    "suggestion": "提高 dropout 或减少通道宽度后建立新 revision。",
                    "rationale": "先缓解过拟合，再比较 validation。",
                    "priority": "high",
                    "requires_human_approval": True,
                }
            ],
            "recommended_action": "adjust_parameters",
        }
        client = _StubChatClient(json.dumps(response, ensure_ascii=False))
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = DefaultDiagnosisProvider(
                Settings(
                    ai_infra_root=Path(temp_dir),
                    llm_api_key="x",
                    llm_model="qwen-test",
                ),
                chat_client=client,
            )
            diagnosis = provider.diagnose(self._facts())

        self.assertFalse(diagnosis.llm_call.used_fallback)
        self.assertEqual(diagnosis.llm_call.total_tokens, 200)
        self.assertEqual(diagnosis.advice.recommended_action, "adjust_parameters")
        self.assertNotIn("test_macro_f1", client.messages[-1]["content"])

    def test_invalid_llm_output_uses_explicit_rule_fallback(self) -> None:
        client = _StubChatClient("not-json")
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = DefaultDiagnosisProvider(
                Settings(
                    ai_infra_root=Path(temp_dir),
                    llm_api_key="x",
                    llm_model="qwen-test",
                ),
                chat_client=client,
            )
            diagnosis = provider.diagnose(self._facts())

        self.assertTrue(diagnosis.llm_call.used_fallback)
        self.assertTrue(diagnosis.llm_call.fallback_reason)
        self.assertEqual(diagnosis.advice.recommended_action, "adjust_parameters")
        self.assertGreaterEqual(len(diagnosis.advice.findings), 2)

    def test_common_llm_enum_aliases_are_normalized_before_schema_validation(self) -> None:
        response = {
            "overall_conclusion": "当前结果可进入完整训练审批。",
            "risk_level": "low",
            "findings": [
                {
                    "finding_id": "泛化 差距",
                    "severity": "medium",
                    "category": "generalization gap",
                    "title": "存在轻微泛化差距",
                    "detail": "仍需关注 validation。",
                    "evidence": {"gap": 0.08},
                }
            ],
            "recommendations": [
                {
                    "recommendation_id": "完整训练",
                    "action_type": "approve_full_train",
                    "target": "current_pipeline",
                    "suggestion": "提交完整训练审批。",
                    "rationale": "validation 已达到条件。",
                    "priority": "medium",
                }
            ],
            "recommended_action": "approve_full_train",
        }
        client = _StubChatClient(json.dumps(response, ensure_ascii=False))
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnosis = DefaultDiagnosisProvider(
                Settings(
                    ai_infra_root=Path(temp_dir),
                    llm_api_key="x",
                    llm_model="qwen-test",
                ),
                chat_client=client,
            ).diagnose(self._facts())

        self.assertFalse(diagnosis.llm_call.used_fallback)
        self.assertEqual(diagnosis.advice.findings[0].severity, "warning")
        self.assertEqual(diagnosis.advice.recommended_action, "approve_full")
        self.assertTrue(
            diagnosis.advice.recommendations[0].requires_human_approval
        )


if __name__ == "__main__":
    unittest.main()
