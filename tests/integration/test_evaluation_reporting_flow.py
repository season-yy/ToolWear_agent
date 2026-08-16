"""评估、结构化诊断、决策和 Markdown 报告的真实持久化测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.core.settings import Settings
from toolwear_agent.schemas import (
    DatasetRef,
    EpochLoss,
    EvaluationDiagnosis,
    EvaluationReport,
    ExperimentRevision,
    ExperimentState,
    LabelPolicy,
    LlmCallAudit,
    MetricBundle,
    ModuleSpec,
    PipelineSpec,
    RunConfig,
    SplitSpec,
    TrainingArtifacts,
    TrainingRunResult,
    TrainingRuntimeInfo,
)
from toolwear_agent.schemas.api import DecisionRequest
from toolwear_agent.schemas.base import utc_now
from toolwear_agent.services.evaluation_diagnosis import build_rule_based_advice
from toolwear_agent.services.evaluation_reporting import EvaluationReportingService
from toolwear_agent.state import RunRecord, SQLiteExperimentRepository


class _FixedDiagnosisProvider:
    """模拟一次已经通过 Schema 的真实 LLM 返回。"""

    def __init__(self) -> None:
        self.calls = 0

    def diagnose(self, facts):
        self.calls += 1
        return EvaluationDiagnosis(
            diagnosis_id="diagnosis-integration-test",
            facts=facts,
            advice=build_rule_based_advice(facts),
            llm_call=LlmCallAudit(
                provider="qwen",
                model="qwen-test",
                status="success",
                used_fallback=False,
                latency_ms=30,
                prompt_template_version="evaluation-governor-v1",
                prompt_sha256="a" * 64,
                total_tokens=210,
            ),
        )


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_id="statistical_features_random_forest",
        display_name="统计特征 + RandomForest",
        source="fixed",
        input_channels=("force_x", "vibration_x"),
        modules=(
            ModuleSpec(module_id="signal_windowing", kind="windowing", order=10),
            ModuleSpec(module_id="statistical_features", kind="feature", order=20),
            ModuleSpec(module_id="random_forest_classifier", kind="model", order=30),
            ModuleSpec(module_id="sklearn_trainer", kind="trainer", order=40),
        ),
        rationale="评估集成测试。",
        risks=("仅验证评估闭环。",),
        expected_cost="low",
    )


class EvaluationReportingFlowTest(unittest.TestCase):
    """验证同一实验 ID 下的文件、状态和 Evidence 能完整恢复。"""

    def test_evaluate_decide_and_report_share_one_structured_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(ai_infra_root=root, llm_api_key="", llm_model="")
            repository = SQLiteExperimentRepository(settings.state_db_path)
            repository.initialize()
            try:
                state = ExperimentState(
                    experiment_id="exp-evaluation-flow",
                    trace_id="trace-evaluation-flow",
                    title="评估闭环测试",
                    objective="只依据 validation 诊断四阶段分类。",
                    dataset_ref=DatasetRef(dataset_id="phm2010", cutter_ids=("C1",)),
                    label_policy=LabelPolicy(),
                    split_spec=SplitSpec(),
                )
                repository.create_experiment(state, actor="human", reason="创建测试实验。")
                pipeline = _pipeline()
                revision = ExperimentRevision(
                    experiment_id=state.experiment_id,
                    revision=1,
                    pipeline=pipeline,
                    run_config=RunConfig(
                        run_id="run-evaluation-flow",
                        experiment_id=state.experiment_id,
                        revision=1,
                        pipeline_id=pipeline.pipeline_id,
                    ),
                    created_by="human",
                    change_reason="锁定评估测试配置。",
                )
                repository.create_revision(revision)
                for target in (
                    "DATA_VALIDATING",
                    "WAITING_PLAN_SELECTION",
                    "PIPELINE_VALIDATING",
                    "CODE_PREPARING",
                    "MINI_TRAINING",
                ):
                    repository.transition_state(
                        state.experiment_id,
                        target,
                        actor="system",
                        reason=f"进入 {target}。",
                    )

                run_dir = root / "experiments" / "runs" / state.experiment_id / "run"
                run_dir.mkdir(parents=True)
                result_file = run_dir / "result.json"
                train = MetricBundle(
                    split="train",
                    sample_count=100,
                    macro_f1=0.96,
                    balanced_accuracy=0.95,
                    per_class={label: {"f1-score": 0.95, "recall": 0.95, "support": 25} for label in ("初期磨损", "正常磨损", "剧烈磨损", "失效磨损")},
                )
                validation = MetricBundle(
                    split="validation",
                    sample_count=80,
                    macro_f1=0.72,
                    balanced_accuracy=0.70,
                    per_class={
                        "初期磨损": {"precision": 0.6, "recall": 0.5, "f1-score": 0.55, "support": 10},
                        "正常磨损": {"precision": 0.8, "recall": 0.82, "f1-score": 0.80, "support": 40},
                        "剧烈磨损": {"precision": 0.78, "recall": 0.75, "f1-score": 0.77, "support": 20},
                        "失效磨损": {"precision": 0.78, "recall": 0.73, "f1-score": 0.76, "support": 10},
                    },
                    confusion_matrix=((5, 5, 0, 0), (2, 33, 5, 0), (0, 3, 15, 2), (0, 0, 3, 7)),
                )
                result = TrainingRunResult(
                    run_id="run-evaluation-flow",
                    experiment_id=state.experiment_id,
                    revision=1,
                    pipeline_id=pipeline.pipeline_id,
                    split_hash="b" * 64,
                    sample_hash="c" * 64,
                    train_sample_count=100,
                    validation_sample_count=80,
                    class_labels=("初期磨损", "正常磨损", "剧烈磨损", "失效磨损"),
                    runtime=TrainingRuntimeInfo(
                        backend="sklearn",
                        requested_device="auto",
                        resolved_device="cpu",
                        elapsed_seconds=1.0,
                    ),
                    evaluation=EvaluationReport(
                        evaluation_id="evaluation-flow",
                        experiment_id=state.experiment_id,
                        run_id="run-evaluation-flow",
                        pipeline_id=pipeline.pipeline_id,
                        metrics=(train, validation),
                        class_labels=("初期磨损", "正常磨损", "剧烈磨损", "失效磨损"),
                    ),
                    epoch_history=(
                        EpochLoss(epoch=1, train_loss=1.0, validation_loss=1.1, learning_rate=0.001),
                        EpochLoss(epoch=2, train_loss=0.7, validation_loss=0.8, learning_rate=0.001),
                    ),
                    artifacts=TrainingArtifacts(
                        run_dir=run_dir,
                        model_file=run_dir / "model.joblib",
                        metrics_file=run_dir / "metrics.json",
                        config_file=run_dir / "run_config.json",
                        pipeline_file=run_dir / "pipeline.json",
                        data_ref_file=run_dir / "data_ref.json",
                        log_file=run_dir / "run.jsonl",
                        code_snapshot_dir=run_dir / "code_snapshot",
                        evidence_index_file=run_dir / "evidence_index.json",
                        result_file=result_file,
                    ),
                )
                result_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                repository.create_run(
                    RunRecord(
                        run_id=result.run_id,
                        experiment_id=state.experiment_id,
                        revision=1,
                        pipeline_id=pipeline.pipeline_id,
                        run_kind="mini_train",
                        status="succeeded",
                        progress=1.0,
                        progress_message="训练完成。",
                        total_epochs=2,
                        current_epoch=2,
                        result_summary={
                            "result_file": str(result_file),
                            "validation_macro_f1": validation.macro_f1,
                            "validation_balanced_accuracy": validation.balanced_accuracy,
                        },
                        completed_at=utc_now(),
                    )
                )
                repository.transition_state(
                    state.experiment_id,
                    "EVALUATING",
                    actor="CodeTrainingEngineerAgent",
                    reason="训练完成。",
                )
                diagnosis_provider = _FixedDiagnosisProvider()
                service = EvaluationReportingService(
                    settings,
                    repository,
                    diagnosis_provider=diagnosis_provider,
                )

                evaluated = service.evaluate(
                    state.experiment_id,
                    rationale="只读取 validation。",
                    idempotency_key="evaluate-flow",
                )
                retried = service.evaluate(
                    state.experiment_id,
                    rationale="只重试 LLM，不重复训练。",
                    idempotency_key="evaluate-flow-retry",
                    force_refresh=True,
                )
                decided = service.decide(
                    state.experiment_id,
                    DecisionRequest(action="auto", rationale="采用诊断建议。"),
                    idempotency_key="decide-flow",
                )
                reported = service.report(
                    state.experiment_id,
                    rationale="生成评估闭环报告。",
                    idempotency_key="report-flow",
                )

                self.assertEqual(evaluated.state.state.value, "DECIDING")
                self.assertEqual(retried.state.state.value, "DECIDING")
                self.assertEqual(diagnosis_provider.calls, 2)
                self.assertEqual(
                    evaluated.payload["diagnosis"]["facts"]["basis_split"],
                    "validation",
                )
                self.assertFalse(
                    evaluated.payload["diagnosis"]["facts"]["final_test_used"]
                )
                self.assertEqual(
                    decided.payload["decision"]["action"],
                    "adjust_parameters",
                )
                self.assertEqual(decided.state.state.value, "WAITING_PLAN_SELECTION")
                report_file = Path(reported.payload["report_file"])
                report = report_file.read_text(encoding="utf-8")
                self.assertIn("EvaluationGovernorAgent 诊断", report)
                self.assertIn("qwen/qwen-test", report)
                self.assertIn("Final test：未运行", report)
                evidence = repository.list_evidence(state.experiment_id)
                descriptions = {item.description for item in evidence}
                diagnosis_versions = [
                    item
                    for item in evidence
                    if item.description == "EvaluationGovernorAgent 的结构化 LLM 诊断"
                ]
                self.assertEqual(len(diagnosis_versions), 2)
                self.assertIn("EvaluationGovernorAgent 的结构化 LLM 诊断", descriptions)
                self.assertIn("EvaluationGovernorAgent LLM 调用审计", descriptions)
                self.assertIn("结构化评估、决策与证据 Markdown 实验报告", descriptions)
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
