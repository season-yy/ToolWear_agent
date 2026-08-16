"""ToolWear 统一 Pydantic Schema 的契约测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from toolwear_agent.schemas import (
    AgentResult,
    AgentTask,
    ApprovalRecord,
    CutterManifest,
    DatasetManifest,
    DatasetRef,
    DecisionRecord,
    EvaluationReport,
    EvidenceRef,
    ExperimentRevision,
    ExperimentState,
    LabelPolicy,
    MemoryCase,
    MetricBundle,
    ModuleSpec,
    PipelineSpec,
    RunConfig,
    SplitSpec,
    ValidationIssue,
    ValidationResult,
)


def _dataset_ref() -> DatasetRef:
    """返回多个测试共用的最小数据集引用。"""

    return DatasetRef(dataset_id="phm2010", cutter_ids=("C1",))


def _pipeline() -> PipelineSpec:
    """返回可执行的最小 sklearn Pipeline。"""

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
        rationale="低成本可解释基线。",
        risks=("跨刀具泛化能力需要验证。",),
        expected_cost="low",
    )


def _run_config(*, run_kind: str = "mini_train", evaluate_test: bool = False) -> RunConfig:
    """返回与 `_pipeline` 对应的运行配置。"""

    return RunConfig(
        run_id="run-demo",
        experiment_id="exp-demo",
        revision=1,
        pipeline_id="statistical_features_random_forest",
        run_kind=run_kind,
        evaluate_test=evaluate_test,
    )


class DatasetSchemaTest(unittest.TestCase):
    """验证数据集、标签和切分契约。"""

    def test_dataset_manifest_round_trip(self) -> None:
        """Manifest 必须能稳定做 JSON 往返。"""

        manifest = DatasetManifest(
            dataset_id="phm2010",
            display_name="PHM 2010",
            adapter="phm2010",
            root=Path("datasets/raw/phm2010"),
            channels=("force_x", "vibration_x"),
            cutters={
                "C1": CutterManifest(
                    cutter_id="C1",
                    relative_path="c1",
                    labeled=True,
                    wear_file="c1_wear.csv",
                )
            },
        )

        restored = DatasetManifest.model_validate_json(manifest.model_dump_json())

        self.assertEqual(restored, manifest)
        self.assertEqual(restored.schema_version, "1.0")

    def test_manifest_rejects_relative_path_escape(self) -> None:
        """Cutter 相对路径不能逃出数据集根目录。"""

        with self.assertRaises(ValidationError):
            CutterManifest(cutter_id="C1", relative_path="../outside", labeled=False)

    def test_label_policy_requires_three_increasing_thresholds(self) -> None:
        """四阶段标签边界必须是三个严格递增值。"""

        with self.assertRaises(ValidationError):
            LabelPolicy(stage_thresholds_um=(90.0, 90.0, 160.0))

    def test_cross_cutter_split_rejects_overlap(self) -> None:
        """跨刀具 train/val/test 刀具集合不能重叠。"""

        with self.assertRaises(ValidationError):
            SplitSpec(
                strategy="cross_cutter",
                train_cutters=("C1", "C4"),
                validation_cutters=("C4",),
                test_cutters=("C6",),
            )


class PipelineSchemaTest(unittest.TestCase):
    """验证模块链和测试集隔离规则。"""

    def test_pipeline_requires_one_model_and_one_trainer(self) -> None:
        """缺少模型模块的 Pipeline 不能进入训练。"""

        with self.assertRaises(ValidationError):
            PipelineSpec(
                pipeline_id="invalid-pipeline",
                display_name="无模型方案",
                source="user",
                input_channels=("force_x",),
                modules=(ModuleSpec(module_id="sklearn_trainer", kind="trainer", order=10),),
                rationale="用于非法输入测试。",
                risks=("无法训练。",),
                expected_cost="low",
            )

    def test_pipeline_rejects_extra_fields(self) -> None:
        """未知字段不能静默进入执行配置。"""

        payload = _pipeline().model_dump()
        payload["unknown_command"] = "run anything"

        with self.assertRaises(ValidationError):
            PipelineSpec.model_validate(payload)

    def test_run_config_only_allows_test_in_final_evaluation(self) -> None:
        """mini/smoke/full train 都不能请求 test 指标。"""

        with self.assertRaises(ValidationError):
            _run_config(run_kind="mini_train", evaluate_test=True)

        final_run = _run_config(run_kind="final_evaluation", evaluate_test=True)
        self.assertTrue(final_run.evaluate_test)


class ExperimentSchemaTest(unittest.TestCase):
    """验证实验修订、审批和决策一致性。"""

    def test_experiment_revision_requires_matching_nested_ids(self) -> None:
        """Revision 内的 RunConfig 必须属于同一实验、修订和 Pipeline。"""

        with self.assertRaises(ValidationError):
            ExperimentRevision(
                experiment_id="exp-demo",
                revision=2,
                pipeline=_pipeline(),
                run_config=_run_config(),
                created_by="human",
                change_reason="非法交叉引用测试。",
            )

    def test_approved_record_requires_decision_identity_and_time(self) -> None:
        """审批完成时必须知道由谁、何时作出。"""

        with self.assertRaises(ValidationError):
            ApprovalRecord(
                approval_id="approval-1",
                experiment_id="exp-demo",
                revision=1,
                action="approve_pipeline",
                status="approved",
                requested_by="AlgorithmArchitectAgent",
            )

        approved = ApprovalRecord(
            approval_id="approval-1",
            experiment_id="exp-demo",
            revision=1,
            action="approve_pipeline",
            status="approved",
            requested_by="AlgorithmArchitectAgent",
            decided_by="human",
            decided_at=datetime.now(timezone.utc),
        )
        self.assertEqual(approved.status.value, "approved")

    def test_decision_basis_is_validation_not_test(self) -> None:
        """调参与停止决策的依据只能是 validation。"""

        with self.assertRaises(ValidationError):
            DecisionRecord(
                decision_id="decision-1",
                experiment_id="exp-demo",
                run_id="run-demo",
                action="adjust_parameters",
                basis_split="test",
                reason="不允许用 test 调参。",
                decided_by="EvaluationGovernorAgent",
            )

    def test_experiment_state_round_trip(self) -> None:
        """页面和数据库共用的状态对象必须可 JSON 往返。"""

        state = ExperimentState(
            experiment_id="exp-demo",
            title="C1 四阶段分类",
            dataset_ref=_dataset_ref(),
            label_policy=LabelPolicy(),
            split_spec=SplitSpec(),
        )

        restored = ExperimentState.model_validate_json(state.model_dump_json())
        self.assertEqual(restored, state)


class EvidenceAgentEvaluationSchemaTest(unittest.TestCase):
    """验证证据、Agent 交接和评估报告契约。"""

    def test_evidence_requires_sha256_shape(self) -> None:
        """Evidence hash 必须是完整 SHA-256 十六进制文本。"""

        with self.assertRaises(ValidationError):
            EvidenceRef(
                evidence_id="evidence-1",
                experiment_id="exp-demo",
                run_id="run-demo",
                kind="metrics",
                uri="runs/run-demo/metrics.json",
                sha256="too-short",
                size_bytes=10,
            )

    def test_agent_name_is_limited_to_six_roles(self) -> None:
        """步骤名或临时角色不能伪装成第七个 Agent。"""

        AgentTask(
            task_id="task-1",
            experiment_id="exp-demo",
            revision=1,
            assigned_to="AlgorithmArchitectAgent",
            requested_by="ExperimentManagerAgent",
            task_type="recommend_pipeline",
            objective="生成候选方案。",
        )

        with self.assertRaises(ValidationError):
            AgentTask(
                task_id="task-2",
                experiment_id="exp-demo",
                revision=1,
                assigned_to="SeventhJudgeAgent",
                requested_by="human",
                task_type="judge",
                objective="非法第七 Agent。",
            )

    def test_failed_agent_result_requires_error_code(self) -> None:
        """Agent 失败时不能只返回一段模糊自然语言。"""

        with self.assertRaises(ValidationError):
            AgentResult(
                task_id="task-1",
                agent_name="AlgorithmArchitectAgent",
                status="failed",
                summary="调用失败。",
            )

    def test_test_metrics_require_final_evaluation_flag(self) -> None:
        """EvaluationReport 中出现 test 指标时必须标记为最终评估。"""

        test_metric = MetricBundle(
            split="test",
            sample_count=100,
            macro_f1=0.8,
            balanced_accuracy=0.82,
        )
        with self.assertRaises(ValidationError):
            EvaluationReport(
                evaluation_id="eval-1",
                experiment_id="exp-demo",
                run_id="run-demo",
                pipeline_id="pipeline-1",
                metrics=(test_metric,),
                class_labels=("initial", "normal", "severe", "failure"),
                final_test=False,
            )

    def test_validation_result_cannot_be_valid_with_error_issue(self) -> None:
        """存在 error 级问题时 ValidationResult 不能声称 valid。"""

        with self.assertRaises(ValidationError):
            ValidationResult(
                valid=True,
                scope="pipeline",
                issues=(ValidationIssue(code="MODEL_MISSING", severity="error", message="缺少模型。"),),
            )

    def test_memory_case_round_trip(self) -> None:
        """轻量 Memory 的文本案例必须可序列化。"""

        case = MemoryCase(
            memory_id="memory-1",
            dataset_id="phm2010",
            task_type="four_stage_classification",
            problem="validation Macro-F1 偏低",
            intervention="降低学习率并增加类别权重",
            outcome="validation Macro-F1 提升",
            summary="一次只基于验证集的调参经验。",
            tags=("cnn1d", "class-imbalance"),
        )
        self.assertEqual(MemoryCase.model_validate_json(case.model_dump_json()), case)


if __name__ == "__main__":
    unittest.main()
