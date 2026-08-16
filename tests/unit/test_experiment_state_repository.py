"""SQLite 实验状态仓库和状态机的行为测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from toolwear_agent.schemas import (
    AgentResult,
    AgentTask,
    ApprovalRecord,
    DatasetRef,
    EvidenceRef,
    ExperimentRevision,
    ExperimentState,
    LabelPolicy,
    MemoryCase,
    ModuleSpec,
    PipelineSpec,
    RunConfig,
    SplitSpec,
)
from toolwear_agent.state import (
    IdempotencyConflictError,
    InvalidStateTransitionError,
    RevisionLockedError,
    RunRecord,
    RunStatus,
    SQLiteExperimentRepository,
)


def _experiment_state(*, title: str = "C1 四阶段分类") -> ExperimentState:
    """构造可写入临时数据库的最小实验状态。"""

    return ExperimentState(
        experiment_id="exp-state-test",
        trace_id="trace-state-test",
        title=title,
        dataset_ref=DatasetRef(dataset_id="phm2010", cutter_ids=("C1",)),
        label_policy=LabelPolicy(),
        split_spec=SplitSpec(),
    )


def _pipeline() -> PipelineSpec:
    """构造与当前 Registry 兼容的统计特征方案。"""

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
        rationale="用于状态仓库测试。",
        risks=("仅为测试配置。",),
        expected_cost="low",
    )


def _revision(revision: int = 1) -> ExperimentRevision:
    """构造不可变实验修订。"""

    pipeline = _pipeline()
    return ExperimentRevision(
        experiment_id="exp-state-test",
        revision=revision,
        parent_revision=None if revision == 1 else revision - 1,
        pipeline=pipeline,
        run_config=RunConfig(
            run_id=f"run-r{revision}",
            experiment_id="exp-state-test",
            revision=revision,
            pipeline_id=pipeline.pipeline_id,
            run_kind="mini_train",
        ),
        created_by="human",
        change_reason="用户确认训练配置。",
    )


class ExperimentStateRepositoryTest(unittest.TestCase):
    """使用真实临时 SQLite 文件验证恢复、幂等和审计语义。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "toolwear.db"
        self.repository = SQLiteExperimentRepository(self.db_path)
        self.repository.initialize()

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_dir.cleanup()

    def _create_experiment(self) -> ExperimentState:
        return self.repository.create_experiment(
            _experiment_state(),
            actor="human",
            reason="创建测试实验。",
            idempotency_key="create-exp-state-test",
        )

    def test_create_is_idempotent_and_records_initial_event(self) -> None:
        """相同幂等键重放不能重复创建实验或事件。"""

        first = self._create_experiment()
        replayed = self._create_experiment()

        self.assertEqual(replayed, first)
        events = self.repository.list_state_events(first.experiment_id)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].before_state)
        self.assertEqual(events[0].after_state.value, "DRAFT")

    def test_reusing_idempotency_key_with_changed_request_is_rejected(self) -> None:
        """同一个幂等键不能伪装成另一份写请求。"""

        self._create_experiment()

        with self.assertRaises(IdempotencyConflictError):
            self.repository.create_experiment(
                _experiment_state(title="被修改的请求"),
                actor="human",
                reason="创建测试实验。",
                idempotency_key="create-exp-state-test",
            )

    def test_transition_persists_auditable_before_and_after(self) -> None:
        """合法转换必须同时更新快照并写入完整审计事件。"""

        self._create_experiment()
        transitioned = self.repository.transition_state(
            "exp-state-test",
            "DATA_VALIDATING",
            actor="DataStewardAgent",
            reason="开始数据登记与体检。",
            evidence_ids=("evidence-data-profile",),
            idempotency_key="transition-data-validating",
        )

        self.assertEqual(transitioned.state.value, "DATA_VALIDATING")
        self.assertEqual(transitioned.last_event_sequence, 2)
        event = self.repository.list_state_events("exp-state-test")[-1]
        self.assertEqual(event.before_state.value, "DRAFT")
        self.assertEqual(event.after_state.value, "DATA_VALIDATING")
        self.assertEqual(event.actor, "DataStewardAgent")
        self.assertEqual(event.evidence_ids, ("evidence-data-profile",))

    def test_invalid_transition_keeps_database_unchanged(self) -> None:
        """状态机拒绝跨阶段跳转，并且不能留下半条事件。"""

        self._create_experiment()

        with self.assertRaises(InvalidStateTransitionError):
            self.repository.transition_state(
                "exp-state-test",
                "MINI_TRAINING",
                actor="human",
                reason="尝试绕过方案选择。",
                idempotency_key="invalid-jump",
            )

        restored = self.repository.get_experiment("exp-state-test")
        self.assertEqual(restored.state.value, "DRAFT")
        self.assertEqual(len(self.repository.list_state_events("exp-state-test")), 1)

    def test_revision_is_immutable_and_training_locks_revision_pointer(self) -> None:
        """修订只允许新增；训练状态下不能切换当前配置。"""

        self._create_experiment()
        created = self.repository.create_revision(
            _revision(),
            idempotency_key="create-revision-1",
        )
        self.assertIsNotNone(created.content_hash)

        for target, key in (
            ("DATA_VALIDATING", "state-data-validating"),
            ("WAITING_PLAN_SELECTION", "state-plan-selection"),
            ("PIPELINE_VALIDATING", "state-pipeline-validating"),
            ("CODE_PREPARING", "state-code-preparing"),
            ("MINI_TRAINING", "state-mini-training"),
        ):
            self.repository.transition_state(
                "exp-state-test",
                target,
                actor="system",
                reason=f"进入 {target}。",
                idempotency_key=key,
            )

        with self.assertRaises(RevisionLockedError):
            self.repository.create_revision(
                _revision(2),
                idempotency_key="create-revision-2-during-training",
            )

        self.assertEqual(self.repository.get_experiment("exp-state-test").revision, 1)

    def test_pending_approval_and_decision_survive_restart(self) -> None:
        """页面刷新或进程重启后仍能恢复审批和当前实验状态。"""

        self._create_experiment()
        pending = ApprovalRecord(
            approval_id="approval-plan-1",
            experiment_id="exp-state-test",
            revision=1,
            action="approve_pipeline",
            requested_by="AlgorithmArchitectAgent",
            rationale="请选择一个候选方案。",
        )
        self.repository.create_approval(
            pending,
            idempotency_key="request-plan-approval",
        )
        decided = self.repository.decide_approval(
            pending.approval_id,
            status="approved",
            decided_by="human",
            rationale="选择统计特征基线。",
            idempotency_key="approve-plan-1",
        )
        self.assertEqual(decided.status.value, "approved")
        self.assertIsNone(self.repository.get_experiment("exp-state-test").pending_approval)

        self.repository.close()
        self.repository = SQLiteExperimentRepository(self.db_path)
        self.repository.initialize()

        restored = self.repository.get_experiment("exp-state-test")
        restored_approval = self.repository.get_approval("approval-plan-1")
        self.assertEqual(restored.trace_id, "trace-state-test")
        self.assertEqual(restored_approval.decided_by, "human")

    def test_run_evidence_task_result_and_memory_have_one_repository_boundary(self) -> None:
        """运行、证据、Agent 任务与经验都由同一个仓库持久化。"""

        self._create_experiment()
        run = RunRecord(
            run_id="run-state-test",
            experiment_id="exp-state-test",
            revision=1,
            pipeline_id="statistical_features_random_forest",
            run_kind="mini_train",
        )
        self.repository.create_run(run, idempotency_key="create-run-state-test")
        completed = self.repository.update_run_status(
            run.run_id,
            status=RunStatus.SUCCEEDED,
            result_summary={"validation_macro_f1": 0.91},
            idempotency_key="finish-run-state-test",
        )

        evidence = EvidenceRef(
            evidence_id="evidence-metrics-1",
            experiment_id="exp-state-test",
            run_id=run.run_id,
            kind="metrics",
            uri="D:/AI_infra/experiments/run-state-test/metrics.json",
            sha256="a" * 64,
            size_bytes=128,
        )
        self.repository.register_evidence(evidence, idempotency_key="evidence-metrics-1")

        task = AgentTask(
            task_id="task-evaluate-1",
            experiment_id="exp-state-test",
            revision=1,
            assigned_to="EvaluationGovernorAgent",
            requested_by="ExperimentManagerAgent",
            task_type="evaluate",
            objective="评估 validation 指标。",
        )
        result = AgentResult(
            task_id=task.task_id,
            agent_name="EvaluationGovernorAgent",
            status="success",
            summary="validation 指标可用。",
            evidence=(evidence,),
        )
        self.repository.save_agent_task(task, idempotency_key="task-evaluate-1")
        self.repository.save_agent_result(result, idempotency_key="result-evaluate-1")

        memory = MemoryCase(
            memory_id="memory-rf-good",
            dataset_id="phm2010",
            task_type="four_stage_classification",
            problem="需要低成本验证阶段可分性。",
            intervention="使用统计特征和随机森林。",
            outcome="validation Macro-F1 达到 0.91。",
            summary="随机森林适合作为首轮低成本基线。",
            tags=("random_forest", "baseline"),
            evidence_ids=(evidence.evidence_id,),
        )
        self.repository.save_memory_case(memory, idempotency_key="memory-rf-good")

        self.assertEqual(completed.status, RunStatus.SUCCEEDED)
        self.assertEqual(self.repository.list_evidence("exp-state-test"), (evidence,))
        self.assertEqual(self.repository.get_agent_result(task.task_id), result)
        self.assertEqual(self.repository.search_memory("随机森林"), (memory,))

    def test_successful_run_budget_is_accounted_only_once_even_after_best_run_changes(self) -> None:
        """旧 Run 不得因后来失去 best_run 身份而被重复计入预算。"""

        self._create_experiment()
        for index, score in ((1, 0.70), (2, 0.90)):
            run = RunRecord(
                run_id=f"run-budget-{index}",
                experiment_id="exp-state-test",
                revision=1,
                pipeline_id="statistical_features_random_forest",
                run_kind="mini_train",
            )
            self.repository.create_run(run)
            self.repository.update_run_status(
                run.run_id,
                status=RunStatus.SUCCEEDED,
                result_summary={"validation_macro_f1": score},
            )
            self.repository.record_successful_run_budget(run.run_id, consumed_epochs=2)

        state_after_two_runs = self.repository.get_experiment("exp-state-test")
        self.assertEqual(state_after_two_runs.best_run_id, "run-budget-2")
        self.assertEqual(state_after_two_runs.budget.completed_mini_runs, 2)
        self.assertEqual(state_after_two_runs.budget.consumed_epochs, 4)

        replayed = self.repository.record_successful_run_budget(
            "run-budget-1",
            consumed_epochs=2,
        )

        self.assertEqual(replayed.budget.completed_mini_runs, 2)
        self.assertEqual(replayed.budget.consumed_epochs, 4)
        self.assertTrue(self.repository.get_run("run-budget-1").budget_accounted)
        self.assertEqual(self.repository.get_run("run-budget-1").consumed_epochs, 2)


if __name__ == "__main__":
    unittest.main()
