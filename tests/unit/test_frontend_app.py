"""状态驱动实验台的纯视图逻辑测试。"""

from __future__ import annotations

import unittest

from toolwear_agent.frontend.ui_state import (
    preparation_progress,
    state_actions,
    workflow_stage_index,
)


class StreamlitUiStateTest(unittest.TestCase):
    """页面按钮必须由后端状态和证据共同决定。"""

    def test_draft_only_allows_data_preparation(self) -> None:
        actions = state_actions("DRAFT", preparation_ready=False, has_succeeded_run=False)

        self.assertTrue(actions.prepare_data)
        self.assertFalse(actions.generate_candidates)
        self.assertFalse(actions.start_training)

    def test_data_validating_requires_all_three_preparation_evidence(self) -> None:
        artifacts = [
            {"evidence_id": "exp-ui-profile-r1"},
            {"evidence_id": "exp-ui-labels-r1"},
            {"evidence_id": "exp-ui-split-r1"},
        ]

        progress = preparation_progress("exp-ui", 1, artifacts)
        actions = state_actions(
            "DATA_VALIDATING",
            preparation_ready=progress.complete,
            has_succeeded_run=False,
        )

        self.assertTrue(progress.complete)
        self.assertTrue(actions.generate_candidates)

    def test_training_and_decision_buttons_follow_state_machine(self) -> None:
        training = state_actions(
            "CODE_PREPARING",
            preparation_ready=True,
            has_succeeded_run=False,
        )
        deciding = state_actions(
            "DECIDING",
            preparation_ready=True,
            has_succeeded_run=True,
        )

        self.assertTrue(training.start_training)
        self.assertFalse(training.decide)
        self.assertTrue(deciding.decide)
        self.assertTrue(deciding.generate_report)

    def test_workflow_stage_index_is_stable_for_branch_states(self) -> None:
        self.assertEqual(workflow_stage_index("DRAFT"), 0)
        self.assertEqual(workflow_stage_index("MINI_TRAINING"), 4)
        self.assertEqual(workflow_stage_index("WAITING_PLAN_SELECTION"), 2)
        self.assertEqual(workflow_stage_index("WAITING_FULL_APPROVAL"), 6)
        self.assertEqual(workflow_stage_index("FAILED"), 4)


if __name__ == "__main__":
    unittest.main()
