from __future__ import annotations

import unittest

from toolwear_agent.agentteams.decision import build_decision_actions, build_parameter_suggestions


class DecisionTests(unittest.TestCase):
    def test_build_decision_actions_keeps_good_baseline_and_adds_next_steps(self) -> None:
        actions = build_decision_actions(
            {
                "decision": {
                    "continue_current_plan": True,
                    "reason": "当前结果足够好，但泛化未验证。",
                }
            }
        )

        action_ids = [action.action_id for action in actions]
        self.assertIn("keep_random_forest_baseline", action_ids)
        self.assertIn("start_cross_cutter_validation", action_ids)
        self.assertIn("start_cnn_comparison", action_ids)

    def test_build_decision_actions_stops_when_diagnosis_says_do_not_continue(self) -> None:
        actions = build_decision_actions(
            {
                "decision": {
                    "continue_current_plan": False,
                    "reason": "指标不足。",
                }
            }
        )

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "stop")

    def test_build_parameter_suggestions_keeps_window_but_raises_sample_fraction(self) -> None:
        suggestions = build_parameter_suggestions(
            train_config={
                "window_size": 4096,
                "overlap_ratio": 0.5,
                "max_windows_per_cut": 32,
            },
            metrics_summary={
                "sample_fraction": 0.2,
            },
        )

        self.assertFalse(suggestions["window_size"]["should_change_now"])
        self.assertTrue(suggestions["sample_fraction"]["should_change_now"])
        self.assertEqual(suggestions["sample_fraction"]["suggested"], 1.0)


if __name__ == "__main__":
    unittest.main()
