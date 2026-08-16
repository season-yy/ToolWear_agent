from __future__ import annotations

import unittest

from toolwear_agent.agentteams.diagnosis import build_decision, build_metric_findings


class DiagnosisTests(unittest.TestCase):
    def test_build_decision_continues_when_metrics_are_high(self) -> None:
        decision = build_decision(
            {
                "validation_macro_f1": 1.0,
            }
        )

        self.assertTrue(decision.continue_current_plan)
        self.assertIn("跨刀具", decision.recommended_next_action)

    def test_build_metric_findings_warns_when_sample_fraction_is_partial(self) -> None:
        findings = build_metric_findings(
            {
                "validation_macro_f1": 1.0,
                "validation_balanced_accuracy": 1.0,
                "sample_fraction": 0.2,
                "sample_count": 2021,
                "full_window_count": 10080,
            }
        )

        titles = [finding.title for finding in findings]
        self.assertIn("当前只是 20% 小范围训练", titles)

    def test_decision_never_uses_final_test_for_tuning(self) -> None:
        good_validation_bad_test = build_decision(
            {
                "validation_macro_f1": 0.95,
                "test_macro_f1": 0.0,
            }
        )
        bad_validation_good_test = build_decision(
            {
                "validation_macro_f1": 0.50,
                "test_macro_f1": 1.0,
            }
        )

        self.assertTrue(good_validation_bad_test.continue_current_plan)
        self.assertFalse(bad_validation_good_test.continue_current_plan)


if __name__ == "__main__":
    unittest.main()
