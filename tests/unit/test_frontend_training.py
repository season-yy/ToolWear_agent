"""训练监控页面的纯逻辑测试。"""

from __future__ import annotations

import unittest

from toolwear_agent.frontend.workspace_training import is_active_run, latest_run


class FrontendTrainingTest(unittest.TestCase):
    """验证页面只轮询仍在执行的当前实验 Run。"""

    def test_latest_run_uses_created_at_instead_of_list_order(self) -> None:
        runs = [
            {"run_id": "new", "created_at": "2026-08-15T12:01:00+00:00"},
            {"run_id": "old", "created_at": "2026-08-15T12:00:00+00:00"},
        ]

        self.assertEqual(latest_run(runs)["run_id"], "new")

    def test_only_queued_and_running_runs_need_polling(self) -> None:
        self.assertTrue(is_active_run({"status": "queued"}))
        self.assertTrue(is_active_run({"status": "running"}))
        self.assertFalse(is_active_run({"status": "succeeded"}))
        self.assertFalse(is_active_run({"status": "failed"}))
        self.assertFalse(is_active_run(None))


if __name__ == "__main__":
    unittest.main()
