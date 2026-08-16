"""Streamlit 工作台与临时 FastAPI 的端到端交互测试。"""

from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import uvicorn
from streamlit.testing.v1 import AppTest

from toolwear_agent.backend.main import create_app
from toolwear_agent.core.settings import Settings
from toolwear_agent.data.registry import DatasetRegistry
from toolwear_agent.schemas import CutterManifest, DatasetManifest


def _available_port() -> int:
    """由系统分配一个测试端口，避免和开发服务争用 18100。"""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class StreamlitWorkbenchIntegrationTest(unittest.TestCase):
    """验证新建实验后页面能从 API 状态恢复为真实工作台。"""

    def test_create_experiment_enters_state_driven_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_root = Path(temp_dir) / "runtime"
            raw_cutter = runtime_root / "datasets" / "raw" / "phm2010" / "c1"
            raw_cutter.mkdir(parents=True)
            port = _available_port()
            settings = Settings(
                ai_infra_root=runtime_root,
                fastapi_port=port,
                train_device="cpu",
                llm_api_key="",
                llm_model="",
            )
            DatasetRegistry(settings.dataset_manifest).register(
                DatasetManifest(
                    dataset_id="phm2010",
                    display_name="PHM 2010",
                    adapter="phm2010",
                    root=settings.phm2010_raw_root,
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
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(settings=settings),
                    host="127.0.0.1",
                    port=port,
                    log_level="error",
                )
            )
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(server.started)

            app_file = Path(__file__).resolve().parents[2] / "toolwear_agent" / "frontend" / "streamlit_app.py"
            try:
                with patch.dict(os.environ, {"FASTAPI_PORT": str(port)}):
                    app = AppTest.from_file(str(app_file)).run(timeout=30)
                    create_button = next(
                        button for button in app.button if button.label == "创建实验"
                    )
                    create_button.click()
                    app.run(timeout=30)

                self.assertEqual(app.exception, [])
                self.assertEqual(app.error, [])
                self.assertIn("PHM2010 刀具磨损四阶段分类", [title.value for title in app.title])
                self.assertIn("数据准备", [tab.label for tab in app.tabs])
                self.assertIn("Agent 协作", [tab.label for tab in app.tabs])
                self.assertTrue(
                    any(
                        "ExperimentManagerAgent" in item.value
                        for item in app.markdown
                    )
                )
                self.assertEqual(
                    app.session_state["experiment_id"],
                    app.session_state["experiment_picker"],
                )
            finally:
                server.should_exit = True
                thread.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
