"""命令行调用初赛白名单提交包构建器。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from toolwear_agent.core.settings import load_settings
from toolwear_agent.delivery.submission import build_submission_package


def main() -> None:
    settings = load_settings()
    default_output = (
        settings.ai_infra_root
        / "submission"
        / f"toolwear-agent-initial-round-{datetime.now():%Y%m%d-%H%M%S}.zip"
    )
    parser = argparse.ArgumentParser(description="生成不含密钥和运行数据的初赛提交包。")
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    result = build_submission_package(REPOSITORY_ROOT, args.output)
    print(f"package={result.output_file}")
    print(f"files={result.file_count}")
    print(f"size_bytes={result.size_bytes}")
    print(f"sha256={result.sha256}")
    print(f"git_commit={result.git_commit}")
    print(f"git_dirty={result.git_dirty}")


if __name__ == "__main__":
    main()
