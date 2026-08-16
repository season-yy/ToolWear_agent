"""核验公开代码包是否满足初赛提交材料的五项要求。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pydantic import ValidationError

from toolwear_agent.schemas.api import CreateExperimentRequest


REQUIRED_FILES = {
    "entrypoint": "scripts/start_local.ps1",
    "dependencies": "pyproject.toml",
    "configuration": ".env.example",
    "sample_input": "examples/create_experiment_request.json",
    "sample_output": "examples/golden_run_summary.json",
    "run_evidence": "docs/run-evidence.md",
}
JSON_EXAMPLES = (
    "examples/create_experiment_request.json",
    "examples/agentteams_task_request.json",
    "examples/golden_run_summary.json",
    "examples/agentteams_task_result.json",
)


def verify_submission_readiness(root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    """返回稳定的提交完整性结果，并验证所有 JSON 样例可解析。"""

    checks = {
        name: (root / relative_path).is_file()
        for name, relative_path in REQUIRED_FILES.items()
    }
    parsed_examples: list[str] = []
    invalid_examples: list[str] = []
    for relative_path in JSON_EXAMPLES:
        path = root / relative_path
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_examples.append(relative_path)
        else:
            parsed_examples.append(relative_path)
    request_example = root / "examples/create_experiment_request.json"
    if request_example.is_file() and "examples/create_experiment_request.json" not in invalid_examples:
        try:
            CreateExperimentRequest.model_validate_json(
                request_example.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            invalid_examples.append("examples/create_experiment_request.json")
    ready = all(checks.values()) and not invalid_examples
    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "parsed_examples": parsed_examples,
        "invalid_examples": invalid_examples,
    }


def main() -> None:
    """打印机器可读结果；未满足要求时以非零状态退出。"""

    result = verify_submission_readiness()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
