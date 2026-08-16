"""修正 CoPaw v1.2.2 在 Team 资源名与运行名不同时的共享目录解析。"""

from pathlib import Path


TARGETS = (
    Path("/opt/venv/standard/lib/python3.11/site-packages/copaw_worker/sync.py"),
    Path("/opt/venv/lite/lib/python3.11/site-packages/copaw_worker/sync.py"),
)

OLD = '''    def _get_team_id(self) -> Optional[str]:
        """Resolve the temporary runtime/storage team name from worker metadata."""
        worker = self._get_worker_info()
        team_ref = worker.get("team")
        if not isinstance(team_ref, str) or not team_ref.strip():
            return None
        return _team_storage_name_from_worker_team(self.bucket, team_ref)
'''

NEW = '''    def _get_team_id(self) -> Optional[str]:
        """Resolve the storage team name, preferring controller runtime config."""
        runtime_path = self.local_dir / "runtime" / "runtime.yaml"
        try:
            in_storage = False
            for raw_line in runtime_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                    continue
                if not raw_line.startswith((" ", "\\t")):
                    in_storage = raw_line.strip() == "storage:"
                    continue
                if not in_storage or ":" not in raw_line:
                    continue
                field, value = raw_line.strip().split(":", 1)
                if field != "sharedPrefix":
                    continue
                prefix = value.strip().strip("'\\\"").strip("/")
                if prefix.startswith("teams/") and prefix.endswith("/shared"):
                    team_id = prefix[len("teams/") : -len("/shared")].strip("/")
                    if team_id:
                        return team_id
        except OSError:
            pass

        worker = self._get_worker_info()
        team_ref = worker.get("team")
        if not isinstance(team_ref, str) or not team_ref.strip():
            return None
        return _team_storage_name_from_worker_team(self.bucket, team_ref)
'''


for target in TARGETS:
    text = target.read_text(encoding="utf-8")
    if OLD not in text:
        raise RuntimeError(f"未找到预期的 CoPaw 同步代码片段: {target}")
    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
