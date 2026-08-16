"""SQLite 连接、事务与一次性建表逻辑。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    revision INTEGER NOT NULL,
    pending_approval TEXT,
    best_run_id TEXT,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_updated_at ON experiments(updated_at DESC);

CREATE TABLE IF NOT EXISTS experiment_revisions (
    experiment_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    pipeline_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (experiment_id, revision),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS state_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT NOT NULL,
    actor TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_state_events_experiment ON state_events(experiment_id, sequence);

CREATE TABLE IF NOT EXISTS candidate_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_recommendations_experiment
    ON candidate_recommendations(experiment_id, created_at DESC);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_approvals_experiment ON approvals(experiment_id, requested_at);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS evidence_refs (
    evidence_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    run_id TEXT,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_evidence_experiment ON evidence_refs(experiment_id, created_at);

CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS agent_results (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES agent_tasks(task_id)
);

CREATE TABLE IF NOT EXISTS memory_cases (
    memory_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    searchable_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_dataset_task ON memory_cases(dataset_id, task_type);

CREATE TABLE IF NOT EXISTS idempotency_records (
    idempotency_key TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class SQLiteDatabase:
    """为 repository 提供线程安全的单连接事务边界。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()
        self.fts5_enabled = False

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("SQLiteDatabase 尚未 initialize。")
        return self._connection

    def initialize(self) -> None:
        """创建数据库目录、连接和幂等 Schema。"""

        with self._lock:
            if self._connection is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.executescript(SCHEMA_SQL)
            connection.execute("PRAGMA user_version = 1")
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_cases_fts "
                    "USING fts5(memory_id UNINDEXED, searchable_text)"
                )
                self.fts5_enabled = True
            except sqlite3.OperationalError:
                self.fts5_enabled = False
            connection.commit()
            self._connection = connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """串行化写事务；异常时完整回滚。"""

        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """保护同一连接上的只读操作。"""

        with self._lock:
            yield self.connection

    def close(self) -> None:
        """提交已完成事务并安全关闭连接，可重复调用。"""

        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
