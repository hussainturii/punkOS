"""
M11: Persistent Memory System (Week 2 stub)
Records every task to SQLite so we have history across sessions.

For 50% submission, just records — no semantic retrieval, no learned workflows.
Those are stretch goals for the second half.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class TaskRecord:
    task_id: str
    goal: str
    plan_summary: Optional[str]
    plan_json: Optional[dict]
    history_json: list[dict]
    success: bool
    final_message: str
    steps_taken: int
    duration_seconds: float
    provider: str
    created_at: str
    metrics: dict = field(default_factory=dict)


class TaskHistoryStore:
    """SQLite-backed task history."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id          TEXT PRIMARY KEY,
            goal             TEXT NOT NULL,
            plan_summary     TEXT,
            plan_json        TEXT,
            history_json     TEXT NOT NULL,
            success          INTEGER NOT NULL,
            final_message    TEXT NOT NULL,
            steps_taken      INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            provider         TEXT NOT NULL,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(self.SCHEMA)
            if 'metrics_json' not in {row[1] for row in c.execute('PRAGMA table_info(tasks)')}:
                c.execute("ALTER TABLE tasks ADD COLUMN metrics_json TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on clean exit, and ALWAYS close it.

        sqlite3's own connection context manager commits but does not close.
        Leaked handles keep the .db file locked on Windows, which breaks any
        caller that later tries to delete or move the database.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_task(self, record: TaskRecord) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO tasks
                   (task_id, goal, plan_summary, plan_json, history_json,
                    success, final_message, steps_taken, duration_seconds,
                    provider, created_at, metrics_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.task_id,
                    record.goal,
                    record.plan_summary,
                    json.dumps(record.plan_json) if record.plan_json else None,
                    json.dumps(record.history_json),
                    1 if record.success else 0,
                    record.final_message,
                    record.steps_taken,
                    record.duration_seconds,
                    record.provider,
                    record.created_at,
                    json.dumps(record.metrics),
                ),
            )
            c.commit()

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT task_id, goal, success, final_message, steps_taken, "
                "duration_seconds, provider, created_at, metrics_json "
                "FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item['metrics'] = json.loads(item.pop('metrics_json', '{}'))
            results.append(item)
        return results

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["plan_json"] = json.loads(d["plan_json"]) if d["plan_json"] else None
        d["history_json"] = json.loads(d["history_json"])
        d["success"] = bool(d["success"])
        d['metrics'] = json.loads(d.pop('metrics_json', '{}'))
        return d

    def task_count(self) -> int:
        with self._conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM tasks").fetchone()
        return n

    def delete_task(self, task_id: str) -> bool:
        """Remove one task from history. Returns True if a row was deleted."""
        with self._conn() as c:
            cur = c.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        return (cur.rowcount or 0) > 0
