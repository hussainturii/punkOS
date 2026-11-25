"""
M11: Persistent Memory System (SDS 1.7.11, FR7.3-FR7.6).

Two kinds of memory, per the SDS:

  Episodic  — what happened. Every task, its plan and its outcome. That already
              lives in core/history.py and stays there.
  Procedural— how to do things. Learned workflows, user preferences, and the
              corrections the user made when the agent got it wrong. That is
              this module.

Storage choice
--------------
The SDS names a JSON file store for DS4 (learned workflows). We use SQLite
instead, for two reasons: REL-8 requires that a crash mid-write cannot corrupt
the store, which SQLite's transactions give us for free and a rewritten JSON
file does not; and Algorithm 6 needs a similarity lookup over stored embeddings,
which means holding BLOB vectors next to the rows. The stored *shape* is
unchanged — workflowId, name, triggerPattern, steps, successRate, lastUsed,
executionCount are all present. `export_workflows()` emits the JSON form the
SDS describes, so nothing downstream is locked in.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .embedding import (
    TextEmbedder, cosine_similarity, deserialize_vector, serialize_vector,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkflowStep:
    action_type: str
    text: str = ""
    target_hint: str = ""
    # How often this step appeared in a successful run of this workflow.
    # Algorithm 6 step 3b weights steps by success frequency.
    success_weight: float = 1.0
    from_correction: bool = False

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type, "text": self.text,
            "target_hint": self.target_hint, "success_weight": self.success_weight,
            "from_correction": self.from_correction,
        }

    @staticmethod
    def from_dict(d: dict) -> "WorkflowStep":
        return WorkflowStep(
            action_type=d.get("action_type", "wait"),
            text=d.get("text", ""),
            target_hint=d.get("target_hint", ""),
            success_weight=float(d.get("success_weight", 1.0)),
            from_correction=bool(d.get("from_correction", False)),
        )


@dataclass
class LearnedWorkflow:
    workflow_id: str
    name: str
    trigger_pattern: str
    steps: list[WorkflowStep]
    execution_count: int = 0
    success_count: int = 0
    last_used: str = ""
    created_at: str = ""
    is_template: bool = False       # FR7.3: user bookmarked it explicitly
    source_task_id: str = ""

    @property
    def success_rate(self) -> float:
        return (self.success_count / self.execution_count) if self.execution_count else 0.0

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id, "name": self.name,
            "trigger_pattern": self.trigger_pattern,
            "steps": [s.to_dict() for s in self.steps],
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "last_used": self.last_used, "created_at": self.created_at,
            "is_template": self.is_template, "source_task_id": self.source_task_id,
        }


@dataclass
class Correction:
    correction_id: str
    task_id: str
    step_index: int
    original_action: str
    corrected_action: str
    note: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "correction_id": self.correction_id, "task_id": self.task_id,
            "step_index": self.step_index, "original_action": self.original_action,
            "corrected_action": self.corrected_action, "note": self.note,
            "created_at": self.created_at,
        }


class ProceduralMemory:
    """Learned workflows, preferences and corrections — persistent across sessions."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS workflows (
            workflow_id     TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            trigger_pattern TEXT NOT NULL,
            steps_json      TEXT NOT NULL,
            embedding       BLOB NOT NULL,
            execution_count INTEGER NOT NULL DEFAULT 0,
            success_count   INTEGER NOT NULL DEFAULT 0,
            last_used       TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL,
            is_template     INTEGER NOT NULL DEFAULT 0,
            source_task_id  TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_wf_last_used ON workflows(last_used DESC);

        CREATE TABLE IF NOT EXISTS preferences (
            key         TEXT PRIMARY KEY,
            value       TEXT NOT NULL,
            hit_count   INTEGER NOT NULL DEFAULT 1,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS corrections (
            correction_id    TEXT PRIMARY KEY,
            task_id          TEXT NOT NULL,
            step_index       INTEGER NOT NULL,
            original_action  TEXT NOT NULL,
            corrected_action TEXT NOT NULL,
            note             TEXT NOT NULL DEFAULT '',
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_corr_task ON corrections(task_id);

        CREATE TABLE IF NOT EXISTS bookmarks (
            task_id     TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL
        );
    """

    def __init__(self, db_path: str | Path, embedder: Optional[TextEmbedder] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or TextEmbedder()
        with self._conn() as c:
            c.executescript(self.SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # REL-8: a crash mid-write rolls back rather than corrupting.
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    def save_workflow(self, workflow: LearnedWorkflow) -> str:
        vec = self.embedder.encode(workflow.trigger_pattern or workflow.name)
        if not workflow.created_at:
            workflow.created_at = _now()
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO workflows
                   (workflow_id, name, trigger_pattern, steps_json, embedding,
                    execution_count, success_count, last_used, created_at,
                    is_template, source_task_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    workflow.workflow_id, workflow.name, workflow.trigger_pattern,
                    json.dumps([s.to_dict() for s in workflow.steps]),
                    serialize_vector(vec),
                    workflow.execution_count, workflow.success_count,
                    workflow.last_used, workflow.created_at,
                    1 if workflow.is_template else 0, workflow.source_task_id,
                ),
            )
        return workflow.workflow_id

    def get_workflow(self, workflow_id: str) -> Optional[LearnedWorkflow]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM workflows WHERE workflow_id = ?",
                            (workflow_id,)).fetchone()
        return self._row_to_workflow(row) if row else None

    def list_workflows(self, limit: int = 100,
                       templates_only: bool = False) -> list[LearnedWorkflow]:
        query = "SELECT * FROM workflows {where} ORDER BY last_used DESC, created_at DESC LIMIT ?"
        where = "WHERE is_template = 1" if templates_only else ""
        with self._conn() as c:
            rows = c.execute(query.format(where=where), (limit,)).fetchall()
        return [self._row_to_workflow(r) for r in rows]

    def find_similar_workflow(
        self, goal: str, threshold: float = 0.85
    ) -> Optional[tuple[LearnedWorkflow, float]]:
        """Algorithm 6 step 2: nearest stored workflow above the threshold."""
        query_vec = self.embedder.encode(goal)
        with self._conn() as c:
            rows = c.execute("SELECT * FROM workflows").fetchall()

        best: Optional[tuple[LearnedWorkflow, float]] = None
        for row in rows:
            score = cosine_similarity(query_vec, deserialize_vector(row["embedding"]))
            if score >= threshold and (best is None or score > best[1]):
                best = (self._row_to_workflow(row), score)
        return best

    def record_execution(self, workflow_id: str, success: bool) -> None:
        """FR7.4 business rule: success rate recalculated after every execution."""
        with self._conn() as c:
            c.execute(
                """UPDATE workflows
                   SET execution_count = execution_count + 1,
                       success_count   = success_count + ?,
                       last_used       = ?
                   WHERE workflow_id = ?""",
                (1 if success else 0, _now(), workflow_id),
            )

    def delete_workflow(self, workflow_id: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
        return (cur.rowcount or 0) > 0

    def _row_to_workflow(self, row: sqlite3.Row) -> LearnedWorkflow:
        return LearnedWorkflow(
            workflow_id=row["workflow_id"],
            name=row["name"],
            trigger_pattern=row["trigger_pattern"],
            steps=[WorkflowStep.from_dict(d) for d in json.loads(row["steps_json"])],
            execution_count=row["execution_count"],
            success_count=row["success_count"],
            last_used=row["last_used"],
            created_at=row["created_at"],
            is_template=bool(row["is_template"]),
            source_task_id=row["source_task_id"],
        )

    def export_workflows(self) -> str:
        """Emit the DS4 JSON form described in the SDS."""
        return json.dumps([w.to_dict() for w in self.list_workflows(limit=10000)],
                          indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Preferences (M11 FE-1)
    # ------------------------------------------------------------------

    def set_preference(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO preferences (key, value, hit_count, updated_at)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     value = excluded.value,
                     hit_count = preferences.hit_count + 1,
                     updated_at = excluded.updated_at""",
                (key, value, _now()),
            )

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def list_preferences(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT key, value, hit_count, updated_at FROM preferences "
                "ORDER BY hit_count DESC, updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def note_app_usage(self, app_name: str) -> None:
        """Track frequently used apps so they can be surfaced in the prompt."""
        if not app_name:
            return
        with self._conn() as c:
            c.execute(
                """INSERT INTO preferences (key, value, hit_count, updated_at)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     hit_count = preferences.hit_count + 1,
                     updated_at = excluded.updated_at""",
                (f"app_usage:{app_name.lower()}", app_name, _now()),
            )

    def frequent_apps(self, limit: int = 5) -> list[tuple[str, int]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT value, hit_count FROM preferences WHERE key LIKE 'app_usage:%' "
                "ORDER BY hit_count DESC LIMIT ?", (limit,)
            ).fetchall()
        return [(r["value"], r["hit_count"]) for r in rows]

    # ------------------------------------------------------------------
    # Corrections (M10 FE-1 / FR7.5)
    # ------------------------------------------------------------------

    def add_correction(self, task_id: str, step_index: int,
                       original_action: str, corrected_action: str,
                       note: str = "") -> Correction:
        correction = Correction(
            correction_id=uuid.uuid4().hex[:12],
            task_id=task_id, step_index=step_index,
            original_action=original_action, corrected_action=corrected_action,
            note=note,
        )
        with self._conn() as c:
            c.execute(
                """INSERT INTO corrections
                   (correction_id, task_id, step_index, original_action,
                    corrected_action, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (correction.correction_id, task_id, step_index, original_action,
                 corrected_action, note, correction.created_at),
            )
        return correction

    def corrections_for_task(self, task_id: str) -> list[Correction]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM corrections WHERE task_id = ? ORDER BY step_index",
                (task_id,)
            ).fetchall()
        return [
            Correction(
                correction_id=r["correction_id"], task_id=r["task_id"],
                step_index=r["step_index"], original_action=r["original_action"],
                corrected_action=r["corrected_action"], note=r["note"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def correction_count(self) -> int:
        with self._conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM corrections").fetchone()
        return n

    # ------------------------------------------------------------------
    # Bookmarks (FR7.3) — persistent, unlike the old in-memory set
    # ------------------------------------------------------------------

    def set_bookmark(self, task_id: str, bookmarked: bool) -> None:
        with self._conn() as c:
            if bookmarked:
                c.execute("INSERT OR REPLACE INTO bookmarks (task_id, created_at) VALUES (?, ?)",
                          (task_id, _now()))
            else:
                c.execute("DELETE FROM bookmarks WHERE task_id = ?", (task_id,))

    def is_bookmarked(self, task_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM bookmarks WHERE task_id = ?", (task_id,)).fetchone()
        return row is not None

    def list_bookmarks(self) -> set[str]:
        with self._conn() as c:
            rows = c.execute("SELECT task_id FROM bookmarks").fetchall()
        return {r["task_id"] for r in rows}

    # ------------------------------------------------------------------
    # FR7.6: memory statistics
    # ------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._conn() as c:
            (workflows,) = c.execute("SELECT COUNT(*) FROM workflows").fetchone()
            (templates,) = c.execute("SELECT COUNT(*) FROM workflows WHERE is_template = 1").fetchone()
            (corrections,) = c.execute("SELECT COUNT(*) FROM corrections").fetchone()
            (prefs,) = c.execute("SELECT COUNT(*) FROM preferences").fetchone()
            (bookmarks,) = c.execute("SELECT COUNT(*) FROM bookmarks").fetchone()
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "learned_workflows": workflows,
            "templates": templates,
            "corrections_applied": corrections,
            "preferences_stored": prefs,
            "bookmarks": bookmarks,
            "store_size_bytes": size_bytes,
            "store_size_human": self._human_size(size_bytes),
            "embedding_backend": self.embedder.backend_name(),
        }

    @staticmethod
    def _human_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {unit}"
            n /= 1024.0
        return f"{n:.1f} TB"

    # ------------------------------------------------------------------
    # Prompt injection (M11 FE-1: "inject into system prompt at task start")
    # ------------------------------------------------------------------

    def context_for_prompt(self, goal: str, max_preferences: int = 8,
                           similarity_threshold: float = 0.85) -> str:
        """Relevant memory, rendered for the executor prompt.

        Kept deliberately short — on a 3B model with a 4096-token context,
        every line spent here is a line not spent on the element list.
        """
        sections: list[str] = []

        match = self.find_similar_workflow(goal, threshold=similarity_threshold)
        if match:
            workflow, score = match
            sections.append(
                f"REMEMBERED PROCEDURE (matched {score:.0%}, "
                f"succeeded {workflow.success_rate:.0%} of {workflow.execution_count} runs):"
            )
            for i, step in enumerate(workflow.steps, start=1):
                marker = " [user-corrected]" if step.from_correction else ""
                detail = step.text or step.target_hint or ""
                sections.append(f"  {i}. {step.action_type} {detail}{marker}".rstrip())
            sections.append("  Historical reference only, not current authorization or proof of success. "
                            "Do not reuse literal paths, recipients or text unless the current user requests them. "
                            "Resolve targets from the current screen; old tags are invalid. "
                            "User constraints and observed state override this sequence. Verify all current outcomes.")

        apps = self.frequent_apps(limit=max_preferences)
        if apps:
            listed = ", ".join(f"{name} ({count}x)" for name, count in apps)
            sections.append(f"USER'S FREQUENT APPS: {listed}")

        return "\n".join(sections) + ("\n" if sections else "")
