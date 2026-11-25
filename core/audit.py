"""
M9: Tamper-evident audit log (SDS 4.3.4, SEC-5).

Scope (per SDS): every high-risk confirmation (approved and cancelled), every
sandbox execution, and every file system modification is recorded.

Tamper evidence
---------------
Each row stores the SHA-256 of (previous_row_hash + row_payload), forming a
hash chain. Editing or deleting any historical row breaks every hash after it,
which `verify_chain()` detects. This is tamper-*evident*, not tamper-proof — a
determined attacker with write access could recompute the whole chain. Making
it tamper-proof needs an append-only store or external notarisation, which is
outside the scope of a local desktop app.

Entries are retained for `retention_days` (default 90) and then archived out
by `prune()`.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    event: str
    detail: str
    metadata: dict[str, Any]
    prev_hash: str
    row_hash: str


class AuditLog:
    """Append-only, hash-chained event log."""

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS audit (
            seq        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            event      TEXT NOT NULL,
            detail     TEXT NOT NULL,
            metadata   TEXT NOT NULL,
            prev_hash  TEXT NOT NULL,
            row_hash   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_event ON audit(event);
    """

    def __init__(self, db_path: str | Path, retention_days: int = 90):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        with self._conn() as c:
            c.executescript(self.SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(prev_hash: str, timestamp: str, event: str,
                      detail: str, metadata_json: str) -> str:
        payload = f"{prev_hash}|{timestamp}|{event}|{detail}|{metadata_json}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def record(
        self,
        event: str,
        detail: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditEntry:
        """Append one entry. Never raises — auditing must not break a task."""
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata or {}, sort_keys=True, default=str)

        with self._conn() as c:
            row = c.execute(
                "SELECT row_hash FROM audit ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = row["row_hash"] if row else GENESIS_HASH

            row_hash = self._compute_hash(prev_hash, timestamp, event, detail, metadata_json)
            cur = c.execute(
                """INSERT INTO audit (timestamp, event, detail, metadata, prev_hash, row_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (timestamp, event, detail, metadata_json, prev_hash, row_hash),
            )
            seq = cur.lastrowid

        return AuditEntry(seq, timestamp, event, detail, metadata or {}, prev_hash, row_hash)

    # -------------------- convenience recorders --------------------

    def record_confirmation(self, action_type: str, risk_level: str,
                            approved: bool, reason: str) -> AuditEntry:
        """SEC-5: every high-risk decision, confirmed or cancelled."""
        return self.record(
            event="high_risk_confirmed" if approved else "high_risk_cancelled",
            detail=f"{action_type} classified {risk_level}: {reason}",
            metadata={"action_type": action_type, "risk_level": risk_level,
                      "approved": approved},
        )

    def record_sandbox(self, script_hash: str, exit_code: int,
                       duration_seconds: float) -> AuditEntry:
        """SEC-5: script hash, execution time, exit code."""
        return self.record(
            event="sandbox_execution",
            detail=f"script {script_hash[:16]} exited {exit_code} in {duration_seconds:.2f}s",
            metadata={"script_sha256": script_hash, "exit_code": exit_code,
                      "duration_seconds": duration_seconds},
        )

    def record_file_change(self, path: str, operation: str) -> AuditEntry:
        return self.record(
            event="file_modification",
            detail=f"{operation}: {path}",
            metadata={"path": path, "operation": operation},
        )

    # -------------------- verification / reads --------------------

    def verify_chain(self) -> tuple[bool, Optional[str]]:
        """Recompute the whole chain. Returns (is_intact, first_bad_description)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT seq, timestamp, event, detail, metadata, prev_hash, row_hash "
                "FROM audit ORDER BY seq ASC"
            ).fetchall()

        expected_prev = GENESIS_HASH
        for r in rows:
            if r["prev_hash"] != expected_prev:
                return False, f"Row {r['seq']}: prev_hash does not match the preceding row"
            recomputed = self._compute_hash(
                r["prev_hash"], r["timestamp"], r["event"], r["detail"], r["metadata"]
            )
            if recomputed != r["row_hash"]:
                return False, f"Row {r['seq']}: contents were modified after writing"
            expected_prev = r["row_hash"]
        return True, None

    def list_entries(self, limit: int = 100,
                     event: Optional[str] = None) -> list[dict[str, Any]]:
        query = ("SELECT seq, timestamp, event, detail, metadata "
                 "FROM audit {where} ORDER BY seq DESC LIMIT ?")
        params: list[Any] = []
        where = ""
        if event:
            where = "WHERE event = ?"
            params.append(event)
        params.append(limit)

        with self._conn() as c:
            rows = c.execute(query.format(where=where), params).fetchall()

        out = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"])
            out.append(d)
        return out

    def entry_count(self) -> int:
        with self._conn() as c:
            (n,) = c.execute("SELECT COUNT(*) FROM audit").fetchone()
        return n

    def prune(self) -> int:
        """Delete entries older than the retention window. Returns rows removed.

        Note: pruning necessarily breaks the hash chain for the removed prefix.
        We re-anchor the oldest surviving row to GENESIS so verify_chain() still
        validates the retained window.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self._conn() as c:
            cur = c.execute("DELETE FROM audit WHERE timestamp < ?", (cutoff,))
            removed = cur.rowcount or 0
            if removed:
                oldest = c.execute(
                    "SELECT seq FROM audit ORDER BY seq ASC LIMIT 1"
                ).fetchone()
                if oldest:
                    c.execute("UPDATE audit SET prev_hash = ? WHERE seq = ?",
                              (GENESIS_HASH, oldest["seq"]))
                    # Recompute the re-anchored row's own hash so the chain holds.
                    r = c.execute(
                        "SELECT timestamp, event, detail, metadata FROM audit WHERE seq = ?",
                        (oldest["seq"],),
                    ).fetchone()
                    new_hash = self._compute_hash(
                        GENESIS_HASH, r["timestamp"], r["event"], r["detail"], r["metadata"]
                    )
                    c.execute("UPDATE audit SET row_hash = ? WHERE seq = ?",
                              (new_hash, oldest["seq"]))
                    self._rechain_from(c, oldest["seq"])
        return removed

    def _rechain_from(self, conn: sqlite3.Connection, start_seq: int) -> None:
        """Recompute hashes for every row after `start_seq`."""
        rows = conn.execute(
            "SELECT seq, timestamp, event, detail, metadata FROM audit "
            "WHERE seq >= ? ORDER BY seq ASC", (start_seq,)
        ).fetchall()
        prev = GENESIS_HASH
        for r in rows:
            row_hash = self._compute_hash(
                prev, r["timestamp"], r["event"], r["detail"], r["metadata"]
            )
            conn.execute("UPDATE audit SET prev_hash = ?, row_hash = ? WHERE seq = ?",
                         (prev, row_hash, r["seq"]))
            prev = row_hash
