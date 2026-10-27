"""SQLite-backed batch job queue shared by the app and the worker process.

Statuses: ``queued → running → done | failed | cancelled``. A cancel request on
a running job sets ``cancel_requested``; the worker checks it between frames.
Claims use ``BEGIN IMMEDIATE`` so two workers never take the same job.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATUSES = ("queued", "running", "done", "failed", "cancelled")
HEARTBEAT_STALE_S = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    status TEXT NOT NULL,
    spec TEXT NOT NULL,
    progress TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    worker_pid INTEGER
);
CREATE TABLE IF NOT EXISTS workers (
    pid INTEGER PRIMARY KEY,
    started_at REAL NOT NULL,
    heartbeat REAL NOT NULL,
    current_job INTEGER
);
"""


@dataclass
class Job:
    """One batch job row."""

    id: int
    created_at: float
    started_at: float | None
    finished_at: float | None
    status: str
    spec: dict[str, Any]
    progress: dict[str, Any]
    error: str | None
    cancel_requested: bool
    worker_pid: int | None


class JobQueue:
    """Small durable queue in one SQLite file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            id=int(row["id"]),
            created_at=float(row["created_at"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            status=str(row["status"]),
            spec=json.loads(row["spec"]),
            progress=json.loads(row["progress"]),
            error=row["error"],
            cancel_requested=bool(row["cancel_requested"]),
            worker_pid=row["worker_pid"],
        )

    def submit(self, spec: dict[str, Any]) -> int:
        """Queue a job; ``spec`` must be JSON-serialisable."""
        payload = json.dumps(spec, allow_nan=False)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs (created_at, status, spec) VALUES (?, 'queued', ?)",
                (time.time(), payload),
            )
            return int(cur.lastrowid or 0)

    def get(self, job_id: int) -> Job:
        """Fetch a job; raise ``KeyError`` if missing."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job(row)

    def recent(self, limit: int = 50) -> list[Job]:
        """Most recent jobs first."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._job(r) for r in rows]

    def claim_next(self, pid: int | None = None) -> Job | None:
        """Atomically move the oldest queued job to ``running``."""
        pid = os.getpid() if pid is None else pid
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, worker_pid = ? WHERE id = ?",
                (time.time(), pid, row["id"]),
            )
            conn.execute("COMMIT")
        return self.get(int(row["id"]))

    def update_progress(self, job_id: int, progress: dict[str, Any]) -> None:
        """Replace the job's progress dict."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ? WHERE id = ?",
                (json.dumps(progress, allow_nan=False), job_id),
            )

    def finish(self, job_id: int, *, status: str = "done", error: str | None = None) -> None:
        """Mark a running job ``done``, ``failed`` or ``cancelled``."""
        if status not in ("done", "failed", "cancelled"):
            raise ValueError(status)
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, time.time(), job_id),
            )

    def request_cancel(self, job_id: int) -> str:
        """Cancel a queued job now, or ask the worker to stop a running one."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(job_id)
            status = str(row["status"])
            if status == "queued":
                conn.execute(
                    "UPDATE jobs SET status = 'cancelled', finished_at = ? WHERE id = ?",
                    (time.time(), job_id),
                )
                status = "cancelled"
            elif status == "running":
                conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
            conn.execute("COMMIT")
        return status

    def cancel_requested(self, job_id: int) -> bool:
        """Return whether a cancel was requested for the job."""
        return self.get(job_id).cancel_requested

    def heartbeat(self, pid: int, current_job: int | None = None) -> None:
        """Record that worker ``pid`` is alive."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO workers (pid, started_at, heartbeat, current_job) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(pid) DO UPDATE SET heartbeat = excluded.heartbeat, current_job = excluded.current_job",
                (pid, now, now, current_job),
            )

    def remove_worker(self, pid: int) -> None:
        """Forget a worker (on clean shutdown)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM workers WHERE pid = ?", (pid,))

    def live_workers(self, stale_after: float = HEARTBEAT_STALE_S) -> list[dict[str, Any]]:
        """Workers whose heartbeat is recent."""
        cutoff = time.time() - stale_after
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM workers WHERE heartbeat >= ?", (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def recover_orphans(self, stale_after: float = HEARTBEAT_STALE_S) -> int:
        """Fail ``running`` jobs whose worker stopped heart-beating."""
        live = {w["pid"] for w in self.live_workers(stale_after)}
        orphaned = [j for j in self.recent(limit=1000) if j.status == "running" and j.worker_pid not in live]
        for job in orphaned:
            self.finish(job.id, status="failed", error="worker stopped (no heartbeat)")
        return len(orphaned)
