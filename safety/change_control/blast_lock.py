"""Lease based blast-radius reservations with a shared SQLite default backend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable, Protocol


@dataclass(frozen=True)
class BlastLockLease:
    owner: str
    resources: frozenset[str]
    expires_at: datetime


class BlastLockBackend(Protocol):
    """Storage contract for process-shared or deployment-specific lock backends."""

    def acquire(
        self, owner: str, resources: frozenset[str], expires_at: datetime
    ) -> None: ...

    def release(self, owner: str) -> None: ...


class SQLiteBlastLockBackend:
    """SQLite lock table; transactions serialize competing workers on one host."""

    def __init__(self, database_path: str | os.PathLike[str], timeout: float = 10.0) -> None:
        self.database_path = str(Path(database_path).expanduser().resolve())
        self.timeout = timeout
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS blast_radius_locks ("
                "resource TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS blast_radius_locks_owner_idx "
                "ON blast_radius_locks(owner)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path, timeout=self.timeout, isolation_level=None
        )
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return connection

    def acquire(
        self, owner: str, resources: frozenset[str], expires_at: datetime
    ) -> None:
        resource_names = sorted(resources)
        expiration = expires_at.timestamp()
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM blast_radius_locks WHERE expires_at <= ?",
                    (datetime.now(timezone.utc).timestamp(),),
                )
                placeholders = ",".join("?" for _ in resource_names)
                collisions = connection.execute(
                    f"SELECT DISTINCT owner FROM blast_radius_locks "
                    f"WHERE resource IN ({placeholders}) AND owner <> ?",
                    (*resource_names, owner),
                ).fetchall()
                if collisions:
                    connection.rollback()
                    owners = ", ".join(sorted(row[0] for row in collisions))
                    raise RuntimeError(
                        "BLAST_RADIUS_LOCKED: overlapping change reservation held by " + owners
                    )
                connection.execute("DELETE FROM blast_radius_locks WHERE owner = ?", (owner,))
                connection.executemany(
                    "INSERT INTO blast_radius_locks(resource, owner, expires_at) VALUES (?, ?, ?)",
                    [(resource, owner, expiration) for resource in resource_names],
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise RuntimeError("BLAST_RADIUS_LOCK_BACKEND_FAILURE: reservation was not acquired") from exc

    def release(self, owner: str) -> None:
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM blast_radius_locks WHERE owner = ?", (owner,))
                connection.commit()
        except sqlite3.Error as exc:
            raise RuntimeError("BLAST_RADIUS_LOCK_BACKEND_FAILURE: reservation release failed") from exc


class BlastRadiusLock:
    """Atomically reserve all impacted devices through a shared backend.

    The default SQLite database is shared by workers on this host. Pass a backend
    implementing ``BlastLockBackend`` to use PostgreSQL advisory locks, Redis, etc.
    """

    def __init__(
        self,
        lease_seconds: int = 4 * 60 * 60,
        *,
        backend: BlastLockBackend | None = None,
        database_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.lease_seconds = lease_seconds
        if backend is not None and database_path is not None:
            raise ValueError("Pass either backend or database_path, not both")
        if backend is None:
            default_path = Path(tempfile.gettempdir()) / "veronica" / "safety-locks.sqlite3"
            path = database_path or os.environ.get("VERONICA_SAFETY_LOCK_DB") or default_path
            backend = SQLiteBlastLockBackend(path)
        self.backend = backend

    def acquire(self, owner: str, resources: Iterable[str]) -> BlastLockLease:
        requested = frozenset(str(item) for item in resources if item)
        if not owner or not requested:
            raise ValueError("A lock owner and at least one resource are required")
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.lease_seconds)
        self.backend.acquire(owner, requested, expires_at)
        return BlastLockLease(owner, requested, expires_at)

    def release(self, owner: str) -> None:
        self.backend.release(owner)

