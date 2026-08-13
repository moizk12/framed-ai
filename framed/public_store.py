"""Durable public-beta persistence, isolated from research state."""

from __future__ import annotations

import os
import uuid
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Protocol


class PublicRepository(Protocol):
    """Persistence contract used by the public application service."""

    def record_analysis(self, analysis_id: str) -> None: ...

    def has_analysis(self, analysis_id: str) -> bool: ...

    def record_feedback(self, feedback: dict) -> None: ...

    def feedback_for(self, analysis_id: str) -> list[dict]: ...


class PublicPersistenceUnavailable(RuntimeError):
    """The public persistence authority could not complete an operation."""


class PublicBetaStore:
    """Public analysis/feedback service independent of database technology."""

    def __init__(self, repository: PublicRepository) -> None:
        self._repository = repository

    def record_analysis(self, analysis_id: str) -> None:
        try:
            self._repository.record_analysis(analysis_id)
        except Exception as exc:
            raise PublicPersistenceUnavailable("record_analysis_failed") from exc

    def has_analysis(self, analysis_id: str) -> bool:
        try:
            return self._repository.has_analysis(analysis_id)
        except Exception as exc:
            raise PublicPersistenceUnavailable("analysis_lookup_failed") from exc

    def record_feedback(self, feedback: dict) -> None:
        try:
            self._repository.record_feedback(feedback)
        except Exception as exc:
            raise PublicPersistenceUnavailable("record_feedback_failed") from exc

    def feedback_for(self, analysis_id: str) -> list[dict]:
        try:
            return self._repository.feedback_for(analysis_id)
        except Exception as exc:
            raise PublicPersistenceUnavailable("feedback_lookup_failed") from exc


class MemoryPublicRepository:
    """Isolated test repository; inject one instance across app recreations."""

    def __init__(self) -> None:
        self._analysis_ids: set[str] = set()
        self._feedback: list[dict] = []
        self._lock = Lock()

    def record_analysis(self, analysis_id: str) -> None:
        with self._lock:
            self._analysis_ids.add(analysis_id)

    def has_analysis(self, analysis_id: str) -> bool:
        with self._lock:
            return analysis_id in self._analysis_ids

    def record_feedback(self, feedback: dict) -> None:
        with self._lock:
            self._feedback.append(deepcopy(feedback))

    def feedback_for(self, analysis_id: str) -> list[dict]:
        with self._lock:
            return [deepcopy(item) for item in self._feedback if item["analysis_id"] == analysis_id]


class PostgresPublicRepository:
    """Small psycopg repository for durable Track A state."""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required for public persistence")
        self._database_url = database_url

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError("psycopg is required for PostgreSQL public persistence") from exc
        return psycopg.connect(self._database_url)

    def bootstrap(self) -> None:
        """Apply pending public-only SQL migrations transactionally."""
        migrations_dir = Path(__file__).with_name("public_migrations")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS public_schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                for migration in sorted(migrations_dir.glob("*.sql")):
                    cursor.execute(
                        "SELECT EXISTS (SELECT 1 FROM public_schema_migrations WHERE version = %s)",
                        (migration.name,),
                    )
                    if cursor.fetchone()[0]:
                        continue
                    cursor.execute(migration.read_text(encoding="utf-8"))
                    cursor.execute(
                        "INSERT INTO public_schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                        (migration.name,),
                    )

    def record_analysis(self, analysis_id: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO public_analyses (analysis_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (analysis_id,),
                )

    def has_analysis(self, analysis_id: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM public_analyses WHERE analysis_id = %s)",
                    (analysis_id,),
                )
                return bool(cursor.fetchone()[0])

    def record_feedback(self, feedback: dict) -> None:
        feedback_id = f"fdb-{uuid.uuid4().hex}"
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO public_feedback
                        (feedback_id, analysis_id, useful, comment, recorded_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        feedback_id,
                        feedback["analysis_id"],
                        feedback["useful"],
                        feedback["comment"],
                        feedback["recorded_at"],
                    ),
                )

    def feedback_for(self, analysis_id: str) -> list[dict]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT analysis_id, useful, comment, recorded_at
                    FROM public_feedback
                    WHERE analysis_id = %s
                    ORDER BY recorded_at, feedback_id
                    """,
                    (analysis_id,),
                )
                return [
                    {
                        "analysis_id": row[0],
                        "useful": row[1],
                        "comment": row[2],
                        "recorded_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                    }
                    for row in cursor.fetchall()
                ]


def build_public_repository(config: dict) -> PublicRepository:
    """Build the configured repository without coupling routes to PostgreSQL."""
    injected = config.get("PUBLIC_REPOSITORY")
    if injected is not None:
        return injected
    if config.get("TESTING"):
        return MemoryPublicRepository()

    database_url = (config.get("DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required outside TESTING for durable public persistence")
    repository = PostgresPublicRepository(database_url)
    if config.get("PUBLIC_AUTO_MIGRATE", True):
        repository.bootstrap()
    return repository
