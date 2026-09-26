from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog


logger = structlog.get_logger("retention")


class RetentionManager:
    """Periodically remove expired public state without racing active sessions."""

    def __init__(
        self,
        *,
        repository: Any,
        postgres_pool: Any | None,
        retention_days: int,
        cleanup_interval_seconds: float,
    ) -> None:
        self.repository = repository
        self.postgres_pool = postgres_pool
        self.retention_days = retention_days
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._active: set[str] = set()
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self.retention_days > 0

    @asynccontextmanager
    async def protect(self, key: str) -> AsyncIterator[None]:
        async with self._lock:
            self._active.add(key)
        try:
            yield
        finally:
            async with self._lock:
                self._active.discard(key)

    def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(
                self._run_loop(),
                name="public-data-retention",
            )

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def run_once(self) -> dict[str, int]:
        if not self.enabled:
            return {"chat_records": 0, "checkpoint_threads": 0}
        before = datetime.now(UTC) - timedelta(days=self.retention_days)
        async with self._lock:
            protected_chats = {
                key.removeprefix("chat:")
                for key in self._active
                if key.startswith("chat:")
            }
            protected_threads = {
                key.removeprefix("agent:")
                for key in self._active
                if key.startswith("agent:")
            }
            chat_records = 0
            cleanup = getattr(self.repository, "cleanup_expired", None)
            if cleanup is not None:
                chat_records = int(
                    await cleanup(
                        before=before,
                        protected_chat_ids=protected_chats,
                    )
                )
            checkpoint_threads = await self._cleanup_checkpoints(
                before,
                protected_threads,
            )
        logger.info(
            "retention.cleanup_completed",
            chat_records=chat_records,
            checkpoint_threads=checkpoint_threads,
        )
        return {
            "chat_records": chat_records,
            "checkpoint_threads": checkpoint_threads,
        }

    async def _cleanup_checkpoints(
        self,
        before: datetime,
        protected_threads: set[str],
    ) -> int:
        if self.postgres_pool is None:
            return 0
        async with self.postgres_pool.acquire(timeout=5.0) as connection:
            async with connection.transaction():
                rows = await connection.fetch(
                    """
                    SELECT thread_id
                    FROM checkpoints
                    WHERE NULLIF(checkpoint->>'ts', '') IS NOT NULL
                    GROUP BY thread_id
                    HAVING MAX((checkpoint->>'ts')::timestamptz) < $1
                    """,
                    before,
                )
                expired = [
                    str(row["thread_id"])
                    for row in rows
                    if str(row["thread_id"]) not in protected_threads
                ]
                if not expired:
                    return 0
                for table in (
                    "checkpoint_writes",
                    "checkpoint_blobs",
                    "checkpoints",
                ):
                    await connection.execute(
                        f"DELETE FROM {table} WHERE thread_id = ANY($1::text[])",
                        expired,
                    )
        return len(expired)

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "retention.cleanup_failed",
                    error_type=type(exc).__name__,
                )
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.cleanup_interval_seconds,
                )
            except TimeoutError:
                pass


@asynccontextmanager
async def protect_retained_state(
    request: Any,
    key: str,
) -> AsyncIterator[None]:
    manager = getattr(request.app.state, "retention_manager", None)
    if manager is None:
        yield
        return
    async with manager.protect(key):
        yield
