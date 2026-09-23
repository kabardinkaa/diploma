from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
import structlog

from app.core.config import Settings


logger = structlog.get_logger("readiness")
_CHECK_TIMEOUT_SECONDS = 3.0


class ReadinessService:
    def __init__(
        self,
        settings: Settings,
        qdrant_client: Any,
        *,
        postgres_pool: Any | None = None,
        postgres_connect: Callable[..., Awaitable[Any]] = asyncpg.connect,
    ) -> None:
        self.settings = settings
        self.qdrant_client = qdrant_client
        self._postgres_pool = postgres_pool
        self._postgres_connect = postgres_connect

    async def _postgres_status(self) -> dict[str, str]:
        if not self.settings.database_url:
            return {"status": "unavailable"}

        connection = None
        try:
            async with asyncio.timeout(_CHECK_TIMEOUT_SECONDS):
                if self._postgres_pool is not None:
                    async with self._postgres_pool.acquire(
                        timeout=_CHECK_TIMEOUT_SECONDS
                    ) as pooled_connection:
                        await pooled_connection.fetchval("SELECT 1")
                    return {"status": "ok"}

                connection = await self._postgres_connect(
                    dsn=self.settings.database_url,
                    timeout=_CHECK_TIMEOUT_SECONDS,
                )
                await connection.fetchval("SELECT 1")
            return {"status": "ok"}
        except Exception as exc:
            logger.warning(
                "readiness.postgres_unavailable",
                error_type=type(exc).__name__,
            )
            return {"status": "unavailable"}
        finally:
            if connection is not None:
                try:
                    async with asyncio.timeout(_CHECK_TIMEOUT_SECONDS):
                        await connection.close()
                except Exception as exc:
                    logger.warning(
                        "readiness.postgres_close_failed",
                        error_type=type(exc).__name__,
                    )

    async def _qdrant_status(self) -> tuple[dict[str, str], dict[str, str]]:
        try:
            async with asyncio.timeout(_CHECK_TIMEOUT_SECONDS):
                response = await self.qdrant_client.get_collections()
            names = {collection.name for collection in response.collections}
        except Exception as exc:
            logger.warning(
                "readiness.qdrant_unavailable",
                error_type=type(exc).__name__,
            )
            return {"status": "unavailable"}, {"status": "unavailable"}

        collection_status = (
            "ok"
            if self.settings.rag_production_collection in names
            else "missing"
        )
        return {"status": "ok"}, {"status": collection_status}

    async def check(self) -> dict[str, Any]:
        postgres, qdrant_result = await asyncio.gather(
            self._postgres_status(),
            self._qdrant_status(),
        )
        qdrant, collection = qdrant_result
        dependencies = {
            "postgres": postgres,
            "qdrant": qdrant,
            "corporate_rag": collection,
        }
        ready = all(item["status"] == "ok" for item in dependencies.values())
        return {
            "status": "ready" if ready else "not_ready",
            "dependencies": dependencies,
        }
