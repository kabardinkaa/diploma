from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
import time

from fastapi import Request

from app.core.exceptions import (
    PublicGenerationDisabledError,
    PublicQuotaExceededError,
)


class PublicGenerationBudget:
    """Atomic process-local request budget shared by public generation routes."""

    def __init__(
        self,
        *,
        enabled: bool,
        requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self.requests = requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._attempts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def consume(self) -> None:
        if not self.enabled:
            raise PublicGenerationDisabledError()
        if self.requests == 0:
            return

        async with self._lock:
            now = self._clock()
            cutoff = now - self.window_seconds
            while self._attempts and self._attempts[0] <= cutoff:
                self._attempts.popleft()
            if len(self._attempts) >= self.requests:
                raise PublicQuotaExceededError()
            self._attempts.append(now)

    @property
    def used(self) -> int:
        return len(self._attempts)


_UNLIMITED_BUDGET = PublicGenerationBudget(
    enabled=True,
    requests=0,
    window_seconds=24 * 60 * 60,
)


def get_public_generation_budget(request: Request) -> PublicGenerationBudget:
    return getattr(request.app.state, "public_generation_budget", _UNLIMITED_BUDGET)
