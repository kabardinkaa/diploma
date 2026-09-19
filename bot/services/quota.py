from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import UTC, date, datetime


class BotQuotaExceeded(RuntimeError):
    pass


class BotUserQuota:
    """Process-local Telegram quota suitable for the single bot demo instance."""

    def __init__(
        self,
        *,
        requests: int,
        window_seconds: float,
        daily_quota: int,
    ) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self.daily_quota = daily_quota
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._daily: dict[str, tuple[date, int]] = {}
        self._lock = asyncio.Lock()

    async def check(self, user_id: str) -> None:
        now = time.monotonic()
        today = datetime.now(UTC).date()
        async with self._lock:
            recent = self._recent[user_id]
            cutoff = now - self.window_seconds
            while recent and recent[0] <= cutoff:
                recent.popleft()
            if len(recent) >= self.requests:
                raise BotQuotaExceeded("bot_rate_limit")

            quota_day, used = self._daily.get(user_id, (today, 0))
            if quota_day != today:
                quota_day, used = today, 0
            if used >= self.daily_quota:
                raise BotQuotaExceeded("bot_daily_quota")

            recent.append(now)
            self._daily[user_id] = (quota_day, used + 1)
