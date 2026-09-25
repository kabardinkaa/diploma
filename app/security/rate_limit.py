from __future__ import annotations

import asyncio
from ipaddress import ip_address, ip_network
import math
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    retry_after: int = 0
    reason: str = ""


class InMemoryRequestLimiter:
    """Single-process sliding-window and concurrency guard for a demo service."""

    def __init__(
        self,
        *,
        requests: int,
        window_seconds: float,
        max_concurrent: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self.max_concurrent = max_concurrent
        self._clock = clock
        self._requests_by_client: dict[str, deque[float]] = defaultdict(deque)
        self._active_requests = 0
        self._operations = 0
        self._lock = asyncio.Lock()

    @property
    def active_requests(self) -> int:
        return self._active_requests

    async def acquire(self, client_id: str) -> LimitDecision:
        now = self._clock()
        async with self._lock:
            self._operations += 1
            if self._operations % 256 == 0:
                cutoff = now - self.window_seconds
                for known_client, known_timestamps in list(
                    self._requests_by_client.items()
                ):
                    while known_timestamps and known_timestamps[0] <= cutoff:
                        known_timestamps.popleft()
                    if not known_timestamps:
                        self._requests_by_client.pop(known_client, None)

            timestamps = self._requests_by_client[client_id]
            cutoff = now - self.window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.requests:
                retry_after = max(
                    1,
                    math.ceil(timestamps[0] + self.window_seconds - now),
                )
                return LimitDecision(False, retry_after, "rate_limit_exceeded")

            if self._active_requests >= self.max_concurrent:
                return LimitDecision(False, 1, "concurrency_limit_exceeded")

            timestamps.append(now)
            self._active_requests += 1
            return LimitDecision(True)

    async def release(self) -> None:
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)


class PublicRateLimitMiddleware:
    """Hold a concurrency slot until the complete ASGI response body is sent."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests: int,
        window_seconds: float,
        max_concurrent: int,
        paths: set[tuple[str, str]],
        enabled: bool = True,
        trusted_proxy_cidrs: str = "",
    ) -> None:
        self.app = app
        self.paths = paths
        self.enabled = enabled
        self.trusted_proxy_networks = tuple(
            ip_network(value.strip(), strict=False)
            for value in trusted_proxy_cidrs.split(",")
            if value.strip()
        )
        self.limiter = InMemoryRequestLimiter(
            requests=requests,
            window_seconds=window_seconds,
            max_concurrent=max_concurrent,
        )

    def _client_id(self, scope: Scope) -> str:
        client = scope.get("client")
        peer = str(client[0]) if client else "unknown"
        try:
            peer_address = ip_address(peer)
        except ValueError:
            return peer

        if not any(
            peer_address in network for network in self.trusted_proxy_networks
        ):
            return peer

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        forwarded_for = headers.get(b"x-forwarded-for", b"").decode(
            "latin-1", errors="ignore"
        )
        candidate = forwarded_for.split(",", 1)[0].strip()
        try:
            return str(ip_address(candidate))
        except ValueError:
            return peer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        route = (str(scope.get("method", "")).upper(), str(scope.get("path", "")))
        if not self.enabled or scope["type"] != "http" or route not in self.paths:
            await self.app(scope, receive, send)
            return

        decision = await self.limiter.acquire(self._client_id(scope))
        if not decision.allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": decision.reason,
                        "message": "Too many expensive requests. Try again later.",
                    }
                },
                headers={"Retry-After": str(decision.retry_after)},
            )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            await self.limiter.release()
