from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class _CacheEntry(Generic[V]):
    value: V
    expires_at: float


class BoundedTTLCache(MutableMapping[K, V], Generic[K, V]):
    """Small process-local LRU cache with TTL and deterministic eviction."""

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._clock = clock
        self._entries: OrderedDict[K, _CacheEntry[V]] = OrderedDict()

    def __getitem__(self, key: K) -> V:
        if not self.enabled:
            raise KeyError(key)
        entry = self._entries[key]
        if entry.expires_at <= self._clock():
            del self._entries[key]
            raise KeyError(key)
        self._entries.move_to_end(key)
        return entry.value

    def __setitem__(self, key: K, value: V) -> None:
        if not self.enabled:
            return
        self._entries[key] = _CacheEntry(
            value=value,
            expires_at=self._clock() + self.ttl_seconds,
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        del self._entries[key]

    def __iter__(self) -> Iterator[K]:
        self._purge_expired()
        return iter(self._entries)

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._entries)

    def get(self, key: K, default: V | None = None) -> V | None:
        try:
            return self[key]
        except KeyError:
            return default

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for key in expired:
            del self._entries[key]
