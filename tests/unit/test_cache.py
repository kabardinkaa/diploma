from app.core.cache import BoundedTTLCache


def test_cache_evicts_least_recently_used_entry() -> None:
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=60)
    cache["first"] = "1"
    cache["second"] = "2"
    assert cache["first"] == "1"

    cache["third"] = "3"

    assert cache.get("second") is None
    assert cache.get("first") == "1"
    assert cache.get("third") == "3"


def test_cache_expires_entries_by_ttl() -> None:
    now = [100.0]
    cache = BoundedTTLCache(
        max_entries=2,
        ttl_seconds=5,
        clock=lambda: now[0],
    )
    cache["key"] = "value"
    now[0] = 105.0

    assert cache.get("key") is None
    assert len(cache) == 0


def test_disabled_cache_never_stores_entries() -> None:
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=60, enabled=False)
    cache["key"] = "value"

    assert cache.get("key") is None
    assert len(cache) == 0
