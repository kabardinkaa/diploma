from __future__ import annotations

import secrets
from typing import Any


PLACEHOLDER_PREFIX = "change-me"


def secret_value(value: Any | None) -> str:
    if value is None:
        return ""
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return str(value).strip()


def is_usable_secret(value: Any | None) -> bool:
    normalized = secret_value(value)
    return bool(normalized) and not normalized.lower().startswith(PLACEHOLDER_PREFIX)


def secret_matches(provided: str | None, configured: Any | None) -> bool:
    expected = secret_value(configured)
    candidate = (provided or "").strip()
    if not candidate or not is_usable_secret(expected):
        return False
    return secrets.compare_digest(candidate, expected)
