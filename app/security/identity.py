from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network, ip_address
import secrets
import time
from typing import Any, Iterable, Mapping

from starlette.requests import Request
from starlette.types import Scope

from app.security.tokens import is_usable_secret, secret_value


PUBLIC_AGENT_SESSION_COOKIE = "diploma_public_agent_session"
PUBLIC_AGENT_SESSION_PATH = "/agent"
DEFAULT_PUBLIC_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
_TOKEN_VERSION = "v1"
_SESSION_ID_BYTES = 32
_CLOCK_SKEW_SECONDS = 60
_PROCESS_SESSION_SECRET = secrets.token_bytes(32)


@dataclass(frozen=True)
class PublicAgentIdentity:
    """Server-controlled public principal and an optional refreshed cookie."""

    principal_id: str
    cookie_value: str | None = None


def resolve_client_ip(
    scope: Scope,
    trusted_proxy_networks: Iterable[IPv4Network | IPv6Network],
) -> str:
    """Resolve a rate-limit/audit IP without trusting arbitrary forwarded headers."""

    client = scope.get("client")
    peer = str(client[0]) if client else "unknown"
    try:
        peer_address = ip_address(peer)
    except ValueError:
        return peer

    if not any(peer_address in network for network in trusted_proxy_networks):
        return peer

    headers = {
        key.lower(): value
        for key, value in scope.get("headers", [])
    }
    forwarded_for = headers.get(b"x-forwarded-for", b"").decode(
        "latin-1",
        errors="ignore",
    )
    candidate = forwarded_for.split(",", 1)[0].strip()
    try:
        return str(ip_address(candidate))
    except ValueError:
        return peer


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(
        value + padding,
        altchars=b"-_",
        validate=True,
    )


def _session_secret(settings: Any) -> bytes:
    configured = getattr(settings, "public_session_secret", None)
    if is_usable_secret(configured):
        return secret_value(configured).encode("utf-8")

    environment = str(getattr(settings, "environment", "dev")).lower()
    if environment in {"prod", "production", "public"}:
        raise RuntimeError("Public session signing secret is not configured")
    return _PROCESS_SESSION_SECRET


def _session_ttl(settings: Any) -> int:
    return int(
        getattr(
            settings,
            "public_session_ttl_seconds",
            DEFAULT_PUBLIC_SESSION_TTL_SECONDS,
        )
    )


def _signature(payload: str, secret: bytes) -> str:
    return _b64encode(
        hmac.new(secret, payload.encode("ascii"), hashlib.sha256).digest()
    )


def _principal_id(session_id: bytes) -> str:
    return hashlib.sha256(session_id).hexdigest()[:32]


def issue_public_agent_identity(
    settings: Any,
    *,
    now: int | None = None,
) -> PublicAgentIdentity:
    issued_at = int(time.time() if now is None else now)
    session_id = secrets.token_bytes(_SESSION_ID_BYTES)
    encoded_id = _b64encode(session_id)
    payload = f"{_TOKEN_VERSION}.{issued_at}.{encoded_id}"
    token = f"{payload}.{_signature(payload, _session_secret(settings))}"
    return PublicAgentIdentity(
        principal_id=_principal_id(session_id),
        cookie_value=token,
    )


def verify_public_agent_cookie(
    token: str,
    settings: Any,
    *,
    now: int | None = None,
) -> PublicAgentIdentity | None:
    try:
        version, issued_raw, encoded_id, supplied_signature = token.split(".")
        if version != _TOKEN_VERSION:
            return None
        issued_at = int(issued_raw)
        current_time = int(time.time() if now is None else now)
        age = current_time - issued_at
        if age < -_CLOCK_SKEW_SECONDS or age > _session_ttl(settings):
            return None
        session_id = _b64decode(encoded_id)
        if len(session_id) != _SESSION_ID_BYTES:
            return None
        payload = f"{version}.{issued_at}.{encoded_id}"
        expected_signature = _signature(payload, _session_secret(settings))
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
    except (binascii.Error, TypeError, ValueError):
        return None

    return PublicAgentIdentity(principal_id=_principal_id(session_id))


def resolve_public_agent_identity(
    request: Request,
    settings: Any,
) -> PublicAgentIdentity:
    token = request.cookies.get(PUBLIC_AGENT_SESSION_COOKIE)
    if token:
        identity = verify_public_agent_cookie(token, settings)
        if identity is not None:
            return identity
    return issue_public_agent_identity(settings)


def public_session_cookie_options(settings: Any) -> Mapping[str, Any]:
    environment = str(getattr(settings, "environment", "dev")).lower()
    return {
        "key": PUBLIC_AGENT_SESSION_COOKIE,
        "max_age": _session_ttl(settings),
        "path": PUBLIC_AGENT_SESSION_PATH,
        "httponly": True,
        "secure": environment in {"prod", "production", "public"},
        "samesite": "lax",
    }
