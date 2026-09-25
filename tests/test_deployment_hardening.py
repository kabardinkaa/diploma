from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.core.config import LLMSettings, Settings
from app.security.rate_limit import PublicRateLimitMiddleware
from scripts import deployment_backup
from scripts.validate_production_config import validate_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _public_settings(**overrides) -> Settings:
    values = {
        "APP_ENV": "public",
        "PUBLIC_DOMAIN": "demo.example.com",
        "PUBLIC_PROXY_IP": "172.31.250.10",
        "TRUSTED_PROXY_CIDRS": "172.31.250.10/32",
        "ADMIN_TOKEN": "real-admin-secret",
        "INTERNAL_TOKEN": "real-internal-secret",
        "QDRANT_API_KEY": "real-qdrant-secret",
        "llm": LLMSettings(
            _env_file=None,
            OPENROUTER_API_KEY="real-provider-secret",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
        ),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _limiter(*, trusted: str = "") -> PublicRateLimitMiddleware:
    async def app(scope, receive, send):
        return None

    return PublicRateLimitMiddleware(
        app,
        requests=10,
        window_seconds=60,
        max_concurrent=2,
        paths=set(),
        trusted_proxy_cidrs=trusted,
    )


def _scope(peer: str, forwarded_for: str) -> dict:
    return {
        "type": "http",
        "client": (peer, 12345),
        "headers": [(b"x-forwarded-for", forwarded_for.encode("ascii"))],
    }


def test_direct_client_cannot_spoof_forwarded_ip() -> None:
    limiter = _limiter(trusted="172.31.250.2/32")

    assert limiter._client_id(_scope("198.51.100.20", "203.0.113.7")) == (
        "198.51.100.20"
    )


def test_trusted_proxy_supplies_actual_client_ip() -> None:
    limiter = _limiter(trusted="172.31.250.2/32")

    assert limiter._client_id(_scope("172.31.250.2", "203.0.113.7")) == (
        "203.0.113.7"
    )


def test_invalid_forwarded_ip_falls_back_to_proxy_peer() -> None:
    limiter = _limiter(trusted="172.31.250.2/32")

    assert limiter._client_id(_scope("172.31.250.2", "not-an-ip")) == (
        "172.31.250.2"
    )


def test_public_configuration_accepts_complete_safe_contract() -> None:
    settings = _public_settings()

    assert settings.public_domain == "demo.example.com"
    assert str(settings.trusted_proxy_networks[0]) == "172.31.250.10/32"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"PUBLIC_DOMAIN": "change-me.example.com"}, "PUBLIC_DOMAIN"),
        ({"TRUSTED_PROXY_CIDRS": ""}, "TRUSTED_PROXY_CIDRS"),
        ({"TRUSTED_PROXY_CIDRS": "0.0.0.0/0"}, "entire Internet"),
        ({"PUBLIC_PROXY_IP": "172.31.250.3"}, "PUBLIC_PROXY_IP"),
        ({"QDRANT_API_KEY": "change-me-qdrant"}, "QDRANT_API_KEY"),
        (
            {
                "llm": LLMSettings(
                    _env_file=None,
                    OPENROUTER_API_KEY="change-me-provider",
                )
            },
            "OPENAI_API_KEY or OPENROUTER_API_KEY",
        ),
    ],
)
def test_public_configuration_rejects_unsafe_values(
    override: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _public_settings(**override)


def test_production_preflight_rejects_placeholder_database_password() -> None:
    values = {
        "PUBLIC_DOMAIN": "demo.example.com",
        "PUBLIC_PROXY_SUBNET": "172.31.250.0/24",
        "PUBLIC_PROXY_IP": "172.31.250.10",
        "TRUSTED_PROXY_CIDRS": "172.31.250.10/32",
        "ADMIN_TOKEN": "real-admin-secret",
        "INTERNAL_TOKEN": "real-internal-secret",
        "POSTGRES_PASSWORD": "change-me-database-password",
        "DATABASE_URL": "postgresql://postgres:secret@postgres:5432/diploma",
        "QDRANT_API_KEY": "real-qdrant-secret",
        "OPENROUTER_API_KEY": "real-provider-secret",
    }

    with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
        validate_values(values)


class _ComposeLoader(yaml.SafeLoader):
    pass


def _compose_sequence(loader, node):
    return loader.construct_sequence(node)


_ComposeLoader.add_constructor("!reset", _compose_sequence)
_ComposeLoader.add_constructor("!override", _compose_sequence)


def test_production_compose_publishes_only_proxy() -> None:
    base = yaml.load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"),
        Loader=_ComposeLoader,
    )["services"]
    services = yaml.load(
        (PROJECT_ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8"),
        Loader=_ComposeLoader,
    )["services"]

    assert services["app"]["ports"] == []
    assert services["qdrant"]["ports"] == []
    assert services["phoenix"]["ports"] == []
    assert "ports" not in base["postgres"]
    assert "ports" not in base["bot"]
    assert "ports" not in base["ingest"]
    assert services["proxy"]["ports"] == ["80:80", "443:443", "443:443/udp"]


def test_proxy_contract_is_secure_and_sse_compatible() -> None:
    config = (PROJECT_ROOT / "deploy" / "Caddyfile").read_text(encoding="utf-8")

    assert "{$PUBLIC_DOMAIN:localhost}" in config
    assert "reverse_proxy app:8000" in config
    assert "flush_interval -1" in config
    assert "write 0s" in config
    assert "response_header_timeout 180s" in config
    assert "max_size {$PROXY_MAX_REQUEST_BODY:12MB}" in config
    assert "Strict-Transport-Security" in config
    assert "X-Content-Type-Options" in config
    assert "X-Frame-Options" in config
    assert "qdrant:" not in config
    assert "postgres:" not in config
    assert "phoenix:" not in config


def test_backup_manifest_validation_and_secret_exclusion(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    (backup_dir / "postgres.dump").write_bytes(b"safe-test-dump")
    deployment_backup._write_manifest(backup_dir)

    deployment_backup.validate_backup(backup_dir)
    manifest = json.loads((backup_dir / "manifest.json").read_text("utf-8"))
    assert ".env" in manifest["excludes"]
    assert ".env.production" in manifest["excludes"]


def test_backup_and_restore_cli_support_non_destructive_dry_run() -> None:
    parser = deployment_backup.build_parser()

    backup = parser.parse_args(["backup", "--dry-run"])
    restore = parser.parse_args(
        ["restore", "--from", "backups/example", "--dry-run"]
    )
    assert backup.dry_run is True
    assert restore.dry_run is True
    assert restore.confirm_restore is False
