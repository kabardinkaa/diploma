"""Validate public deployment settings without printing credential values."""

from __future__ import annotations

import argparse
import os
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from app.core.config import LLMSettings, Settings
from app.security.tokens import is_usable_secret


def validate_values(source: Mapping[str, str]) -> Settings:
    values = dict(source)
    values["APP_ENV"] = "public"
    postgres_password = values.get("POSTGRES_PASSWORD")
    if not is_usable_secret(postgres_password):
        raise ValueError(
            "Public deployment requires non-placeholder POSTGRES_PASSWORD"
        )
    database_url = values.get("DATABASE_URL", "").strip()
    if not database_url or "change-me" in database_url.lower():
        raise ValueError(
            "Public deployment requires a non-placeholder DATABASE_URL"
        )

    proxy_ip = ip_address(values.get("PUBLIC_PROXY_IP", ""))
    proxy_subnet = ip_network(values.get("PUBLIC_PROXY_SUBNET", ""), strict=False)
    if proxy_ip not in proxy_subnet:
        raise ValueError("PUBLIC_PROXY_IP must be inside PUBLIC_PROXY_SUBNET")

    llm = LLMSettings(_env_file=None, **values)
    return Settings(_env_file=None, llm=llm, **values)


def validate(env_file: Path) -> Settings:
    if not env_file.is_file():
        raise ValueError(f"Production env file does not exist: {env_file}")
    values = {
        key: value
        for key, value in dotenv_values(env_file).items()
        if value is not None
    }
    return validate_values(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.production"))
    parser.add_argument("--from-environment", action="store_true")
    args = parser.parse_args()
    try:
        settings = (
            validate_values(os.environ)
            if args.from_environment
            else validate(args.env_file)
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Production configuration invalid: {exc}\n")
    print(
        "Production configuration valid: "
        f"domain={settings.public_domain}, trusted_proxy_configured=yes"
    )


if __name__ == "__main__":
    main()
