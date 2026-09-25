"""Export or restore one Qdrant collection snapshot from inside the app image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import httpx


def _client() -> tuple[httpx.Client, str]:
    base_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    headers = {"api-key": api_key} if api_key else {}
    return httpx.Client(headers=headers, timeout=120.0, trust_env=False), base_url


def export_snapshot(collection: str, output: Path) -> None:
    client, base_url = _client()
    with client:
        response = client.post(f"{base_url}/collections/{collection}/snapshots")
        response.raise_for_status()
        snapshot_name = response.json()["result"]["name"]
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with client.stream(
                "GET",
                f"{base_url}/collections/{collection}/snapshots/{snapshot_name}",
            ) as download:
                download.raise_for_status()
                with output.open("wb") as stream:
                    for block in download.iter_bytes():
                        stream.write(block)
        finally:
            client.delete(
                f"{base_url}/collections/{collection}/snapshots/{snapshot_name}"
            ).raise_for_status()


def restore_snapshot(collection: str, snapshot: Path) -> None:
    client, base_url = _client()
    with snapshot.open("rb") as stream, client:
        response = client.post(
            f"{base_url}/collections/{collection}/snapshots/upload",
            params={"priority": "snapshot"},
            files={"snapshot": (snapshot.name, stream, "application/octet-stream")},
        )
        response.raise_for_status()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "restore"):
        operation = subparsers.add_parser(command)
        operation.add_argument("--collection", default="corporate_rag")
        operation.add_argument("--path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.collection):
        raise SystemExit("Collection name contains unsupported characters")
    if args.command == "export":
        export_snapshot(args.collection, args.path)
    else:
        restore_snapshot(args.collection, args.path)


if __name__ == "__main__":
    main()
