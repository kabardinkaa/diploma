"""Cross-platform backup/restore orchestration for the public Compose stack."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPOSE_FILES = ("docker-compose.yml", "docker-compose.prod.yml")
SAFE_CONFIG_FILES = (
    ".env.production.example",
    "constraints.txt",
    "docker-compose.yml",
    "docker-compose.prod.yml",
    "deploy/Caddyfile",
)


def _compose_prefix(files: list[str], env_file: Path) -> list[str]:
    command = ["docker", "compose", "--env-file", str(env_file)]
    for file_name in files or DEFAULT_COMPOSE_FILES:
        command.extend(("-f", file_name))
    return command


def _display(command: list[str]) -> str:
    return " ".join(command)


def _run(
    command: list[str],
    *,
    dry_run: bool,
    stdin_path: Path | None = None,
    stdout_path: Path | None = None,
) -> None:
    print(f"+ {_display(command)}")
    if dry_run:
        return
    stdin = stdin_path.open("rb") if stdin_path else None
    stdout = stdout_path.open("wb") if stdout_path else None
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdin=stdin,
            stdout=stdout,
            check=True,
        )
    finally:
        if stdin:
            stdin.close()
        if stdout:
            stdout.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(backup_dir: Path) -> None:
    files = {
        path.relative_to(backup_dir).as_posix(): _sha256(path)
        for path in sorted(backup_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "format": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "includes": [
            "postgres custom-format dump",
            "corporate_rag Qdrant collection snapshot",
            "RAG docstore state",
            "production corpus",
            "non-secret deployment configuration",
        ],
        "excludes": [
            "provider credentials and tokens",
            ".env",
            ".env.production",
            "embedding model cache",
            "Phoenix traces",
            "Docker images and logs",
        ],
        "sha256": files,
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_backup(backup_dir: Path) -> None:
    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != 1:
        raise ValueError("Unsupported backup manifest format")
    for relative, expected in manifest["sha256"].items():
        path = backup_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Backup checksum mismatch: {relative}")
    forbidden = {".env", ".env.production"}
    if any(path.name in forbidden for path in backup_dir.rglob("*")):
        raise ValueError("Backup must not contain live environment secret files")


def create_backup(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.output.resolve() / timestamp
    compose = _compose_prefix(args.compose_file, args.env_file)
    if args.dry_run:
        print(f"Backup target: {backup_dir}")
    else:
        backup_dir.mkdir(parents=True, exist_ok=False)

    _run(
        compose
        + [
            "exec",
            "-T",
            "postgres",
            "pg_dump",
            "-U",
            args.postgres_user,
            "-d",
            args.postgres_db,
            "-Fc",
        ],
        dry_run=args.dry_run,
        stdout_path=backup_dir / "postgres.dump",
    )

    container_snapshot = f"/tmp/{args.collection}.snapshot"
    _run(
        compose
        + [
            "exec",
            "-T",
            "app",
            "python",
            "scripts/qdrant_snapshot.py",
            "export",
            "--collection",
            args.collection,
            "--path",
            container_snapshot,
        ],
        dry_run=args.dry_run,
    )
    _run(
        compose
        + [
            "cp",
            f"app:{container_snapshot}",
            str(backup_dir / "qdrant.snapshot"),
        ],
        dry_run=args.dry_run,
    )
    _run(
        compose + ["exec", "-T", "app", "rm", "-f", container_snapshot],
        dry_run=args.dry_run,
    )
    _run(
        compose + ["cp", "app:/app/.cache/rag", str(backup_dir / "rag-state")],
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        shutil.make_archive(
            str(backup_dir / "corpus"),
            "zip",
            root_dir=PROJECT_ROOT / "data",
        )
        config_dir = backup_dir / "config"
        for relative in SAFE_CONFIG_FILES:
            source = PROJECT_ROOT / relative
            target = config_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _write_manifest(backup_dir)
        validate_backup(backup_dir)
    return backup_dir


def restore_backup(args: argparse.Namespace) -> None:
    backup_dir = args.from_dir.resolve()
    if not args.dry_run:
        validate_backup(backup_dir)
    if not args.confirm_restore and not args.dry_run:
        raise SystemExit("Restore requires --confirm-restore")

    compose = _compose_prefix(args.compose_file, args.env_file)
    _run(
        compose
        + [
            "exec",
            "-T",
            "postgres",
            "pg_restore",
            "-U",
            args.postgres_user,
            "-d",
            args.postgres_db,
            "--clean",
            "--if-exists",
            "--no-owner",
        ],
        dry_run=args.dry_run,
        stdin_path=backup_dir / "postgres.dump",
    )

    container_snapshot = f"/tmp/{args.collection}.snapshot"
    _run(
        compose
        + [
            "cp",
            str(backup_dir / "qdrant.snapshot"),
            f"app:{container_snapshot}",
        ],
        dry_run=args.dry_run,
    )
    _run(
        compose
        + [
            "exec",
            "-T",
            "app",
            "python",
            "scripts/qdrant_snapshot.py",
            "restore",
            "--collection",
            args.collection,
            "--path",
            container_snapshot,
        ],
        dry_run=args.dry_run,
    )
    _run(
        compose
        + ["cp", str(backup_dir / "rag-state" / "."), "app:/app/.cache/rag"],
        dry_run=args.dry_run,
    )

    if args.restore_corpus:
        if args.dry_run:
            print(f"+ extract {backup_dir / 'corpus.zip'} -> {PROJECT_ROOT / 'data'}")
        else:
            with zipfile.ZipFile(backup_dir / "corpus.zip") as archive:
                target = (PROJECT_ROOT / "data").resolve()
                if any(
                    not (target / member.filename).resolve().is_relative_to(target)
                    for member in archive.infolist()
                ):
                    raise ValueError("Corpus archive contains an unsafe path")
                archive.extractall(PROJECT_ROOT / "data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("backup", "restore"):
        operation = subparsers.add_parser(command)
        operation.add_argument(
            "--env-file",
            type=Path,
            default=PROJECT_ROOT / ".env.production",
        )
        operation.add_argument("--compose-file", action="append", default=[])
        operation.add_argument("--collection", default="corporate_rag")
        operation.add_argument("--postgres-user", default="postgres")
        operation.add_argument("--postgres-db", default="diploma")
        operation.add_argument("--dry-run", action="store_true")
    backup = subparsers.choices["backup"]
    backup.add_argument("--output", type=Path, default=PROJECT_ROOT / "backups")
    restore = subparsers.choices["restore"]
    restore.add_argument("--from", dest="from_dir", type=Path, required=True)
    restore.add_argument("--confirm-restore", action="store_true")
    restore.add_argument("--restore-corpus", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "backup":
            backup_dir = create_backup(args)
            print(f"Backup ready: {backup_dir}")
        else:
            restore_backup(args)
            print("Restore completed")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Deployment backup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
