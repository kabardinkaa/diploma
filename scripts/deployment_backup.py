"""Cross-platform backup/restore orchestration for the public Compose stack."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPOSE_FILES = ("docker-compose.yml", "docker-compose.prod.yml")
CURRENT_PROJECT_NAME = PROJECT_ROOT.name.lower().replace(" ", "-")
SAFE_CONFIG_FILES = (
    ".env.production.example",
    "constraints.txt",
    "docker-compose.yml",
    "docker-compose.prod.yml",
    "docker-compose.restore.yml",
    "deploy/Caddyfile",
)


POSTGRES_TABLES = (
    "chats",
    "chat_messages",
    "feedback",
    "broadcast_tasks",
    "system_prompts",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
)
REQUIRED_BACKUP_FILES = {
    "postgres.dump",
    "postgres.json",
    "qdrant.snapshot",
    "qdrant.json",
    "corpus.zip",
    "rag-state/docstore.json",
    "rag-state/docstore.manifest.json",
}


def _compose_prefix(
    files: list[str],
    env_file: Path,
    project_name: str,
) -> list[str]:
    command = [
        "docker",
        "compose",
        "--project-name",
        project_name,
        "--env-file",
        str(env_file),
    ]
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


def _capture(command: list[str], *, dry_run: bool) -> str:
    print(f"+ {_display(command)}")
    if dry_run:
        return ""
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _backup_mount(backup_dir: Path, *, read_only: bool = False) -> str:
    suffix = ":ro" if read_only else ""
    return f"{backup_dir.resolve().as_posix()}:/backup{suffix}"


def _postgres_counts_sql() -> str:
    pairs = ", ".join(
        f"'{table}', (SELECT COUNT(*) FROM {table})"
        for table in POSTGRES_TABLES
    )
    return f"SELECT json_build_object({pairs})::text"


def _postgres_counts(compose: list[str], args: argparse.Namespace) -> dict[str, int]:
    output = _capture(
        compose
        + [
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            args.postgres_user,
            "-d",
            args.postgres_db,
            "-Atc",
            _postgres_counts_sql(),
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {}
    return {key: int(value) for key, value in json.loads(output).items()}


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


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
        "format": 2,
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
    if manifest.get("format") != 2:
        raise ValueError("Unsupported backup manifest format")
    checksums = manifest.get("sha256")
    if not isinstance(checksums, dict):
        raise ValueError("Backup manifest does not contain checksums")
    missing = REQUIRED_BACKUP_FILES.difference(checksums)
    if missing:
        raise ValueError(f"Backup is incomplete: {', '.join(sorted(missing))}")
    root = backup_dir.resolve()
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(checksums):
        raise ValueError("Backup contents do not match its manifest")
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Backup manifest contains an unsafe path")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Backup checksum mismatch: {relative}")
    forbidden = {".env", ".env.production"}
    if any(path.name in forbidden for path in backup_dir.rglob("*")):
        raise ValueError("Backup must not contain live environment secret files")
    postgres = _load_json(backup_dir / "postgres.json")
    if set(postgres) != set(POSTGRES_TABLES):
        raise ValueError("PostgreSQL backup metadata is incomplete")
    qdrant = _load_json(backup_dir / "qdrant.json")
    collection = qdrant.get("collection")
    if not isinstance(collection, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", collection
    ):
        raise ValueError("Qdrant backup metadata has an invalid collection")
    if not isinstance(qdrant.get("points_count"), int):
        raise ValueError("Qdrant backup metadata has no point count")


def create_backup(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.output.resolve() / timestamp
    compose = _compose_prefix(args.compose_file, args.env_file, args.project_name)
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
    if not args.dry_run:
        (backup_dir / "postgres.json").write_text(
            json.dumps(_postgres_counts(compose, args), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        _postgres_counts(compose, args)

    run_prefix = compose + [
        "run",
        "--rm",
        "--no-deps",
        "-T",
        "-v",
        _backup_mount(backup_dir),
        "app",
    ]
    _run(
        run_prefix
        + [
            "python",
            "scripts/qdrant_snapshot.py",
            "export",
            "--collection",
            args.collection,
            "--path",
            "/backup/qdrant.snapshot",
            "--metadata",
            "/backup/qdrant.json",
        ],
        dry_run=args.dry_run,
    )
    _run(
        run_prefix
        + [
            "python",
            "scripts/deployment_state.py",
            "backup",
        ],
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


def _validate_project_name(args: argparse.Namespace) -> None:
    if not args.project_name:
        raise ValueError("Restore requires an explicit --project-name")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.project_name):
        raise ValueError("Compose project name contains unsupported characters")
    if (
        args.project_name.casefold() == CURRENT_PROJECT_NAME.casefold()
        and not args.allow_current_project
    ):
        raise ValueError(
            "Refusing restore into the current project without --allow-current-project"
        )


def _validate_postgres(
    compose: list[str],
    args: argparse.Namespace,
    expected: dict,
) -> dict[str, int]:
    actual = _postgres_counts(compose, args)
    if not args.dry_run and actual != expected:
        raise ValueError("Restored PostgreSQL row counts do not match backup")
    return actual


def _qdrant_metadata(
    compose: list[str],
    args: argparse.Namespace,
    backup_dir: Path,
) -> dict:
    output = _capture(
        compose
        + [
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "-v",
            _backup_mount(backup_dir, read_only=True),
            "app",
            "python",
            "scripts/qdrant_snapshot.py",
            "inspect",
            "--collection",
            args.collection,
        ],
        dry_run=args.dry_run,
    )
    return {} if args.dry_run else json.loads(output)


def _wait_ready(
    compose: list[str],
    *,
    timeout_seconds: float,
    dry_run: bool,
) -> dict:
    command = compose + [
        "exec",
        "-T",
        "app",
        "python",
        "scripts/deployment_state.py",
        "readiness",
    ]
    print(f"+ wait up to {timeout_seconds:g}s: {_display(command)}")
    if dry_run:
        return {}
    deadline = time.monotonic() + timeout_seconds
    last_error: subprocess.CalledProcessError | None = None
    while time.monotonic() < deadline:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(completed.stdout)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            time.sleep(2)
    raise ValueError("Application readiness validation timed out") from last_error


def restore_backup(args: argparse.Namespace) -> dict:
    backup_dir = args.from_dir.resolve()
    validate_backup(backup_dir)
    _validate_project_name(args)
    if not args.confirm_restore and not args.dry_run:
        raise SystemExit("Restore requires --confirm-restore")

    expected_postgres = _load_json(backup_dir / "postgres.json")
    expected_qdrant = _load_json(backup_dir / "qdrant.json")
    if expected_qdrant.get("collection") != args.collection:
        raise ValueError("Requested collection does not match backup metadata")
    compose = _compose_prefix(args.compose_file, args.env_file, args.project_name)
    _run(
        compose + ["stop", "app", "bot", "ingest"],
        dry_run=args.dry_run,
    )
    _run(
        compose + ["up", "-d", "--wait", "postgres", "qdrant"],
        dry_run=args.dry_run,
    )
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
    postgres_result = _validate_postgres(compose, args, expected_postgres)

    _run(
        compose
        + [
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "-v",
            _backup_mount(backup_dir, read_only=True),
            "app",
            "python",
            "scripts/qdrant_snapshot.py",
            "restore",
            "--collection",
            args.collection,
            "--path",
            "/backup/qdrant.snapshot",
        ],
        dry_run=args.dry_run,
    )
    qdrant_result = _qdrant_metadata(compose, args, backup_dir)
    if not args.dry_run and qdrant_result != expected_qdrant:
        raise ValueError("Restored Qdrant metadata does not match backup")

    _run(
        compose
        + [
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "-v",
            _backup_mount(backup_dir, read_only=True),
            "app",
            "python",
            "scripts/deployment_state.py",
            "restore",
        ],
        dry_run=args.dry_run,
    )
    state_output = _capture(
        compose
        + [
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "-v",
            _backup_mount(backup_dir, read_only=True),
            "app",
            "python",
            "scripts/deployment_state.py",
            "validate",
        ],
        dry_run=args.dry_run,
    )
    state_result = {} if args.dry_run else json.loads(state_output)

    _run(
        compose + ["up", "-d", "--no-deps", "app"],
        dry_run=args.dry_run,
    )
    readiness_result = _wait_ready(
        compose,
        timeout_seconds=args.readiness_timeout,
        dry_run=args.dry_run,
    )
    return {
        "project_name": args.project_name,
        "postgres": postgres_result,
        "qdrant": qdrant_result,
        "state": state_result,
        "readiness": readiness_result,
    }


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
        operation.add_argument("--project-name", default=CURRENT_PROJECT_NAME)
        operation.add_argument("--collection", default="corporate_rag")
        operation.add_argument("--postgres-user", default="postgres")
        operation.add_argument("--postgres-db", default="diploma")
        operation.add_argument("--dry-run", action="store_true")
    backup = subparsers.choices["backup"]
    backup.add_argument("--output", type=Path, default=PROJECT_ROOT / "backups")
    restore = subparsers.choices["restore"]
    restore.set_defaults(project_name=None)
    restore.add_argument("--from", dest="from_dir", type=Path, required=True)
    restore.add_argument("--confirm-restore", action="store_true")
    restore.add_argument("--allow-current-project", action="store_true")
    restore.add_argument("--readiness-timeout", type=float, default=300.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "backup":
            backup_dir = create_backup(args)
            print(f"Backup ready: {backup_dir}")
        else:
            result = restore_backup(args)
            print("Restore completed: " + json.dumps(result, sort_keys=True))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Deployment backup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
