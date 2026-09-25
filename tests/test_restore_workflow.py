from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import deployment_backup, deployment_state


def _make_backup(root: Path) -> Path:
    backup = root / "backup"
    rag = backup / "rag-state"
    rag.mkdir(parents=True)
    (backup / "postgres.dump").write_bytes(b"postgres-dump")
    (backup / "postgres.json").write_text(
        json.dumps({table: 0 for table in deployment_backup.POSTGRES_TABLES}),
        encoding="utf-8",
    )
    (backup / "qdrant.snapshot").write_bytes(b"qdrant-snapshot")
    (backup / "qdrant.json").write_text(
        json.dumps({"collection": "corporate_rag", "points_count": 73}),
        encoding="utf-8",
    )
    (rag / "docstore.json").write_text('{"docstore": true}', encoding="utf-8")
    (rag / "docstore.manifest.json").write_text(
        '{"manifest": true}', encoding="utf-8"
    )
    with zipfile.ZipFile(backup / "corpus.zip", "w") as archive:
        archive.writestr("vpn.md", "VPN instructions")
    deployment_backup._write_manifest(backup)
    return backup


def _restore_args(backup: Path, *extra: str):
    return deployment_backup.build_parser().parse_args(
        [
            "restore",
            "--from",
            str(backup),
            "--project-name",
            "diploma-restore-test",
            "--compose-file",
            "docker-compose.yml",
            "--compose-file",
            "docker-compose.restore.yml",
            "--env-file",
            ".env",
            "--confirm-restore",
            *extra,
        ]
    )


def _successful_restore_mocks(monkeypatch: pytest.MonkeyPatch):
    commands: list[list[str]] = []

    def record(command, **_kwargs):
        commands.append(command)

    monkeypatch.setattr(deployment_backup, "_run", record)
    monkeypatch.setattr(
        deployment_backup,
        "_postgres_counts",
        lambda _compose, _args: {
            table: 0 for table in deployment_backup.POSTGRES_TABLES
        },
    )
    monkeypatch.setattr(
        deployment_backup,
        "_qdrant_metadata",
        lambda _compose, _args, _backup: {
            "collection": "corporate_rag",
            "points_count": 73,
        },
    )
    monkeypatch.setattr(
        deployment_backup,
        "_capture",
        lambda _command, **_kwargs: json.dumps(
            {"rag_state_files": 2, "corpus_files": 1}
        ),
    )
    monkeypatch.setattr(
        deployment_backup,
        "_wait_ready",
        lambda _compose, **_kwargs: {"status": "ready"},
    )
    return commands


def test_restore_propagates_project_name_and_uses_one_shot_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = _make_backup(tmp_path)
    commands = _successful_restore_mocks(monkeypatch)

    result = deployment_backup.restore_backup(_restore_args(backup))

    assert result["project_name"] == "diploma-restore-test"
    assert commands
    for command in commands:
        project_index = command.index("--project-name")
        assert command[project_index + 1] == "diploma-restore-test"
    app_commands = [command for command in commands if "app" in command]
    assert any("run" in command and "restore" in command for command in app_commands)
    assert not any(
        "exec" in command and command[command.index("exec") + 2] == "app"
        for command in app_commands
    )


def test_invalid_checksum_fails_before_any_compose_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = _make_backup(tmp_path)
    (backup / "postgres.dump").write_bytes(b"tampered")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        deployment_backup, "_run", lambda command, **_kwargs: commands.append(command)
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        deployment_backup.restore_backup(_restore_args(backup))

    assert commands == []


def test_incomplete_backup_fails_before_any_compose_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = _make_backup(tmp_path)
    (backup / "qdrant.snapshot").unlink()
    deployment_backup._write_manifest(backup)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        deployment_backup, "_run", lambda command, **_kwargs: commands.append(command)
    )

    with pytest.raises(ValueError, match="incomplete"):
        deployment_backup.restore_backup(_restore_args(backup))

    assert commands == []


def test_exact_corpus_restore_removes_newer_files(tmp_path: Path) -> None:
    archive = tmp_path / "corpus.zip"
    target = tmp_path / "restore-target"
    target.mkdir()
    (target / "newer-local-file.md").write_text("must disappear", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as corpus:
        corpus.writestr("expected/vpn.md", "VPN instructions")

    deployment_state.restore_corpus_exact(archive, target)

    assert not (target / "newer-local-file.md").exists()
    assert (target / "expected" / "vpn.md").read_text(encoding="utf-8") == (
        "VPN instructions"
    )


@pytest.mark.parametrize("failing_program", ["pg_restore", "qdrant_snapshot.py"])
def test_restore_failure_is_not_masked(
    failing_program: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup = _make_backup(tmp_path)
    monkeypatch.setattr(
        deployment_backup,
        "_postgres_counts",
        lambda _compose, _args: {
            table: 0 for table in deployment_backup.POSTGRES_TABLES
        },
    )

    def fail_selected(command, **_kwargs):
        if any(failing_program in part for part in command) and (
            failing_program == "pg_restore" or "restore" in command
        ):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(deployment_backup, "_run", fail_selected)

    with pytest.raises(subprocess.CalledProcessError):
        deployment_backup.restore_backup(_restore_args(backup))


def test_post_restore_validation_failure_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backup = _make_backup(tmp_path)
    monkeypatch.setattr(
        deployment_backup,
        "restore_backup",
        lambda _args: (_ for _ in ()).throw(ValueError("validation failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "deployment_backup.py",
            "restore",
            "--from",
            str(backup),
            "--project-name",
            "diploma-restore-test",
            "--confirm-restore",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        deployment_backup.main()

    assert exc_info.value.code == 1
    assert "validation failed" in capsys.readouterr().err


def test_restore_refuses_current_project_without_override(tmp_path: Path) -> None:
    backup = _make_backup(tmp_path)
    args = _restore_args(backup)
    args.project_name = deployment_backup.CURRENT_PROJECT_NAME

    with pytest.raises(ValueError, match="current project"):
        deployment_backup.restore_backup(args)


def test_restore_dry_run_is_non_destructive_and_requires_valid_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backup = _make_backup(tmp_path)
    args = _restore_args(backup, "--dry-run")
    args.confirm_restore = False
    commands: list[list[str]] = []
    monkeypatch.setattr(
        deployment_backup, "_run", lambda command, **_kwargs: commands.append(command)
    )
    monkeypatch.setattr(deployment_backup, "_capture", lambda *_args, **_kwargs: "")

    result = deployment_backup.restore_backup(args)

    assert result["project_name"] == "diploma-restore-test"
    assert commands
