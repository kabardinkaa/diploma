"""Copy, restore, and validate RAG filesystem state inside one-shot containers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _safe_target(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError(f"Refusing broad restore target: {resolved}")
    if resolved.is_symlink():
        raise ValueError(f"Restore target must not be a symlink: {resolved}")
    return resolved


def _clear_directory(path: Path) -> Path:
    target = _safe_target(path)
    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    return target


def _tree_digests(root: Path) -> dict[str, str]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"State directory does not exist: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"State tree contains a symlink: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = _sha256_stream(stream)
    return result


def copy_tree_exact(source: Path, target: Path) -> None:
    source = source.resolve()
    expected = _tree_digests(source)
    destination = _clear_directory(target)
    for relative in expected:
        source_file = source / relative
        target_file = destination / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
    if _tree_digests(destination) != expected:
        raise ValueError("Copied state tree does not match its source")


def _zip_digests(archive_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or (member.external_attr >> 16) & 0o170000 == stat.S_IFLNK
            ):
                raise ValueError("Corpus archive contains an unsafe entry")
            if member.is_dir():
                continue
            relative = member_path.as_posix()
            if relative in result:
                raise ValueError(f"Corpus archive contains duplicate entry: {relative}")
            with archive.open(member) as stream:
                result[relative] = _sha256_stream(stream)
    if not result:
        raise ValueError("Corpus archive is empty")
    return result


def restore_corpus_exact(archive_path: Path, target: Path) -> None:
    expected = _zip_digests(archive_path)
    destination = _clear_directory(target)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    if _tree_digests(destination) != expected:
        raise ValueError("Restored corpus does not match the backup archive")


def restore_state(
    backup_root: Path,
    rag_target: Path,
    corpus_target: Path,
) -> None:
    copy_tree_exact(backup_root / "rag-state", rag_target)
    restore_corpus_exact(backup_root / "corpus.zip", corpus_target)


def validate_state(
    backup_root: Path,
    rag_target: Path,
    corpus_target: Path,
) -> dict[str, int]:
    expected_rag = _tree_digests(backup_root / "rag-state")
    actual_rag = _tree_digests(rag_target)
    if actual_rag != expected_rag:
        raise ValueError("Restored RAG state does not match backup")
    expected_corpus = _zip_digests(backup_root / "corpus.zip")
    actual_corpus = _tree_digests(corpus_target)
    if actual_corpus != expected_corpus:
        raise ValueError("Restored corpus does not match backup")
    return {
        "rag_state_files": len(actual_rag),
        "corpus_files": len(actual_corpus),
    }


def readiness(url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        payload = json.load(response)
    if payload.get("status") != "ready":
        raise ValueError("Application readiness check failed")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup = subparsers.add_parser("backup")
    backup.add_argument("--source", type=Path, default=Path("/app/.cache/rag"))
    backup.add_argument("--target", type=Path, default=Path("/backup/rag-state"))

    for name in ("restore", "validate"):
        operation = subparsers.add_parser(name)
        operation.add_argument("--backup-root", type=Path, default=Path("/backup"))
        operation.add_argument("--rag-target", type=Path, default=Path("/app/.cache/rag"))
        operation.add_argument("--corpus-target", type=Path, default=Path("/app/data"))

    ready = subparsers.add_parser("readiness")
    ready.add_argument("--url", default="http://127.0.0.1:8000/health/ready")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "backup":
        copy_tree_exact(args.source, args.target)
        result = {"rag_state_files": len(_tree_digests(args.target))}
    elif args.command == "restore":
        restore_state(args.backup_root, args.rag_target, args.corpus_target)
        result = validate_state(args.backup_root, args.rag_target, args.corpus_target)
    elif args.command == "validate":
        result = validate_state(args.backup_root, args.rag_target, args.corpus_target)
    else:
        result = readiness(args.url)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
