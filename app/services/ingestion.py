from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from llama_index.core import Document
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import (
    DocxReader,
    HTMLTagReader,
    MarkdownReader,
    PyMuPDFReader,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.core.config import Settings
from app.services.chunking import build_chunk_parser, build_e5_embedding

logger = structlog.get_logger("rag-ingestion")

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".html", ".htm", ".md"})
EXCLUDED_EMBED_METADATA_KEYS = [
    "source_path",
    "last_modified",
    "author",
    "version",
    "doc_type",
]
_VERSION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(v\d+(?:\.\d+)?|20\d{2})(?!\d)",
    re.IGNORECASE,
)
_CATEGORY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class IngestFailure(BaseModel):
    file: str
    error: str


class IngestReport(BaseModel):
    discovered_files: int = 0
    parsed_files: int = 0
    changed_files: int = 0
    unchanged_files: int = 0
    failed_files: int = 0
    generated_nodes: int = 0
    collection: str
    points_count: int = 0
    elapsed_seconds: float = 0.0
    failures: list[IngestFailure] = Field(default_factory=list)


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def stable_document_id(relative_path: str, part: int | None = None) -> str:
    normalized = relative_path.replace("\\", "/").lower()
    if part is not None:
        normalized = f"{normalized}#part={part}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_version(file_name: str) -> str | None:
    match = _VERSION_PATTERN.search(Path(file_name).stem)
    return match.group(1) if match else None


def category_from_path(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    directories = list(relative.parts[:-1])
    if directories and directories[0].lower() == "corporate":
        directories = directories[1:]
    return directories[0] if directories else "uncategorized"


def validate_category(category: str) -> str:
    value = category.strip()
    if not value or not _CATEGORY_PATTERN.fullmatch(value):
        raise ValueError("category may contain only letters, digits, '_' and '-'")
    return value


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        *,
        sync_client: QdrantClient | Any | None = None,
        async_client: AsyncQdrantClient | Any | None = None,
        embed_model: Any | None = None,
        pipeline_factory: Callable[..., Any] = IngestionPipeline,
    ) -> None:
        api_key_value = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        api_key = api_key_value or None
        self.settings = settings
        self._client = sync_client or QdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
        self._aclient = async_client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
        self._owns_client = sync_client is None
        self._owns_async_client = async_client is None
        self._embed_model = embed_model
        self._pipeline_factory = pipeline_factory
        self._docstore: SimpleDocumentStore | Any | None = None
        self._pipeline: Any | None = None

    @property
    def collection_name(self) -> str:
        return self.settings.rag_production_collection

    @property
    def docstore_path(self) -> Path:
        return Path(self.settings.rag_docstore_path)

    @property
    def manifest_path(self) -> Path:
        return self.docstore_path.with_suffix(".manifest.json")

    @staticmethod
    def reader_for_extension(extension: str) -> Any:
        readers: dict[str, Callable[[], Any]] = {
            ".pdf": PyMuPDFReader,
            ".docx": DocxReader,
            ".html": lambda: HTMLTagReader(tag="section", ignore_no_id=False),
            ".htm": lambda: HTMLTagReader(tag="section", ignore_no_id=False),
            ".md": MarkdownReader,
        }
        try:
            return readers[extension.lower()]()
        except KeyError as exc:
            raise ValueError(f"Unsupported document extension: {extension}") from exc

    def discover_files(self, root: Path) -> list[Path]:
        if not root.is_dir():
            raise FileNotFoundError(f"Ingestion root not found: {root}")
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.endswith(".failed")
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _load_manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _persist_manifest(self, manifest: dict[str, str]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_docstore(self) -> Any:
        if self.docstore_path.exists():
            return SimpleDocumentStore.from_persist_path(str(self.docstore_path))
        return SimpleDocumentStore()

    def _build_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        self._embed_model = self._embed_model or build_e5_embedding(self.settings)
        parser = build_chunk_parser(
            self.settings.rag_chunking_strategy,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
            embed_model=self._embed_model,
        )
        self._docstore = self._load_docstore()
        vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self._client,
            aclient=self._aclient,
        )
        self._pipeline = self._pipeline_factory(
            transformations=[parser, self._embed_model],
            vector_store=vector_store,
            docstore=self._docstore,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        return self._pipeline

    @staticmethod
    def _docx_author(path: Path) -> str | None:
        if path.suffix.lower() != ".docx":
            return None
        from docx import Document as DocxDocument

        return DocxDocument(path).core_properties.author or None

    def _base_metadata(self, path: Path, root: Path) -> dict[str, Any]:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        return {
            "source": path.name,
            "source_path": relative,
            "file_name": path.name,
            "category": category_from_path(path, root),
            "doc_type": path.suffix.lower().lstrip("."),
            "last_modified": datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=UTC,
            ).isoformat(),
            "version": extract_version(path.name) or "unspecified",
            "author": self._docx_author(path) or "",
            "language": "ru",
        }

    def _parse_file_sync(self, path: Path, root: Path) -> list[Document]:
        reader = self.reader_for_extension(path.suffix)
        metadata = self._base_metadata(path, root)
        if path.suffix.lower() in {".pdf", ".md"}:
            parsed = reader.load_data(str(path), extra_info=metadata)
        else:
            parsed = reader.load_data(path, extra_info=metadata)

        relative = metadata["source_path"]
        documents: list[Document] = []
        for index, item in enumerate(parsed):
            text = clean_text(item.get_content(metadata_mode=MetadataMode.NONE))
            if not text:
                continue
            item_metadata = {**metadata, **(item.metadata or {})}
            page = item_metadata.get("page") or item_metadata.get("page_label")
            if page is not None:
                try:
                    item_metadata["page"] = int(page)
                except (TypeError, ValueError):
                    item_metadata["page"] = None
            documents.append(
                Document(
                    text=text,
                    id_=stable_document_id(relative, index if len(parsed) > 1 else None),
                    metadata=item_metadata,
                    excluded_embed_metadata_keys=list(EXCLUDED_EMBED_METADATA_KEYS),
                    excluded_llm_metadata_keys=["source_path", "last_modified", "author"],
                )
            )
        if not documents:
            raise ValueError("parser returned no text")
        return documents

    async def parse_file(self, path: Path, root: Path) -> list[Document]:
        return await asyncio.to_thread(self._parse_file_sync, path, root)

    async def _points_count(self) -> int:
        try:
            info = await self._aclient.get_collection(self.collection_name)
        except Exception:
            return 0
        return int(info.points_count or 0)

    async def _reset(self) -> None:
        try:
            await self._aclient.delete_collection(self.collection_name)
        except Exception:
            pass
        for path in (self.docstore_path, self.manifest_path):
            if path.exists():
                path.unlink()
        self._pipeline = None
        self._docstore = None

    async def ingest_files(
        self,
        files: Sequence[Path],
        *,
        root: Path,
        mode: Literal["incremental", "full"] = "incremental",
    ) -> IngestReport:
        started_at = time.perf_counter()
        if mode == "full":
            await self._reset()
        manifest = self._load_manifest()
        report = IngestReport(
            discovered_files=len(files),
            collection=self.collection_name,
        )
        documents: list[Document] = []

        for path in files:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            digest = await asyncio.to_thread(self._file_hash, path)
            if manifest.get(relative) == digest:
                report.unchanged_files += 1
                continue
            try:
                parsed = await self.parse_file(path, root)
                documents.extend(parsed)
                report.parsed_files += 1
                report.changed_files += 1
                manifest[relative] = digest
                logger.info("rag.ingest.file", file=relative, status="changed")
            except Exception as exc:
                failed_path = path.with_name(f"{path.name}.failed")
                await asyncio.to_thread(path.replace, failed_path)
                report.failed_files += 1
                report.failures.append(
                    IngestFailure(file=relative, error=type(exc).__name__)
                )
                logger.warning(
                    "rag.ingest.file_failed",
                    file=relative,
                    error_type=type(exc).__name__,
                )

        if documents:
            pipeline = self._build_pipeline()
            nodes = await pipeline.arun(documents=documents, show_progress=True)
            report.generated_nodes = len(nodes)
            self.docstore_path.parent.mkdir(parents=True, exist_ok=True)
            self._docstore.persist(persist_path=str(self.docstore_path))
        self._persist_manifest(manifest)
        report.points_count = await self._points_count()
        report.elapsed_seconds = round(time.perf_counter() - started_at, 2)
        return report

    async def ingest_path(
        self,
        root: str | Path,
        *,
        mode: Literal["incremental", "full"] = "incremental",
    ) -> IngestReport:
        root_path = Path(root)
        return await self.ingest_files(
            self.discover_files(root_path),
            root=root_path,
            mode=mode,
        )

    async def incremental_reindex(self, root: str | Path) -> IngestReport:
        return await self.ingest_path(root, mode="incremental")

    async def full_reindex(self, root: str | Path) -> IngestReport:
        return await self.ingest_path(root, mode="full")

    async def close(self) -> None:
        if self._owns_client:
            await asyncio.to_thread(self._client.close)
        if self._owns_async_client:
            await self._aclient.close()
