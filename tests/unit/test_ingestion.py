from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from llama_index.readers.file import (
    DocxReader,
    HTMLTagReader,
    MarkdownReader,
    PyMuPDFReader,
)

from app.core.config import LLMSettings, Settings
from app.services.ingestion import (
    EXCLUDED_EMBED_METADATA_KEYS,
    IngestionService,
    category_from_path,
    clean_text,
    extract_version,
    stable_document_id,
)


def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        RAG_DATA_DIR=tmp_path,
        RAG_DOCSTORE_PATH=tmp_path / "state" / "docstore.json",
        llm=LLMSettings(_env_file=None, OPENAI_API_KEY="test"),
    )


def service(tmp_path: Path) -> IngestionService:
    return IngestionService(
        settings(tmp_path),
        sync_client=Mock(),
        async_client=AsyncMock(),
        embed_model=Mock(),
    )


def test_parser_routing_supports_all_required_formats(tmp_path: Path) -> None:
    instance = service(tmp_path)
    assert isinstance(instance.reader_for_extension(".pdf"), PyMuPDFReader)
    assert isinstance(instance.reader_for_extension(".docx"), DocxReader)
    assert isinstance(instance.reader_for_extension(".html"), HTMLTagReader)
    assert isinstance(instance.reader_for_extension(".md"), MarkdownReader)
    with pytest.raises(ValueError):
        instance.reader_for_extension(".json")


def test_cleaning_metadata_helpers_and_stable_id(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = root / "corporate" / "vpn" / "guide_v1.3_2026.md"
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")

    assert clean_text("a  \r\n\r\n\r\n\r\n# Header\n") == "a\n\n# Header"
    assert category_from_path(path, root) == "vpn"
    assert extract_version(path.name) == "v1.3"
    assert stable_document_id("vpn/guide.md") == stable_document_id(
        "VPN\\GUIDE.md"
    )

    metadata = service(tmp_path)._base_metadata(path, root)
    assert metadata["last_modified"].endswith("+00:00")
    assert metadata["author"] == ""
    assert metadata["doc_type"] == "md"


def test_real_corpus_inventory_and_reader_metadata() -> None:
    root = Path("data")
    instance = IngestionService(
        Settings(
            _env_file=None,
            llm=LLMSettings(_env_file=None, OPENAI_API_KEY="test"),
        ),
        sync_client=Mock(),
        async_client=AsyncMock(),
    )
    files = instance.discover_files(root)
    suffixes = {path.suffix for path in files}
    assert len(files) >= 50
    assert suffixes == {".md", ".html", ".docx", ".pdf"}

    docx_path = next(path for path in files if path.suffix == ".docx")
    document = instance._parse_file_sync(docx_path, root)[0]
    assert document.metadata["category"]
    assert document.metadata["author"] == "Diploma course corpus generator"
    assert document.excluded_embed_metadata_keys == EXCLUDED_EMBED_METADATA_KEYS


class FakeDocstore:
    def __init__(self) -> None:
        self.persist = Mock()


class FakePipeline:
    def __init__(self, docstore: FakeDocstore) -> None:
        self.docstore = docstore
        self.arun = AsyncMock(return_value=[object(), object()])


@pytest.mark.asyncio
async def test_second_ingest_is_unchanged_and_does_not_duplicate(
    tmp_path: Path,
    mocker,
) -> None:
    root = tmp_path / "data"
    path = root / "vpn" / "guide_v2026.md"
    path.parent.mkdir(parents=True)
    path.write_text("# VPN\n\nПроверка подключения.", encoding="utf-8")
    instance = service(tmp_path)
    document = SimpleNamespace()
    mocker.patch.object(
        instance,
        "parse_file",
        AsyncMock(return_value=[document]),
    )
    docstore = FakeDocstore()
    pipeline = FakePipeline(docstore)
    instance._docstore = docstore
    mocker.patch.object(instance, "_build_pipeline", return_value=pipeline)
    mocker.patch.object(instance, "_points_count", AsyncMock(return_value=2))

    first = await instance.ingest_path(root)
    second = await instance.ingest_path(root)

    assert first.changed_files == 1
    assert first.generated_nodes == 2
    assert second.changed_files == 0
    assert second.unchanged_files == 1
    assert second.generated_nodes == 0
    assert pipeline.arun.await_count == 1
    assert json.loads(instance.manifest_path.read_text())["vpn/guide_v2026.md"]


@pytest.mark.asyncio
async def test_failed_file_is_renamed_and_does_not_abort(
    tmp_path: Path,
    mocker,
) -> None:
    root = tmp_path / "data"
    path = root / "vpn" / "broken.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a pdf")
    instance = service(tmp_path)
    mocker.patch.object(
        instance,
        "parse_file",
        AsyncMock(side_effect=ValueError("broken")),
    )
    mocker.patch.object(instance, "_points_count", AsyncMock(return_value=0))

    report = await instance.ingest_path(root)

    assert report.failed_files == 1
    assert not path.exists()
    assert path.with_name("broken.pdf.failed").exists()
