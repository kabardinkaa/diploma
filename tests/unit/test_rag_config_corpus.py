from pathlib import Path

import pytest

from app.core.config import LLMSettings, Settings


def test_rag_settings_are_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", "custom/rag")
    monkeypatch.setenv("RAG_COLLECTION", "custom_llama")
    monkeypatch.setenv("RAG_BAREMETAL_COLLECTION", "custom_bare")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "256")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "32")
    monkeypatch.setenv("RAG_SIMILARITY_TOP_K", "4")
    monkeypatch.setenv("RAG_MIN_SCORE", "0.77")

    settings = Settings(
        _env_file=None,
        llm=LLMSettings(_env_file=None, OPENAI_API_KEY="test"),
    )

    assert settings.rag_data_dir == Path("custom/rag")
    assert settings.rag_collection == "custom_llama"
    assert settings.rag_baremetal_collection == "custom_bare"
    assert settings.rag_chunk_size == 256
    assert settings.rag_chunk_overlap == 32
    assert settings.rag_similarity_top_k == 4
    assert settings.rag_min_score == 0.77


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
        Settings(
            _env_file=None,
            RAG_CHUNK_SIZE=64,
            RAG_CHUNK_OVERLAP=64,
            llm=LLMSettings(_env_file=None, OPENAI_API_KEY="test"),
        )


def test_rag_corpus_has_ten_files_and_unrelated_document() -> None:
    data_dir = Path("data/rag-block-03")
    files = sorted(path for path in data_dir.iterdir() if path.is_file())

    assert len(files) == 10
    assert all(path.suffix.lower() in {".md", ".txt"} for path in files)
    unrelated = data_dir / "10_office_plants.txt"
    assert unrelated in files
    assert "растени" in unrelated.read_text(encoding="utf-8").lower()
