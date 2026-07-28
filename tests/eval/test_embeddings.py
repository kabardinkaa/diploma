from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from app.services.embeddings import EmbeddingService


class FakeSentenceTransformer:
    def __init__(self, model_name: str, calls: list[dict[str, Any]]) -> None:
        self.model_name = model_name
        self.calls = calls

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append({"texts": list(texts), **kwargs})
        return [[3.0, 4.0, float(index + 1)] for index, _ in enumerate(texts)]


def make_service(
    tmp_path: Path,
    calls: list[dict[str, Any]],
    *,
    model_name: str = "intfloat/multilingual-e5-base",
    batch_size: int = 32,
) -> EmbeddingService:
    return EmbeddingService(
        model_name=model_name,
        batch_size=batch_size,
        cache_dir=tmp_path / "embeddings",
        model_factory=lambda name: FakeSentenceTransformer(name, calls),
    )


def test_mini_benchmark_uses_real_knowledge_base_documents() -> None:
    root = Path(__file__).resolve().parents[2]
    benchmark = json.loads(
        (root / "tests/eval/mini_benchmark.json").read_text(encoding="utf-8")
    )
    knowledge_base = json.loads(
        (root / "app/tools/knowledge_base.json").read_text(encoding="utf-8-sig")
    )
    source_documents = {document["content"] for document in knowledge_base}

    assert 5 <= len(benchmark) <= 10
    for item in benchmark:
        assert set(item) == {"query", "relevant", "irrelevant"}
        assert all(isinstance(item[key], str) and item[key] for key in item)
        assert item["relevant"] in source_documents
        assert item["irrelevant"] in source_documents
        assert item["relevant"] != item["irrelevant"]


def test_embeddings_are_batched_and_normalized(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    with make_service(tmp_path, calls, batch_size=2) as service:
        vectors = service.embed_texts(["one", "two", "three", "four", "five"])

    assert [len(call["texts"]) for call in calls] == [2, 2, 1]
    assert all(call["batch_size"] == 2 for call in calls)
    assert all(call["normalize_embeddings"] is True for call in calls)
    assert all(call["show_progress_bar"] is False for call in calls)
    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    assert all(math.isclose(norm, 1.0) for norm in norms)


def test_e5_query_and_passage_prefixes_are_internal(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    with make_service(tmp_path, calls) as service:
        service.embed_query("как восстановить VPN")
        service.embed_documents(["инструкция VPN", "инструкция CRM"])
        service.embed_texts(["обычный текст"])

    assert calls[0]["texts"] == ["query: как восстановить VPN"]
    assert calls[1]["texts"] == [
        "passage: инструкция VPN",
        "passage: инструкция CRM",
    ]
    assert calls[2]["texts"] == ["обычный текст"]


def test_cache_hit_skips_repeated_computation(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    with make_service(tmp_path, calls) as service:
        first = service.embed_query("не работает VPN")
        second = service.embed_query("не работает VPN")

    assert first == second
    assert len(calls) == 1


def test_cache_survives_service_restart_without_model_loading(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    service = make_service(tmp_path, calls)
    expected = service.embed_documents(["документ о почте"])
    service.close()

    def fail_if_loaded(_: str) -> Any:
        raise AssertionError("model must not load on a persistent cache hit")

    restarted = EmbeddingService(
        model_name="intfloat/multilingual-e5-base",
        cache_dir=tmp_path / "embeddings",
        model_factory=fail_if_loaded,
    )
    try:
        actual = restarted.embed_documents(["документ о почте"])
    finally:
        restarted.close()

    assert actual == expected
    assert len(calls) == 1


def test_changing_model_invalidates_cached_vector(tmp_path: Path) -> None:
    calls_a: list[dict[str, Any]] = []
    calls_b: list[dict[str, Any]] = []

    with make_service(tmp_path, calls_a, model_name="model-a") as first:
        first.embed_texts(["same text"])
    with make_service(tmp_path, calls_b, model_name="model-b") as second:
        second.embed_texts(["same text"])

    assert len(calls_a) == 1
    assert len(calls_b) == 1


def test_returned_dimensions_and_types_are_stable(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    with make_service(tmp_path, calls) as service:
        query = service.embed_query("CRM")
        documents = service.embed_documents(["CRM", "VPN"])
        empty = service.embed_texts([])

    assert isinstance(query, list)
    assert len(query) == 3
    assert all(isinstance(value, float) for value in query)
    assert [len(vector) for vector in documents] == [3, 3]
    assert empty == []


def test_invalid_batch_size_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingService(
            model_name="test",
            batch_size=0,
            cache_dir=tmp_path / "embeddings",
        )
