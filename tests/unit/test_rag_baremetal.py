from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr
from qdrant_client.models import ScoredPoint

from app.services.rag_baremetal import (
    BareMetalRAGService,
    chunk_text,
    deterministic_point_id,
)
from app.services.rag_common import FALLBACK_ANSWER


def settings(min_score: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_url="http://qdrant.test:6333",
        qdrant_api_key=None,
        rag_baremetal_collection="rag_block_03_baremetal",
        rag_data_dir=Path("data/rag-block-03"),
        rag_chunk_size=4,
        rag_chunk_overlap=1,
        rag_similarity_top_k=3,
        rag_min_score=min_score,
        embedding_model="test",
        embedding_batch_size=4,
        embedding_cache_dir=Path(".cache/test"),
        embedding_dim=3,
        llm=SimpleNamespace(
            api_key=SecretStr("test"),
            base_url="https://example.test/v1",
            default_model="test-model",
            request_timeout=1.0,
            max_retries=0,
        ),
    )


def test_chunking_and_ids_are_deterministic() -> None:
    first_chunks = chunk_text("one two three four five six", 4, 1)
    second_chunks = chunk_text("one two three four five six", 4, 1)

    assert first_chunks == second_chunks == ["one two three four", "four five six"]
    assert deterministic_point_id("01_vpn.md", 0) == deterministic_point_id(
        "01_vpn.md", 0
    )
    assert deterministic_point_id("01_vpn.md", 0) != deterministic_point_id(
        "01_vpn.md", 1
    )


@pytest.mark.asyncio
async def test_build_uses_document_embeddings_and_stable_points(mocker) -> None:
    qdrant = AsyncMock()
    embeddings = SimpleNamespace(
        embed_documents=Mock(return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        embed_query=Mock(),
    )
    service = BareMetalRAGService(
        settings(),
        qdrant_client=qdrant,
        embedding_service=embeddings,
        openai_client=AsyncMock(),
    )
    service._vector_store = SimpleNamespace(
        ensure_collection=AsyncMock(),
        points_count=AsyncMock(return_value=0),
        upsert=AsyncMock(),
    )
    mocker.patch.object(
        service,
        "_read_chunks",
        return_value=[
            ("01_vpn.md", 0, "vpn text"),
            ("02_crm.md", 0, "crm text"),
        ],
    )

    await service.build()
    await service.build()

    embeddings.embed_documents.assert_called_once_with(["vpn text", "crm text"])
    service._vector_store.upsert.assert_awaited_once()
    points = service._vector_store.upsert.await_args.args[0]
    assert [point.id for point in points] == [
        deterministic_point_id("01_vpn.md", 0),
        deterministic_point_id("02_crm.md", 0),
    ]


def scored_point(score: float = 0.9, text: str = "Проверьте DNS.") -> ScoredPoint:
    return ScoredPoint(
        id=deterministic_point_id("01_vpn.md", 0),
        version=0,
        score=score,
        payload={
            "text": text,
            "source": "01_vpn.md",
            "chunk_index": 0,
        },
    )


@pytest.mark.asyncio
async def test_answer_uses_query_embedding_and_query_points() -> None:
    qdrant = AsyncMock()
    full_text = "Проверьте DNS. " + ("детали " * 60) + "КОНЕЦ"
    qdrant.query_points.return_value = SimpleNamespace(
        points=[scored_point(text=full_text)]
    )
    embeddings = SimpleNamespace(
        embed_documents=Mock(),
        embed_query=Mock(return_value=[0.1, 0.2, 0.3]),
    )
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="Проверьте DNS-кеш."))
        ]
    )
    openai = AsyncMock()
    openai.chat.completions.create.return_value = completion
    service = BareMetalRAGService(
        settings(),
        qdrant_client=qdrant,
        embedding_service=embeddings,
        openai_client=openai,
    )
    service._built = True

    result = await service.answer("Почему не работает VPN?")

    embeddings.embed_query.assert_called_once_with("Почему не работает VPN?")
    qdrant.query_points.assert_awaited_once()
    assert qdrant.query_points.await_args.kwargs["limit"] == 3
    prompt = openai.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "КОНЕЦ" in prompt
    assert result == {
        "answer": "Проверьте DNS-кеш.",
        "top_score": 0.9,
        "sources": [
            {
                "text": full_text[:300],
                "source": "01_vpn.md",
                "score": 0.9,
            }
        ],
    }


@pytest.mark.asyncio
async def test_baremetal_fallback_skips_generation() -> None:
    qdrant = AsyncMock()
    qdrant.query_points.return_value = SimpleNamespace(
        points=[scored_point(score=0.3)]
    )
    embeddings = SimpleNamespace(
        embed_documents=Mock(),
        embed_query=Mock(return_value=[0.1, 0.2, 0.3]),
    )
    openai = AsyncMock()
    service = BareMetalRAGService(
        settings(min_score=0.8),
        qdrant_client=qdrant,
        embedding_service=embeddings,
        openai_client=openai,
    )
    service._built = True

    result = await service.answer("Как оформить отпуск?")

    assert result["answer"] == FALLBACK_ANSWER
    openai.chat.completions.create.assert_not_awaited()
