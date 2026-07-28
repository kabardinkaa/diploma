from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.routers.rag import router
from app.services.rag import RAGService
from app.services.rag_common import FALLBACK_ANSWER


def rag_settings(min_score: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_url="http://qdrant.test:6333",
        qdrant_api_key=None,
        rag_collection="rag_block_03",
        rag_data_dir=Path("data/rag-block-03"),
        rag_chunk_size=512,
        rag_chunk_overlap=64,
        rag_similarity_top_k=3,
        rag_min_score=min_score,
        embedding_model="intfloat/multilingual-e5-base",
        embedding_batch_size=8,
        embedding_cache_dir=Path(".cache/test"),
        embedding_dim=768,
        llm=SimpleNamespace(
            api_key=SecretStr("test"),
            base_url="https://example.test/v1",
            default_model="test-model",
            request_timeout=1.0,
            max_retries=0,
        ),
    )


def make_service(min_score: float = 0.5) -> RAGService:
    return RAGService(
        rag_settings(min_score),
        qdrant_client=Mock(),
        async_qdrant_client=AsyncMock(),
        openai_client=AsyncMock(),
    )


class FakeIndex:
    def __init__(self) -> None:
        self.retriever = Mock()
        self.query_engine = Mock()

    def as_retriever(self, **_: object) -> Mock:
        return self.retriever

    def as_query_engine(self, **_: object) -> Mock:
        return self.query_engine


def source_node(text: str, source: str, score: float) -> SimpleNamespace:
    node = SimpleNamespace(
        text=text,
        metadata={"file_name": source},
        get_content=lambda: text,
    )
    return SimpleNamespace(node=node, score=score)


@pytest.mark.asyncio
async def test_build_creates_components_once_and_does_not_reindex(mocker) -> None:
    service = make_service()
    index = FakeIndex()
    mocker.patch.object(service, "_configure_llama_index")
    mocker.patch.object(service, "_collection_count", AsyncMock(return_value=0))
    create = mocker.patch.object(
        service,
        "_create_index",
        AsyncMock(return_value=index),
    )

    await service.build()
    await service.build()

    create.assert_awaited_once()
    assert service.indexed_on_build is True
    assert service._retriever is index.retriever
    assert service._query_engine is index.query_engine


@pytest.mark.asyncio
async def test_existing_collection_attaches_without_reindexing(mocker) -> None:
    service = make_service()
    index = FakeIndex()
    mocker.patch.object(service, "_configure_llama_index")
    mocker.patch.object(service, "_collection_count", AsyncMock(return_value=10))
    validate = mocker.patch.object(
        service,
        "_validate_existing_collection",
        AsyncMock(),
    )
    attach = mocker.patch.object(
        service,
        "_attach_index",
        AsyncMock(return_value=index),
    )
    create = mocker.patch.object(service, "_create_index", AsyncMock())

    await service.build()

    validate.assert_awaited_once()
    attach.assert_awaited_once()
    create.assert_not_awaited()
    assert service.indexed_on_build is False


@pytest.mark.asyncio
async def test_answer_contract_sources_and_top_score() -> None:
    service = make_service()
    nodes = [
        source_node("VPN details", "01_vpn.md", 0.81),
        source_node("DNS details", "01_vpn.md", 0.93),
        source_node("Ticket details", "09_support_ticket.md", 0.72),
    ]
    service._built = True
    service._retriever = SimpleNamespace(aretrieve=AsyncMock(return_value=nodes))
    response = SimpleNamespace(
        source_nodes=nodes,
        __str__=lambda _: "Проверьте DNS.",
    )
    service._query_engine = SimpleNamespace(
        aquery=AsyncMock(return_value=response)
    )

    result = await service.answer("Почему не открывается внутренний сайт?")

    assert set(result) == {"answer", "top_score", "sources"}
    assert result["top_score"] == 0.93
    assert len(result["sources"]) == 3
    assert result["sources"][1] == {
        "text": "DNS details",
        "source": "01_vpn.md",
        "score": 0.93,
    }
    service._query_engine.aquery.assert_awaited_once()


@pytest.mark.asyncio
async def test_low_score_fallback_does_not_call_llm() -> None:
    service = make_service(min_score=0.8)
    nodes = [source_node("weak", "10_office_plants.txt", 0.42)]
    service._built = True
    service._retriever = SimpleNamespace(aretrieve=AsyncMock(return_value=nodes))
    service._query_engine = SimpleNamespace(aquery=AsyncMock())

    result = await service.answer("Как оформить отпуск?")

    assert result["answer"] == FALLBACK_ANSWER
    assert result["top_score"] == 0.42
    service._query_engine.aquery.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_sources_fallback() -> None:
    service = make_service()
    service._built = True
    service._retriever = SimpleNamespace(aretrieve=AsyncMock(return_value=[]))
    service._query_engine = SimpleNamespace(aquery=AsyncMock())

    result = await service.answer("Нет данных")

    assert result == {
        "answer": FALLBACK_ANSWER,
        "top_score": 0.0,
        "sources": [],
    }


class FakeAPIService:
    def __init__(self) -> None:
        self.answer = AsyncMock(
            return_value={
                "answer": "Ответ",
                "top_score": 0.91,
                "sources": [
                    {"text": "Фрагмент", "source": "01_vpn.md", "score": 0.91}
                ],
            }
        )
        self.build = AsyncMock()


def test_rag_endpoint_uses_startup_service_and_validates_question() -> None:
    api = FastAPI()
    api.include_router(router)
    api.state.rag_service = FakeAPIService()
    client = TestClient(api)

    response = client.post("/rag/query", json={"question": "  VPN?  "})
    blank = client.post("/rag/query", json={"question": "   "})

    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "01_vpn.md"
    api.state.rag_service.answer.assert_awaited_once_with("VPN?")
    api.state.rag_service.build.assert_not_awaited()
    assert blank.status_code == 422
