from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.routers.rag import router
from app.services.rag import RAGService, normalize_citations, sanitize_sse_payload
from app.services.rag_common import FALLBACK_ANSWER


def rag_settings(min_score: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_url="http://qdrant.test:6333",
        qdrant_api_key=None,
        rag_collection="old_collection",
        rag_production_collection="corporate_rag",
        rag_similarity_top_k=3,
        rag_retrieval_top_k=10,
        rag_min_score=min_score,
        rag_reranker_enabled=False,
        rag_rerank_top_n=5,
        rag_max_sources=5,
        rag_condense_enabled=True,
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
        embed_model=Mock(),
    )


class FakeIndex:
    def __init__(self) -> None:
        self.retriever = Mock()

    def as_retriever(self, **_: object) -> Mock:
        return self.retriever


def source_node(
    text: str,
    source: str,
    score: float,
    page: int | None = None,
) -> SimpleNamespace:
    node = SimpleNamespace(
        text=text,
        metadata={"file_name": source, "page": page},
        get_content=lambda: text,
    )
    return SimpleNamespace(node=node, score=score)


class AsyncChunks:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def iterate():
            for text in self._chunks:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content=text))
                    ]
                )

        return iterate()


@pytest.mark.asyncio
async def test_build_attaches_query_components_once_without_ingestion(mocker) -> None:
    service = make_service()
    index = FakeIndex()
    configure = mocker.patch.object(service, "_configure_llama_index")
    mocker.patch.object(service, "_collection_count", AsyncMock(return_value=0))
    attach = mocker.patch.object(
        service,
        "_attach_index",
        AsyncMock(return_value=index),
    )

    await service.build()
    await service.build()

    configure.assert_called_once()
    attach.assert_awaited_once()
    assert service.indexed_on_build is False
    assert service._retriever is index.retriever
    assert service.retrieval_top_k == 10
    assert service.collection_name == "corporate_rag"


@pytest.mark.asyncio
async def test_existing_collection_is_validated_before_attach(mocker) -> None:
    service = make_service()
    mocker.patch.object(service, "_configure_llama_index")
    mocker.patch.object(service, "_collection_count", AsyncMock(return_value=10))
    validate = mocker.patch.object(
        service,
        "_validate_existing_collection",
        AsyncMock(),
    )
    mocker.patch.object(
        service,
        "_attach_index",
        AsyncMock(return_value=FakeIndex()),
    )

    await service.build()

    validate.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_contexts_returns_top_k_without_generation() -> None:
    service = make_service()
    nodes = [
        source_node("Top fragment", "vpn.md", 0.93),
        source_node("Second fragment", "network.md", 0.81),
    ]
    retriever = SimpleNamespace(aretrieve=AsyncMock(return_value=nodes))
    as_retriever = Mock(return_value=retriever)
    generation = AsyncMock()
    service._built = True
    service._index = SimpleNamespace(as_retriever=as_retriever)
    service._retriever = SimpleNamespace(aretrieve=AsyncMock())
    service._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=generation))
    )

    contexts = await service.retrieve_contexts("VPN error", top_k=1)

    assert contexts == [
        {
            "text": "Top fragment",
            "file_name": "vpn.md",
            "page": None,
            "dense_score": 0.93,
        }
    ]
    as_retriever.assert_called_once_with(similarity_top_k=1)
    generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_contract_citations_sources_and_top_score() -> None:
    service = make_service()
    nodes = [
        source_node("VPN details", "vpn.pdf", 0.93, 2),
        source_node("DNS details", "network.md", 0.81),
    ]
    service._built = True
    service._retriever = SimpleNamespace(aretrieve=AsyncMock(return_value=nodes))
    create = AsyncMock(return_value=AsyncChunks(["Проверьте DNS."]))
    service._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await service.answer("Почему не открывается внутренний сайт?")

    assert result["top_score"] == 0.93
    assert result["confident"] is True
    assert result["answer"].endswith("[1]")
    assert result["sources"][0] == {
        "id": 1,
        "file_name": "vpn.pdf",
        "page": 2,
        "score": 0.93,
        "snippet": "VPN details",
    }
    assert create.await_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_low_score_fallback_skips_llm_and_hides_sources() -> None:
    service = make_service(min_score=0.8)
    nodes = [source_node("weak", "irrelevant.md", 0.42)]
    service._built = True
    service._retriever = SimpleNamespace(aretrieve=AsyncMock(return_value=nodes))
    create = AsyncMock()
    service._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = await service.answer("Как оформить отпуск?")

    assert result == {
        "answer": FALLBACK_ANSWER,
        "top_score": 0.42,
        "confident": False,
        "sources": [],
    }
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_condense_follow_up_and_safe_failure_fallback() -> None:
    service = make_service()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Что делать, если VPN всё ещё не подключается?"
                )
            )
        ]
    )
    create = AsyncMock(return_value=response)
    service._openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    history = [{"role": "user", "content": "Как восстановить VPN?"}]

    condensed, used = await service._condense(
        "А если после этого всё равно не пускает?",
        history,
    )
    assert used is True
    assert "VPN" in condensed

    create.side_effect = RuntimeError("provider unavailable")
    fallback, used = await service._condense("А если после этого?", history)
    assert fallback == "А если после этого?"
    assert used is False


def test_invalid_citations_are_removed_and_missing_citation_is_added() -> None:
    assert normalize_citations("Ответ [1], ошибка [4].", {1, 2}) == "Ответ [1], ошибка ."
    assert normalize_citations("Ответ без ссылки.", {1}) == "Ответ без ссылки. [1]"


def test_sse_sanitizer_keeps_newline_inside_json_string() -> None:
    payload = sanitize_sse_payload({"type": "token", "delta": "a\nb"})
    assert "\\n" in payload
    assert "\n" not in payload


class FakeAPIService:
    def __init__(self) -> None:
        self.answer = AsyncMock(
            return_value={
                "answer": "Ответ [1]",
                "top_score": 0.91,
                "confident": True,
                "sources": [
                    {
                        "id": 1,
                        "file_name": "vpn.pdf",
                        "page": None,
                        "score": 0.91,
                        "snippet": "Фрагмент",
                    }
                ],
            }
        )


def test_rag_endpoint_uses_startup_service_and_validates_question() -> None:
    api = FastAPI()
    api.include_router(router)
    api.state.rag_service = FakeAPIService()
    client = TestClient(api)

    response = client.post("/rag/query", json={"question": "  VPN?  "})
    blank = client.post("/rag/query", json={"question": "   "})

    assert response.status_code == 200
    assert response.json()["sources"][0]["file_name"] == "vpn.pdf"
    assert response.json()["confident"] is True
    api.state.rag_service.answer.assert_awaited_once_with("VPN?")
    assert blank.status_code == 422
