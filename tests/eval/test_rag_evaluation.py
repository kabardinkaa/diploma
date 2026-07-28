from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.eval.artifacts import (
    aggregate_rows,
    dataset_sha256,
    load_golden_dataset,
    result_paths,
)
from app.eval.config import (
    baseline_variant,
    chunk_256_variant,
    top_k_5_variant,
)
from app.services.rag import RAGService


GOLDEN_PATH = Path("tests/eval/golden_dataset.json")


def test_golden_dataset_contract_and_stable_sha() -> None:
    rows = load_golden_dataset(GOLDEN_PATH)

    assert len(rows) == 30
    assert dataset_sha256(GOLDEN_PATH) == dataset_sha256(GOLDEN_PATH)
    assert len(dataset_sha256(GOLDEN_PATH)) == 64


def test_local_eval_defaults_do_not_change_production_model() -> None:
    settings = Settings(_env_file=None)

    assert settings.eval_judge_provider == "local"
    assert settings.eval_judge_model == "local-qwen-judge"
    assert settings.eval_embedding_model == "intfloat/multilingual-e5-base"
    assert settings.rag_generation_model == "openai/gpt-5.4-mini"


def test_result_paths_include_timestamp_and_label() -> None:
    csv_path, aggregate_path = result_paths(
        Path("results"),
        "top_k_5",
        now=datetime(2026, 7, 28, 12, 34, 56, tzinfo=UTC),
    )

    assert csv_path.name == "2026-07-28_123456_top_k_5.csv"
    assert aggregate_path.name == "2026-07-28_123456_top_k_5_aggregate.json"


def test_aggregate_ignores_failed_and_non_numeric_metric_values() -> None:
    aggregate = aggregate_rows(
        [
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.6,
                "latency_ms": 100,
                "error": "",
            },
            {
                "faithfulness": None,
                "answer_relevancy": float("nan"),
                "latency_ms": 300,
                "error": "provider error",
            },
        ]
    )

    assert aggregate["mean_metrics"]["faithfulness"] == 0.8
    assert aggregate["mean_metrics"]["answer_relevancy"] == 0.6
    assert aggregate["failed_count"] == 1
    assert aggregate["latency_ms"] == {"avg": 200.0, "p50": 200.0, "p95": 290.0}


def test_ab_variants_change_only_intended_parameter() -> None:
    settings = Settings(_env_file=None)
    baseline = baseline_variant(settings)
    chunk = chunk_256_variant(settings)
    top_k = top_k_5_variant(settings)

    assert baseline.chunk_size == 512
    assert baseline.chunk_overlap == 64
    assert baseline.top_k == 10
    assert baseline.reranker_enabled is False
    assert chunk.chunk_size == 256
    assert chunk.collection == "corporate_rag_eval_chunk256"
    assert {
        key
        for key in baseline.__dict__
        if baseline.__dict__[key] != chunk.__dict__[key]
    } == {"label", "collection", "docstore_path", "chunk_size"}
    assert top_k.top_k == 5
    assert {
        key
        for key in baseline.__dict__
        if baseline.__dict__[key] != top_k.__dict__[key]
    } == {"label", "top_k"}


def rag_settings() -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_url="http://qdrant.test:6333",
        qdrant_api_key=None,
        rag_collection="old",
        rag_production_collection="corporate_rag",
        rag_similarity_top_k=3,
        rag_retrieval_top_k=10,
        rag_min_score=0.5,
        rag_reranker_enabled=False,
        rag_rerank_top_n=5,
        rag_max_sources=5,
        rag_condense_enabled=False,
        rag_generation_model="fixed-production-model",
        embedding_model="intfloat/multilingual-e5-base",
        embedding_batch_size=8,
        embedding_cache_dir=Path(".cache/test"),
        embedding_dim=768,
        llm=SimpleNamespace(
            api_key=SecretStr("test"),
            base_url="https://example.test/v1",
            default_model="legacy-model",
            request_timeout=1.0,
            max_retries=0,
        ),
    )


@pytest.mark.asyncio
async def test_evaluate_inputs_returns_full_context_without_double_retrieval() -> None:
    long_context = "Полный контекст. " * 40
    node = SimpleNamespace(
        text=long_context,
        metadata={"file_name": "vpn.md"},
        get_content=lambda: long_context,
    )
    retriever = SimpleNamespace(
        aretrieve=AsyncMock(
            return_value=[SimpleNamespace(node=node, score=0.91)]
        )
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ [1]"))]
    )
    create = AsyncMock(return_value=completion)
    service = RAGService(
        rag_settings(),
        qdrant_client=Mock(),
        async_qdrant_client=AsyncMock(),
        openai_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        embed_model=Mock(),
    )
    service._built = True
    service._retriever = retriever

    result = await service.evaluate_inputs("Как подключить VPN?")

    retriever.aretrieve.assert_awaited_once_with("Как подключить VPN?")
    assert result["retrieved_contexts"] == [long_context]
    assert len(result["retrieved_contexts"][0]) > 300
    assert result["retrieved_doc_ids"] == ["vpn.md"]
    assert result["response"] == "Ответ [1]"
    assert create.await_args.kwargs["model"] == "fixed-production-model"


def test_citation_marker_shape_matches_numbered_sources() -> None:
    assert re.search(r"\[\d+]", "Ответ подтверждён источником [1].")
