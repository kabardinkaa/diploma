from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)

from app.services.chunking import (
    build_chunk_parser,
    build_fixed_size_parser,
    build_recursive_parser,
    build_semantic_parser,
    russian_sentence_tokenizer,
)
from app.services.retrieval_eval import (
    evaluate_retrieval,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
)


class FakeEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> list[float]:
        return [1.0, 0.0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_fixed_parser_uses_requested_parameters() -> None:
    parser = build_fixed_size_parser(256, 32)

    assert isinstance(parser, TokenTextSplitter)
    assert parser.chunk_size == 256
    assert parser.chunk_overlap == 32


def test_recursive_parser_uses_russian_sentence_splitter() -> None:
    parser = build_recursive_parser(512, 64)

    assert isinstance(parser, SentenceSplitter)
    assert parser.chunk_size == 512
    assert parser.chunk_overlap == 64
    assert parser.paragraph_separator == "\n\n"
    assert russian_sentence_tokenizer in parser._split_fns
    assert russian_sentence_tokenizer("Первое. Второе! Третье?") == [
        "Первое.",
        "Второе!",
        "Третье?",
    ]


def test_semantic_parser_uses_required_parameters() -> None:
    embedding = FakeEmbedding(model_name="fake")
    parser = build_semantic_parser(embedding)

    assert isinstance(parser, SemanticSplitterNodeParser)
    assert parser.embed_model is embedding
    assert parser.buffer_size == 1
    assert parser.breakpoint_percentile_threshold == 95


def test_strategy_factory_returns_distinct_parsers() -> None:
    embedding = FakeEmbedding(model_name="fake")

    assert isinstance(build_chunk_parser("fixed"), TokenTextSplitter)
    assert isinstance(build_chunk_parser("recursive"), SentenceSplitter)
    assert isinstance(
        build_chunk_parser("semantic", embed_model=embedding),
        SemanticSplitterNodeParser,
    )


def test_document_metadata_is_preserved_in_nodes() -> None:
    document = Document(
        text=("VPN работает. " * 80),
        metadata={
            "doc_id": "01_vpn.md",
            "source": "01_vpn.md",
            "file_name": "01_vpn.md",
        },
    )
    parser = build_fixed_size_parser(64, 8)

    nodes = parser.get_nodes_from_documents([document])

    assert len(nodes) > 1
    assert all(node.metadata["doc_id"] == "01_vpn.md" for node in nodes)
    assert all(node.metadata["source"] == "01_vpn.md" for node in nodes)
    assert all(node.metadata["file_name"] == "01_vpn.md" for node in nodes)


def test_metrics_exact_single_relevant_example() -> None:
    relevant = ["doc_b"]
    retrieved = ["doc_a", "doc_b", "doc_c"]

    assert hit_rate_at_k(relevant, retrieved, 5) == 1.0
    assert mrr_at_k(relevant, retrieved, 10) == 0.5
    assert recall_at_k(relevant, retrieved, 10) == 1.0


def test_metrics_multi_relevant_and_deduplication() -> None:
    relevant = ["doc_b", "doc_d"]
    retrieved = ["doc_a", "doc_b", "doc_b", "doc_c", "doc_d"]

    assert hit_rate_at_k(relevant, retrieved, 5) == 1.0
    assert mrr_at_k(relevant, retrieved, 10) == 0.5
    assert recall_at_k(relevant, retrieved, 10) == 1.0


def test_metrics_no_hit() -> None:
    relevant = ["doc_z"]
    retrieved = ["doc_a", "doc_b", "doc_c"]

    assert hit_rate_at_k(relevant, retrieved, 5) == 0.0
    assert mrr_at_k(relevant, retrieved, 10) == 0.0
    assert recall_at_k(relevant, retrieved, 10) == 0.0


def test_evaluate_retrieval_contract() -> None:
    result = evaluate_retrieval(
        [
            {
                "question": "q1",
                "relevant_doc_ids": ["doc_b"],
                "retrieved_doc_ids": ["doc_a", "doc_b"],
                "latency_ms": 10.0,
            },
            {
                "question": "q2",
                "relevant_doc_ids": ["doc_x"],
                "retrieved_doc_ids": ["doc_x", "doc_y"],
                "latency_ms": 20.0,
            },
        ]
    )

    assert result["hit_rate_at_5"] == 1.0
    assert result["mrr_at_10"] == 0.75
    assert result["recall_at_10"] == 1.0
    assert result["avg_retrieval_ms"] == 15.0
    assert result["p50_retrieval_ms"] == 15.0
    assert result["p95_retrieval_ms"] == 19.5


def test_golden_dataset_contract() -> None:
    dataset = json.loads(
        Path("tests/eval/retrieval_dataset.json").read_text(encoding="utf-8")
    )
    corpus_ids = {
        path.name
        for path in Path("data/rag-block-03").iterdir()
        if path.is_file()
    }

    assert len(dataset) == 24
    assert all(item["question"].strip() for item in dataset)
    assert all(1 <= len(item["relevant_doc_ids"]) <= 3 for item in dataset)
    assert all(set(item["relevant_doc_ids"]) <= corpus_ids for item in dataset)
    assert sum(len(item["relevant_doc_ids"]) == 2 for item in dataset) >= 4
    assert sum(len(item["relevant_doc_ids"]) == 3 for item in dataset) >= 2
    assert all(
        "10_office_plants.txt" not in item["relevant_doc_ids"]
        for item in dataset
    )
