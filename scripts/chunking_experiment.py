from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import get_tokenizer
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings, get_settings
from app.services.chunking import (
    ChunkingStrategy,
    build_chunk_parser,
    build_e5_embedding,
)
from app.services.embeddings import EmbeddingService
from app.services.reranker import Reranker
from app.services.retrieval_eval import evaluate_retrieval

DATASET_PATH = PROJECT_ROOT / "tests/eval/retrieval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "tests/eval/chunking_results.json"
COLLECTIONS: dict[ChunkingStrategy, str] = {
    "fixed": "docs_fixed",
    "recursive": "docs_recursive",
    "semantic": "docs_semantic",
}
POINT_NAMESPACE = uuid.UUID("38f07c73-17ed-5541-a126-9ad14c2d70dd")

TUNING_CONFIGS = [
    {
        "experiment": "baseline",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "candidate_top_k": 10,
    },
    {
        "experiment": "smaller_chunks",
        "chunk_size": 256,
        "chunk_overlap": 64,
        "candidate_top_k": 10,
    },
    {
        "experiment": "larger_chunks",
        "chunk_size": 1024,
        "chunk_overlap": 64,
        "candidate_top_k": 10,
    },
    {
        "experiment": "less_overlap",
        "chunk_size": 512,
        "chunk_overlap": 32,
        "candidate_top_k": 10,
    },
    {
        "experiment": "wider_retrieval",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "candidate_top_k": 20,
    },
]


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_sha256(path: Path = DATASET_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(settings: Settings) -> list[Document]:
    data_dir = PROJECT_ROOT / settings.rag_data_dir
    files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    return [
        Document(
            text=path.read_text(encoding="utf-8"),
            id_=path.name,
            metadata={
                "doc_id": path.name,
                "source": path.name,
                "file_name": path.name,
            },
        )
        for path in files
    ]


def validate_dataset(
    dataset: list[dict[str, Any]],
    corpus_doc_ids: set[str],
    *,
    minimum_questions: int = 20,
) -> None:
    if len(dataset) < minimum_questions:
        raise ValueError(
            f"Golden dataset must contain at least {minimum_questions} questions"
        )
    for index, item in enumerate(dataset):
        question = item.get("question", "")
        relevant = item.get("relevant_doc_ids", [])
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {index} is empty")
        if not 1 <= len(relevant) <= 3:
            raise ValueError(
                f"Question {index} must have between 1 and 3 relevant_doc_ids"
            )
        unknown = set(relevant) - corpus_doc_ids
        if unknown:
            raise ValueError(
                f"Question {index} references unknown doc ids: {sorted(unknown)}"
            )


async def create_nodes(
    strategy: ChunkingStrategy,
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Any,
    semantic_embed_model: Any | None,
) -> list[Any]:
    parser = build_chunk_parser(
        strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
        embed_model=semantic_embed_model,
    )
    if strategy == "semantic":
        nodes = await parser.aget_nodes_from_documents(
            documents,
            show_progress=False,
        )
    else:
        nodes = await asyncio.to_thread(
            parser.get_nodes_from_documents,
            documents,
            show_progress=False,
        )

    for node in nodes:
        doc_id = str(
            node.metadata.get("doc_id")
            or node.metadata.get("file_name")
            or node.ref_doc_id
        )
        node.metadata.update(
            {
                "doc_id": doc_id,
                "source": doc_id,
                "file_name": doc_id,
            }
        )
    return list(nodes)


def chunk_statistics(
    nodes: list[Any],
    document_count: int,
    tokenizer: Any,
) -> dict[str, Any]:
    token_lengths = [
        len(tokenizer(node.get_content(metadata_mode=MetadataMode.NONE)))
        for node in nodes
    ]
    ordered = sorted(token_lengths)
    middle = len(ordered) // 2
    median = (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "document_count": document_count,
        "total_chunks": len(nodes),
        "avg_chunks_per_document": len(nodes) / document_count,
        "min_chunk_tokens": min(token_lengths),
        "max_chunk_tokens": max(token_lengths),
        "avg_chunk_tokens": sum(token_lengths) / len(token_lengths),
        "median_chunk_tokens": median,
    }


async def reset_collection(
    client: AsyncQdrantClient,
    name: str,
    dimension: int,
) -> None:
    if await client.collection_exists(name):
        await client.delete_collection(name)
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    await client.create_payload_index(
        collection_name=name,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )


def point_id(collection: str, doc_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return str(
        uuid.uuid5(
            POINT_NAMESPACE,
            f"{collection}:{doc_id}:{chunk_index}:{digest}",
        )
    )


async def index_nodes(
    client: AsyncQdrantClient,
    collection: str,
    nodes: list[Any],
    embedding_service: EmbeddingService,
) -> int:
    texts = [
        node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    ]
    vectors = await asyncio.to_thread(
        embedding_service.embed_documents,
        texts,
    )
    per_document_indexes: dict[str, int] = {}
    points: list[PointStruct] = []
    for node, text, vector in zip(nodes, texts, vectors, strict=True):
        doc_id = str(node.metadata["doc_id"])
        chunk_index = per_document_indexes.get(doc_id, 0)
        per_document_indexes[doc_id] = chunk_index + 1
        points.append(
            PointStruct(
                id=point_id(collection, doc_id, chunk_index, text),
                vector=vector,
                payload={
                    "text": text,
                    "doc_id": doc_id,
                    "source": str(node.metadata["source"]),
                    "file_name": str(node.metadata["file_name"]),
                    "chunk_index": chunk_index,
                },
            )
        )
    for start in range(0, len(points), 256):
        await client.upsert(
            collection_name=collection,
            points=points[start : start + 256],
            wait=True,
        )
    info = await client.get_collection(collection)
    return int(info.points_count or 0)


async def retrieve_candidates(
    client: AsyncQdrantClient,
    collection: str,
    question: str,
    embedding_service: EmbeddingService,
    top_k: int,
) -> list[dict[str, Any]]:
    vector = await asyncio.to_thread(embedding_service.embed_query, question)
    response = await client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": str((point.payload or {}).get("text", "")),
            "doc_id": str((point.payload or {}).get("doc_id", "")),
            "source": str((point.payload or {}).get("source", "")),
            "file_name": str((point.payload or {}).get("file_name", "")),
            "dense_score": float(point.score),
        }
        for point in response.points
    ]


async def evaluate_collection(
    client: AsyncQdrantClient,
    collection: str,
    dataset: list[dict[str, Any]],
    embedding_service: EmbeddingService,
    *,
    candidate_k: int,
    reranker: Reranker | None = None,
    rerank_top_n: int = 10,
) -> dict[str, Any]:
    warmup = await retrieve_candidates(
        client,
        collection,
        dataset[0]["question"],
        embedding_service,
        candidate_k,
    )
    if reranker is not None:
        await asyncio.to_thread(
            reranker.rerank,
            dataset[0]["question"],
            warmup,
            rerank_top_n,
        )

    question_results: list[dict[str, Any]] = []
    for item in dataset:
        started_at = time.perf_counter()
        candidates = await retrieve_candidates(
            client,
            collection,
            item["question"],
            embedding_service,
            candidate_k,
        )
        if reranker is not None:
            candidates = await asyncio.to_thread(
                reranker.rerank,
                item["question"],
                candidates,
                rerank_top_n,
            )
        latency_ms = (time.perf_counter() - started_at) * 1000
        question_results.append(
            {
                "question": item["question"],
                "relevant_doc_ids": item["relevant_doc_ids"],
                "retrieved_doc_ids": [
                    candidate["doc_id"] for candidate in candidates
                ],
                "scores": [
                    {
                        "doc_id": candidate["doc_id"],
                        "dense_score": candidate["dense_score"],
                        **(
                            {"reranker_score": candidate["reranker_score"]}
                            if "reranker_score" in candidate
                            else {}
                        ),
                    }
                    for candidate in candidates
                ],
                "latency_ms": latency_ms,
            }
        )
    return evaluate_retrieval(question_results)


async def run_index_experiment(
    *,
    strategy: ChunkingStrategy,
    collection: str,
    documents: list[Document],
    dataset: list[dict[str, Any]],
    tokenizer: Any,
    semantic_embed_model: Any | None,
    embedding_service: EmbeddingService,
    client: AsyncQdrantClient,
    settings: Settings,
    chunk_size: int,
    chunk_overlap: int,
    candidate_top_k: int,
) -> dict[str, Any]:
    nodes = await create_nodes(
        strategy,
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
        semantic_embed_model=semantic_embed_model,
    )
    stats = chunk_statistics(nodes, len(documents), tokenizer)
    await reset_collection(client, collection, settings.embedding_dim)
    points_count = await index_nodes(
        client,
        collection,
        nodes,
        embedding_service,
    )
    if points_count != stats["total_chunks"]:
        raise RuntimeError(
            f"{collection}: points_count={points_count}, "
            f"total_chunks={stats['total_chunks']}"
        )
    metrics = await evaluate_collection(
        client,
        collection,
        dataset,
        embedding_service,
        candidate_k=candidate_top_k,
    )
    return {
        "strategy": strategy,
        "collection": collection,
        "params": {
            "chunk_size": chunk_size if strategy != "semantic" else None,
            "chunk_overlap": chunk_overlap if strategy != "semantic" else None,
            "buffer_size": 1 if strategy == "semantic" else None,
            "breakpoint_percentile_threshold": (
                95 if strategy == "semantic" else None
            ),
            "candidate_top_k": candidate_top_k,
        },
        "points_count": points_count,
        "chunk_stats": stats,
        "metrics": metrics,
    }


def choose_best_strategy(
    strategy_results: dict[str, dict[str, Any]],
) -> str:
    candidates = list(strategy_results.items())
    for metric_name in ("hit_rate_at_5", "mrr_at_10", "recall_at_10"):
        best_value = max(item[1]["metrics"][metric_name] for item in candidates)
        candidates = [
            item
            for item in candidates
            if abs(item[1]["metrics"][metric_name] - best_value) < 1e-12
        ]

    minimum_latency = min(
        item[1]["metrics"]["avg_retrieval_ms"] for item in candidates
    )
    latency_tolerance = max(1.0, minimum_latency * 0.05)
    candidates = [
        item
        for item in candidates
        if item[1]["metrics"]["avg_retrieval_ms"]
        <= minimum_latency + latency_tolerance
    ]
    return min(
        candidates,
        key=lambda item: (
            item[1]["points_count"],
            item[1]["metrics"]["avg_retrieval_ms"],
            item[0],
        ),
    )[0]


def choose_best_tuning(
    tuning_results: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = list(tuning_results)
    for metric_name in ("hit_rate_at_5", "mrr_at_10", "recall_at_10"):
        best_value = max(item["metrics"][metric_name] for item in candidates)
        candidates = [
            item
            for item in candidates
            if abs(item["metrics"][metric_name] - best_value) < 1e-12
        ]
    minimum_latency = min(
        item["metrics"]["avg_retrieval_ms"] for item in candidates
    )
    candidates = [
        item
        for item in candidates
        if item["metrics"]["avg_retrieval_ms"] <= minimum_latency + 1.0
    ]
    minimum_points = min(item["points_count"] for item in candidates)
    candidates = [
        item for item in candidates if item["points_count"] == minimum_points
    ]
    preference = {
        config["experiment"]: index
        for index, config in enumerate(TUNING_CONFIGS)
    }
    return min(candidates, key=lambda item: preference[item["experiment"]])


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    dataset = load_dataset()
    documents = load_corpus(settings)
    corpus_ids = {str(document.metadata["doc_id"]) for document in documents}
    validate_dataset(dataset, corpus_ids)

    tokenizer = get_tokenizer()
    semantic_embed_model = build_e5_embedding(settings)
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.embedding_cache_dir,
    )
    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key
        else None
    )
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
    )

    try:
        await client.get_collections()
        strategy_results: dict[str, dict[str, Any]] = {}
        for strategy, collection in COLLECTIONS.items():
            print(f"Running strategy: {strategy}")
            strategy_results[strategy] = await run_index_experiment(
                strategy=strategy,
                collection=collection,
                documents=documents,
                dataset=dataset,
                tokenizer=tokenizer,
                semantic_embed_model=semantic_embed_model,
                embedding_service=embedding_service,
                client=client,
                settings=settings,
                chunk_size=512,
                chunk_overlap=64,
                candidate_top_k=10,
            )

        best_strategy = choose_best_strategy(strategy_results)
        best_nonsemantic = choose_best_strategy(
            {
                key: value
                for key, value in strategy_results.items()
                if key != "semantic"
            }
        )
        tuning_strategy = (
            best_nonsemantic if best_strategy == "semantic" else best_strategy
        )

        tuning_results: list[dict[str, Any]] = []
        for config in TUNING_CONFIGS:
            collection = (
                f"tuning_{tuning_strategy}_{config['experiment']}"
            )
            print(f"Running tuning: {config['experiment']}")
            result = await run_index_experiment(
                strategy=tuning_strategy,
                collection=collection,
                documents=documents,
                dataset=dataset,
                tokenizer=tokenizer,
                semantic_embed_model=semantic_embed_model,
                embedding_service=embedding_service,
                client=client,
                settings=settings,
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                candidate_top_k=config["candidate_top_k"],
            )
            result["experiment"] = config["experiment"]
            tuning_results.append(result)
            await client.delete_collection(collection)

        best_collection = COLLECTIONS[best_strategy]
        reranker_before = await evaluate_collection(
            client,
            best_collection,
            dataset,
            embedding_service,
            candidate_k=20,
        )
        reranker_result: dict[str, Any]
        if args.skip_reranker:
            reranker_result = {
                "status": "skipped",
                "model": args.reranker_model,
                "before": reranker_before,
                "after": None,
                "error": "Skipped by command-line option",
            }
        else:
            reranker = Reranker(
                args.reranker_model,
                cache_folder=settings.embedding_cache_dir,
            )
            try:
                print(f"Running reranker: {args.reranker_model}")
                after = await evaluate_collection(
                    client,
                    best_collection,
                    dataset,
                    embedding_service,
                    candidate_k=20,
                    reranker=reranker,
                    rerank_top_n=10,
                )
                reranker_result = {
                    "status": "completed",
                    "model": args.reranker_model,
                    "candidate_k": 20,
                    "top_n": 10,
                    "before": reranker_before,
                    "after": after,
                    "error": None,
                }
            except Exception as exc:
                reranker_result = {
                    "status": "failed",
                    "model": args.reranker_model,
                    "candidate_k": 20,
                    "top_n": 10,
                    "before": reranker_before,
                    "after": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        best_tuning = choose_best_tuning(tuning_results)
        before_quality = (
            reranker_before["hit_rate_at_5"],
            reranker_before["mrr_at_10"],
            reranker_before["recall_at_10"],
        )
        after_metrics = reranker_result.get("after")
        after_quality = (
            (
                after_metrics["hit_rate_at_5"],
                after_metrics["mrr_at_10"],
                after_metrics["recall_at_10"],
            )
            if after_metrics
            else None
        )
        reranker_enabled = bool(
            after_quality is not None and after_quality > before_quality
        )
        result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": {
                "path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
                "sha256": dataset_sha256(),
                "question_count": len(dataset),
                "single_doc_questions": sum(
                    len(item["relevant_doc_ids"]) == 1 for item in dataset
                ),
                "multi_doc_questions": sum(
                    len(item["relevant_doc_ids"]) > 1 for item in dataset
                ),
            },
            "corpus": {
                "path": str(settings.rag_data_dir),
                "document_count": len(documents),
                "doc_ids": sorted(corpus_ids),
            },
            "models": {
                "embedding": settings.embedding_model,
                "dimension": settings.embedding_dim,
                "distance": "COSINE",
                "reranker": args.reranker_model,
            },
            "strategies": strategy_results,
            "best_strategy": best_strategy,
            "tuning_strategy": tuning_strategy,
            "tuning": tuning_results,
            "reranker": reranker_result,
            "final_configuration": {
                "strategy": tuning_strategy,
                "experiment": best_tuning["experiment"],
                "chunk_size": best_tuning["params"]["chunk_size"],
                "chunk_overlap": best_tuning["params"]["chunk_overlap"],
                "retrieval_top_k": best_tuning["params"]["candidate_top_k"],
                "reranker_enabled": reranker_enabled,
                "reranker_model": args.reranker_model,
                "rerank_top_n": 10,
                "hit_rate_at_5": best_tuning["metrics"]["hit_rate_at_5"],
            },
        }
        RESULTS_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
    finally:
        embedding_service.close()
        await client.close()


def print_summary(result: dict[str, Any]) -> None:
    print("\nStrategy summary")
    print(
        "strategy | chunks | avg_tokens | hit@5 | mrr@10 | "
        "recall@10 | avg_ms"
    )
    for strategy, item in result["strategies"].items():
        stats = item["chunk_stats"]
        metrics = item["metrics"]
        print(
            f"{strategy} | {item['points_count']} | "
            f"{stats['avg_chunk_tokens']:.2f} | "
            f"{metrics['hit_rate_at_5']:.3f} | "
            f"{metrics['mrr_at_10']:.3f} | "
            f"{metrics['recall_at_10']:.3f} | "
            f"{metrics['avg_retrieval_ms']:.2f}"
        )
    print(f"best_strategy: {result['best_strategy']}")
    print(f"reranker_status: {result['reranker']['status']}")
    print(f"results: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--skip-reranker", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print_summary(asyncio.run(run(parse_args())))
