from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from scripts.load_to_qdrant import build_points, embed_corpus, load_documents


EXPERIMENT_QUERIES = [
    "Не могу подключиться к рабочему VPN",
    "Как разблокировать учётную запись в CRM?",
    "Письма из корпоративной почты не отправляются",
    "В гарнитуре не работает микрофон",
    "Как запросить доступ к внутренней системе?",
]
COSINE_COLLECTION = "documents_cosine"
DOT_COLLECTION = "documents_dot"


def serialize_hits(points: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(point.id),
            "source": point.payload["source"],
            "title": point.payload["title"],
            "score": round(float(point.score), 6),
        }
        for point in points
    ]


async def delete_if_exists(client: AsyncQdrantClient, collection_name: str) -> None:
    collections = await client.get_collections()
    if collection_name in {item.name for item in collections.collections}:
        await client.delete_collection(collection_name)


async def run_experiments(local: bool) -> dict[str, Any]:
    settings = get_settings()
    documents = load_documents()
    vectors = embed_corpus(documents, settings)
    points = build_points(documents, vectors, settings.embedding_dim)

    local_client = AsyncQdrantClient(location=":memory:") if local else None
    production = VectorStore(settings, client=local_client)
    client = production.client

    cosine_store = VectorStore(
        settings,
        client=client,
        collection_name=COSINE_COLLECTION,
        distance=Distance.COSINE,
    )
    dot_store = VectorStore(
        settings,
        client=client,
        collection_name=DOT_COLLECTION,
        distance=Distance.DOT,
    )

    try:
        await production.ensure_collection()
        await production.upsert(points, batch_size=128)
        points_count_first = await production.points_count()
        await production.upsert(points, batch_size=128)
        points_count_second = await production.points_count()

        await delete_if_exists(client, COSINE_COLLECTION)
        await delete_if_exists(client, DOT_COLLECTION)
        await cosine_store.ensure_collection()
        await dot_store.ensure_collection()
        await cosine_store.upsert(points, batch_size=128)
        await dot_store.upsert(points, batch_size=128)

        metric_rows = []
        with EmbeddingService(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            cache_dir=settings.embedding_cache_dir,
        ) as embeddings:
            for query in EXPERIMENT_QUERIES:
                query_vector = embeddings.embed_query(query)
                cosine_hits = await cosine_store.search(query_vector, top_k=5)
                dot_hits = await dot_store.search(query_vector, top_k=5)
                cosine_ids = [str(point.id) for point in cosine_hits]
                dot_ids = [str(point.id) for point in dot_hits]
                metric_rows.append(
                    {
                        "query": query,
                        "cosine_ids": cosine_ids,
                        "dot_ids": dot_ids,
                        "same_ranking": cosine_ids == dot_ids,
                    }
                )

            match_query = embeddings.embed_query(
                "VPN не подключается и показывает ошибку"
            )
            match_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value="vpn"),
                    )
                ]
            )
            match_hits = await production.search(
                match_query,
                top_k=3,
                query_filter=match_filter,
            )

            date_query = embeddings.embed_query(
                "Как импортировать vpn-office-2024.conf после обновления VPN-клиента?"
            )
            without_date_filter = await production.search(date_query, top_k=3)
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            date_filter = Filter(
                must=[
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(gte=cutoff),
                    )
                ]
            )
            with_date_filter = await production.search(
                date_query,
                top_k=3,
                query_filter=date_filter,
            )

            composite_query = embeddings.embed_query(
                "Как создать обращение и приложить данные об ошибке?"
            )
            composite_filter = Filter(
                must=[
                    FieldCondition(
                        key="department",
                        match=MatchValue(value="contact_center"),
                    )
                ],
                must_not=[
                    FieldCondition(
                        key="is_archived",
                        match=MatchValue(value=True),
                    )
                ],
            )
            composite_hits = await production.search(
                composite_query,
                top_k=3,
                query_filter=composite_filter,
            )

        return {
            "mode": "in-memory" if local else "qdrant-server",
            "collection": settings.qdrant_collection,
            "points_count_first": points_count_first,
            "points_count_second": points_count_second,
            "idempotent": points_count_first == points_count_second,
            "metric_experiment": metric_rows,
            "filters": {
                "match_category_vpn": serialize_hits(match_hits),
                "datetime_cutoff": cutoff.isoformat(),
                "without_datetime_filter": serialize_hits(without_date_filter),
                "with_datetime_filter": serialize_hits(with_date_filter),
                "contact_center_without_archived": serialize_hits(composite_hits),
            },
        }
    finally:
        await delete_if_exists(client, COSINE_COLLECTION)
        await delete_if_exists(client, DOT_COLLECTION)
        if local_client is not None:
            await local_client.close()
        else:
            await production.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use qdrant-client in-memory mode instead of a Qdrant server.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = asyncio.run(run_experiments(local=arguments.local))
    print(json.dumps(result, ensure_ascii=False, indent=2))
