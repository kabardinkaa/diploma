from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings, get_settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


CORPUS_PATH = PROJECT_ROOT / "data/support_kb.json"
SUPPORT_KB_NAMESPACE = uuid.UUID("ed4a5cb4-f202-5eb8-9bd3-a1404c1d13c6")
REQUIRED_FIELDS = {
    "source",
    "title",
    "text",
    "created_at",
    "category",
    "department",
    "is_archived",
}


def stable_point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(SUPPORT_KB_NAMESPACE, f"{source}:{chunk_index}"))


def load_documents(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    documents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(documents, list) or len(documents) < 100:
        raise ValueError("support corpus must contain at least 100 documents")

    business_keys: set[tuple[str, int]] = set()
    for index, document in enumerate(documents):
        missing = REQUIRED_FIELDS - set(document)
        if missing:
            raise ValueError(
                f"document {index} is missing fields: {sorted(missing)}"
            )
        if not all(document[field] for field in REQUIRED_FIELDS - {"is_archived"}):
            raise ValueError(f"document {index} contains an empty required field")
        if not isinstance(document["is_archived"], bool):
            raise ValueError(f"document {index} has non-boolean is_archived")

        datetime.fromisoformat(document["created_at"].replace("Z", "+00:00"))
        chunk_index = int(document.get("chunk_index", 0))
        business_key = (document["source"], chunk_index)
        if business_key in business_keys:
            raise ValueError(f"duplicate document business key: {business_key}")
        business_keys.add(business_key)

    return documents


def validate_embedding_dimension(vector: list[float], expected_dim: int) -> None:
    actual_dim = len(vector)
    if actual_dim != expected_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"model produced {actual_dim} dimensions, "
            f"but EMBEDDING_DIM={expected_dim} and collection expects {expected_dim}."
        )


def embed_corpus(
    documents: list[dict[str, Any]],
    settings: Settings,
) -> list[list[float]]:
    with EmbeddingService(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.embedding_cache_dir,
    ) as embeddings:
        sample_vector = embeddings.embed_documents([documents[0]["text"]])[0]
        validate_embedding_dimension(sample_vector, settings.embedding_dim)

        vectors: list[list[float]] = []
        starts = range(0, len(documents), settings.embedding_batch_size)
        for start in tqdm(
            starts,
            desc="Embedding support corpus",
            unit="batch",
        ):
            batch = documents[start : start + settings.embedding_batch_size]
            batch_vectors = embeddings.embed_documents(
                [document["text"] for document in batch]
            )
            for vector in batch_vectors:
                validate_embedding_dimension(vector, settings.embedding_dim)
            vectors.extend(batch_vectors)
    return vectors


def build_points(
    documents: list[dict[str, Any]],
    vectors: list[list[float]],
    expected_dim: int,
) -> list[PointStruct]:
    if len(documents) != len(vectors):
        raise ValueError("documents and vectors counts do not match")

    points: list[PointStruct] = []
    for document, vector in zip(documents, vectors, strict=True):
        validate_embedding_dimension(vector, expected_dim)
        chunk_index = int(document.get("chunk_index", 0))
        payload = {
            field: document[field]
            for field in (
                "source",
                "title",
                "text",
                "created_at",
                "category",
                "department",
                "is_archived",
            )
        }
        payload["chunk_index"] = chunk_index
        points.append(
            PointStruct(
                id=stable_point_id(document["source"], chunk_index),
                vector=vector,
                payload=payload,
            )
        )
    return points


async def load_to_qdrant() -> int:
    settings = get_settings()
    documents = load_documents()
    vectors = embed_corpus(documents, settings)
    points = build_points(documents, vectors, settings.embedding_dim)

    store = VectorStore(settings)
    try:
        await store.ensure_collection()
        await store.upsert(points, batch_size=128)
        points_count = await store.points_count()
    finally:
        await store.close()

    print(f"collection: {settings.qdrant_collection}")
    print(f"embedding model: {settings.embedding_model}")
    print(f"dimension: {settings.embedding_dim}")
    print(f"processed documents: {len(documents)}")
    print(f"points_count: {points_count}")
    return points_count


if __name__ == "__main__":
    asyncio.run(load_to_qdrant())
