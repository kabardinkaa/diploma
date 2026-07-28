from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.core.config import Settings


PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,
    "created_at": PayloadSchemaType.DATETIME,
    "category": PayloadSchemaType.KEYWORD,
    "department": PayloadSchemaType.KEYWORD,
    "is_archived": PayloadSchemaType.BOOL,
}


class VectorStoreConfigurationError(RuntimeError):
    """Raised when an existing collection violates the embedding contract."""


class VectorStore:
    """Thin async boundary around one reusable Qdrant client."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: AsyncQdrantClient | Any | None = None,
        collection_name: str | None = None,
        distance: Distance = Distance.COSINE,
    ) -> None:
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        self.client = (
            client
            if client is not None
            else AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=api_key,
            )
        )
        self.collection_name = collection_name or settings.qdrant_collection
        self.embedding_dim = settings.embedding_dim
        self.distance = distance
        self._owns_client = client is None

    @staticmethod
    def _enum_value(value: Any) -> str:
        raw = getattr(value, "value", value)
        return str(raw).lower()

    def _validate_collection(self, collection_info: Any) -> None:
        vectors = collection_info.config.params.vectors
        if isinstance(vectors, dict):
            raise VectorStoreConfigurationError(
                f"Collection {self.collection_name!r} uses named vectors; "
                "one unnamed dense vector is required."
            )

        actual_size = getattr(vectors, "size", None)
        if actual_size != self.embedding_dim:
            raise VectorStoreConfigurationError(
                "Embedding dimension mismatch: "
                f"collection {self.collection_name!r} has {actual_size} dimensions, "
                f"but EMBEDDING_DIM={self.embedding_dim}."
            )

        actual_distance = getattr(vectors, "distance", None)
        if self._enum_value(actual_distance) != self._enum_value(self.distance):
            raise VectorStoreConfigurationError(
                f"Distance mismatch for collection {self.collection_name!r}: "
                f"expected {self.distance.value}, got {actual_distance}."
            )

    async def ensure_collection(self) -> None:
        collections = await self.client.get_collections()
        names = {collection.name for collection in collections.collections}

        if self.collection_name not in names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=self.distance,
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )

        collection_info = await self.client.get_collection(self.collection_name)
        self._validate_collection(collection_info)

        payload_schema = collection_info.payload_schema or {}
        for field_name, field_schema in PAYLOAD_INDEXES.items():
            if field_name in payload_schema:
                continue
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    async def upsert(
        self,
        points: list[PointStruct],
        batch_size: int = 256,
    ) -> None:
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=start + batch_size >= len(points),
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return list(result.points)

    async def points_count(self) -> int:
        collection_info = await self.client.get_collection(self.collection_name)
        return int(collection_info.points_count or 0)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()
