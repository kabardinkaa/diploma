from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from app.core.config import Settings, get_settings
from app.services.embeddings import EmbeddingService
from app.services.rag_common import FALLBACK_ANSWER, build_rag_prompt
from app.services.vector_store import VectorStore

logger = structlog.get_logger("rag-baremetal-service")

RAG_POINT_NAMESPACE = uuid.UUID("326115b1-22d8-51a6-9ea0-5867c2d20ef7")


def deterministic_point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(RAG_POINT_NAMESPACE, f"{source}:{chunk_index}"))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = chunk_size - chunk_overlap
    return [
        " ".join(words[start : start + chunk_size])
        for start in range(0, len(words), step)
    ]


class BareMetalRAGService:
    def __init__(
        self,
        settings: Settings,
        *,
        qdrant_client: AsyncQdrantClient | Any | None = None,
        embedding_service: EmbeddingService | Any | None = None,
        openai_client: AsyncOpenAI | Any | None = None,
    ) -> None:
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        self.settings = settings
        self._qdrant_client = qdrant_client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
        self._vector_store = VectorStore(
            settings,
            client=self._qdrant_client,
            collection_name=settings.rag_baremetal_collection,
        )
        self._embedding_service = embedding_service or EmbeddingService(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            cache_dir=settings.embedding_cache_dir,
        )
        openai_kwargs: dict[str, Any] = {
            "api_key": settings.llm.api_key.get_secret_value(),
            "timeout": settings.llm.request_timeout,
            "max_retries": settings.llm.max_retries,
        }
        if settings.llm.base_url:
            openai_kwargs["base_url"] = settings.llm.base_url
        self._openai_client = openai_client or AsyncOpenAI(**openai_kwargs)
        self._owns_qdrant = qdrant_client is None
        self._owns_embeddings = embedding_service is None
        self._owns_openai = openai_client is None
        self._built = False
        self.indexed_on_build = False

    def _read_chunks(self) -> list[tuple[str, int, str]]:
        data_dir = Path(self.settings.rag_data_dir)
        files = sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        )
        if len(files) < 10:
            raise ValueError("RAG corpus must contain at least 10 text files")

        chunks: list[tuple[str, int, str]] = []
        for path in files:
            source = path.name
            text = path.read_text(encoding="utf-8")
            for index, chunk in enumerate(
                chunk_text(
                    text,
                    self.settings.rag_chunk_size,
                    self.settings.rag_chunk_overlap,
                )
            ):
                chunks.append((source, index, chunk))
        return chunks

    async def build(self) -> None:
        if self._built:
            return

        await self._vector_store.ensure_collection()
        if await self._vector_store.points_count() == 0:
            chunks = await asyncio.to_thread(self._read_chunks)
            texts = [text for _, _, text in chunks]
            vectors = await asyncio.to_thread(
                self._embedding_service.embed_documents,
                texts,
            )
            points = [
                PointStruct(
                    id=deterministic_point_id(source, chunk_index),
                    vector=vector,
                    payload={
                        "text": text,
                        "source": source,
                        "chunk_index": chunk_index,
                    },
                )
                for (source, chunk_index, text), vector in zip(
                    chunks,
                    vectors,
                    strict=True,
                )
            ]
            await self._vector_store.upsert(points)
            self.indexed_on_build = True
        self._built = True

    @staticmethod
    def _sources(points: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "text": str((point.payload or {}).get("text", ""))[:300],
                "source": str((point.payload or {}).get("source", "unknown")),
                "score": round(float(point.score), 3),
            }
            for point in points
        ]

    async def answer(self, question: str) -> dict[str, Any]:
        if not self._built:
            raise RuntimeError(
                "BareMetalRAGService.build() must be called before answer()"
            )

        started_at = time.perf_counter()
        query_vector = await asyncio.to_thread(
            self._embedding_service.embed_query,
            question,
        )
        points = await self._vector_store.search(
            query_vector=query_vector,
            top_k=self.settings.rag_similarity_top_k,
        )
        sources = self._sources(points)
        top_score = max((source["score"] for source in sources), default=0.0)
        fallback = not points or top_score < self.settings.rag_min_score

        if fallback:
            answer = FALLBACK_ANSWER
        else:
            prompt = build_rag_prompt(
                question,
                (
                    "Источник: "
                    f"{(point.payload or {}).get('source', 'unknown')}\n"
                    f"{(point.payload or {}).get('text', '')}"
                    for point in points
                ),
            )
            response = await self._openai_client.chat.completions.create(
                model=self.settings.llm.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            answer = response.choices[0].message.content or FALLBACK_ANSWER

        result = {
            "answer": answer,
            "top_score": round(top_score, 3),
            "sources": sources,
        }
        logger.info(
            "rag.query",
            implementation="bare-metal",
            top_score=result["top_score"],
            source_count=len(sources),
            fallback=fallback,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return result

    async def points_count(self) -> int:
        return await self._vector_store.points_count()

    async def close(self) -> None:
        if self._owns_embeddings:
            await asyncio.to_thread(self._embedding_service.close)
        if self._owns_qdrant:
            await self._qdrant_client.close()
        if self._owns_openai:
            await self._openai_client.close()


async def _main() -> None:
    service = BareMetalRAGService(get_settings())
    try:
        await service.build()
        result = await service.answer(
            "Что проверить, если VPN подключился, но внутренние ресурсы не открываются?"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(_main())
