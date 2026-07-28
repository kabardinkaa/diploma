from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import structlog
from llama_index.core import (
    Settings as LlamaSettings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.core.config import Settings, get_settings
from app.services.chunking import build_e5_embedding
from app.services.rag_common import FALLBACK_ANSWER, RAG_SYSTEM_PROMPT

logger = structlog.get_logger("rag-service")

TEXT_QA_TEMPLATE = PromptTemplate(
    f"""{RAG_SYSTEM_PROMPT}

Контекст:
{{context_str}}

Вопрос пользователя:
{{query_str}}

Ответ:"""
)


class RAGCollectionError(RuntimeError):
    """Raised when an existing collection is not a LlamaIndex collection."""


class RAGService:
    def __init__(
        self,
        settings: Settings,
        *,
        qdrant_client: QdrantClient | Any | None = None,
        async_qdrant_client: AsyncQdrantClient | Any | None = None,
        openai_client: AsyncOpenAI | Any | None = None,
    ) -> None:
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        self.settings = settings
        self._client = qdrant_client or QdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
        self._aclient = async_qdrant_client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
        openai_kwargs: dict[str, Any] = {
            "api_key": settings.llm.api_key.get_secret_value(),
            "timeout": settings.llm.request_timeout,
            "max_retries": settings.llm.max_retries,
        }
        if settings.llm.base_url:
            openai_kwargs["base_url"] = settings.llm.base_url
        self._openai_client = openai_client or AsyncOpenAI(**openai_kwargs)
        self._owns_client = qdrant_client is None
        self._owns_async_client = async_qdrant_client is None
        self._owns_openai_client = openai_client is None

        self._embed_model: Any | None = None
        self._llm: Any | None = None
        self._node_parser: Any | None = None
        self._vector_store: Any | None = None
        self._index: Any | None = None
        self._retriever: Any | None = None
        self._query_engine: Any | None = None
        self._built = False
        self.indexed_on_build = False

    def _configure_llama_index(self) -> None:
        self._embed_model = build_e5_embedding(self.settings)
        llm_kwargs: dict[str, Any] = {
            "model": self.settings.llm.default_model,
            "temperature": 0.0,
            "api_key": self.settings.llm.api_key.get_secret_value(),
            "timeout": self.settings.llm.request_timeout,
            "max_retries": self.settings.llm.max_retries,
            "async_openai_client": self._openai_client,
        }
        if self.settings.llm.base_url:
            llm_kwargs["api_base"] = self.settings.llm.base_url
        if self.settings.llm.base_url:
            self._llm = OpenAILike(
                **llm_kwargs,
                context_window=32768,
                is_chat_model=True,
                is_function_calling_model=False,
            )
        else:
            self._llm = LlamaOpenAI(**llm_kwargs)
        self._node_parser = SentenceSplitter(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )
        LlamaSettings.embed_model = self._embed_model
        LlamaSettings.llm = self._llm
        LlamaSettings.node_parser = self._node_parser

    async def _collection_count(self) -> int:
        collections = await self._aclient.get_collections()
        names = {collection.name for collection in collections.collections}
        if self.settings.rag_collection not in names:
            return 0
        info = await self._aclient.get_collection(self.settings.rag_collection)
        return int(info.points_count or 0)

    async def _validate_existing_collection(self) -> None:
        info = await self._aclient.get_collection(self.settings.rag_collection)
        vectors = info.config.params.vectors
        configs = list(vectors.values()) if isinstance(vectors, dict) else [vectors]
        if not any(
            config.size == self.settings.embedding_dim
            and str(getattr(config.distance, "value", config.distance)).lower() == "cosine"
            for config in configs
        ):
            raise RAGCollectionError(
                "Existing LlamaIndex collection has incompatible vectors"
            )

        records, _ = await self._aclient.scroll(
            collection_name=self.settings.rag_collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        payload = records[0].payload if records else {}
        if "_node_content" not in (payload or {}):
            raise RAGCollectionError(
                "Existing collection does not contain LlamaIndex node payloads"
            )

    def _load_documents(self) -> list[Any]:
        data_dir = Path(self.settings.rag_data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"RAG data directory not found: {data_dir}")

        def metadata_for(path: str) -> dict[str, str]:
            file_name = Path(path).name
            return {"file_name": file_name, "source": file_name}

        documents = SimpleDirectoryReader(
            input_dir=str(data_dir),
            recursive=True,
            file_metadata=metadata_for,
        ).load_data()
        if len(documents) < 10:
            raise ValueError("RAG corpus must contain at least 10 documents")
        return documents

    async def _create_index(self) -> Any:
        documents = await asyncio.to_thread(self._load_documents)
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        return await asyncio.to_thread(
            VectorStoreIndex.from_documents,
            documents,
            storage_context=storage_context,
            transformations=[self._node_parser],
            show_progress=False,
        )

    async def _attach_index(self) -> Any:
        return await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            self._vector_store,
            embed_model=self._embed_model,
        )

    async def build(self) -> None:
        if self._built:
            return

        self._configure_llama_index()
        self._vector_store = QdrantVectorStore(
            collection_name=self.settings.rag_collection,
            client=self._client,
            aclient=self._aclient,
        )

        count = await self._collection_count()
        if count == 0:
            self._index = await self._create_index()
            self.indexed_on_build = True
        else:
            await self._validate_existing_collection()
            self._index = await self._attach_index()

        self._retriever = self._index.as_retriever(
            similarity_top_k=self.settings.rag_similarity_top_k
        )
        self._query_engine = self._index.as_query_engine(
            similarity_top_k=self.settings.rag_similarity_top_k,
            text_qa_template=TEXT_QA_TEMPLATE,
            response_mode="compact",
        )
        self._built = True

    @staticmethod
    def _sources(nodes: list[Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for item in nodes:
            node = item.node
            metadata = node.metadata or {}
            text = getattr(node, "text", None) or node.get_content()
            score = float(item.score or 0.0)
            sources.append(
                {
                    "text": text[:300],
                    "source": str(
                        metadata.get("file_name")
                        or metadata.get("source")
                        or "unknown"
                    ),
                    "score": round(score, 3),
                }
            )
        return sources

    async def answer(self, question: str) -> dict[str, Any]:
        if not self._built:
            raise RuntimeError("RAGService.build() must be called before answer()")

        started_at = time.perf_counter()
        candidates = list(await self._retriever.aretrieve(question))
        candidate_sources = self._sources(candidates)
        top_score = max((source["score"] for source in candidate_sources), default=0.0)
        fallback = not candidates or top_score < self.settings.rag_min_score

        if fallback:
            result = {
                "answer": FALLBACK_ANSWER,
                "top_score": round(top_score, 3),
                "sources": candidate_sources,
            }
        else:
            response = await self._query_engine.aquery(question)
            response_nodes = list(getattr(response, "source_nodes", []) or candidates)
            sources = self._sources(response_nodes)
            response_top_score = max(
                (source["score"] for source in sources),
                default=top_score,
            )
            result = {
                "answer": str(response),
                "top_score": round(response_top_score, 3),
                "sources": sources,
            }

        logger.info(
            "rag.query",
            top_score=result["top_score"],
            source_count=len(result["sources"]),
            fallback=fallback,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return result

    async def points_count(self) -> int:
        return await self._collection_count()

    async def close(self) -> None:
        if self._owns_client:
            await asyncio.to_thread(self._client.close)
        if self._owns_async_client:
            await self._aclient.close()
        if self._owns_openai_client:
            await self._openai_client.close()


async def _main() -> None:
    service = RAGService(get_settings())
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
