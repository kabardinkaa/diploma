from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import structlog
from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.core.config import Settings, get_settings
from app.services.chunking import build_e5_embedding
from app.services.rag_common import FALLBACK_ANSWER
from app.services.reranker import Reranker

logger = structlog.get_logger("rag-service")
_CITATION_PATTERN = re.compile(r"\[(\d+)]")
_FOLLOW_UP_MARKERS = (
    "а если",
    "а для",
    "после этого",
    "тогда",
    "это",
    "такое",
    "их",
    "ним",
)

RAG_GENERATION_PROMPT = """\
Ты — ассистент внутренней технической поддержки.
Отвечай только по приведённому контексту и не используй внешние знания.
Каждое фактическое утверждение сопровождай ссылкой [N] на источник.
Не придумывай номера источников. Если источники противоречат друг другу,
прямо сообщи об этом. Отвечай по-русски, кратко и по существу."""

CONDENSE_PROMPT = """\
Перепиши последний вопрос пользователя в самодостаточный поисковый запрос.
Учитывай только историю диалога. Не отвечай на вопрос, не добавляй факты.
Верни только одну строку поискового запроса."""


class RAGCollectionError(RuntimeError):
    """Raised when an existing collection has an incompatible vector schema."""


def normalize_citations(answer: str, valid_ids: set[int]) -> str:
    def replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid_ids else ""

    normalized = _CITATION_PATTERN.sub(replace, answer)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized).strip()
    if valid_ids and not any(
        int(item) in valid_ids for item in _CITATION_PATTERN.findall(normalized)
    ):
        normalized = f"{normalized} [1]".strip()
    return normalized


def sanitize_sse_payload(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class RAGService:
    def __init__(
        self,
        settings: Settings,
        *,
        qdrant_client: QdrantClient | Any | None = None,
        async_qdrant_client: AsyncQdrantClient | Any | None = None,
        openai_client: AsyncOpenAI | Any | None = None,
        embed_model: Any | None = None,
        reranker: Reranker | Any | None = None,
    ) -> None:
        api_key_value = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        api_key = api_key_value or None
        self.settings = settings
        self._client = qdrant_client or QdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
        self._aclient = async_qdrant_client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
        kwargs: dict[str, Any] = {
            "api_key": settings.llm.api_key.get_secret_value(),
            "timeout": settings.llm.request_timeout,
            "max_retries": settings.llm.max_retries,
        }
        if settings.llm.base_url:
            kwargs["base_url"] = settings.llm.base_url
        self._openai_client = openai_client or AsyncOpenAI(**kwargs)
        self._owns_client = qdrant_client is None
        self._owns_async_client = async_qdrant_client is None
        self._owns_openai_client = openai_client is None
        self._embed_model = embed_model
        self._reranker = reranker
        self._vector_store: Any | None = None
        self._index: Any | None = None
        self._retriever: Any | None = None
        self._built = False
        self.indexed_on_build = False
        self.last_condensed_query: str | None = None

    @property
    def collection_name(self) -> str:
        return getattr(
            self.settings,
            "rag_production_collection",
            self.settings.rag_collection,
        )

    @property
    def retrieval_top_k(self) -> int:
        return getattr(
            self.settings,
            "rag_retrieval_top_k",
            self.settings.rag_similarity_top_k,
        )

    def _configure_llama_index(self) -> None:
        self._embed_model = self._embed_model or build_e5_embedding(self.settings)
        LlamaSettings.embed_model = self._embed_model

    async def _collection_count(self) -> int:
        try:
            info = await self._aclient.get_collection(self.collection_name)
        except Exception:
            return 0
        return int(info.points_count or 0)

    async def _validate_existing_collection(self) -> None:
        info = await self._aclient.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        configs = list(vectors.values()) if isinstance(vectors, dict) else [vectors]
        if not any(
            config.size == self.settings.embedding_dim
            and str(getattr(config.distance, "value", config.distance)).lower()
            == "cosine"
            for config in configs
        ):
            raise RAGCollectionError("corporate_rag has incompatible vectors")

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
            collection_name=self.collection_name,
            client=self._client,
            aclient=self._aclient,
        )
        if await self._collection_count():
            await self._validate_existing_collection()
        self._index = await self._attach_index()
        self._retriever = self._index.as_retriever(
            similarity_top_k=self.retrieval_top_k
        )
        self._built = True

    @staticmethod
    def _history_messages(history: Sequence[Any] | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history or []:
            if isinstance(item, dict):
                role, content = item.get("role"), item.get("content")
            else:
                role, content = getattr(item, "role", None), getattr(item, "content", None)
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _is_follow_up(question: str) -> bool:
        lowered = question.strip().lower()
        return len(lowered.split()) <= 14 and any(
            marker in lowered for marker in _FOLLOW_UP_MARKERS
        )

    async def _condense(
        self,
        question: str,
        history: Sequence[Any] | None,
    ) -> tuple[str, bool]:
        messages = self._history_messages(history)
        enabled = getattr(self.settings, "rag_condense_enabled", True)
        if not enabled or not messages or not self._is_follow_up(question):
            return question, False
        dialogue = "\n".join(
            f"{item['role']}: {item['content']}" for item in messages[-8:]
        )
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.settings.llm.default_model,
                temperature=0,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    {
                        "role": "user",
                        "content": f"История:\n{dialogue}\n\nВопрос:\n{question}",
                    },
                ],
            )
            condensed = (response.choices[0].message.content or "").strip()
            if condensed:
                self.last_condensed_query = condensed
                return condensed, True
        except Exception:
            logger.warning("rag.condense_failed", exc_info=True)
        return question, False

    @staticmethod
    def _candidate(item: Any) -> dict[str, Any]:
        node = item.node
        metadata = node.metadata or {}
        text = getattr(node, "text", None) or node.get_content()
        return {
            "text": text,
            "file_name": str(
                metadata.get("file_name") or metadata.get("source") or "unknown"
            ),
            "page": metadata.get("page"),
            "dense_score": float(item.score or 0.0),
        }

    def _sources(self, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        maximum = getattr(self.settings, "rag_max_sources", 5)
        return [
            {
                "id": index,
                "file_name": candidate["file_name"],
                "page": candidate.get("page"),
                "score": round(candidate["dense_score"], 3),
                "snippet": candidate["text"][:300],
            }
            for index, candidate in enumerate(candidates[:maximum], start=1)
        ]

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not getattr(self.settings, "rag_reranker_enabled", False):
            return candidates
        if self._reranker is None:
            self._reranker = Reranker(
                self.settings.rag_reranker_model,
                cache_folder=self.settings.embedding_cache_dir,
            )
        return self._reranker.rerank(
            query,
            candidates,
            top_n=getattr(self.settings, "rag_rerank_top_n", 5),
        )

    async def _retrieve(
        self,
        question: str,
        history: Sequence[Any] | None,
    ) -> dict[str, Any]:
        search_query, condensed = await self._condense(question, history)
        started = time.perf_counter()
        try:
            nodes = list(await self._retriever.aretrieve(search_query))
        except Exception:
            logger.warning("rag.retrieve_failed", exc_info=True)
            nodes = []
        candidates = [self._candidate(item) for item in nodes]
        top_score = max(
            (candidate["dense_score"] for candidate in candidates),
            default=0.0,
        )
        candidates = self._rerank_candidates(search_query, candidates)
        confident = bool(candidates) and top_score >= self.settings.rag_min_score
        return {
            "search_query": search_query,
            "condensed": condensed,
            "candidates": candidates,
            "sources": self._sources(candidates) if confident else [],
            "top_score": round(top_score, 3),
            "confident": confident,
            "retrieve_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    @staticmethod
    def _numbered_context(candidates: Sequence[dict[str, Any]], maximum: int) -> str:
        return "\n\n".join(
            f"[{index}]\n{candidate['text']}"
            for index, candidate in enumerate(candidates[:maximum], start=1)
        )

    async def stream_answer(
        self,
        question: str,
        history: Sequence[Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if not self._built:
            raise RuntimeError("RAGService.build() must be called before stream_answer()")
        total_started = time.perf_counter()
        retrieval = await self._retrieve(question, history)
        if not retrieval["confident"]:
            yield {"type": "token", "delta": FALLBACK_ANSWER}
            answer = FALLBACK_ANSWER
            generate_ms = 0.0
        else:
            maximum = getattr(self.settings, "rag_max_sources", 5)
            context = self._numbered_context(retrieval["candidates"], maximum)
            messages = [
                {"role": "system", "content": RAG_GENERATION_PROMPT},
                *self._history_messages(history)[-8:],
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос:\n{question}",
                },
            ]
            generate_started = time.perf_counter()
            stream = await self._openai_client.chat.completions.create(
                model=self.settings.llm.default_model,
                temperature=0.1,
                max_tokens=500,
                messages=messages,
                stream=True,
            )
            chunks: list[str] = []
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    chunks.append(delta)
                    yield {"type": "token", "delta": delta}
            answer = normalize_citations(
                "".join(chunks).strip(),
                {source["id"] for source in retrieval["sources"]},
            )
            generate_ms = round((time.perf_counter() - generate_started) * 1000, 2)

        event = {
            "type": "sources",
            "answer": answer,
            "sources": retrieval["sources"],
            "top_score": retrieval["top_score"],
            "confident": retrieval["confident"],
            "condensed_query": (
                retrieval["search_query"] if retrieval["condensed"] else None
            ),
        }
        yield event
        logger.info(
            "rag.query",
            question_hash=hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            top_score=retrieval["top_score"],
            confident=retrieval["confident"],
            source_count=len(retrieval["sources"]),
            condensed=retrieval["condensed"],
            reranker_enabled=getattr(self.settings, "rag_reranker_enabled", False),
            retrieve_ms=retrieval["retrieve_ms"],
            generate_ms=generate_ms,
            total_ms=round((time.perf_counter() - total_started) * 1000, 2),
        )

    async def answer(
        self,
        question: str,
        history: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        final: dict[str, Any] | None = None
        async for event in self.stream_answer(question, history):
            if event["type"] == "sources":
                final = event
        if final is None:
            raise RuntimeError("RAG stream ended without a final event")
        return {
            "answer": final["answer"],
            "top_score": final["top_score"],
            "confident": final["confident"],
            "sources": final["sources"],
        }

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
        print(await service.answer("Как восстановить доступ к VPN?"))
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(_main())
