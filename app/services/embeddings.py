from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from diskcache import Cache

from app.core.config import get_settings

EmbeddingKind = Literal["plain", "query", "passage"]
ModelFactory = Callable[[str], Any]


class EmbeddingService:
    """Local E5 embeddings with batching and a persistent, model-aware cache."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        cache_dir: str | Path = ".cache/embeddings",
        model_factory: ModelFactory | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self._model_factory = model_factory or self._load_sentence_transformer
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._cache = Cache(str(self.cache_dir))

    @staticmethod
    def _load_sentence_transformer(model_name: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_factory(self.model_name)
        return self._model

    def _cache_key(self, kind: EmbeddingKind, text: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "kind": kind,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _prefixed_text(kind: EmbeddingKind, text: str) -> str:
        if kind == "query":
            return f"query: {text}"
        if kind == "passage":
            return f"passage: {text}"
        return text

    @staticmethod
    def _as_float_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    @classmethod
    def _normalize(cls, vector: Any) -> list[float]:
        values = cls._as_float_list(vector)
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

    def _embed(self, texts: Sequence[str], kind: EmbeddingKind) -> list[list[float]]:
        if not texts:
            return []

        raw_texts = list(texts)
        if any(not isinstance(text, str) for text in raw_texts):
            raise TypeError("all texts must be strings")

        results: list[list[float] | None] = [None] * len(raw_texts)
        misses: dict[str, tuple[str, list[int]]] = {}

        for index, text in enumerate(raw_texts):
            key = self._cache_key(kind, text)
            cached = self._cache.get(key)
            if cached is not None:
                results[index] = self._as_float_list(cached)
                continue

            if key not in misses:
                misses[key] = (text, [])
            misses[key][1].append(index)

        missing_items = list(misses.items())
        for start in range(0, len(missing_items), self.batch_size):
            batch_items = missing_items[start : start + self.batch_size]
            batch_texts = [
                self._prefixed_text(kind, raw_text)
                for _, (raw_text, _) in batch_items
            ]
            encoded = self._get_model().encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            if len(encoded) != len(batch_items):
                raise ValueError("embedding model returned an unexpected batch size")

            for (key, (_, indexes)), vector in zip(batch_items, encoded, strict=True):
                normalized = self._normalize(vector)
                self._cache.set(key, normalized)
                for index in indexes:
                    results[index] = normalized.copy()

        if any(vector is None for vector in results):
            raise RuntimeError("failed to produce all requested embeddings")
        return [vector for vector in results if vector is not None]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "plain")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> EmbeddingService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_default_service: EmbeddingService | None = None
_default_service_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    global _default_service
    if _default_service is None:
        with _default_service_lock:
            if _default_service is None:
                settings = get_settings()
                _default_service = EmbeddingService(
                    model_name=settings.embedding_model,
                    batch_size=settings.embedding_batch_size,
                    cache_dir=settings.embedding_cache_dir,
                )
    return _default_service


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_service().embed_texts(texts)


def embed_query(text: str) -> list[float]:
    return get_embedding_service().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embedding_service().embed_documents(texts)
