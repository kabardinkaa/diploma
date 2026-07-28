from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ModelFactory = Callable[[str], Any]


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        model_factory: ModelFactory | None = None,
        batch_size: int = 8,
        cache_folder: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_folder = str(cache_folder) if cache_folder else None
        self._model_factory = model_factory
        self._model: Any | None = None

    def _load_model(self) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            self.model_name,
            cache_folder=self.cache_folder,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = (
                self._model_factory(self.model_name)
                if self._model_factory
                else self._load_model()
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not candidates:
            return []

        scores = self._get_model().predict(
            [(query, candidate["text"]) for candidate in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = [
            {**candidate, "reranker_score": float(score)}
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        return ranked[:top_n]
