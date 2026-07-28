from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from app.core.config import Settings


@dataclass(frozen=True)
class EvaluationVariant:
    label: str
    collection: str
    docstore_path: Path
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    reranker_enabled: bool

    def apply(self, settings: Settings) -> Settings:
        return settings.model_copy(
            update={
                "rag_production_collection": self.collection,
                "rag_docstore_path": self.docstore_path,
                "rag_chunking_strategy": self.chunking_strategy,
                "rag_chunk_size": self.chunk_size,
                "rag_chunk_overlap": self.chunk_overlap,
                "rag_retrieval_top_k": self.top_k,
                "rag_reranker_enabled": self.reranker_enabled,
            }
        )


def baseline_variant(settings: Settings) -> EvaluationVariant:
    return EvaluationVariant(
        label="baseline",
        collection=settings.rag_production_collection,
        docstore_path=settings.rag_docstore_path,
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=64,
        top_k=10,
        reranker_enabled=False,
    )


def chunk_256_variant(settings: Settings) -> EvaluationVariant:
    return replace(
        baseline_variant(settings),
        label="chunk_256",
        collection="corporate_rag_eval_chunk256",
        docstore_path=Path(".cache/rag/eval_chunk256_docstore.json"),
        chunk_size=256,
    )


def top_k_5_variant(settings: Settings) -> EvaluationVariant:
    return replace(
        baseline_variant(settings),
        label="top_k_5",
        top_k=5,
    )


def variant_for_label(settings: Settings, label: str) -> EvaluationVariant:
    variants = {
        "baseline": baseline_variant,
        "chunk_256": chunk_256_variant,
        "top_k_5": top_k_5_variant,
    }
    try:
        return variants[label](settings)
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation label: {label}") from exc
