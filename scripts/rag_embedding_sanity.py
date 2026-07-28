from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def main() -> None:
    settings = get_settings()
    model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        query_instruction="query: ",
        text_instruction="passage: ",
        normalize=True,
        embed_batch_size=settings.embedding_batch_size,
        cache_folder=str(settings.embedding_cache_dir),
        show_progress_bar=False,
    )
    query = model.get_query_embedding(
        "Что проверить, если VPN подключен, но внутренние сайты не открываются?"
    )
    relevant = model.get_text_embedding(
        Path("data/rag-block-03/01_vpn.md").read_text(encoding="utf-8")
    )
    irrelevant = model.get_text_embedding(
        Path("data/rag-block-03/10_office_plants.txt").read_text(encoding="utf-8")
    )

    result = {
        "model": settings.embedding_model,
        "dimension": len(query),
        "query_norm": round(math.sqrt(sum(value * value for value in query)), 6),
        "relevant_score": round(cosine(query, relevant), 6),
        "irrelevant_score": round(cosine(query, irrelevant), 6),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if len(query) != settings.embedding_dim:
        raise SystemExit("Embedding dimension mismatch")
    if abs(result["query_norm"] - 1.0) > 1e-3:
        raise SystemExit("Query embedding is not L2-normalized")
    if result["relevant_score"] <= result["irrelevant_score"]:
        raise SystemExit("Relevant document did not outrank irrelevant document")


if __name__ == "__main__":
    main()
