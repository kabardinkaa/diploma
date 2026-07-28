from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.embeddings import EmbeddingService


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def main() -> None:
    settings = get_settings()
    benchmark_path = PROJECT_ROOT / "tests/eval/mini_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    sample = benchmark[0]

    with EmbeddingService(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.embedding_cache_dir,
    ) as service:
        started = perf_counter()
        query_vector = service.embed_query(sample["query"])
        first_seconds = perf_counter() - started

        started = perf_counter()
        cached_query_vector = service.embed_query(sample["query"])
        cached_seconds = perf_counter() - started

        scores: list[tuple[float, float, float]] = []
        for item in benchmark:
            benchmark_query = service.embed_query(item["query"])
            plain_query = service.embed_texts([item["query"]])[0]
            relevant_vector, irrelevant_vector = service.embed_documents(
                [item["relevant"], item["irrelevant"]]
            )
            scores.append(
                (
                    cosine(benchmark_query, relevant_vector),
                    cosine(benchmark_query, irrelevant_vector),
                    cosine(plain_query, relevant_vector),
                )
            )

    norm = math.sqrt(sum(value * value for value in query_vector))

    print(f"model: {settings.embedding_model}")
    print(f"dimension: {len(query_vector)}")
    print(f"query vector norm: {norm:.6f}")
    print(f"first query call: {first_seconds:.4f}s")
    print(f"second query call (cache hit): {cached_seconds:.4f}s")
    print(f"cached vector matches: {query_vector == cached_query_vector}")
    for index, (relevant, irrelevant, without_prefix) in enumerate(scores, start=1):
        print(
            f"pair {index}: relevant={relevant:.6f}, "
            f"irrelevant={irrelevant:.6f}, "
            f"without_query_prefix={without_prefix:.6f}, "
            f"prefix_delta={relevant - without_prefix:+.6f}"
        )

    passed = sum(relevant > irrelevant for relevant, irrelevant, _ in scores)
    print(f"benchmark pairs with relevant > irrelevant: {passed}/{len(scores)}")
    if passed != len(scores):
        raise SystemExit("smoke test failed: at least one benchmark pair is misranked")


if __name__ == "__main__":
    main()
