from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any


def hit_rate_at_k(
    relevant_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    k: int = 5,
) -> float:
    relevant = set(relevant_doc_ids)
    return float(any(doc_id in relevant for doc_id in retrieved_doc_ids[:k]))


def mrr_at_k(
    relevant_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    k: int = 10,
) -> float:
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    relevant_doc_ids: Sequence[str],
    retrieved_doc_ids: Sequence[str],
    k: int = 10,
) -> float:
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    retrieved_unique = set(retrieved_doc_ids[:k])
    return len(relevant & retrieved_unique) / len(relevant)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def evaluate_retrieval(question_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not question_results:
        raise ValueError("question_results must not be empty")

    evaluated: list[dict[str, Any]] = []
    for result in question_results:
        relevant = result["relevant_doc_ids"]
        retrieved = result["retrieved_doc_ids"]
        evaluated.append(
            {
                **result,
                "hit_at_5": hit_rate_at_k(relevant, retrieved, 5),
                "reciprocal_rank_at_10": mrr_at_k(relevant, retrieved, 10),
                "recall_at_10": recall_at_k(relevant, retrieved, 10),
            }
        )

    latencies = [float(item["latency_ms"]) for item in evaluated]
    return {
        "hit_rate_at_5": statistics.fmean(
            item["hit_at_5"] for item in evaluated
        ),
        "mrr_at_10": statistics.fmean(
            item["reciprocal_rank_at_10"] for item in evaluated
        ),
        "recall_at_10": statistics.fmean(
            item["recall_at_10"] for item in evaluated
        ),
        "avg_retrieval_ms": statistics.fmean(latencies),
        "p50_retrieval_ms": percentile(latencies, 0.5),
        "p95_retrieval_ms": percentile(latencies, 0.95),
        "questions": evaluated,
    }
