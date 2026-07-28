from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.core.config import Settings, get_settings
from app.eval.artifacts import (
    aggregate_rows,
    dataset_sha256,
    load_golden_dataset,
    result_paths,
)
from app.eval.config import EvaluationVariant, variant_for_label
from app.eval.metrics import (
    build_eval_client,
    build_evaluator_embeddings,
    build_judge,
    build_metrics,
    eval_row,
)
from app.services.rag import RAGService


def parse_args() -> argparse.Namespace:
    defaults = get_settings()
    parser = argparse.ArgumentParser(description="Run the offline RAGAS evaluation.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--golden", type=Path, default=defaults.eval_golden_path)
    parser.add_argument("--results-dir", type=Path, default=defaults.eval_results_dir)
    parser.add_argument("--collection")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configured_variant(settings: Settings, args: argparse.Namespace) -> EvaluationVariant:
    try:
        variant = variant_for_label(settings, args.label)
    except ValueError:
        variant = replace(variant_for_label(settings, "baseline"), label=args.label)
    updates: dict[str, Any] = {}
    if args.collection:
        updates["collection"] = args.collection
    if args.top_k is not None:
        updates["top_k"] = args.top_k
    if args.chunk_size is not None:
        updates["chunk_size"] = args.chunk_size
    if not updates:
        return variant
    return EvaluationVariant(**{**variant.__dict__, **updates})


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for field in ("retrieved_contexts", "retrieved_doc_ids", "retrieved_scores"):
        result[field] = json.dumps(result.get(field, []), ensure_ascii=False)
    return result


async def evaluate(
    settings: Settings,
    variant: EvaluationVariant,
    golden_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    client = build_eval_client(settings)
    service = RAGService(settings, openai_client=client)
    judge = build_judge(settings, client)
    embeddings = build_evaluator_embeddings(settings, client)
    metrics = build_metrics(judge, embeddings)
    semaphore = asyncio.Semaphore(settings.eval_concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            output: dict[str, Any] = {
                "user_input": row["user_input"],
                "reference": row["reference"],
                "response": "",
                "retrieved_contexts": [],
                "retrieved_doc_ids": [],
                "retrieved_scores": [],
                "top_score": None,
                "confident": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "has_citation": None,
                "latency_ms": None,
                "error": "",
            }
            try:
                rag_result = await service.evaluate_inputs(
                    row["user_input"],
                    top_k=variant.top_k,
                )
                output.update(rag_result)
                output.update(
                    await eval_row(
                        metrics,
                        user_input=row["user_input"],
                        reference=row["reference"],
                        response=rag_result["response"],
                        retrieved_contexts=rag_result["retrieved_contexts"],
                    )
                )
            except Exception as exc:
                output["error"] = f"{type(exc).__name__}: {exc}"
            return output

    try:
        await service.build()
        return await asyncio.gather(*(one(row) for row in golden_rows))
    finally:
        await service.close()
        await client.close()


async def run(args: argparse.Namespace) -> None:
    base_settings = get_settings()
    variant = configured_variant(base_settings, args)
    minimum = 1 if args.limit else 30
    golden_rows = load_golden_dataset(args.golden, minimum=minimum)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        golden_rows = golden_rows[: args.limit]
    csv_path, aggregate_path = result_paths(args.results_dir, args.label)
    eval_settings = variant.apply(base_settings)

    summary = {
        "question_count": len(golden_rows),
        "metric_count": 5,
        "judge_model": eval_settings.eval_judge_model,
        "estimated_evaluation_operations": len(golden_rows) * 5,
        "variant": variant.__dict__,
        "csv_path": str(csv_path),
        "aggregate_path": str(aggregate_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if args.dry_run:
        print("Dry run complete; no LLM calls made and no artifacts written.")
        return

    rows = await evaluate(eval_settings, variant, golden_rows)
    aggregate = aggregate_rows(rows)
    aggregate.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "label": args.label,
            "dataset_sha256": dataset_sha256(args.golden),
            "production_model": eval_settings.rag_generation_model,
            "judge_model": eval_settings.eval_judge_model,
            "evaluator_embedding_model": eval_settings.eval_embedding_model,
            "rag_embedding_model": eval_settings.embedding_model,
            "chunk_config": {
                "strategy": variant.chunking_strategy,
                "size": variant.chunk_size,
                "overlap": variant.chunk_overlap,
            },
            "collection": variant.collection,
            "top_k": variant.top_k,
            "reranker": variant.reranker_enabled,
        }
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(serializable_row(row) for row in rows).to_csv(
        csv_path,
        index=False,
    )
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
