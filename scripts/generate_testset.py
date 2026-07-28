from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

from app.core.config import Settings, get_settings
from app.eval.metrics import (
    build_eval_client,
    build_evaluator_embeddings,
    build_judge,
)
from app.services.ingestion import IngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an uncurated RAGAS golden dataset from the RAG corpus."
    )
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corporate"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/eval/golden_dataset_raw.csv"),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Fixed OpenAI-compatible model; defaults to EVAL_JUDGE_MODEL.",
    )
    return parser.parse_args()


async def load_documents(
    settings: Settings,
    corpus: Path,
) -> list[Any]:
    service = IngestionService(
        settings,
        sync_client=object(),
        async_client=object(),
    )
    try:
        files = service.discover_files(corpus)
        documents: list[Any] = []
        for path in files:
            documents.extend(await service.parse_file(path, corpus))
        return documents
    finally:
        await service.close()


async def generate(args: argparse.Namespace) -> None:
    if args.size < 1:
        raise ValueError("--size must be positive")

    settings = get_settings()
    judge_model = args.judge_model or settings.eval_judge_model
    if judge_model != settings.eval_judge_model:
        settings = settings.model_copy(update={"eval_judge_model": judge_model})
    client = build_eval_client(settings)
    documents = await load_documents(settings, args.corpus)
    if not documents:
        raise RuntimeError(f"No documents parsed from {args.corpus}")

    judge = build_judge(settings, client)
    embeddings = build_evaluator_embeddings(settings)
    generator = TestsetGenerator(
        llm=judge,
        embedding_model=embeddings,
    )
    print(
        {
            "documents": len(documents),
            "requested_samples": args.size,
            "judge_model": judge_model,
            "embedding_model": settings.eval_embedding_model,
        }
    )
    try:
        testset = await asyncio.to_thread(
            generator.generate_with_llamaindex_docs,
            documents,
            testset_size=args.size,
            run_config=RunConfig(
                timeout=settings.eval_request_timeout,
                max_retries=settings.llm.max_retries,
                max_workers=settings.eval_concurrency,
            ),
        )
        frame = testset.to_pandas()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print({"output": str(args.output), "rows": len(frame)})
    finally:
        embeddings.close()
        try:
            await client.close()
        except RuntimeError as exc:
            if "Event loop is closed" not in str(exc):
                raise


def main() -> None:
    asyncio.run(generate(parse_args()))


if __name__ == "__main__":
    main()
