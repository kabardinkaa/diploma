from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.rag import RAGService
from app.services.rag_baremetal import BareMetalRAGService
from app.services.rag_common import FALLBACK_ANSWER

QUESTIONS = [
    {
        "category": "good",
        "question": (
            "Что проверить, если VPN подключился, "
            "но внутренние ресурсы не открываются?"
        ),
    },
    {
        "category": "good",
        "question": "Как разблокировать учетную запись в CRM?",
    },
    {
        "category": "good",
        "question": "Что делать, если в гарнитуре не работает микрофон?",
    },
    {
        "category": "medium",
        "question": (
            "После смены пароля перестал работать вход сразу в несколько "
            "внутренних систем. Что проверить?"
        ),
    },
    {
        "category": "out-of-base",
        "question": "Как оформить ежегодный отпуск на две недели?",
    },
]


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    sources = result["sources"]
    return {
        "answer": result["answer"],
        "top_score": result["top_score"],
        "top_1_source": sources[0]["source"] if sources else None,
        "source_scores": [
            {"source": source["source"], "score": source["score"]}
            for source in sources
        ],
        "fallback": result["answer"] == FALLBACK_ANSWER,
    }


async def build_twice(
    service_type: type[RAGService] | type[BareMetalRAGService],
) -> tuple[Any, int, int]:
    first = service_type(get_settings())
    await first.build()
    first_count = await first.points_count()
    await first.close()

    second = service_type(get_settings())
    await second.build()
    second_count = await second.points_count()
    return second, first_count, second_count


async def main() -> None:
    llama, llama_first, llama_second = await build_twice(RAGService)
    bare, bare_first, bare_second = await build_twice(BareMetalRAGService)
    try:
        results: list[dict[str, Any]] = []
        for item in QUESTIONS:
            llama_result = await llama.answer(item["question"])
            bare_result = await bare.answer(item["question"])
            results.append(
                {
                    **item,
                    "llama_index": compact_result(llama_result),
                    "bare_metal": compact_result(bare_result),
                    "same_top_1": (
                        llama_result["sources"][0]["source"]
                        if llama_result["sources"]
                        else None
                    )
                    == (
                        bare_result["sources"][0]["source"]
                        if bare_result["sources"]
                        else None
                    ),
                }
            )

        report = {
            "llama_index": {
                "collection": get_settings().rag_collection,
                "points_count_first": llama_first,
                "points_count_second": llama_second,
                "idempotent": llama_first == llama_second,
            },
            "bare_metal": {
                "collection": get_settings().rag_baremetal_collection,
                "points_count_first": bare_first,
                "points_count_second": bare_second,
                "idempotent": bare_first == bare_second,
            },
            "questions": results,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        await llama.close()
        await bare.close()


if __name__ == "__main__":
    asyncio.run(main())
