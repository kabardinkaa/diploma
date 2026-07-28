from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from phoenix.client import Client

from app.eval.artifacts import load_golden_dataset


RESULT_PATH = Path("tests/eval/results/trace_summary.json")
EXTRA_QUERIES = [
    "Как безопасно передать в поддержку диагностические данные?",
    "Что делать перед изменением настроек рабочего приложения?",
    "Как оформить проблему, затронувшую группу сотрудников?",
    "Какая погода будет завтра в Казани?",
    "Кто выиграл последний чемпионат мира по футболу?",
    "Как приготовить яблочный пирог?",
]


async def main() -> None:
    backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    phoenix_url = os.environ.get("PHOENIX_HTTP_ENDPOINT", "http://localhost:6006")
    project_name = os.environ.get("PHOENIX_PROJECT_NAME", "diploma-fastapi")
    golden = load_golden_dataset(
        Path(os.environ.get("EVAL_GOLDEN_PATH", "tests/eval/golden_dataset.json"))
    )
    queries = [row["user_input"] for row in golden[:18]] + EXTRA_QUERIES
    started_at = datetime.now(UTC)
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=120) as client:
        for question in queries:
            try:
                response = await client.post(
                    f"{backend_url}/rag/query",
                    json={"question": question},
                )
                response.raise_for_status()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    await asyncio.sleep(5)
    phoenix = Client(base_url=phoenix_url)
    frame = phoenix.spans.get_spans_dataframe(
        project_name=project_name,
        start_time=started_at - timedelta(seconds=2),
        limit=5000,
        timeout=30,
    )
    names = frame["name"].astype(str).str.lower() if not frame.empty else []
    trace_ids = (
        frame["context.trace_id"].dropna().astype(str).unique()
        if not frame.empty
        else []
    )

    def count_matching(*tokens: str) -> int:
        if frame.empty:
            return 0
        return int(names.apply(lambda name: any(token in name for token in tokens)).sum())

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "queries_sent": len(queries),
        "successful_requests": len(queries) - len(errors),
        "traces_found": len(trace_ids),
        "spans_found": len(frame),
        "retrieval_spans": count_matching("retrieve"),
        "llm_spans": count_matching("chatcompletion", "llm", "generate"),
        "embedding_spans": count_matching("embedding"),
        "errors": dict(Counter(errors)),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(queries) < 20:
        raise SystemExit("Trace smoke requires at least 20 requests")
    if summary["traces_found"] < 20:
        raise SystemExit("Phoenix did not expose at least 20 traces")


if __name__ == "__main__":
    asyncio.run(main())
