from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from langchain.agents import create_agent

from app.core.config import get_settings
from app.services.rag import RAGService
from experiments.common import (
    RESULTS_PATH,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    ExperimentResult,
    LocalFaithfulnessEvaluator,
    QualityEvaluator,
    TestQuestion,
    activate_search,
    build_experiment_model,
    build_result,
    save_results,
    selected_questions,
    stream_agent,
)


SINGLE_AGENT_PROMPT = """\
Ты единый агент корпоративной технической поддержки. Для каждого вопроса сначала
используй search_knowledge_base и найди необходимые факты. Затем дай связный ответ
на русском языке, используя только найденные сведения. Каждое фактическое
утверждение подтверждай доступной ссылкой [1], [2] и не придумывай источники.
Если база не содержит ответа, прямо сообщи об этом.
"""


def build_single_agent(
    model: Any,
    *,
    search_tool: Any = SEARCH_KNOWLEDGE_BASE_TOOL,
) -> Any:
    return create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=SINGLE_AGENT_PROMPT,
        name="single_agent_baseline",
    )


async def run_question(
    app: Any,
    question: TestQuestion,
    rag_service: RAGService,
    *,
    quality_evaluator: QualityEvaluator | None = None,
    print_updates: bool = True,
) -> ExperimentResult:
    async with activate_search(rag_service) as trace:
        events, latency_ms = await stream_agent(
            app,
            question.question,
            thread_id=f"exp-single-{question.id}",
            print_updates=print_updates,
        )
        contexts = [str(item.get("text", "")) for item in trace.contexts]
        provisional = build_result(
            implementation="single_agent",
            question=question,
            events=events,
            latency_ms=latency_ms,
            trace=trace,
            quality=None,
        )
        quality = (
            await quality_evaluator.score(
                question=question,
                answer=provisional.answer,
                contexts=contexts,
            )
            if quality_evaluator
            else None
        )
        provisional.quality = quality
        return provisional


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Block 6.5 single-agent baseline")
    parser.add_argument("--question-id", choices=[item.id for item in selected_questions(None)])
    parser.add_argument("--quality", action="store_true", help="Use the configured local RAGAS judge")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--quiet", action="store_true", help="Do not print LangGraph updates")
    return parser


async def _main(args: argparse.Namespace) -> None:
    settings = get_settings()
    model = build_experiment_model(settings)
    app = build_single_agent(model)
    rag_service = RAGService(settings)
    evaluator = LocalFaithfulnessEvaluator(settings) if args.quality else None
    try:
        await rag_service.build()
        for question in selected_questions(args.question_id):
            result = await run_question(
                app,
                question,
                rag_service,
                quality_evaluator=evaluator,
                print_updates=not args.quiet,
            )
            save_results([result], args.results)
    finally:
        await rag_service.close()
        if evaluator:
            await evaluator.close()


if __name__ == "__main__":
    asyncio.run(_main(_parser().parse_args()))
