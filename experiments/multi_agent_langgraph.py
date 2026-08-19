from __future__ import annotations

import argparse
import asyncio
import re
from uuid import uuid4
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command
from langgraph_supervisor import create_supervisor

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


RESEARCHER_PROMPT = """\
Ты researcher корпоративной технической поддержки. Используй только
search_knowledge_base, собери факты и верни маркированный список фактов вместе с
точными номерами источников [1], [2]. Не формируй финальный ответ пользователю и
не добавляй сведения, которых нет в найденных фрагментах. Сначала вызови поиск
ровно один раз. Если он вернул непустые релевантные контексты, считай их
достаточными и не вызывай поиск повторно. Повторный запрос допустим только если
первый поиск не вернул контекстов.
"""

WRITER_PROMPT = """\
Ты writer корпоративной технической поддержки. У тебя нет инструментов поиска.
На основе фактов researcher составь связный финальный ответ на русском языке.
Каждое фактическое утверждение подтверждай переданными ссылками [1], [2]. Если
фактов недостаточно, честно сообщи, что база знаний не содержит ответа.
"""

SUPERVISOR_PROMPT = """\
Ты supervisor команды поддержки. Координируй работу, но не выполняй поиск и не
пиши ответ самостоятельно. Сначала передай вопрос researcher, затем передай его
факты writer. Не пропускай writer и не меняй содержание найденных источников.
Ответ writer автоматически передаётся пользователю без дополнительного LLM-вызова.
"""


def guard_supervisor_handoff(state: dict[str, Any]) -> dict[str, Any] | Command:
    messages = list(state.get("messages", []))
    last_ai = next(
        (message for message in reversed(messages) if isinstance(message, AIMessage)),
        None,
    )
    if last_ai is None or any(
        call.get("name") == "transfer_to_writer" for call in last_ai.tool_calls
    ):
        return {}

    writer_completed = any(
        isinstance(message, AIMessage)
        and message.name == "writer"
        and str(message.content).strip()
        for message in messages
    )
    researcher_result = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AIMessage)
            and message.name == "researcher"
            and str(message.content).strip()
            and re.search(r"\[\d+\]", str(message.content))
        ),
        None,
    )
    if writer_completed or researcher_result is None:
        return {}

    tool_call_id = str(uuid4())
    handoff = AIMessage(
        content="",
        name="supervisor",
        tool_calls=[
            {
                "name": "transfer_to_writer",
                "args": {},
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
        response_metadata={"handoff_fallback": True},
    )
    confirmation = ToolMessage(
        content="Successfully transferred to writer",
        name="transfer_to_writer",
        tool_call_id=tool_call_id,
        response_metadata={
            "__handoff_destination": "writer",
            "handoff_fallback": True,
        },
    )
    return Command(
        goto="writer",
        graph=Command.PARENT,
        update={"messages": [*messages, handoff, confirmation]},
    )


def finish_after_writer(state: dict[str, Any]) -> dict[str, Any]:
    writer_answer = next(
        (
            message
            for message in reversed(state.get("messages", []))
            if isinstance(message, AIMessage)
            and message.name == "writer"
            and not message.tool_calls
            and str(message.content).strip()
        ),
        None,
    )
    if writer_answer is None:
        raise RuntimeError("writer did not return a final answer")
    return {}


def build_multi_agent(
    model: Any,
    *,
    search_tool: Any = SEARCH_KNOWLEDGE_BASE_TOOL,
    checkpointer: Any | None = None,
) -> Any:
    researcher = create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=RESEARCHER_PROMPT,
        name="researcher",
    )
    writer = create_agent(
        model=model,
        tools=[],
        system_prompt=WRITER_PROMPT,
        name="writer",
    )
    workflow = create_supervisor(
        agents=[researcher, writer],
        model=model,
        prompt=SUPERVISOR_PROMPT,
        post_model_hook=guard_supervisor_handoff,
        output_mode="last_message",
        add_handoff_back_messages=False,
        supervisor_name="supervisor",
    )
    writer_edge = ("writer", "supervisor")
    if writer_edge not in workflow.edges:
        raise RuntimeError("langgraph-supervisor writer edge contract changed")
    workflow.edges.remove(writer_edge)
    workflow.add_node("supervisor_final", finish_after_writer)
    workflow.add_edge("writer", "supervisor_final")
    workflow.add_edge("supervisor_final", END)
    return workflow.compile(
        checkpointer=checkpointer or InMemorySaver(),
        name="multi_agent_experiment",
    )


def save_mermaid(app: Any, path: Path = Path("docs/architecture-multi-agent.md")) -> str:
    mermaid = app.get_graph().draw_mermaid()
    document = (
        "# Архитектура multi-agent эксперимента\n\n"
        "Схема получена из скомпилированного LangGraph через "
        "`app.get_graph().draw_mermaid()`. Supervisor маршрутизирует запрос к "
        "researcher с RAG-инструментом, затем к writer без инструментов.\n\n"
        "```mermaid\n"
        f"{mermaid.rstrip()}\n"
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return mermaid


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
            thread_id=f"exp-langgraph-{question.id}",
            print_updates=print_updates,
        )
        contexts = [str(item.get("text", "")) for item in trace.contexts]
        provisional = build_result(
            implementation="multi_agent",
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
    parser = argparse.ArgumentParser(description="Run the Block 6.5 LangGraph supervisor")
    parser.add_argument("--question-id", choices=[item.id for item in selected_questions(None)])
    parser.add_argument("--quality", action="store_true", help="Use the configured local RAGAS judge")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--architecture", type=Path, default=Path("docs/architecture-multi-agent.md"))
    parser.add_argument("--quiet", action="store_true", help="Do not print LangGraph updates")
    return parser


async def _main(args: argparse.Namespace) -> None:
    settings = get_settings()
    model = build_experiment_model(settings)
    app = build_multi_agent(model)
    save_mermaid(app, args.architecture)
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
