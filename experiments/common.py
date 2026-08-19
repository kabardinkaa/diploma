from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.services.rag import RAGService


RESULTS_PATH = Path("experiments/results.json")
EXPERIMENT_MODEL = "gpt-5.4-mini"
SEARCH_TOP_K = 3


@dataclass(frozen=True)
class TestQuestion:
    id: str
    category: str
    question: str
    reference: str


TEST_QUESTIONS: tuple[TestQuestion, ...] = (
    TestQuestion(
        id="1",
        category="corpus",
        question=(
            "Как подключиться к корпоративному VPN и что проверить, если "
            "подключение не устанавливается?"
        ),
        reference=(
            "Проверить интернет, корпоративную учётную запись, адрес VPN, MFA "
            "и повторить подключение по инструкции."
        ),
    ),
    TestQuestion(
        id="2",
        category="corpus",
        question=(
            "Что нужно указать в заявке на доступ к внутренней системе и кто "
            "должен её согласовать?"
        ),
        reference=(
            "В заявке указывают пользователя, систему, требуемую роль, деловое "
            "обоснование и срок; доступ согласует ответственный руководитель "
            "или владелец системы."
        ),
    ),
    TestQuestion(
        id="3",
        category="corpus",
        question=(
            "Что делать, если в гарнитуре не работает микрофон в приложении "
            "для звонков?"
        ),
        reference=(
            "Проверить выбранное устройство ввода, разрешение на микрофон, "
            "подключение гарнитуры и тестовую запись; затем перезапустить приложение."
        ),
    ),
    TestQuestion(
        id="4",
        category="multi_step",
        question=(
            "После смены пароля заблокировалась CRM и перестала подключаться "
            "почта. Какие шаги выполнить и какие данные приложить в обращение "
            "в поддержку?"
        ),
        reference=(
            "Обновить сохранённые учётные данные и активные SSO-сессии в CRM и "
            "почте, проверить MFA, а в тикете указать пользователя, системы, "
            "время ошибки, шаги воспроизведения и приложить безопасный скриншот."
        ),
    ),
    TestQuestion(
        id="5",
        category="out_of_scope",
        question="Как оформить командировку на Марс и получить компенсацию расходов?",
        reference="В корпоративной базе техподдержки такой информации нет.",
    ),
)


@dataclass
class SearchTrace:
    tool_calls: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentResult:
    implementation: str
    question_id: str
    question: str
    total_tokens: int
    llm_calls: int
    latency_ms: float
    handoff_count: int
    quality: float | None
    answer: str
    category: str = ""
    route: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieved_contexts: list[str] = field(default_factory=list)
    search_tool_calls: int = 0
    rag_retrieval_calls: int = 0
    search_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QualityEvaluator(Protocol):
    async def score(
        self,
        *,
        question: TestQuestion,
        answer: str,
        contexts: list[str],
    ) -> float | None: ...


_ACTIVE_RAG: ContextVar[tuple[RAGService, SearchTrace] | None] = ContextVar(
    "multi_agent_active_rag",
    default=None,
)


@asynccontextmanager
async def activate_search(rag_service: RAGService) -> AsyncIterator[SearchTrace]:
    trace = SearchTrace()
    token = _ACTIVE_RAG.set((rag_service, trace))
    try:
        yield trace
    finally:
        _ACTIVE_RAG.reset(token)


async def search_knowledge_base(query: str) -> str:
    """Search the existing corporate RAG and return numbered source fragments."""
    active = _ACTIVE_RAG.get()
    if active is None:
        raise RuntimeError("search_knowledge_base requires an active RAGService")
    rag_service, trace = active
    trace.tool_calls.append(query)
    if trace.contexts:
        return _format_search_contexts(trace.contexts)

    trace.calls.append(query)
    contexts = await rag_service.retrieve_contexts(query, top_k=SEARCH_TOP_K)
    if not contexts:
        return "В корпоративной базе знаний релевантные сведения не найдены."

    for context in contexts:
        source_id = len(trace.contexts) + 1
        item = dict(context)
        item["id"] = source_id
        trace.contexts.append(item)
    return _format_search_contexts(trace.contexts)


def _format_search_contexts(contexts: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for item in contexts:
        page = item.get("page")
        location = f", страница {page}" if page is not None else ""
        blocks.append(
            f"[{item['id']}] Источник: {item.get('file_name', 'unknown')}{location}\n"
            f"{item.get('text', '')}"
        )
    return "\n\n".join(blocks)


SEARCH_KNOWLEDGE_BASE_TOOL = StructuredTool.from_function(
    coroutine=search_knowledge_base,
    name="search_knowledge_base",
    description=(
        "Ищет сведения только в существующей корпоративной базе знаний. "
        "Возвращает до трёх пронумерованных фрагментов с источниками для цитат [1], [2]."
    ),
)


def is_local_endpoint(url: str | None) -> bool:
    if not url:
        return False
    hostname = (urlparse(url).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def build_experiment_model(settings: Settings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    base_url = settings.llm.base_url
    if not is_local_endpoint(base_url):
        raise RuntimeError(
            "Experiment LLM must use a local OpenAI-compatible endpoint; "
            "set OPENAI_BASE_URL to LM Studio or Ollama."
        )
    return ChatOpenAI(
        model=os.getenv("EXPERIMENT_MODEL", EXPERIMENT_MODEL),
        temperature=0,
        api_key=settings.llm.api_key,
        base_url=base_url,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )


def _walk_messages(value: Any) -> Iterable[BaseMessage]:
    if isinstance(value, BaseMessage):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_messages(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _walk_messages(nested)


def unique_messages(events: Iterable[Any]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    seen: set[str | int] = set()
    for event in events:
        for message in _walk_messages(event):
            marker: str | int = message.id or id(message)
            if marker not in seen:
                seen.add(marker)
                messages.append(message)
    return messages


def message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def collect_metrics(events: Iterable[Any]) -> dict[str, int]:
    prompt = completion = calls = handoffs = 0
    for message in unique_messages(events):
        if not isinstance(message, AIMessage):
            continue
        calls += 1
        usage = dict(message.usage_metadata or {})
        if not usage and isinstance(message.response_metadata, Mapping):
            raw = message.response_metadata.get("token_usage") or message.response_metadata.get("usage")
            usage = dict(raw) if isinstance(raw, Mapping) else {}
        prompt += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        handoffs += sum(
            str(call.get("name", "")).startswith(("transfer_to_", "delegate_to_"))
            for call in message.tool_calls
        )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "llm_calls": calls,
        "handoff_count": handoffs,
    }


def final_answer(events: Iterable[Any]) -> str:
    for message in reversed(unique_messages(events)):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = message_text(message)
            if text:
                return text
    return ""


def event_to_json(event: Any) -> Any:
    if isinstance(event, BaseMessage):
        return event.model_dump(mode="json")
    if isinstance(event, Mapping):
        return {str(key): event_to_json(value) for key, value in event.items()}
    if isinstance(event, Sequence) and not isinstance(event, (str, bytes)):
        return [event_to_json(value) for value in event]
    return event


async def stream_agent(
    app: Any,
    question: str,
    *,
    thread_id: str,
    print_updates: bool = True,
) -> tuple[list[Any], float]:
    events: list[Any] = []
    started = time.perf_counter()
    config = {"configurable": {"thread_id": thread_id}}
    async for event in app.astream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        events.append(event)
        if print_updates:
            print(json.dumps(event_to_json(event), ensure_ascii=False, default=str))
    return events, round((time.perf_counter() - started) * 1000, 2)


def build_result(
    *,
    implementation: str,
    question: TestQuestion,
    events: list[Any],
    latency_ms: float,
    trace: SearchTrace,
    quality: float | None,
    error: str | None = None,
) -> ExperimentResult:
    metrics = collect_metrics(events)
    return ExperimentResult(
        implementation=implementation,
        question_id=question.id,
        question=question.question,
        total_tokens=metrics["total_tokens"],
        llm_calls=metrics["llm_calls"],
        latency_ms=latency_ms,
        handoff_count=metrics["handoff_count"] if implementation == "multi_agent" else 0,
        quality=quality,
        answer=final_answer(events),
        category=question.category,
        route=[
            str(node)
            for event in events
            if isinstance(event, Mapping)
            for node in event
            if node in {"supervisor", "researcher", "writer", "supervisor_final"}
        ],
        prompt_tokens=metrics["prompt_tokens"],
        completion_tokens=metrics["completion_tokens"],
        retrieved_contexts=[str(item.get("text", "")) for item in trace.contexts],
        search_tool_calls=len(trace.tool_calls),
        rag_retrieval_calls=len(trace.calls),
        search_queries=list(trace.tool_calls),
        sources=[
            {
                "id": item.get("id"),
                "file_name": item.get("file_name"),
                "page": item.get("page"),
                "score": item.get("dense_score"),
            }
            for item in trace.contexts
        ],
        error=error,
    )


def save_results(
    results: Iterable[ExperimentResult | Mapping[str, Any]],
    path: Path = RESULTS_PATH,
) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("results file must contain a JSON array")
        existing = [dict(item) for item in loaded]
    merged = {
        (str(item["implementation"]), str(item["question_id"])): item
        for item in existing
    }
    for result in results:
        item = result.to_dict() if isinstance(result, ExperimentResult) else dict(result)
        merged[(str(item["implementation"]), str(item["question_id"]))] = item
    payload = sorted(merged.values(), key=lambda item: (item["question_id"], item["implementation"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def aggregate_results(results: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["implementation"]), []).append(result)

    summary: dict[str, dict[str, float | None]] = {}
    for implementation, items in grouped.items():
        quality_values = [
            float(item["quality"])
            for item in items
            if item.get("quality") is not None
        ]
        summary[implementation] = {
            "runs": float(len(items)),
            "avg_total_tokens": statistics.fmean(
                float(item["total_tokens"]) for item in items
            ),
            "avg_llm_calls": statistics.fmean(
                float(item["llm_calls"]) for item in items
            ),
            "latency_p50_ms": statistics.median(
                float(item["latency_ms"]) for item in items
            ),
            "avg_handoff_count": statistics.fmean(
                float(item["handoff_count"]) for item in items
            ),
            "avg_quality": statistics.fmean(quality_values) if quality_values else None,
        }
    return summary


def selected_questions(question_id: str | None) -> tuple[TestQuestion, ...]:
    if question_id is None:
        return TEST_QUESTIONS
    selected = tuple(item for item in TEST_QUESTIONS if item.id == question_id)
    if not selected:
        raise ValueError(f"unknown question id: {question_id}")
    return selected


class LocalFaithfulnessEvaluator:
    """Lazy local RAGAS faithfulness evaluator; construction performs no request."""

    def __init__(self, settings: Settings | None = None) -> None:
        from ragas.metrics.collections import Faithfulness

        from app.eval.metrics import build_eval_client, build_judge

        self.settings = settings or get_settings()
        if not is_local_endpoint(self.settings.eval_judge_base_url):
            raise RuntimeError("Quality judge must use a local endpoint")
        self.client = build_eval_client(self.settings)
        self.metric = Faithfulness(llm=build_judge(self.settings, self.client))

    async def score(
        self,
        *,
        question: TestQuestion,
        answer: str,
        contexts: list[str],
    ) -> float | None:
        if not answer or not contexts:
            return 0.0
        result = await self.metric.ascore(
            user_input=question.question,
            response=answer,
            retrieved_contexts=contexts,
        )
        return float(result.value) if result.value is not None else None

    async def close(self) -> None:
        await self.client.close()
