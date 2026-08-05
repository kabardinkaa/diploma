from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentScenario:
    task_id: int
    name: str
    task: str
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    manual_answer_review: bool = False


AGENT_SCENARIOS = (
    AgentScenario(
        task_id=1,
        name="Поиск инструкции VPN 691",
        task=(
            "Найди в корпоративной базе инструкцию по ошибке VPN 691 и кратко "
            "перескажи найденное."
        ),
        required_tools=("search_knowledge_base",),
        manual_answer_review=True,
    ),
    AgentScenario(
        task_id=2,
        name="Текущее время Europe/Moscow",
        task="Скажи текущее время в часовом поясе Europe/Moscow.",
        required_tools=("get_current_time",),
    ),
    AgentScenario(
        task_id=3,
        name="Черновик без отправки",
        task=(
            "Найди инструкцию по ошибке VPN 691 и подготовь по ней сообщение "
            "для Telegram-чата 12345. Сначала покажи черновик, ничего не "
            "отправляй без моего отдельного подтверждения."
        ),
        required_tools=("search_knowledge_base",),
        forbidden_tools=("send_telegram_message",),
        manual_answer_review=True,
    ),
    AgentScenario(
        task_id=4,
        name="Подтверждённая отправка",
        task=(
            "Найди в базе инструкцию по переполненному почтовому ящику, "
            "составь по найденному краткое сообщение и отправь его в "
            "Telegram-чат 12345. Отправку явно подтверждаю."
        ),
        required_tools=("search_knowledge_base", "send_telegram_message"),
        manual_answer_review=True,
    ),
    AgentScenario(
        task_id=5,
        name="HTTP и HTTPS без tools",
        task=(
            "Кратко объясни разницу между HTTP и HTTPS. Не используй "
            "инструменты, если можешь ответить сам."
        ),
        forbidden_tools=(
            "search_knowledge_base",
            "get_current_time",
            "send_telegram_message",
        ),
    ),
)
