from langchain_core.tools import tool

from app.tools.naive_agent_tools import (
    get_current_time as get_current_time_impl,
    search_knowledge_base as search_knowledge_base_impl,
    send_telegram_message as send_telegram_message_impl,
)


@tool
def search_knowledge_base(query: str) -> str:
    """Ищет top-1 фрагмент во внутренней базе знаний техподдержки. Вызывай этот
    tool, когда ответ зависит от корпоративной инструкции. Возвращает найденный
    текст или явное сообщение об отсутствии данных для следующего шага. Не ищет
    в интернете и не выполняет пишущих действий.
    """

    return search_knowledge_base_impl(query)


@tool
def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Получает текущее время в заданной IANA timezone. Вызывай этот tool, когда
    задача зависит от текущей даты или времени. Возвращает ISO datetime с
    часовым смещением. Не обращается к сети и не планирует будущие события.
    """

    return get_current_time_impl(timezone)


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """Имитирует отправку готового текста в указанный Telegram-чат. Вызывай этот
    write-tool только после явного подтверждения пользователя. Возвращает
    строковый статус выполненного действия. Не использует Telegram API и не
    изменяет содержание сообщения.
    """

    return send_telegram_message_impl(chat_id, text)


TOOLS = [search_knowledge_base, get_current_time, send_telegram_message]
TOOLS_BY_NAME = {graph_tool.name: graph_tool for graph_tool in TOOLS}
