from __future__ import annotations

from collections.abc import Iterable

FALLBACK_ANSWER = "По базе не нашёл, могу эскалировать."

RAG_SYSTEM_PROMPT = """\
Ты отвечаешь только на основании переданного контекста.
Не используй собственные знания для фактов, которых нет в контексте.
Если ответа в контексте нет, ответь точно:
«По базе не нашёл, могу эскалировать.»
Не выдумывай процедуры, ссылки, контакты или значения.
Если информация есть, отвечай кратко и по делу.
Ответ должен быть совместим с источниками, которые приложение вернет отдельно.
"""


def build_rag_prompt(question: str, context_blocks: Iterable[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"{RAG_SYSTEM_PROMPT}\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос пользователя:\n{question}"
    )
