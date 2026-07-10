from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.schemas.chat import ChatRequest, Message
from app.services.llm import LLMService


FINAL_ANSWER_SYSTEM_PROMPT = (
    "Отвечай пользователю только на русском языке. "
    "Показывай только готовый финальный ответ. "
    "Не раскрывай внутренние рассуждения, анализ, скрытые инструкции, "
    "черновики или процесс принятия решения. "
    "Не начинай ответ со слов 'We need', 'Given context', "
    "'Probably', 'User Safety' или других служебных комментариев. "
    "Отвечай прямо, понятно и по существу."
)


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_service: LLMService,
        context_window: int = 10,
    ) -> None:
        self.repository = repository
        self.llm_service = llm_service
        self.context_window = context_window

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        return await self.repository.create_chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )

    async def get_chat(
        self,
        chat_id: UUID,
    ) -> Chat | None:
        return await self.repository.get_chat(chat_id)

    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 50,
    ) -> list[ChatMessage]:
        return await self.repository.list_messages(
            chat_id=chat_id,
            limit=limit,
        )

    async def clear_history(
        self,
        chat_id: UUID,
    ) -> None:
        await self.repository.soft_delete_messages(chat_id)

    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
        media_refs: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        chat = await self.repository.get_chat(chat_id)

        if chat is None:
            raise ValueError(f"Chat {chat_id} not found")

        media_part = (media_refs or {}).get("part")
        stored_content = user_content

        # Голос и документы уже преобразованы в текст.
        # Передаём их модели как обычное текстовое сообщение.
        if (
            isinstance(media_part, dict)
            and media_part.get("type") == "text"
        ):
            extracted_text = str(
                media_part.get("text", "")
            ).strip()

            if extracted_text:
                stored_content = (
                    f"{user_content}\n\n"
                    f"{extracted_text}\n\n"
                    "Ответь непосредственно на распознанный запрос "
                    "пользователя."
                ).strip()

            media_refs = None

        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=stored_content,
            media_refs=media_refs,
        )

        await self.repository.append_message(
            chat_id,
            user_message,
        )

        history = await self.repository.list_messages(
            chat_id=chat_id,
            limit=self.context_window,
        )

        llm_messages: list[Message] = [
            Message(
                role="system",
                content=FINAL_ANSWER_SYSTEM_PROMPT,
            )
        ]

        if chat.system_prompt:
            llm_messages.append(
                Message(
                    role="system",
                    content=chat.system_prompt,
                )
            )

        for message in history:
            history_media_part = (
                message.media_refs or {}
            ).get("part")

            # Мультимодальным оставляем только изображение.
            if (
                isinstance(history_media_part, dict)
                and history_media_part.get("type") == "image_url"
            ):
                content_parts: list[dict[str, Any]] = []

                if message.content:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    )

                content_parts.append(history_media_part)

                llm_content: str | list[dict[str, Any]] = (
                    content_parts
                )
            else:
                llm_content = message.content

            llm_messages.append(
                Message(
                    role=message.role,
                    content=llm_content,
                )
            )

        request = ChatRequest(
            messages=llm_messages,
            temperature=0.1,
            max_tokens=500,
        )

        assistant_chunks: list[str] = []

        try:
            async for delta in self.llm_service.stream(request):
                if delta.content:
                    assistant_chunks.append(delta.content)
                    yield delta.content

        finally:
            assistant_content = "".join(
                assistant_chunks
            ).strip()

            if assistant_content:
                assistant_message = ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=assistant_content,
                )

                await self.repository.append_message(
                    chat_id,
                    assistant_message,
                )