from collections.abc import AsyncIterator
import hashlib
from typing import Any
from uuid import UUID

from app.chat.domain import Chat, ChatMessage, SystemPrompt
from app.chat.repository import ChatRepository
from app.moderation import ModerationResult, ModerationService
from app.schemas.chat import ChatRequest, Message
from app.services.llm import LLMService


MODERATION_SAFE_MESSAGE = "Не могу показать ответ — он мог нарушить правила"


class ModerationBlockedError(Exception):
    def __init__(self, result: ModerationResult) -> None:
        self.result = result
        super().__init__("moderation_blocked")


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
        moderation_service: ModerationService | None = None,
        context_window: int = 10,
        rag_service: Any | None = None,
    ) -> None:
        self.repository = repository
        self.llm_service = llm_service
        self.moderation_service = moderation_service or ModerationService()
        self.context_window = context_window
        self.rag_service = rag_service
        self.last_assistant_message_id: UUID | None = None
        self.last_sources: list[dict[str, Any]] = []
        self.last_rag_meta: dict[str, Any] = {
            "top_score": 0.0,
            "confident": False,
        }

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

    async def check_user_content(self, content: str) -> None:
        result = await self.moderation_service.check_input(content)

        if not result.allowed:
            raise ModerationBlockedError(result)

    async def set_handoff_status(
        self,
        chat_id: UUID,
        status: str,
    ) -> Chat | None:
        return await self.repository.set_handoff_status(chat_id, status)  # type: ignore[arg-type]

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        value: str,
    ):
        chat = await self.repository.get_chat(chat_id)

        if chat is None:
            raise ValueError(f"Chat {chat_id} not found")

        return await self.repository.save_feedback(
            chat_id=chat_id,
            message_id=message_id,
            owner_external_id=chat.owner_external_id,
            value=value,  # type: ignore[arg-type]
        )

    async def select_prompt(self, owner_external_id: str) -> SystemPrompt | None:
        try:
            prompts = await self.repository.list_active_prompts()
        except Exception:
            return None

        if not prompts:
            return None

        total = sum(prompt.traffic_pct for prompt in prompts)
        if total != 100:
            return prompts[0]

        bucket = int(
            hashlib.sha256(owner_external_id.encode("utf-8")).hexdigest(),
            16,
        ) % 100
        cursor = 0

        for prompt in prompts:
            cursor += prompt.traffic_pct
            if bucket < cursor:
                return prompt

        return prompts[0]

    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
        media_refs: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        chat = await self.repository.get_chat(chat_id)

        if chat is None:
            raise ValueError(f"Chat {chat_id} not found")

        if chat.handoff_status == "paused_for_human":
            user_message = ChatMessage(
                chat_id=chat_id,
                role="user",
                content=user_content,
                media_refs=media_refs,
            )
            await self.repository.append_message(chat_id, user_message)

            assistant_message = ChatMessage(
                chat_id=chat_id,
                role="assistant",
                content="Диалог ожидает оператора. Мы вернемся с ответом позже.",
            )
            await self.repository.append_message(chat_id, assistant_message)
            self.last_assistant_message_id = assistant_message.id

            yield assistant_message.content
            return

        media_part = (media_refs or {}).get("part")
        use_rag = media_refs is None and self.rag_service is not None
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

        await self.check_user_content(stored_content)

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

        selected_prompt = await self.select_prompt(chat.owner_external_id)

        if use_rag:
            provider_chunks: list[str] = []
            final_event: dict[str, Any] | None = None
            async for event in self.rag_service.stream_answer(
                stored_content,
                history=history[:-1],
            ):
                if event.get("type") == "token" and event.get("delta"):
                    provider_chunks.append(str(event["delta"]))
                elif event.get("type") == "sources":
                    final_event = event

            if final_event is None:
                raise RuntimeError("RAG stream ended without sources event")

            assistant_content = str(final_event["answer"]).strip()
            output_result = await self.moderation_service.check_output(
                assistant_content
            )
            sources = list(final_event.get("sources") or [])
            if not output_result.allowed:
                assistant_content = MODERATION_SAFE_MESSAGE
                sources = []

            assistant_message = ChatMessage(
                chat_id=chat_id,
                role="assistant",
                content=assistant_content,
                sources=sources,
                prompt_id=selected_prompt.id if selected_prompt else None,
            )
            await self.repository.append_message(chat_id, assistant_message)
            self.last_assistant_message_id = assistant_message.id
            self.last_sources = sources
            self.last_rag_meta = {
                "top_score": final_event.get("top_score", 0.0),
                "confident": bool(final_event.get("confident")),
                "condensed_query": final_event.get("condensed_query"),
            }

            offset = 0
            lengths = [len(chunk) for chunk in provider_chunks if chunk]
            if not lengths:
                lengths = [64]
            for length in lengths:
                delta = assistant_content[offset : offset + length]
                if delta:
                    yield delta
                offset += length
            if offset < len(assistant_content):
                yield assistant_content[offset:]
            return

        if selected_prompt is not None:
            llm_messages.append(
                Message(
                    role="system",
                    content=selected_prompt.body,
                )
            )

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

        finally:
            assistant_content = "".join(
                assistant_chunks
            ).strip()

            if assistant_content:
                output_result = await self.moderation_service.check_output(
                    assistant_content
                )

                if not output_result.allowed:
                    assistant_content = MODERATION_SAFE_MESSAGE

                assistant_message = ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=assistant_content,
                    prompt_id=selected_prompt.id if selected_prompt else None,
                )

                await self.repository.append_message(
                    chat_id,
                    assistant_message,
                )

                self.last_assistant_message_id = assistant_message.id
                self.last_sources = []
                self.last_rag_meta = {
                    "top_score": 0.0,
                    "confident": False,
                }

                yield assistant_content
