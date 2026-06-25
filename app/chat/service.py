from collections.abc import AsyncIterator
from uuid import UUID

from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository
from app.schemas.chat import ChatRequest, Message
from app.services.llm import LLMService


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_service: LLMService,
        context_window: int = 10,
    ):
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

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return await self.repository.get_chat(chat_id)

    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 50,
    ) -> list[ChatMessage]:
        return await self.repository.list_messages(chat_id=chat_id, limit=limit)

    async def clear_history(self, chat_id: UUID) -> None:
        await self.repository.soft_delete_messages(chat_id)

    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
    ) -> AsyncIterator[str]:
        chat = await self.repository.get_chat(chat_id)

        if chat is None:
            raise ValueError(f"Chat {chat_id} not found")

        user_message = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
        )
        await self.repository.append_message(chat_id, user_message)

        history = await self.repository.list_messages(
            chat_id=chat_id,
            limit=self.context_window,
        )

        llm_messages: list[Message] = []

        if chat.system_prompt:
            llm_messages.append(
                Message(
                    role="system",
                    content=chat.system_prompt,
                )
            )

        for message in history:
            llm_messages.append(
                Message(
                    role=message.role,
                    content=message.content,
                )
            )

        request = ChatRequest(
            messages=llm_messages,
            temperature=0.2,
            max_tokens=500,
        )

        assistant_chunks: list[str] = []

        try:
            async for delta in self.llm_service.stream(request):
                if delta.content:
                    assistant_chunks.append(delta.content)
                    yield delta.content
        finally:
            assistant_content = "".join(assistant_chunks).strip()

            if assistant_content:
                assistant_message = ChatMessage(
                    chat_id=chat_id,
                    role="assistant",
                    content=assistant_content,
                )
                await self.repository.append_message(chat_id, assistant_message)