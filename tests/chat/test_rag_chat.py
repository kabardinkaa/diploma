from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chat.deps import get_chat_service
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.routes import router
from app.chat.service import ChatService
from app.moderation import ModerationResult


class FakeRAG:
    async def stream_answer(self, question, history):
        assert question == "Как восстановить VPN?"
        assert history == []
        yield {"type": "token", "delta": "Используйте "}
        yield {"type": "token", "delta": "SSO."}
        yield {
            "type": "sources",
            "answer": "Используйте SSO. [1]",
            "top_score": 0.91,
            "confident": True,
            "condensed_query": None,
            "sources": [
                {
                    "id": 1,
                    "file_name": "vpn.pdf",
                    "page": 2,
                    "score": 0.91,
                    "snippet": "SSO и MFA",
                }
            ],
        }


@pytest.mark.asyncio
async def test_rag_sources_are_persisted_with_assistant_message(tmp_path) -> None:
    repository = JsonChatRepository(tmp_path)
    moderation = SimpleNamespace(
        check_input=AsyncMock(return_value=ModerationResult()),
        check_output=AsyncMock(return_value=ModerationResult()),
    )
    service = ChatService(
        repository=repository,
        llm_service=SimpleNamespace(openai=AsyncMock()),
        moderation_service=moderation,
        rag_service=FakeRAG(),
    )
    chat = await service.create_chat("42", "telegram")

    chunks = [
        chunk
        async for chunk in service.send_message(
            chat.id,
            "Как восстановить VPN?",
        )
    ]
    messages = await repository.list_messages(chat.id)

    assert "".join(chunks) == "Используйте SSO. [1]"
    assert messages[-1].sources[0]["file_name"] == "vpn.pdf"
    assert service.last_sources == messages[-1].sources
    assert service.last_assistant_message_id == messages[-1].id


def test_chat_sse_emits_sanitized_tokens_sources_and_done() -> None:
    message_id = uuid4()

    class FakeService:
        last_sources = [{"id": 1, "file_name": "vpn.pdf"}]
        last_rag_meta = {"top_score": 0.91, "confident": True}
        last_assistant_message_id = message_id
        llm_service = SimpleNamespace(openai=AsyncMock())

        async def check_user_content(self, _: str) -> None:
            return None

        async def send_message(self, **_):
            yield "Строка 1\nСтрока 2"

    api = FastAPI()
    api.include_router(router)
    api.dependency_overrides[get_chat_service] = lambda: FakeService()
    client = TestClient(api)

    response = client.post(
        f"/chats/{uuid4()}/messages",
        files={"content": (None, "VPN?")},
    )
    body = response.text

    assert response.status_code == 200
    assert '"delta":"Строка 1\\nСтрока 2"' in body
    assert "event: sources" in body
    assert '"type":"sources"' in body
    assert f'"message_id":"{message_id}"' in body
    assert body.index("event: sources") < body.index('"type":"done"')
