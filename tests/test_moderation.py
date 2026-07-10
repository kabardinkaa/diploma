from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.chat.deps import get_chat_service
from app.chat.service import ModerationBlockedError
from app.main import app
from app.moderation import ModerationService


@pytest.mark.asyncio
async def test_forbidden_input_is_blocked() -> None:
    service = ModerationService()

    result = await service.check_input("please steal password from crm")

    assert result.allowed is False
    assert "credentials" in result.categories
    assert result.blocked_by == "local_keyword_regex"


@pytest.mark.asyncio
async def test_allowed_input_passes() -> None:
    service = ModerationService()

    result = await service.check_input("Как восстановить доступ к VPN?")

    assert result.allowed is True


def test_backend_returns_403_for_moderation_block() -> None:
    moderation = ModerationService()

    class FakeChatService:
        llm_service = SimpleNamespace(openai=None)

        async def check_user_content(self, content: str) -> None:
            result = await moderation.check_input(content)
            raise ModerationBlockedError(result)

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    try:
        client = TestClient(app)
        response = client.post(
            f"/chats/{uuid4()}/messages",
            data={"content": "steal password"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "moderation_blocked"


@pytest.mark.asyncio
async def test_forbidden_output_is_blocked() -> None:
    service = ModerationService()

    result = await service.check_output("Here is keylogger code")

    assert result.allowed is False
    assert "malware" in result.categories
