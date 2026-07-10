from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from app.chat.deps import get_repository
from app.chat.domain import AdminStats, AdminUser, BroadcastTask
from app.core.config import get_settings
from app.main import app


class FakeAdminRepository:
    async def admin_stats(self):
        return AdminStats(
            total_messages=1,
            active_users=1,
            avg_latency_ms=None,
            moderation_block_rate=0.0,
            feedback_up_ratio=None,
        )

    async def list_admin_users(self, limit: int = 50):
        return [
            AdminUser(
                owner_external_id="123",
                chats_count=1,
                last_seen_at=datetime.now(UTC),
            )
        ][:limit]

    async def create_broadcast(self, message: str, interface_filter: str | None = None):
        return BroadcastTask(
            id=uuid4(),
            message=message,
            interface_filter=interface_filter,
            recipients=["123"],
        )

def test_admin_without_token_returns_403(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    get_settings.cache_clear()
    app.dependency_overrides[get_repository] = lambda: FakeAdminRepository()

    try:
        response = TestClient(app).get("/chats/admin/stats")
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 403


def test_admin_stats_users_and_broadcast(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    get_settings.cache_clear()
    app.dependency_overrides[get_repository] = lambda: FakeAdminRepository()
    client = TestClient(app)

    try:
        headers = {"X-Admin-Token": "secret"}
        stats = client.get("/chats/admin/stats", headers=headers)
        users = client.get("/chats/admin/users", headers=headers)
        broadcast = client.post(
            "/chats/admin/broadcast",
            headers=headers,
            json={
                "message": "hello",
                "interface_filter": "telegram",
            },
        )
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert stats.status_code == 200
    assert stats.json()["total_messages"] == 1
    assert users.status_code == 200
    assert users.json()[0]["owner_external_id"] == "123"
    assert broadcast.status_code == 200
    assert broadcast.json()["status"] == "pending"
