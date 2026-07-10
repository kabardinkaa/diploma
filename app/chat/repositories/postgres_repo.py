import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

try:
    import asyncpg
except ImportError:  # pragma: no cover - exercised only without optional dependency
    asyncpg = None  # type: ignore[assignment]

from app.chat.domain import (
    AdminStats,
    AdminUser,
    BroadcastTask,
    Chat,
    ChatMessage,
    Feedback,
    FeedbackValue,
    HandoffStatus,
    SystemPrompt,
)
from app.chat.repositories.json_repo import DEFAULT_PROMPTS


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chats (
    id uuid PRIMARY KEY,
    owner_external_id text NOT NULL,
    interface text NOT NULL,
    system_prompt text,
    handoff_status text NOT NULL DEFAULT 'active',
    created_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id uuid PRIMARY KEY,
    chat_id uuid NOT NULL REFERENCES chats(id),
    role text NOT NULL,
    content text NOT NULL,
    media_refs jsonb,
    prompt_id uuid,
    tokens integer,
    created_at timestamptz NOT NULL,
    deleted_at timestamptz
);

CREATE TABLE IF NOT EXISTS feedback (
    id uuid PRIMARY KEY,
    chat_id uuid NOT NULL REFERENCES chats(id),
    message_id uuid NOT NULL,
    owner_external_id text NOT NULL,
    value text NOT NULL,
    created_at timestamptz NOT NULL,
    UNIQUE(owner_external_id, message_id)
);

CREATE TABLE IF NOT EXISTS broadcast_tasks (
    id uuid PRIMARY KEY,
    message text NOT NULL,
    interface_filter text,
    status text NOT NULL,
    sent integer NOT NULL DEFAULT 0,
    failed integer NOT NULL DEFAULT 0,
    recipients jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL,
    updated_at timestamptz
);

CREATE TABLE IF NOT EXISTS system_prompts (
    id uuid PRIMARY KEY,
    version text NOT NULL,
    body text NOT NULL,
    created_at timestamptz NOT NULL,
    active boolean NOT NULL,
    traffic_pct integer NOT NULL,
    notes text
);
"""


class PostgresChatRepository:
    def __init__(self, database_url: str) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for PostgresChatRepository")

        self.database_url = database_url
        self._schema_ready = False

    async def _connect(self):
        return await asyncpg.connect(self.database_url)

    async def _ensure_schema(self, conn) -> None:
        if self._schema_ready:
            return

        await conn.execute(SCHEMA_SQL)
        count = await conn.fetchval("SELECT COUNT(*) FROM system_prompts")

        if count == 0:
            for prompt in DEFAULT_PROMPTS:
                await conn.execute(
                    """
                    INSERT INTO system_prompts
                    (id, version, body, created_at, active, traffic_pct, notes)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    """,
                    prompt.id,
                    prompt.version,
                    prompt.body,
                    prompt.created_at,
                    prompt.active,
                    prompt.traffic_pct,
                    prompt.notes,
                )

        self._schema_ready = True

    async def _run(self, callback):
        conn = await self._connect()
        try:
            await self._ensure_schema(conn)
            return await callback(conn)
        finally:
            await conn.close()

    def _chat_from_row(self, row) -> Chat:
        return Chat(
            id=row["id"],
            owner_external_id=row["owner_external_id"],
            interface=row["interface"],
            system_prompt=row["system_prompt"],
            handoff_status=row["handoff_status"],
            created_at=row["created_at"],
        )

    def _message_from_row(self, row) -> ChatMessage:
        media_refs = row["media_refs"]
        if isinstance(media_refs, str):
            media_refs = json.loads(media_refs)

        return ChatMessage(
            id=row["id"],
            chat_id=row["chat_id"],
            role=row["role"],
            content=row["content"],
            media_refs=media_refs,
            prompt_id=row["prompt_id"],
            tokens=row["tokens"],
            created_at=row["created_at"],
        )

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )

        async def op(conn):
            await conn.execute(
                """
                INSERT INTO chats
                (id, owner_external_id, interface, system_prompt, handoff_status, created_at)
                VALUES ($1,$2,$3,$4,$5,$6)
                """,
                chat.id,
                chat.owner_external_id,
                chat.interface,
                chat.system_prompt,
                chat.handoff_status,
                chat.created_at,
            )
            return chat

        return await self._run(op)

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        async def op(conn):
            row = await conn.fetchrow("SELECT * FROM chats WHERE id=$1", chat_id)
            return self._chat_from_row(row) if row else None

        return await self._run(op)

    async def append_message(
        self,
        chat_id: UUID,
        message: ChatMessage,
    ) -> ChatMessage:
        async def op(conn):
            await conn.execute(
                """
                INSERT INTO chat_messages
                (id, chat_id, role, content, media_refs, prompt_id, tokens, created_at)
                VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8)
                """,
                message.id,
                chat_id,
                message.role,
                message.content,
                json.dumps(message.media_refs, ensure_ascii=False)
                if message.media_refs is not None
                else None,
                message.prompt_id,
                message.tokens,
                message.created_at,
            )
            return message

        return await self._run(op)

    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 50,
    ) -> list[ChatMessage]:
        async def op(conn):
            rows = await conn.fetch(
                """
                SELECT * FROM chat_messages
                WHERE chat_id=$1 AND deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $2
                """,
                chat_id,
                limit,
            )
            return [
                self._message_from_row(row)
                for row in reversed(rows)
            ]

        return await self._run(op)

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        async def op(conn):
            await conn.execute(
                "UPDATE chat_messages SET deleted_at=$2 WHERE chat_id=$1 AND deleted_at IS NULL",
                chat_id,
                datetime.now(UTC),
            )

        await self._run(op)

    async def save_feedback(
        self,
        chat_id: UUID,
        message_id: UUID,
        owner_external_id: str,
        value: FeedbackValue,
    ) -> Feedback | None:
        feedback = Feedback(
            chat_id=chat_id,
            message_id=message_id,
            owner_external_id=owner_external_id,
            value=value,
        )

        async def op(conn):
            row = await conn.fetchrow(
                """
                INSERT INTO feedback
                (id, chat_id, message_id, owner_external_id, value, created_at)
                VALUES ($1,$2,$3,$4,$5,$6)
                ON CONFLICT (owner_external_id, message_id) DO NOTHING
                RETURNING id
                """,
                feedback.id,
                feedback.chat_id,
                feedback.message_id,
                feedback.owner_external_id,
                feedback.value,
                feedback.created_at,
            )
            return feedback if row else None

        return await self._run(op)

    async def set_handoff_status(
        self,
        chat_id: UUID,
        status: HandoffStatus,
    ) -> Chat | None:
        async def op(conn):
            row = await conn.fetchrow(
                """
                UPDATE chats SET handoff_status=$2
                WHERE id=$1
                RETURNING *
                """,
                chat_id,
                status,
            )
            return self._chat_from_row(row) if row else None

        return await self._run(op)

    async def admin_stats(self) -> AdminStats:
        async def op(conn):
            total_messages = await conn.fetchval(
                """
                SELECT COUNT(*) FROM chat_messages
                WHERE created_at >= now() - interval '24 hours'
                """
            )
            active_users = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT c.owner_external_id)
                FROM chat_messages m
                JOIN chats c ON c.id = m.chat_id
                WHERE m.created_at >= now() - interval '24 hours'
                """
            )
            feedback_total = await conn.fetchval(
                "SELECT COUNT(*) FROM feedback WHERE created_at >= now() - interval '24 hours'"
            )
            feedback_up = await conn.fetchval(
                """
                SELECT COUNT(*) FROM feedback
                WHERE value='up' AND created_at >= now() - interval '24 hours'
                """
            )
            return AdminStats(
                total_messages=int(total_messages or 0),
                active_users=int(active_users or 0),
                avg_latency_ms=None,
                moderation_block_rate=0.0,
                feedback_up_ratio=(
                    float(feedback_up) / float(feedback_total)
                    if feedback_total
                    else None
                ),
            )

        return await self._run(op)

    async def list_admin_users(self, limit: int = 50) -> list[AdminUser]:
        async def op(conn):
            rows = await conn.fetch(
                """
                SELECT
                  c.owner_external_id,
                  COUNT(DISTINCT c.id) AS chats_count,
                  GREATEST(MAX(c.created_at), COALESCE(MAX(m.created_at), MAX(c.created_at))) AS last_seen_at
                FROM chats c
                LEFT JOIN chat_messages m ON m.chat_id = c.id
                GROUP BY c.owner_external_id
                ORDER BY last_seen_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [
                AdminUser(
                    owner_external_id=row["owner_external_id"],
                    chats_count=row["chats_count"],
                    last_seen_at=row["last_seen_at"],
                )
                for row in rows
            ]

        return await self._run(op)

    async def create_broadcast(
        self,
        message: str,
        interface_filter: str | None = None,
    ) -> BroadcastTask:
        async def op(conn):
            rows = await conn.fetch(
                """
                SELECT DISTINCT owner_external_id FROM chats
                WHERE $1::text IS NULL OR interface=$1
                ORDER BY owner_external_id
                """,
                interface_filter,
            )
            task = BroadcastTask(
                message=message,
                interface_filter=interface_filter,
                recipients=[row["owner_external_id"] for row in rows],
            )
            await conn.execute(
                """
                INSERT INTO broadcast_tasks
                (id, message, interface_filter, status, sent, failed, recipients, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8)
                """,
                task.id,
                task.message,
                task.interface_filter,
                task.status,
                task.sent,
                task.failed,
                json.dumps(task.recipients),
                task.created_at,
            )
            return task

        return await self._run(op)

    async def list_pending_broadcasts(self, limit: int = 10) -> list[BroadcastTask]:
        async def op(conn):
            rows = await conn.fetch(
                """
                SELECT * FROM broadcast_tasks
                WHERE status='pending'
                ORDER BY created_at
                LIMIT $1
                """,
                limit,
            )
            return [
                BroadcastTask.model_validate(dict(row))
                for row in rows
            ]

        return await self._run(op)

    async def update_broadcast_result(
        self,
        task_id: UUID,
        sent: int,
        failed: int,
        status: str = "done",
    ) -> BroadcastTask | None:
        async def op(conn):
            row = await conn.fetchrow(
                """
                UPDATE broadcast_tasks
                SET sent=$2, failed=$3, status=$4, updated_at=$5
                WHERE id=$1
                RETURNING *
                """,
                task_id,
                sent,
                failed,
                status,
                datetime.now(UTC),
            )
            return BroadcastTask.model_validate(dict(row)) if row else None

        return await self._run(op)

    async def list_active_prompts(self) -> list[SystemPrompt]:
        async def op(conn):
            rows = await conn.fetch(
                """
                SELECT * FROM system_prompts
                WHERE active=true
                ORDER BY created_at, version
                """
            )
            return [
                SystemPrompt.model_validate(dict(row))
                for row in rows
            ]

        return await self._run(op)
