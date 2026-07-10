from app.chat.repositories.postgres_repo import PostgresChatRepository, SCHEMA_SQL


def test_postgres_repository_exposes_required_contract() -> None:
    required_methods = [
        "create_chat",
        "get_chat",
        "append_message",
        "list_messages",
        "soft_delete_messages",
        "save_feedback",
        "admin_stats",
        "list_admin_users",
        "create_broadcast",
        "list_pending_broadcasts",
        "set_handoff_status",
        "list_active_prompts",
    ]

    for method in required_methods:
        assert hasattr(PostgresChatRepository, method)


def test_postgres_schema_contains_required_tables() -> None:
    for table in (
        "chats",
        "chat_messages",
        "feedback",
        "broadcast_tasks",
        "system_prompts",
    ):
        assert table in SCHEMA_SQL
