from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


METHODS = {"get", "post", "put", "patch", "delete"}
PUBLIC_OPERATIONS = {
    ("/health", "get"),
    ("/health/live", "get"),
    ("/health/ready", "get"),
    ("/models", "get"),
    ("/chat", "post"),
    ("/chat/stream", "post"),
    ("/rag/query", "post"),
}
ADMIN_OPERATIONS = {
    ("/chat/batch", "post"),
    ("/documents/upload", "post"),
    ("/documents/reindex", "post"),
    ("/chats/admin/stats", "get"),
    ("/chats/admin/users", "get"),
    ("/chats/admin/broadcast", "post"),
}
INTERNAL_OPERATIONS = {
    ("/chats", "post"),
    ("/chats/{chat_id}", "get"),
    ("/chats/{chat_id}/messages", "get"),
    ("/chats/{chat_id}/messages", "post"),
    ("/chats/{chat_id}/messages", "delete"),
    ("/chats/{chat_id}/messages/{message_id}/feedback", "post"),
    ("/chats/{chat_id}/handoff", "post"),
    ("/chats/admin/internal/broadcasts/pending", "get"),
    ("/chats/admin/internal/broadcasts/{task_id}/result", "post"),
}


def _schema() -> dict:
    return app.openapi()


def _operations(schema: dict):
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method in METHODS:
                yield path, method, operation


def test_docs_and_openapi_are_public_and_valid() -> None:
    client = TestClient(app)

    docs = client.get("/docs")
    response = client.get("/openapi.json")

    assert docs.status_code == 200
    assert response.status_code == 200
    schema = response.json()
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "Diploma AI Assistant API"
    assert schema["info"]["version"]


def test_openapi_tags_and_operation_metadata_cover_every_route() -> None:
    schema = _schema()
    assert [tag["name"] for tag in schema["tags"]] == [
        "Health",
        "Models",
        "Chat",
        "RAG",
        "Agent",
        "Documents / Ingestion",
        "Chat History",
        "Admin",
        "Internal",
    ]

    operations = list(_operations(schema))
    assert {(path, method) for path, method, _ in operations} == (
        PUBLIC_OPERATIONS
        | ADMIN_OPERATIONS
        | INTERNAL_OPERATIONS
        | {("/agent/stream", "post")}
    )
    operation_ids = [operation["operationId"] for _, _, operation in operations]
    assert len(operation_ids) == len(set(operation_ids))
    assert all(operation.get("summary") for _, _, operation in operations)
    assert all(operation.get("description") for _, _, operation in operations)


def test_openapi_security_matches_public_admin_and_internal_boundaries() -> None:
    schema = _schema()
    schemes = schema["components"]["securitySchemes"]
    assert schemes["AdminToken"] == {
        "type": "apiKey",
        "description": "Operator token for administrative and approved write operations.",
        "in": "header",
        "name": "X-Admin-Token",
    }
    assert schemes["InternalToken"] == {
        "type": "apiKey",
        "description": "Service-to-service token used by trusted internal clients.",
        "in": "header",
        "name": "X-Internal-Token",
    }

    for path, method in PUBLIC_OPERATIONS:
        assert "security" not in schema["paths"][path][method]
    for path, method in ADMIN_OPERATIONS:
        assert schema["paths"][path][method]["security"] == [{"AdminToken": []}]
    for path, method in INTERNAL_OPERATIONS:
        assert schema["paths"][path][method]["security"] == [
            {"InternalToken": []}
        ]

    assert schema["paths"]["/agent/stream"]["post"]["security"] == [
        {},
        {"AdminToken": []},
    ]


def test_openapi_marks_server_controlled_compatibility_fields() -> None:
    schema = _schema()
    chat_request = schema["components"]["schemas"]["ChatRequest"]["properties"]
    assert chat_request["model"]["deprecated"] is True
    assert "server always uses" in chat_request["model"]["description"]
    assert chat_request["max_tokens"]["deprecated"] is True
    assert chat_request["max_tokens"]["default"] == 256
    assert chat_request["max_tokens"].get("maximum") != 16000

    agent_request = schema["components"]["schemas"]["AgentStreamRequest"]
    assert agent_request["properties"]["user_role"]["deprecated"] is True
    assert "not an owner credential" in agent_request["properties"]["thread_id"][
        "description"
    ]


def test_openapi_documents_sse_and_primary_error_contracts() -> None:
    schema = _schema()
    for path in (
        "/chat/stream",
        "/agent/stream",
        "/chats/{chat_id}/messages",
    ):
        response = schema["paths"][path]["post"]["responses"]["200"]
        assert set(response["content"]) == {"text/event-stream"}
        examples = response["content"]["text/event-stream"]["examples"]
        assert "safeError" in examples
        assert "event: error" in examples["safeError"]["value"]
        assert "event: done" in examples["safeError"]["value"]

    assert {"200", "422", "429", "502", "503", "504"} <= set(
        schema["paths"]["/chat"]["post"]["responses"]
    )
    assert {"200", "422", "429", "502", "503", "504"} <= set(
        schema["paths"]["/rag/query"]["post"]["responses"]
    )
    assert {"202", "400", "403", "409", "413", "415", "422"} <= set(
        schema["paths"]["/documents/upload"]["post"]["responses"]
    )
