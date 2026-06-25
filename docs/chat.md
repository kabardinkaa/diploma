# Архитектура чата и хранение истории

## Что реализовано

В проект добавлен отдельный backend-модуль `app/chat/`, который хранит историю диалогов на стороне сервера.

Клиент больше не обязан каждый раз передавать весь массив `messages`. Вместо этого он:

1. создаёт чат через `POST /chats`;
2. отправляет новое сообщение через `POST /chats/{chat_id}/messages`;
3. backend сам загружает историю, собирает контекст и отправляет его в LLM.

Старые ручки `/chat`, `/chat/stream`, `/chat/batch` сохранены и не ломались.

## Архитектура

```mermaid
flowchart LR
    Client[CLI / Web / Bot] --> Routes[app/chat/routes.py]

    Routes --> Service[ChatService]

    Service --> Repo[ChatRepository Protocol]
    Service --> LLM[LLMService]

    Repo --> JsonRepo[JsonChatRepository]
    JsonRepo --> Files[chat.json + messages.jsonl]

    LLM --> OpenRouter[OpenAI-compatible API / OpenRouter]