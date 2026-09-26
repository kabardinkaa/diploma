# Chat and history

## Runtime contract

Chat subsystem lives in `app/chat/`. Telegram and other trusted clients create a server-side chat, then send messages by chat ID; the backend loads history, applies moderation/RAG and persists the assistant response.

```text
Client -> /chats routes -> ChatService
                         -> ChatRepository -> PostgreSQL (Compose)
                                           -> JSON files (local option)
                         -> RAGService / LLMService
```

The generic public `/chat` and `/chat/stream` endpoints remain stateless request/response APIs. `/chats/*` is the persistent history API and requires `X-Internal-Token` in public mode.

## Storage

Compose forces `CHAT_REPOSITORY=postgres` and uses one shared application pool. Local development may select `json` with `CHAT_STORAGE_DIR`.

Persisted records include chats, messages, media metadata, RAG sources, selected prompt IDs, feedback, handoff state and broadcast tasks. Public retention removes expired records while protecting active streams.

## Message flow

1. Validate text/media size, filename, MIME/signature and archive structure.
2. Run input moderation.
3. Load bounded conversation context.
4. For text-only messages, retrieve `corporate_rag` context and apply confidence guard.
5. Generate and apply output moderation.
6. Persist user/assistant messages and shown sources.
7. Emit SSE token payloads, `sources` event and `done` with message ID.

After stream start, provider/infrastructure failures use safe SSE `error` + `done` without raw exception details.

## Media

Supported message inputs include images, PDF, DOCX, voice and audio. Audio transcription uses the local bounded Whisper resource; document extraction is bounded by upload/archive limits. Telegram applies its own smaller photo/media limits before backend upload.

## Internal and admin operations

- Feedback is idempotent per owner/message.
- Handoff pauses automated processing for operator flow.
- Admin endpoints expose aggregate stats/users and create broadcasts.
- Broadcast worker polls internal pending/result endpoints only when `INTERNAL_TOKEN` is configured.
