# Telegram bot

Telegram bot — тонкий aiogram-клиент production backend. Исторический отчёт ранней реализации сохранён в [archive/telegram-bot-course-report.md](archive/telegram-bot-course-report.md).

## Architecture

```text
Telegram user
  -> aiogram handlers
  -> BackendClient
  -> internal /chats API
  -> ChatService
  -> RAG / LLM / PostgreSQL
```

Bot не создаёт отдельный LLM/RAG stack и не хранит полную историю. Backend отвечает за persistence, retrieval, moderation, generation и sources.

## Startup

`BOT_TOKEN` опционален. Если token пуст или равен placeholder, процесс пишет безопасное предупреждение и завершается без restart loop. При реальном token:

```powershell
python -m bot
```

В Compose `BACKEND_URL=http://app:8000`. В public mode bot и app должны иметь одинаковый реальный `INTERNAL_TOKEN`. Broadcast worker запускается только с настроенным internal token; без него bot продолжает основные функции без повторяющихся 403.

## User features

- `/start`, `/help`, `/ask`, `/clear`, `/cancel` и `/operator`.
- Обычный text chat через backend SSE.
- Photo, voice, audio, PDF и DOCX с отдельными server-side size/type limits.
- RAG sources, confidence metadata и feedback buttons.
- Handoff к оператору.
- Per-user short-window и daily quota.
- Bounded TTL mappings между Telegram user/chat и backend chat.

Admin IDs из `BOT_ADMIN_IDS` дополнительно получают `/stats`, `/users` и `/broadcast`.

## Backend contract

Основные вызовы:

- `POST /chats` — создать backend chat;
- `POST /chats/{chat_id}/messages` — text/media SSE;
- `DELETE /chats/{chat_id}/messages` — очистить историю;
- `POST /chats/{chat_id}/messages/{message_id}/feedback`;
- `POST /chats/{chat_id}/handoff`;
- `/chats/admin/internal/*` — polling/result broadcast worker.

Во внешнем public deployment эти endpoints не являются публичным клиентским API и требуют `X-Internal-Token`. Admin endpoints используют отдельный `X-Admin-Token`.

## Limits and privacy

| Setting | Purpose |
| --- | --- |
| `BOT_RATE_LIMIT_REQUESTS` / `BOT_RATE_LIMIT_WINDOW_SECONDS` | Per-user burst limit |
| `BOT_DAILY_QUOTA` | Per-user daily generation quota |
| `BOT_QUOTA_MAX_USERS` / `BOT_QUOTA_STATE_TTL_SECONDS` | Bound quota state |
| `BOT_CHAT_CACHE_MAX_ENTRIES` / `BOT_CHAT_CACHE_TTL_SECONDS` | Bound chat mappings |
| `BOT_PHOTO_MAX_BYTES` | Photo download limit |
| `BOT_MEDIA_MAX_BYTES` | Document/voice/audio download limit |

Telegram end-user generation также расходует общий backend `PUBLIC_GENERATION_BUDGET_REQUESTS`. Token и internal/admin credentials не включаются в user-facing errors или logs.

## Verification

Bot client, FSM, media limits, lifecycle, auth/error mapping и production handlers покрыты в `tests/bot/`. Полный актуальный test count приводится только в корневом README; исторические counts находятся в архивном отчёте.
