# Persistent LangGraph agent

Этот документ описывает текущий production path. Исторический отчёт блока 6.4 сохранён в [archive/agent-persistent-report.md](archive/agent-persistent-report.md).

## Runtime

`app/services/agent_persistent.py` создаёт и компилирует один `StateGraph` на lifespan приложения. Model client и checkpointer переиспользуются между запросами и закрываются при shutdown.

| Environment | Checkpointer |
| --- | --- |
| Local default | SQLite: `AGENT_SQLITE_PATH` |
| Docker Compose | PostgreSQL через `DATABASE_URL` |
| Isolated tests | In-memory |

HTTP endpoint: `POST /agent/stream`. Он выдаёт SSE updates/messages, interrupt и финальный `done`. Ошибка после старта stream маппится в безопасные `error` + `done` без raw exception.

## Session isolation

Клиент передаёт логический `thread_id`, но не получает прямой доступ к checkpoint namespace.

- Public request получает подписанную server-generated session identity в HttpOnly cookie.
- Эффективный namespace содержит server principal и client thread ID.
- Подмена `thread_id` без cookie не открывает чужое состояние.
- Cookie secret и token не пишутся в logs/checkpoints.
- Public cookie имеет `SameSite=Lax`, `Path=/agent` и `Secure` в public mode.

## Roles

| Caller | Effective role | Write-tool policy |
| --- | --- | --- |
| Public, без admin token | `read-only` | Пишущие инструменты отклоняются без side effect |
| Валидный `X-Admin-Token` | `write-with-approve` | `interrupt()` и выполнение только после `resume=true` |

Поле роли из request body не повышает привилегии: public caller остаётся `read-only`.

## Tool execution

Read-only tools выполняются в отдельном safe node. Неизвестная или упавшая tool возвращает контролируемый tool error и логирует только тип ошибки.

Текущий write boundary — `send_telegram_message`:

1. Model предлагает tool call.
2. Graph валидирует и сохраняет preview без side effect.
3. Для admin role graph создаёт interrupt.
4. `resume=false` отклоняет действие.
5. `resume=true` вызывает handler один раз после подтверждения.

Это confirmation contract демонстрационного агента, а не универсальная транзакционная система для произвольных внешних интеграций.

## Persistence and retention

Compose использует таблицы LangGraph в том же PostgreSQL instance. Retention manager удаляет только просроченные checkpoint threads и сериализуется с активными agent streams, поэтому активная сессия не удаляется посреди запроса.

Backup включает checkpoint tables в PostgreSQL dump. Restore workflow проверяет PostgreSQL state и readiness; подробности — в [deployment.md](deployment.md).

## Public controls

Public agent request:

- использует server-selected `AGENT_MODEL` и `CHAT_MAX_TOKENS`;
- учитывается общим public generation budget;
- ограничивается rate/concurrency middleware;
- не получает privileged role через request payload;
- не раскрывает provider/internal exception text через SSE.

Точные request/resume schemas доступны в Swagger `/docs`.
