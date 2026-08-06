# Блок 6.4: персистентный LangGraph-агент

## 1. Backend локально и в Docker

Локальный backend по умолчанию - `sqlite`: `AGENT_CHECKPOINTER=sqlite` и
`AGENT_SQLITE_PATH=./agent.db`. Он не требует отдельного сервиса, переживает рестарт
локального процесса и закрывается контекстным менеджером `AsyncSqliteSaver`.
`agent.db` и его служебные файлы исключены из Git.

В Docker сервис `app` получает `AGENT_CHECKPOINTER=postgres`. Используется
`AsyncPostgresSaver` и уже существующая БД `diploma`; новая БД для LangGraph не
создаётся. Для быстрых изолированных тестов также поддержан `memory` через
`InMemorySaver`.

`agent_lifespan()` создаёт saver, один раз вызывает `setup()`, компилирует граф и
закрывает соединение при остановке приложения. HTTP-router получает один
долгоживущий граф через `app.state.persistent_agent`.

## 2. Postgres в compose, URI и проверка таблиц

Сервис `postgres` в `compose.yaml` использует образ `postgres:16-alpine`, БД
`diploma`, пользователя `postgres`, healthcheck `pg_isready` и постоянный volume
`pg-data`. Приложение подключается по уже существующему URI:

```text
postgresql://postgres:postgres@postgres:5432/diploma
```

`DATABASE_URL` является основным источником. Если он отсутствует, URI собирается
из `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST` и
`POSTGRES_PORT`.

В текущей среде Docker daemon не был запущен: клиент сообщил, что pipe
`docker_engine` не найден. Поэтому фактический вывод `psql` не приводится. После
запуска Docker проверка выполняется так:

```powershell
docker compose up -d postgres
docker compose up -d app
docker compose exec postgres psql -U postgres -d diploma -c "\dt checkpoint*"
```

Ожидаемые служебные таблицы создаёт `AsyncPostgresSaver.setup()`:
`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`,
`checkpoint_migrations`. Alembic в проекте отсутствует, поэтому искусственная
конфигурация миграций не добавлялась.

## 3. Почему send_telegram_message опасен

`send_telegram_message` имитирует внешнее пишущее действие. Даже учебная
`print`-реализация представляет границу side effect: в реальном адаптере на этом
месте будет сетевой вызов Telegram.

До `interrupt()` выполняется только `prepare_send_telegram_message`: он извлекает
и проверяет `chat_id` и `text`, сохраняет детерминированный preview и
`tool_call_id`, выставляет `sent=False`. Повторный запуск подготовки безопасен.

После `interrupt()` узел `confirm_and_execute_send_telegram_message` получает
решение. Только при `Command(resume=True)` вызывается send-handler. При
`Command(resume=False)` handler не вызывается. Это важно, потому что при resume
LangGraph запускает interrupt-узел сначала: side effect до `interrupt()` мог бы
выполниться дважды.

## 4. Interrupt и одобрение

Фактический фрагмент запуска `python scripts/time_travel_demo.py`:

```text
__interrupt__ payload:
[Interrupt(value={'type': 'approve_send_telegram_message',
 'preview': {'chat_id': 'demo-support',
 'text': 'Срочно нужна помощь с VPN'}}, ...)]

branch-approve __interrupt__: [Interrupt(...)]
SIDE EFFECT: [TELEGRAM -> demo-support] Срочно нужна помощь с VPN
branch-approve resume=True: sent=True
```

Тест `test_resume_true_executes_side_effect_once` дополнительно проверяет через
`AsyncMock`, что handler вызван ровно один раз и только после resume.

## 5. Time travel

Демонстрация использует временный SQLite-файл и детерминированную модель, поэтому
не требует Postgres, LLM или Telegram API. Реальная история содержала четыре
checkpoint:

```text
checkpoint_id | next
... | ('confirm_and_execute_send_telegram_message',)
... | ('prepare_send_telegram_message',)
... | ('call_model',)
... | ('__start__',)
```

Состояние было прочитано повторно через конкретный `checkpoint_id`:

```text
preview={'chat_id': 'demo-support', 'text': 'Срочно нужна помощь с VPN', ...}
sent=False
next=('confirm_and_execute_send_telegram_message',)
```

Две ветки запускаются с одинаковым входом, но разными стабильными `thread_id`,
поскольку resume-значение фиксируется в lineage:

```text
branch-reject resume=False: sent=False
branch-approve resume=True: sent=True
```

## 6. Streaming modes и curl

Endpoint `POST /agent/stream` использует
`stream_mode=["updates", "messages"]`. `updates` передаёт изменения отдельных
узлов и `__interrupt__`, не дублируя полное состояние. `messages` передаёт
сообщения или токены модели. Каждое событие сериализуется как
`data: <JSON>\n\n`; дополнительно выдаются события `done` и читаемое `error`.

Новый запуск:

```bash
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","input":{"messages":[{"role":"user","content":"Отправь сообщение в чат support: нужна помощь"}]},"user_role":"write-with-approve"}'
```

Продолжение того же thread после interrupt:

```bash
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo-1","resume":true,"user_role":"write-with-approve"}'
```

## 7. Permission policy

| Роль | Политика write-tool |
|---|---|
| `read-only` | Отказ без interrupt и без side effect, `sent=False` |
| `write-with-approve` | Обязательный `interrupt()`, выполнение только после `resume=True` |
| `full` | Выполнение без interrupt; решение явно отражается как одобренное |

`thread_id` всегда приходит извне через API и не генерируется через `uuid4()`.
Он стабилен между первоначальным запросом и resume. `user_role` передаётся и в
state, и в `configurable`, причём configurable-значение является приоритетным.

## 8. Хрупкие места и улучшения

Фактические проверки текущей реализации:

```text
python -m pytest tests/test_agent_persistent.py -q  -> 9 passed
python -m pytest -q                                 -> 228 passed, 57 warnings
python -m compileall app scripts                    -> success
python scripts/time_travel_demo.py                  -> success
docker compose config --quiet                       -> success
```

Полный pytest запускался с временным `PHOENIX_WORKING_DIR=.tmp-phoenix`, потому
что sandbox запрещает Phoenix создавать каталог в профиле пользователя. Warnings
относятся к deprecated API сторонних `ldap3` и Pydantic-зависимостей.

- Реальный Postgres lifecycle и таблицы не проверены из-за остановленного Docker
  daemon; синтаксис compose проверен командой `docker compose config --quiet`.
- Текущий write-handler остаётся учебной `print`-заглушкой. Перед реальным
  Telegram API понадобятся idempotency key, audit log и обработка transient errors.
- При одном model response с несколькими разнотипными tool calls write-tool имеет
  приоритет. Production-версия должна хранить очередь pending tool calls.
- SSE отдаёт ошибки клиенту в безопасном текстовом виде, но production следует
  дополнить correlation ID и метриками прерванных thread.
- История сообщений пока не сжимается; для длинных thread нужен отдельный узел
  compact/summarize и retention policy для checkpoint-таблиц.
