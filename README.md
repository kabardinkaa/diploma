# Diploma AI Assistant

## Что это

Diploma AI Assistant — backend ИИ-ассистента внутренней техподдержки, подготовленный для публичной демонстрации. Он принимает обычные и потоковые chat-запросы, ищет ответы в корпоративной базе знаний, хранит историю диалогов и поддерживает persistent LangGraph-агента с подтверждением действий. Telegram-бот работает как тонкий клиент того же backend.

Продукт поддерживает SSE streaming, изображения, PDF/DOCX, аудио, модерацию, RAG-источники, health/readiness probes, observability, безопасный upload/ingestion и воспроизводимый backup/restore.

## Architecture

```text
Internet
  -> Caddy TLS :80/:443                  (production only)
  -> FastAPI app :8000
       -> OpenAI/OpenRouter-compatible provider
       -> PostgreSQL                     (chat + LangGraph checkpoints)
       -> Qdrant / corporate_rag         (retrieval)
       -> Phoenix / OTLP                 (optional tracing)

Telegram user
  -> aiogram bot
  -> internal HTTP API
  -> FastAPI app

One-shot ingest
  -> data/ + persistent manifest/docstore
  -> Qdrant / corporate_rag
```

Production публикует только Caddy. PostgreSQL, Qdrant, Phoenix, app и bot остаются во внутренних Docker networks. Redis и LiteLLM в текущий runtime не входят. Подробно: [docs/architecture.md](docs/architecture.md).

## Main capabilities

- Async chat API, admin batch и SSE streaming.
- Production RAG по `corporate_rag` с confidence guard и sources.
- Идемпотентный ingestion PDF, DOCX, HTML и Markdown.
- Persistent chat/history в PostgreSQL; JSON repository для локального режима.
- Persistent LangGraph agent с server-controlled roles и confirmation/resume.
- Telegram-бот для text/media, feedback, handoff и admin broadcast.
- Local и optional provider moderation, prompt-injection guard и output moderation.
- Phoenix/OpenTelemetry tracing и structured JSON logs.
- Liveness/readiness, structured infrastructure errors и safe SSE errors.
- Admin/internal perimeter, upload validation, rate/concurrency и hard generation budget.
- TLS deployment, persistent volumes, backup и isolated restore rehearsal.

## Quick start — local

Требуются Docker Engine и Docker Compose v2.

```powershell
git clone <repository-url>
Set-Location diploma
Copy-Item .env.example .env
```

В `.env` задайте один provider key: `OPENROUTER_API_KEY` или `OPENAI_API_KEY`. `BOT_TOKEN` необязателен: без реального token bot тихо завершается. Не коммитьте `.env`.

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose ps -a
```

`ingest` — one-shot service: он должен завершиться с code `0`; `app` ждёт этого условия. Incremental ingestion повторно не строит embeddings для неизменившихся файлов.

```powershell
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/health/ready
```

- Swagger UI: <http://localhost:8000/docs>
- OpenAPI: <http://localhost:8000/openapi.json>
- Phoenix UI: <http://127.0.0.1:6006>

Остановка с сохранением named volumes:

```powershell
docker compose down
```

Не используйте `down --volumes`, если не нужно сознательно удалить данные.

## Environment

| Группа | Ключевые переменные | Назначение |
| --- | --- | --- |
| Provider | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENAI_MODEL` | Один OpenAI-compatible provider и server-selected model |
| Database | `DATABASE_URL`, `DB_POOL_MIN_SIZE`, `DB_POOL_MAX_SIZE`, `CHAT_REPOSITORY`, `AGENT_CHECKPOINTER` | Chat history, pool и LangGraph persistence |
| Qdrant/RAG | `QDRANT_URL`, `QDRANT_API_KEY`, `RAG_PRODUCTION_COLLECTION`, `RAG_MIN_SCORE`, `RAG_RETRIEVAL_TOP_K`, `RAG_MAX_TOKENS` | Retrieval и bounded generation; production collection — `corporate_rag` |
| Telegram | `BOT_TOKEN`, `BOT_ADMIN_IDS`, `BACKEND_URL`, `BOT_DAILY_QUOTA` | Optional bot и per-user quota |
| Security | `ADMIN_TOKEN`, `INTERNAL_TOKEN`, `PUBLIC_SESSION_SECRET`, `CORS_ORIGINS` | Admin/internal auth, public agent identity и browser allowlist |
| Public budget | `PUBLIC_GENERATION_ENABLED`, `PUBLIC_GENERATION_BUDGET_REQUESTS`, `PUBLIC_GENERATION_BUDGET_WINDOW_SECONDS` | Hard process-local budget public generation |
| Rate/concurrency | `PUBLIC_RATE_LIMIT_*`, `PUBLIC_MAX_CONCURRENT_REQUESTS`, `CHAT_MAX_TOKENS` | Per-client protection и server-side token cap |
| Retention/privacy | `PUBLIC_DATA_RETENTION_DAYS`, `RETENTION_CLEANUP_INTERVAL_SECONDS`, `LOG_PROMPT_PREVIEW_ENABLED` | Cleanup persistent public state и safe logs |
| Tracing | `RAG_TRACING_ENABLED`, `TRACING_CAPTURE_CONTENT`, `PHOENIX_COLLECTOR_ENDPOINT` | Optional OTLP traces; production content capture disabled |
| Public deployment | `APP_ENV`, `PUBLIC_DOMAIN`, `PUBLIC_PROXY_IP`, `TRUSTED_PROXY_CIDRS`, `POSTGRES_PASSWORD` | Fail-fast perimeter и Caddy network |

Полные defaults: [.env.example](.env.example) и [.env.production.example](.env.production.example). Реальные secrets в репозитории не хранятся.

## API

| Контур | Основные endpoints |
| --- | --- |
| Public | `GET /health`, `/health/live`, `/health/ready`, `/models`; `POST /chat`, `/chat/stream`, `/rag/query`, `/agent/stream` |
| Admin | `POST /chat/batch`, `/documents/upload`, `/documents/reindex`; `/chats/admin/*`; privileged agent call с `X-Admin-Token` |
| Internal | `/chats/*` для Telegram/backend history и `/chats/admin/internal/*` с `X-Internal-Token` в public mode |

Точные schemas доступны в `/docs` и `/openapi.json`. Public clients не должны использовать internal chat-history API.

## RAG

Production RAG использует Qdrant collection `corporate_rag`, normalized `intfloat/multilingual-e5-base` embeddings, recursive chunking и dense retrieval. Ingestion поддерживает PDF, DOCX, HTML/HTM и Markdown, ведёт persistent manifest/docstore и не перезаписывает существующий corpus-файл через upload.

`POST /rag/query` возвращает `answer`, `sources`, `top_score` и `confident`. Пустой retrieval или score ниже threshold — штатный refusal без LLM generation. Qdrant network/auth/timeout/server failures возвращают safe structured `503`, а не маскируются под empty retrieval. Generation ограничена `RAG_MAX_TOKENS`.

Подробно: [docs/rag.md](docs/rag.md). Evaluation evidence: [docs/rag_evaluation.md](docs/rag_evaluation.md).

## Persistent Agent

`POST /agent/stream` использует один долгоживущий LangGraph graph и persistent checkpointer: SQLite локально, PostgreSQL в Compose. Public session identity создаётся сервером и подписывается HttpOnly cookie; клиентский `thread_id` не даёт доступ к чужой сессии.

Публичный запрос всегда получает `read-only`, даже если body пытается задать `full`. Валидный `X-Admin-Token` даёт `write-with-approve`: write-tool выполняется только после interrupt и явного `resume=true`.

Подробно: [docs/agent-persistent-report.md](docs/agent-persistent-report.md).

## Security / Public deployment

- Caddy завершает TLS; internal services не публикуются на host.
- Admin/internal endpoints защищены разными tokens.
- Model, role и token budget выбираются server-side.
- Public endpoints защищены rate limit, concurrency cap и общим process-local generation budget.
- Production запрещает wildcard/HTTP/localhost CORS origins и fail-fast проверяет secrets, proxy, budget, retention и privacy settings.
- Upload/reindex ограничены по size/count, filename, MIME/signature, archive structure и ingestion lock.
- Public chat/checkpoint state и bot caches имеют bounded retention/TTL.

Public runbook: [docs/deployment.md](docs/deployment.md). Security evidence: [docs/security/README.md](docs/security/README.md).

## Observability

Structured logs содержат request ID, latency, model/usage metadata и prompt hash. Phoenix принимает OpenTelemetry/OpenInference traces для OpenAI-compatible и LlamaIndex вызовов. В production tracing по умолчанию выключен; при включении `TRACING_CAPTURE_CONTENT=false` скрывает inputs, outputs, messages, prompts, choices и embeddings. Prompt preview в production logs отключён.

Примеры: [docs/observability/README.md](docs/observability/README.md).

## Backup / Restore

`scripts/deployment_backup.py` создаёт единый manifest-набор:

- PostgreSQL custom-format dump и row counts;
- Qdrant snapshot `corporate_rag` и point count;
- production corpus;
- RAG docstore/manifest;
- checksums и safe configuration без secrets.

Restore проверяет checksums до записи, восстанавливает Postgres/Qdrant/corpus/RAG state и проверяет readiness. Workflow репетировался в изолированном Compose project с `docker-compose.restore.yml`. Команды и ограничения: [docs/deployment.md](docs/deployment.md#7-backup).

## Tests

```powershell
python -m pytest -q
python -m compileall -q app bot scripts
docker compose config --quiet
```

Полный regression-набор запускается командой `pytest -q`; актуальный результат фиксируется в отчёте о финальной технической проверке. Evaluation suites и их артефакты описаны отдельно в [docs/rag_evaluation.md](docs/rag_evaluation.md).

## Public deployment

```powershell
Copy-Item .env.production.example .env.production
# Replace every change-me value; do not commit this file.
python -m scripts.validate_production_config --env-file .env.production
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production config --quiet
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production up -d --build
```

Production публикует только Caddy `80/443`. Перед запуском прочитайте [docs/deployment.md](docs/deployment.md).

## Documentation

Карта active runbooks, evaluation evidence и historical materials: [docs/README.md](docs/README.md).

## Course history / Evolution

Исходная хронология блоков 3–6, промежуточные команды, benchmark- и тестовые результаты сохранены без потерь в [docs/archive/course-history.md](docs/archive/course-history.md). Это evidence, а не текущая production-инструкция.
