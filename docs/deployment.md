# Public deployment quick start

Этот документ описывает финальный public runtime. Исторические учебные блоки в
README остаются доказательной базой курса, но не являются инструкцией по
публичному развёртыванию.

## 1. Подготовка

Нужны Docker Engine с Compose v2, публичный DNS `A`/`AAAA` record на сервер и
доступные входящие TCP `80/443` (плюс UDP `443` для HTTP/3). Реальный домен в
репозитории не задан.

```powershell
Copy-Item .env.production.example .env.production
```

В `.env.production` обязательно заменить:

- `PUBLIC_DOMAIN`;
- `ADMIN_TOKEN` и `INTERNAL_TOKEN`;
- `PUBLIC_SESSION_SECRET` (отдельный случайный секрет не короче 32 символов);
- `POSTGRES_PASSWORD` и согласованный `DATABASE_URL`;
- `QDRANT_API_KEY`;
- один provider key: `OPENAI_API_KEY` или `OPENROUTER_API_KEY`;
- `CORS_ORIGINS` на HTTPS origin публичного интерфейса.

Генерируйте независимые длинные случайные секреты. Значения `change-me-*`,
пустые обязательные credentials, URL вместо host name, глобальный trusted proxy
`0.0.0.0/0`, короткий/placeholder public session secret и несогласованные
proxy IP/subnet отклоняются. `.env.production`
игнорируется Git и не включается в backup по умолчанию.

Проверка до запуска не печатает секреты:

```powershell
python -m scripts.validate_production_config --env-file .env.production
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production config --quiet
```

## 2. Запуск и проверка

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production ps -a
```

Caddy получает/обновляет TLS certificate автоматически. Открывайте:

- `https://<PUBLIC_DOMAIN>/health/live` — процесс app жив;
- `https://<PUBLIC_DOMAIN>/health/ready` — Postgres, Qdrant и `corporate_rag` готовы;
- `https://<PUBLIC_DOMAIN>/docs` — Swagger/OpenAPI;
- `https://<PUBLIC_DOMAIN>/openapi.json` — OpenAPI JSON.

Без платного вызова инфраструктуру проверяют health/OpenAPI endpoints. RAG
проверяется одним осознанным запросом только после настройки provider budget:

```powershell
curl.exe -X POST "https://<PUBLIC_DOMAIN>/rag/query" `
  -H "Content-Type: application/json" `
  -d '{"question":"Как подключиться к корпоративному VPN?"}'
```

Остановка сохраняет named volumes:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production down
```

Не используйте `down --volumes`, если не требуется сознательно удалить данные.

## 3. Сетевая и TLS-модель

Production overlay публикует только Caddy:

- TCP `80` — ACME/redirect на HTTPS;
- TCP `443` — HTTPS;
- UDP `443` — HTTP/3;
- `app:8000` — только Docker network;
- Postgres, Qdrant и Phoenix — только Docker network;
- bot не имеет входящего порта.

Caddy подключён только к выделенной `public-edge` сети и проксирует только
`app:8000`. App соединяет edge и внутреннюю default network. Доступ proxy к
Postgres/Qdrant/Phoenix отсутствует.

TLS включается Caddy автоматически для `PUBLIC_DOMAIN`. HSTS отправляется на
HTTPS-ответах; также включены `X-Content-Type-Options`, `Referrer-Policy` и
`X-Frame-Options`. Request body ограничен `12MB`, что оставляет overhead поверх
backend upload limit 10 MiB. Таймауты ограничивают медленные подключения, а
`flush_interval -1` сохраняет немедленную доставку SSE events.

Для локального production-like smoke допустим `PUBLIC_DOMAIN=localhost`; Caddy
использует локальный certificate, поэтому тестовый клиент должен доверять Caddy
local CA или явно использовать `curl -k`. Это не конфигурация публичного DNS.

## 4. Trusted proxy и client IP

Rate limiter использует socket peer IP напрямую. `X-Forwarded-For` учитывается
только когда непосредственный peer входит в `TRUSTED_PROXY_CIDRS`. Production
defaults согласованы с единственным Caddy container:

```env
PUBLIC_PROXY_SUBNET=172.31.250.0/24
PUBLIC_PROXY_IP=172.31.250.10
TRUSTED_PROXY_CIDRS=172.31.250.10/32
```

Caddy по умолчанию формирует upstream `X-Forwarded-For` из реального клиента и
не доверяет входному spoofed значению. App проверяет, что `PUBLIC_PROXY_IP`
попадает в trusted CIDR. В dev `TRUSTED_PROXY_CIDRS` пуст, поэтому произвольный
заголовок клиента игнорируется.

Public persistent-agent не использует IP как owner identity. При первом
`/agent/stream` сервер выдаёт подписанную opaque cookie с `HttpOnly`,
`SameSite=Lax`, `Path=/agent` и `Secure` в public mode. Checkpoint namespace
строится из server-controlled principal и client `thread_id`; cookie token и его
секрет не пишутся в logs/checkpoints.

При конфликте `172.31.250.0/24` с сетью хоста выберите другой приватный subnet и
одновременно согласуйте все три значения. Не используйте `0.0.0.0/0` или
`::/0`.

## 5. Public generation budget, CORS and privacy

Public generation routes `/chat`, `/chat/stream`, `/rag/query` and public
`/agent/stream` share one hard request budget:

```env
PUBLIC_GENERATION_ENABLED=true
PUBLIC_GENERATION_BUDGET_REQUESTS=100
PUBLIC_GENERATION_BUDGET_WINDOW_SECONDS=86400
```

One accepted public request consumes one unit before provider execution. Web
generation and Telegram end-user text/media generation share the budget. Health,
OpenAPI, admin batch and non-generation internal/admin routes do not consume it.
Exhaustion returns structured `429 public_quota_exhausted`; setting
`PUBLIC_GENERATION_ENABLED=false` returns `503 public_generation_disabled`
without calling the provider. Streaming routes send the same safe code as an
SSE `error` event followed by `done`.

The counter is intentionally process-local for this single-replica diploma
deployment. It resets on app restart and each replica would have its own
counter. A multi-replica or strict billing deployment must replace it with a
shared atomic store such as Redis or a database-backed quota.

Production `CORS_ORIGINS` is a JSON list of explicit HTTPS origins. Wildcards,
plain HTTP, localhost, credentials in URLs, paths, queries and fragments are
rejected during startup. Development keeps the permissive wildcard default.

Public persisted chat/message/feedback/broadcast state and PostgreSQL agent
checkpoint threads are cleaned using:

```env
PUBLIC_DATA_RETENTION_DAYS=30
RETENTION_CLEANUP_INTERVAL_SECONDS=3600
```

Cleanup is serialized with active chat and agent streams, so their state is not
removed mid-request. System prompts and RAG corpus/evidence are not part of this
cleanup. The LLM response cache is already bounded by entry count and TTL;
Telegram chat-id mappings are bounded by `BOT_CHAT_CACHE_MAX_ENTRIES` and
`BOT_CHAT_CACHE_TTL_SECONDS`; per-user Telegram quota state is bounded by
`BOT_QUOTA_MAX_USERS` and `BOT_QUOTA_STATE_TTL_SECONDS`. Docker JSON logs are
size/file rotated.

Production defaults disable Phoenix export and prompt previews. If tracing is
enabled deliberately, keep `TRACING_CAPTURE_CONTENT=false`: OpenInference then
hides inputs, outputs, messages, prompts, choices and embedding text/vectors.
Phoenix's persistent volume has no automatic age-based cleanup in this project;
with tracing disabled no new application traces are written, while an operator
must explicitly prune historical Phoenix data under the deployment's data
policy. Never delete `phoenix-data` as part of an ordinary restart.

No browser-visible demo token is used. For an open demo it would be a public
client-side value and would add a second authentication scheme without material
protection beyond the server-side hard budget, rate limit and concurrency cap.

## 6. Persistent data

Compose использует named volumes:

- `pg-data` — Postgres chat/checkpoint state;
- `qdrant-storage` — vector collections;
- `rag-state` — ingestion docstore;
- `phoenix-data` — локальные traces;
- `caddy-data` — TLS certificates/account;
- `caddy-config` — runtime state Caddy.

Production corpus монтируется из `./data`. Embedding weights/cache находятся в
`./.cache/embeddings` и могут быть повторно загружены/построены.

## 7. Backup

Во время backup не запускайте upload/reindex/ingest. `pg_dump` и Qdrant snapshot
создаются штатными online-механизмами; текущие volumes не удаляются.

```powershell
python scripts/deployment_backup.py backup `
  --project-name diploma `
  --env-file .env.production
```

Архив появляется в `backups/<UTC timestamp>/`; `backups/` игнорируется Git.
Внутри:

- `postgres.dump` — custom-format PostgreSQL dump;
- `postgres.json` — контрольные row counts основных таблиц;
- `qdrant.snapshot` — snapshot collection `corporate_rag`;
- `qdrant.json` — имя collection и ожидаемый point count;
- `rag-state/` — persistent docstore;
- `corpus.zip` — production corpus `data/`;
- `config/` — Compose/Caddy/constraints и env template без секретов;
- `manifest.json` — состав, исключения и SHA-256 checksums.

Не сохраняются реальные `.env`/`.env.production`, provider keys, embedding cache,
Phoenix traces, Docker images и logs. Production secret file храните отдельно в
зашифрованном secret manager/backup.

План команд без записи backup:

```powershell
python scripts/deployment_backup.py backup `
  --project-name diploma `
  --env-file .env.production `
  --dry-run
```

## 8. Restore

Restore полностью заменяет состояние Postgres, collection `corporate_rag`, RAG
docstore и corpus. Наложения архива поверх более нового `data/` нет: содержимое
целевого corpus и RAG state сначала очищается внутри их конкретных mount points,
затем сверяется с backup. Manifest и все SHA-256 checksums проверяются до первой
операции с Compose.

CLI требует явный `--project-name` и передаёт его каждой внутренней команде
Compose. Восстановление в текущий project `diploma` по умолчанию запрещено. Для
сознательного production restore нужны оба подтверждения:

```powershell
python scripts/deployment_backup.py restore `
  --project-name diploma `
  --allow-current-project `
  --env-file .env.production `
  --from backups/<UTC timestamp> `
  --confirm-restore
```

До запуска сохраните отдельную копию текущих данных и остановите внешние writers.
Сам runner останавливает `app`, `bot` и `ingest`, запускает/проверяет Postgres и
Qdrant, выполняет `pg_restore`, а Qdrant и файловое состояние восстанавливает
через одноразовые `app` containers. Поэтому restore не зависит от остановленного
контейнера `app`. Затем runner запускает `app` и автоматически проверяет:

- row counts таблиц Postgres против `postgres.json`;
- наличие `corporate_rag` и point count против `qdrant.json`;
- точное совпадение corpus и RAG docstore/manifest с backup;
- `GET /health/ready` без LLM/provider-запроса.

`bot` и `proxy` намеренно не запускаются автоматически: поднимите их после
успешной проверки результата. Перед реальным restore доступна non-destructive
проверка плана; backup при этом всё равно должен быть полным и валидным:

```powershell
python scripts/deployment_backup.py restore `
  --project-name diploma-restore-plan `
  --env-file .env.production `
  --from backups/<UTC timestamp> `
  --dry-run
```

### Isolated restore rehearsal

Никогда не репетируйте destructive restore на production project. Overlay
`docker-compose.restore.yml` отключает host ports, заменяет corpus bind mount на
named volume и оставляет все state volumes в namespace уникального project name.
Пример безопасной репетиции для локального backup:

```powershell
$restoreProject = "diploma-restore-20260925-01"
$backup = "backups/<UTC timestamp>"

python scripts/deployment_backup.py restore `
  --project-name $restoreProject `
  --env-file .env `
  --compose-file docker-compose.yml `
  --compose-file docker-compose.restore.yml `
  --from $backup `
  --confirm-restore

docker compose `
  --project-name $restoreProject `
  --env-file .env `
  -f docker-compose.yml `
  -f docker-compose.restore.yml `
  ps -a
```

Убедитесь, что `$restoreProject` — новый уникальный идентификатор и не равен
`diploma`. После проверки удаляйте только этот namespace и его volumes:

```powershell
docker compose `
  --project-name $restoreProject `
  --env-file .env `
  -f docker-compose.yml `
  -f docker-compose.restore.yml `
  down --volumes --remove-orphans
```

Команда `down --volumes` допустима только после повторной визуальной проверки
точного временного project name. Volumes текущего project `diploma` в этом
workflow не используются и не удаляются.

## 9. Dev workflow

Базовый Compose остаётся локальным стендом:

```powershell
docker compose up -d --build
```

В dev app доступен на `localhost:8000`, Qdrant/Phoenix — только через
`127.0.0.1`, Postgres не публикуется. Production всегда запускайте с обоими
Compose-файлами и `.env.production`.
