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
python scripts/validate_production_config.py --env-file .env.production
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

## 5. Persistent data

Compose использует named volumes:

- `pg-data` — Postgres chat/checkpoint state;
- `qdrant-storage` — vector collections;
- `rag-state` — ingestion docstore;
- `phoenix-data` — локальные traces;
- `caddy-data` — TLS certificates/account;
- `caddy-config` — runtime state Caddy.

Production corpus монтируется из `./data`. Embedding weights/cache находятся в
`./.cache/embeddings` и могут быть повторно загружены/построены.

## 6. Backup

Во время backup не запускайте upload/reindex/ingest. `pg_dump` и Qdrant snapshot
создаются штатными online-механизмами; текущие volumes не удаляются.

```powershell
python scripts/deployment_backup.py backup --env-file .env.production
```

Архив появляется в `backups/<UTC timestamp>/`; `backups/` игнорируется Git.
Внутри:

- `postgres.dump` — custom-format PostgreSQL dump;
- `qdrant.snapshot` — snapshot collection `corporate_rag`;
- `rag-state/` — persistent docstore;
- `corpus.zip` — production corpus `data/`;
- `config/` — Compose/Caddy/constraints и env template без секретов;
- `manifest.json` — состав, исключения и SHA-256 checksums.

Не сохраняются реальные `.env`/`.env.production`, provider keys, embedding cache,
Phoenix traces, Docker images и logs. Production secret file храните отдельно в
зашифрованном secret manager/backup.

План команд без записи backup:

```powershell
python scripts/deployment_backup.py backup --env-file .env.production --dry-run
```

## 7. Restore

Restore перезаписывает состояние Postgres/Qdrant и требует явный флаг. Сначала
проверьте архив и остановите writers, не удаляя volumes:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production stop app bot ingest
python scripts/deployment_backup.py restore `
  --env-file .env.production `
  --from backups/<UTC timestamp> `
  --confirm-restore
docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production up -d app bot proxy
```

Corpus по умолчанию не перезаписывается. Для сознательного восстановления
файлов в `data/` добавьте `--restore-corpus` после отдельной копии текущего
каталога. Перед реальным restore доступна non-destructive проверка плана:

```powershell
python scripts/deployment_backup.py restore `
  --env-file .env.production `
  --from backups/<UTC timestamp> `
  --dry-run
```

Restore не тестируется на текущих production volumes. Для полного rehearsal
используйте отдельный Compose project name и отдельные test volumes.

## 8. Dev workflow

Базовый Compose остаётся локальным стендом:

```powershell
docker compose up -d --build
```

В dev app доступен на `localhost:8000`, Qdrant/Phoenix — только через
`127.0.0.1`, Postgres не публикуется. Production всегда запускайте с обоими
Compose-файлами и `.env.production`.
