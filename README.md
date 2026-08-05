# Дипломный проект: ИИ-ассистент техподдержки

## Тема проекта

ИИ-ассистент для техподдержки сотрудников контактного центра.

Ассистент помогает сотрудникам с типовыми вопросами: восстановление доступов, ошибки в корпоративных системах, VPN, почта, CRM, рабочее место и создание обращений в техподдержку.

## Что реализовано в блоке 3.1

В рамках домашнего задания реализован полный цикл Function Calling:

1. Пользователь отправляет запрос.
2. LLM принимает решение: ответить самостоятельно или вызвать инструмент.
3. Если нужен инструмент, модель возвращает `tool_call` с именем функции и аргументами.
4. Python-код выполняет функцию-обработчик.
5. Результат функции возвращается обратно в модель.
6. Модель формирует финальный ответ пользователю.

## Реализованный инструмент

### `search_knowledge_base(query, product)`

Инструмент ищет релевантную инструкцию во внутренней базе знаний техподдержки.

Инструмент используется, когда пользователь спрашивает:

- как восстановить доступ;
- что делать с ошибкой;
- как оформить заявку;
- как решить проблему с VPN, CRM, почтой, рабочим местом или корпоративной системой.

Инструмент не используется для приветствий, small talk и общих вопросов, где можно ответить без поиска по базе знаний.

## Структура проекта

```text
app/
  prompts/
    system_v1.j2
    loader.py
    tools/
      search_knowledge_base.md

  tools/
    schemas.py
    handlers.py
    knowledge_base.json

  llm/
    client.py

examples/
  __init__.py
  run_tool_call.py

logs/
  tool_calls.jsonl

.env.example
.gitignore
requirements.txt
README.md
```

## Где что находится

- `app/prompts/system_v1.j2` — system prompt ассистента.
- `app/prompts/tools/search_knowledge_base.md` — описание инструмента для модели.
- `app/prompts/loader.py` — загрузчик prompt-файлов.
- `app/tools/schemas.py` — JSON Schema инструмента.
- `app/tools/handlers.py` — функция-обработчик инструмента.
- `app/tools/knowledge_base.json` — локальная база знаний техподдержки.
- `app/llm/client.py` — полный цикл Function Calling.
- `examples/run_tool_call.py` — запуск трёх тестовых сценариев.
- `logs/tool_calls.jsonl` — логирование шагов выполнения.

## Установка

Создание виртуального окружения:

```bash
python -m venv .venv
```

Активация виртуального окружения на Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Настройка переменных окружения

Нужно создать файл `.env` на основе `.env.example`.

Пример для OpenRouter:

```env
OPENAI_API_KEY=your_openrouter_key
OPENAI_MODEL=openrouter/free
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Файл `.env` не добавляется в Git, потому что содержит API-ключ.

## Запуск

Запуск тестовых сценариев:

```bash
python -m examples.run_tool_call
```

## Тестовые запросы

### Тест 1 — запрос требует tool

Запрос:

```text
Как сотруднику восстановить доступ к VPN?
```

Результат:

Модель вызвала инструмент `search_knowledge_base` с аргументами:

```json
{
  "query": "восстановить доступ к VPN",
  "product": "Корпоративный VPN"
}
```

После выполнения функции модель вернула финальный ответ с инструкцией: проверить интернет-соединение, перезапустить VPN-клиент, убедиться, что учётная запись не заблокирована, а при сохранении проблемы создать обращение в техподдержку.

Вывод: сценарий отработал корректно, потому что вопрос требовал поиска инструкции в базе знаний.

### Тест 2 — запрос не требует tool

Запрос:

```text
Привет, кто ты?
```

Результат:

Модель не вызвала инструмент и ответила самостоятельно:

```text
Я — ИИ-ассистент внутренней техподдержки, помогаю сотрудникам с доступами, ошибками и запросами в корпоративных системах.
```

Вывод: сценарий отработал корректно, потому что для приветствия и объяснения роли ассистента поиск в базе знаний не нужен.

### Тест 3 — пограничный случай

Запрос:

```text
У меня что-то не открывается, что делать?
```

Результат:

Модель не стала сразу вызывать инструмент и задала уточняющий вопрос:

```text
Пожалуйста, уточните, какую конкретно программу или сервис вы пытаетесь открыть и какое сообщение об ошибке появляется?
```

Вывод: это самый интересный результат. Вопрос связан с техподдержкой, но в нём недостаточно информации: неизвестна система, ошибка и контекст. Поэтому модель корректно не стала гадать и сначала запросила уточнение.

## Логирование

Каждый запуск пишет события в файл:

```text
logs/tool_calls.jsonl
```

Логируются:

- пользовательский ввод;
- выбранная модель;
- имя tool;
- аргументы tool;
- результат функции;
- финальный ответ;
- количество токенов.

Примеры событий:

```text
input
tool_call
final_with_tool
final_without_tool
```

## Наблюдение по самому неожиданному результату

Самым показательным оказался третий тест: `У меня что-то не открывается, что делать?`

Ожидалось, что модель может вызвать поиск по базе знаний, потому что запрос похож на обращение в техподдержку. Но модель решила сначала уточнить, какая именно система не открывается. Это правильное поведение для ассистента техподдержки: без названия системы и текста ошибки можно дать слишком общий или неверный ответ.

## Итог

В проекте реализован базовый механизм Function Calling для будущего дипломного ИИ-ассистента техподдержки.

Текущая версия умеет:

- хранить prompt-файлы отдельно от кода;
- описывать инструмент через JSON Schema;
- вызывать Python-функцию по решению модели;
- читать данные из локальной базы знаний;
- возвращать результат инструмента обратно в модель;
- логировать полный цикл работы;
- обрабатывать сценарии с tool, без tool и с неоднозначным запросом.

## Домашнее задание 3.3 — Асинхронная обработка запросов к ИИ

В рамках домашнего задания синхронный LLM-клиент был дополнен асинхронной реализацией.

### Что реализовано

* Добавлен класс `AsyncLLMClient` в файле `app/llm/async_client.py`.
* Используется `AsyncOpenAI`, а не синхронный `OpenAI`.
* Реализован метод `complete(prompt)` для одного асинхронного запроса.
* Реализован метод `batch_chat(prompts)` через `asyncio.gather(..., return_exceptions=True)`.
* Ограничение конкурентности сделано через `asyncio.Semaphore`.
* `Semaphore` хранится как атрибут экземпляра `self._sem`.
* Реализован метод `stream_chat(prompt)` как async-генератор.
* Добавлен FastAPI endpoint `POST /chat/stream`, который отдаёт ответ через SSE.
* Добавлен скрипт `scripts/benchmark.py` для сравнения sync и async режимов.
* Результаты бенчмарка сохраняются в `scripts/benchmark_results.md`.

### Установка зависимостей

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Настройка `.env`

Для OpenRouter:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openrouter/free
```

Для OpenAI:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

### Запуск FastAPI

```powershell
uvicorn app.api:app --reload
```

Проверка health-check:

```text
http://127.0.0.1:8000/health
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Проверка обычного chat endpoint

```powershell
$body = @{
  prompt = "Что такое event loop? Ответь одним коротким абзацем."
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Проверка streaming через SSE

```powershell
'{"prompt":"Кратко объясни, зачем нужен async для LLM-запросов."}' | Set-Content request.json -Encoding UTF8

curl.exe -N -X POST "http://127.0.0.1:8000/chat/stream" -H "Content-Type: application/json" --data-binary "@request.json"
```

В ответ приходят SSE-события:

```text
event: token
data: Async позволяет...

event: token
data: отправлять несколько LLM-запросов...

event: done
data: [DONE]
```

### Бенчмарк sync vs async

Для учебного прогона использовалось 5 промптов и модель `openrouter/free`.

Команда запуска:

```powershell
$env:BENCHMARK_PROMPT_COUNT="5"
python scripts\benchmark.py
```

Результат:

| Режим            | Concurrency / Semaphore | Кол-во запросов | Успешно | Ошибки | Время, сек |
| ---------------- | ----------------------: | --------------: | ------: | -----: | ---------: |
| sync sequential  |                       1 |               5 |       5 |      0 |      66.98 |
| async batch_chat |                       1 |               5 |       5 |      0 |      70.79 |
| async batch_chat |                       5 |               5 |       4 |      1 |      19.26 |
| async batch_chat |                      10 |               5 |       5 |      0 |      20.65 |

### Вывод по бенчмарку

Асинхронная обработка показала ускорение относительно последовательного sync-режима. Лучший стабильный результат в этом прогоне показал режим `concurrency=10`: все 5 запросов завершились успешно, а общее время составило 20.65 секунды против 66.98 секунды в sync-режиме.

Режим `concurrency=5` был немного быстрее по времени, но один запрос получил ошибку `429 Rate limit exceeded` от бесплатного провайдера OpenRouter. Это показывает, что внешний rate limit провайдера влияет на результаты бенчмарка.

При этом `batch_chat` не уронил весь батч: ошибка была возвращена как отдельный результат благодаря `asyncio.gather(..., return_exceptions=True)`. Это подтверждает корректную обработку частичных ошибок.

Для учебного проекта выбран concurrency=10 как лучший стабильный результат этого прогона: все 5 запросов завершились успешно, время составило 20.65 сек против 66.98 сек в sync-режиме. В production лимит конкурентности нужно выбирать с учётом реальных rate limits провайдера по формуле:

```text
Semaphore = floor(RPM / 60 * 0.8)
```

### Файлы, добавленные в рамках задания

* `app/llm/async_client.py` — асинхронный LLM-клиент.
* `app/api.py` — FastAPI endpoints `/chat`, `/chat/stream`, `/health`.
* `scripts/benchmark.py` — скрипт бенчмарка.
* `scripts/benchmark_results.md` — сохранённые результаты бенчмарка.
* `requirements.txt` — обновлённые зависимости.

## Домашнее задание 3.4 — FastAPI-сервис для LLM

В рамках домашнего задания проект был расширен до FastAPI-сервиса с production-подходом к структуре, конфигурации, зависимостям и обработке запросов к LLM.

### Что реализовано

* Создана структура проекта:

  * `app/main.py`
  * `app/core/config.py`
  * `app/core/exceptions.py`
  * `app/deps/providers.py`
  * `app/routers/chat.py`
  * `app/routers/models.py`
  * `app/routers/health.py`
  * `app/services/llm.py`
  * `app/schemas/chat.py`
  * `app/schemas/models.py`
* Реализован запуск сервиса через `uvicorn app.main:app --reload`.
* Добавлен конфиг через `pydantic-settings`.
* API-ключи хранятся через `SecretStr`, реальные ключи не добавляются в git.
* Добавлен `lifespan` для инициализации и закрытия `AsyncOpenAI`.
* Добавлена Dependency Injection через `Annotated[..., Depends(...)]`.
* Реализован сервисный слой `LLMService`.
* Добавлены endpoints:

  * `GET /health`
  * `GET /models`
  * `POST /chat`
  * `POST /chat/stream`
  * `POST /chat/batch`
* Добавлен кеш ответов для `POST /chat`.
* Для локального MVP используется in-memory cache, чтобы сервис работал без обязательного запуска Redis.
* Добавлен streaming через `StreamingResponse` и формат SSE `data: ...\n\n`.
* В конце streaming-ответа отправляется `data: [DONE]`.
* Добавлен middleware для `request_id`, latency и логирования запроса.
* Добавлен CORS middleware.
* Добавлены обработчики ошибок:

  * доменные LLM-ошибки;
  * ошибки валидации `RequestValidationError`.

### Запуск проекта

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Проверка health endpoint

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Ожидаемый ответ:

```json
{
  "status": "ok"
}
```

### Проверка models endpoint

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/models"
```

Endpoint возвращает список доступных моделей и их параметры.

### Проверка chat endpoint

```powershell
$body = @{
  messages = @(
    @{
      role = "system"
      content = "Отвечай строго на русском языке. Одним коротким предложением."
    },
    @{
      role = "user"
      content = "Что такое FastAPI? Ответь одной фразой."
    }
  )
  temperature = 0
  max_tokens = 80
} | ConvertTo-Json -Depth 5 -Compress

$response1 = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

$response1 | Select-Object model, finish_reason, cached, usage
```

Повторный такой же запрос возвращает `cached: true`.

### Проверка streaming endpoint

```powershell
@'
{
  "messages": [
    {
      "role": "system",
      "content": "Отвечай строго на русском языке. Коротко."
    },
    {
      "role": "user",
      "content": "Считай до пяти."
    }
  ],
  "temperature": 0,
  "max_tokens": 80
}
'@ | Set-Content stream_request.json -Encoding UTF8

curl.exe -N -X POST "http://127.0.0.1:8000/chat/stream" -H "Content-Type: application/json" --data-binary "@stream_request.json"
```

В ответе приходят SSE-события:

```text
data: 1

data: 2

data: 3

data: {"usage":{"prompt_tokens":89,"completion_tokens":36,"total_tokens":125}}

data: [DONE]
```

### Проверка batch endpoint

```powershell
$batchBody = @{
  requests = @(
    @{
      messages = @(
        @{
          role = "system"
          content = "Отвечай строго одной короткой фразой на русском."
        },
        @{
          role = "user"
          content = "Что такое FastAPI?"
        }
      )
      temperature = 0
      max_tokens = 40
    },
    @{
      messages = @(
        @{
          role = "system"
          content = "Отвечай строго одной короткой фразой на русском."
        },
        @{
          role = "user"
          content = "Что такое DI?"
        }
      )
      temperature = 0
      max_tokens = 40
    }
  )
} | ConvertTo-Json -Depth 10 -Compress

$batchResponse = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat/batch" `
  -Method POST `
  -ContentType "application/json" `
  -Body $batchBody

$batchResponse.results | Select-Object model, usage, finish_reason, cached | ConvertTo-Json -Depth 10
```

### Swagger

Swagger UI доступен по адресу:

```text
http://127.0.0.1:8000/docs
```

В Swagger отображаются endpoints:

* `GET /health`
* `GET /models`
* `POST /chat`
* `POST /chat/stream`
* `POST /chat/batch`

### Итог

В результате выполнения ДЗ получился FastAPI-сервис для LLM, который использует async-клиент, отдельный сервисный слой, DI, lifespan, конфиг через окружение, middleware, кеш, Swagger и streaming endpoint. Эта структура будет использоваться как основа для дальнейшего дипломного проекта.

## Мультимодальный Telegram-бот

В проект добавлена поддержка мультимодальных сообщений через единый backend endpoint:

`POST /chats/{chat_id}/messages`

Endpoint принимает `multipart/form-data`:

- `content` — текст сообщения;
- `media` — необязательный файл.

Поддерживаемые типы:

- изображения — преобразуются в `image_url` content-part;
- голосовые сообщения и аудио — распознаются локально через `faster-whisper`;
- PDF — текст извлекается через `pypdf`;
- DOCX — текст извлекается через `python-docx`.

Telegram-бот поддерживает:

- обычный текст;
- фото;
- голосовые сообщения;
- аудиофайлы;
- PDF;
- DOCX.

Ответ backend передаётся через SSE-события:

```json
{"type": "token", "delta": "..."}
{"type": "done"}
В Telegram ответ отображается постепенно через редактирование сообщения.

Проверка

pytest -q

Ожидаемый результат:

30 passed

## Homework B4.4 Production Operations

### Environment

Copy `.env.example` to `.env` and fill secrets locally. Do not commit `.env`.

Required production-operation settings:

```env
ADMIN_TOKEN=change-me-admin-token
INTERNAL_TOKEN=change-me-internal-token
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/diploma
CHAT_REPOSITORY=json
BOT_ADMIN_IDS=123456789
MODERATION_OPENAI_ENABLED=false
```

`ADMIN_TOKEN` protects `/chats/admin/*`. `INTERNAL_TOKEN` protects internal broadcast worker endpoints. `BOT_ADMIN_IDS` is a comma-separated Telegram user id list.

### Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
python -m bot
```

### Docker Compose

```bash
docker compose up --build
```

Compose starts `app`, `bot`, `postgres`, `redis`, and `phoenix`. Postgres data is persisted in the `pg-data` volume.

### Implemented B4.4 Features

- Backend moderation in `app/moderation` with local keyword/regex rules from `moderation_keywords.yaml`.
- Admin endpoints under `/chats/admin` protected by `X-Admin-Token`.
- JSON and Postgres chat repositories selected via `CHAT_REPOSITORY=json|postgres`.
- Feedback endpoint: `POST /chats/{chat_id}/messages/{message_id}/feedback`.
- Telegram feedback buttons with callback data `fb:up:<message_id>` and `fb:down:<message_id>`.
- Telegram admin commands: `/stats`, `/users`, `/broadcast <text>`.
- Broadcast queue consumed by the bot through `INTERNAL_TOKEN` protected endpoints.
- Handoff command `/operator` sets `handoff_status=paused_for_human`.
- System prompt A/B split with deterministic owner-based routing and `prompt_id` saved on assistant messages.

### Checks

```bash
python -m compileall app bot
pytest -q
docker compose config
```

## Домашнее задание 5.1 - Эмбеддинги и семантический поиск

Для русскоязычной базы знаний выбрана локальная модель
`intfloat/multilingual-e5-base` с 768-мерными нормализованными векторами.
Обоснование выбора и оценка стоимости находятся в
`docs/embedding-model-choice.md`.

- Mini-benchmark: `tests/eval/mini_benchmark.json`.
- Сервис: `app/services/embeddings.py`.
- E5-префиксы `query:` и `passage:` добавляются сервисом автоматически.
- Размер батча по умолчанию: 32.
- Постоянный кеш: `.cache/embeddings` или путь из `EMBEDDING_CACHE_DIR`.
- Ключ кеша учитывает модель, тип embedding и исходный текст.

Запуск smoke-теста:

```powershell
python scripts/embedding_smoke.py
```

Smoke-тест выводит модель, размерность и норму вектора, время первого и
повторного вызова, а также cosine similarity для релевантного и нерелевантного
документов. Первый live-прогон дал размерность 768, норму 1.0, score 0.876590
для релевантного документа против 0.787760 для нерелевантного. Холодный вызов
с загрузкой модели занял 183.7152 с, повторный вызов из кеша - 0.0002 с;
релевантный документ оказался выше нерелевантного во всех 5 из 5 пар.
Первый запуск скачивает веса модели и зависит от скорости сети.

## Домашнее задание 5.2 - Векторные базы данных

В проект добавлен self-hosted Qdrant с production-коллекцией `documents`:

- `intfloat/multilingual-e5-base`, 768 измерений;
- метрика COSINE и HNSW `m=16`, `ef_construct=100`;
- 128 уникальных учебных документов техподдержки;
- async-обёртка `app/services/vector_store.py`;
- идемпотентная загрузка `scripts/load_to_qdrant.py`;
- эксперименты `scripts/vector_store_smoke.py`;
- отчёт `docs/vector_store.md`.

Настройте `QDRANT_API_KEY` в `.env`, затем запустите:

```powershell
docker compose up -d qdrant
docker compose ps
python scripts/load_to_qdrant.py
python scripts/load_to_qdrant.py
python scripts/vector_store_smoke.py
```

Qdrant dashboard доступен по адресу `http://localhost:6333/dashboard`.
Для проверки без Docker можно выполнить:

```powershell
python scripts/vector_store_smoke.py --local
```

## Домашнее задание 5.3 - RAG с LlamaIndex

Добавлены основная LlamaIndex RAG-реализация и совместимая bare-metal версия на
одном учебном корпусе из `data/rag-block-03/`. Используются отдельные коллекции
`rag_block_03` и `rag_block_03_baremetal`; коллекция Б5.2 `documents` не
изменяется.

- API: `POST /rag/query`
- LlamaIndex CLI: `python -m app.services.rag`
- Bare-metal CLI: `python -m app.services.rag_baremetal`
- Сравнительный smoke: `python scripts/rag_smoke.py`
- Архитектура и фактические результаты: `docs/rag.md`

## Домашнее задание 5.4 - Chunking и оптимизация retrieval

Retrieval benchmark сравнивает fixed, recursive и semantic chunking на одном
golden dataset из 24 вопросов:

- dataset: `tests/eval/retrieval_dataset.json`;
- factories: `app/services/chunking.py`;
- metrics: `app/services/retrieval_eval.py`;
- local BGE reranker: `app/services/reranker.py`;
- runner: `scripts/chunking_experiment.py`;
- raw results: `tests/eval/chunking_results.json`;
- отчет: `docs/chunking_experiment.md`.

Финально выбраны recursive `512/64`, retrieval top-K `10`, reranker выключен:
качество `Hit@5=1.000`, `MRR@10=1.000`, `Recall@10=1.000`, а live BGE не
улучшил метрики и добавил около 1.36 секунды latency.

```powershell
python scripts/chunking_experiment.py
```
## Домашнее задание 5.5 - Корпоративный RAG-ассистент

Проект расширен до production-like RAG с 73 учебными multi-format документами
(Markdown, HTML, DOCX, PDF), отдельным offline `IngestionService`, persistent
docstore и incremental UPSERTS в Qdrant collection `corporate_rag`.

Online-контур использует retrieval top-5, optional reranker, dense score
guard без LLM-вызова, multi-turn condense, явные `[N]` citations и sources в
синхронном `/rag/query`, chat SSE и Telegram. Документы можно принять через
`POST /documents/upload` или переиндексировать через
`POST /documents/reindex`; Telegram остаётся thin client.

Подробности: [docs/rag.md](docs/rag.md) и
[docs/data_inventory.md](docs/data_inventory.md).

```powershell
docker compose up -d --build
python scripts/ingest.py data/
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Домашнее задание 5.6 - Оценка качества и мониторинг RAG

Добавлен zero-cost evaluation-контур: 42 raw TestsetGenerator-пары, curated
golden dataset из 30 записей, локальный Qwen3.5-35B-A3B judge через LM Studio,
локальные E5 embeddings и RAGAS 0.4 с пятью метриками. Два завершённых A/B
сценария сравнивают chunk size `512`/`256` и top-K `10`/`5`; каждый сохраняет
timestamped per-row CSV и aggregate JSON. Winner evaluation: top-K `5`.
По результатам Б5.6 этот вариант применён как финальная production-конфигурация.

Полный протокол, фактические метрики, failure analysis и точные команды:
[docs/rag_evaluation.md](docs/rag_evaluation.md).

```powershell
pip install ".[eval,tracing]"
python scripts/verify_eval.py
python scripts/run_eval.py --label baseline --dry-run
```

## Домашнее задание 6.1 - Наивный агентный цикл

Добавлен синхронный agent loop без LangChain и LangGraph. На каждой итерации
модель `gpt-5.4-mini` получает полную историю запуска и может вызвать один или
несколько инструментов из явного allowlist:

- `search_knowledge_base` использует production RAG и возвращает top-1 фрагмент;
- `get_current_time` получает локальное время через `zoneinfo` без сети;
- `send_telegram_message` является учебной `print`-заглушкой и не обращается к
  Telegram API.

Обычный запуск печатает финальный ответ или причину остановки:

```powershell
python -m app.services.agent_naive "Найди инструкцию по ошибке VPN 691"
```

Флаг `--trace` дополнительно выводит JSON-трассу с инструментами, аргументами,
результатами, токенами и длительностью каждого шага:

```powershell
python -m app.services.agent_naive "Проверь текущее время в Москве" --trace
```

Агент использует существующие настройки `OPENAI_API_KEY` или
`OPENROUTER_API_KEY`, соответствующий `OPENAI_BASE_URL` или
`OPENROUTER_BASE_URL`, а также `LLM_REQUEST_TIMEOUT` и `LLM_MAX_RETRIES`.
Модель зафиксирована заданием и не создаёт второй независимый контур настроек.
Пять подготовленных ручных сценариев приведены в
[docs/agent-naive-traces/README.md](docs/agent-naive-traces/README.md); реальные
платные прогоны и искусственные trace-файлы не выполнялись.

## Домашнее задание 6.2 - ReAct и critic

Рядом с неизменённым baseline `agent_naive` добавлен нативный ReAct-цикл на
`client.chat.completions.create`. `agent_react` исполняет ровно один tool-call
за итерацию, ведёт append-only историю и после каждого observation вызывает
отдельного critic с ответом `OK` или `REVISE`. Общий лимит применённых ревизий
за запуск равен двум.

```powershell
python -m app.services.agent_react "Найди инструкцию по ошибке VPN 691"
python -m app.services.agent_react "Найди инструкцию по ошибке VPN 691" --trace
```

Доступны флаги `--max-iterations` (8–20), `--timeout-per-iteration` (5–15
секунд), `--max-revisions` (0–2), `--model-main` и `--model-critic`. Один общий
дедлайн итерации охватывает actor LLM, исполнение инструмента и critic LLM;
LLM-запросы также получают оставшийся request timeout. На Windows LLM-вызовы
ограничены daemon-потоком и SDK timeout, а быстрые локальные tools исполняются
синхронно после проверки deadline. Поэтому пишущий tool не продолжит работу
после возврата `Timeout`. Произвольный долгий Python-tool нельзя безопасно
принудительно остановить внутри процесса; такой tool задержит возврат до своего
завершения и должен выноситься в отдельный worker/process.

Trace содержит инструмент, аргументы, сокращённый observation, задержку,
вердикт critic и usage каждой итерации. Итоговый usage суммирует токены actor и
critic; если OpenAI-compatible провайдер не вернул usage, записываются нули.
Схемы трёх инструментов вынесены отдельно и используют строгий allowlist.
`search_knowledge_base` возвращает top-1 фрагмент для следующего действия,
`get_current_time` возвращает ISO datetime, а `send_telegram_message` остаётся
локальной `print`-заглушкой и требует явного подтверждения.

Пять одинаковых прогонов для naive и ReAct, команды и ссылки на неизменённые raw
stdout описаны в [docs/agent-react-scenarios.md](docs/agent-react-scenarios.md).
Фактические метрики, средние значения и ограничения локального запуска собраны
в [docs/agent-react-report.md](docs/agent-react-report.md); искусственные
метрики не использовались.

## Домашнее задание 6.3 - LangGraph

ReAct orchestration перенесён в LangGraph 1.x двумя способами: custom
`StateGraph` с явными state/reducers/router/stop-краном и prebuilt-граф через
актуальный `langchain.agents.create_agent`. Оба используют один настроенный
`ChatOpenAI`, модель `gpt-5.4-mini`, temperature `0` и единый набор доменных
tools без дублирования бизнес-логики.

```powershell
python -m app.services.agent_graph custom "Скажи время в Europe/Moscow"
python -m app.services.agent_graph prebuilt "Скажи время в Europe/Moscow" --trace
python -m app.services.agent_graph custom --mermaid
python scripts/visualize_graph.py
```

`--thread-id` заранее сохраняет интерфейс для будущего checkpointer. Mermaid
генерируется локально в `docs/agent-graph-*.mmd` и `.md`, без сетевого renderer.
Benchmark-runner сравнивает baseline блока 6.2, custom и prebuilt на пяти
задачах с тремя повторами. Запуски выполняются последовательно, каждый raw
сохраняется сразу, а `gc.collect()` освобождает память между runs:

```powershell
python scripts/bench_agents.py --dry-run
python scripts/bench_agents.py --resume --task-id 2 --implementation custom --repeats 3 --react-timeout-per-iteration 15 --run-timeout 60
python scripts/bench_agents.py --plan-rerun-failed
python scripts/bench_agents.py --resume --rerun-failed --max-attempts 2 --task-id 4 --implementation custom --repeats 3 --react-timeout-per-iteration 15 --run-timeout 60
python scripts/bench_agents.py --aggregate-only
python scripts/visualize_graph.py
```

Финальный локальный benchmark содержит 45 канонических результатов: 40
технически успешных и 5 окончательных timeout ReAct. OpenRouter не
использовался. Raw и история attempts лежат в `docs/agent-graph-results/`, smoke
— в `docs/agent-graph-smoke/`, полный анализ — в
[docs/agent-graph-report.md](docs/agent-graph-report.md). Baseline-файлы
`agent_naive.py` и `agent_react.py` не изменены. `send_telegram_message`
остаётся локальной print-заглушкой и не обращается к Telegram API.
