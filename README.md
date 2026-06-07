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
