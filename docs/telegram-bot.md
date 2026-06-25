# Telegram-бот как тонкий клиент

## Что реализовано

В проект добавлен Telegram-бот на aiogram 3.

Бот работает как тонкий клиент к backend chat-сервису из Б4.1:

- не знает про LLM;
- не хранит историю локально;
- создаёт чат через backend;
- отправляет сообщения в backend;
- получает ответ через SSE;
- очищает историю через backend;
- поддерживает FSM-сценарий `/ask`.

## Архитектура

Пользователь пишет в Telegram-бота.  
Бот обрабатывает сообщение через aiogram handlers.  
Handlers используют `BackendClient`.  
`BackendClient` вызывает backend:

- `POST /chats`
- `POST /chats/{chat_id}/messages`
- `DELETE /chats/{chat_id}/messages`

Backend хранит историю, собирает контекст и обращается к LLM.

## Структура

bot/
  __main__.py
  config.py
  states.py
  handlers/
    commands.py
    fsm.py
    text.py
  keyboards/
    inline.py
  services/
    backend_client.py

## Настройки

Настройки бота описаны в `bot/config.py`.

Используются переменные окружения:

BOT_TOKEN=
BACKEND_URL=http://127.0.0.1:8000
BOT_ADMIN_IDS=[]

`BOT_TOKEN` хранится только в `.env`.

Файл `.env` не коммитится в Git.

## BackendClient

`BackendClient` находится в `bot/services/backend_client.py`.

Он реализует методы:

- `get_or_create_chat(owner_external_id, interface)`
- `send_message(chat_id, content)`
- `clear_messages(chat_id)`

`get_or_create_chat` создаёт чат в backend через `POST /chats`.

`send_message` отправляет сообщение через `POST /chats/{chat_id}/messages` и парсит SSE-ответ.

`clear_messages` очищает историю через `DELETE /chats/{chat_id}/messages`.

## Команды

Реализованы команды:

- `/start` — создаёт чат в backend и показывает приветствие;
- `/help` — показывает список команд;
- `/clear` — очищает историю через backend;
- `/cancel` — сбрасывает активный FSM-сценарий.

## Текстовый handler

Обычные текстовые сообщения отправляются в backend.

Бот не хранит историю локально. Повторный вопрос работает за счёт backend-истории.

Проверка:

Пользователь: Привет, меня зовут Диана  
Бот: Привет, Диана! Как я могу вам помочь?

Пользователь: Как меня зовут?  
Бот: Диана.

## FSM-сценарий /ask

Реализован сценарий `/ask`.

Сценарий:

1. Пользователь пишет `/ask`.
2. Бот показывает inline-клавиатуру с темами.
3. Пользователь выбирает тему.
4. FSM сохраняет тему и переходит в состояние `waiting_for_question`.
5. Пользователь пишет вопрос.
6. Бот собирает prompt вида: `Тема: <topic>. Вопрос: <text>`.
7. Prompt отправляется в backend как обычное сообщение.
8. Ответ приходит стримом.
9. FSM очищается.

Темы выбраны из домена дипломного проекта:

- Доступы
- Рабочее место
- Корпоративное ПО
- Ошибки и сбои
- Эскалации

## Запуск

Сначала запускается backend:

.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

Потом запускается бот:

.\.venv\Scripts\python.exe -m bot

## Проверка

Проверены сценарии:

- `/start`
- `/help`
- обычное текстовое сообщение
- память через backend
- `/ask`
- `/cancel`
- `/clear`

После `/clear` бот больше не использует старый контекст.

## Тесты

Добавлены тесты:

- `tests/bot/test_backend_client.py`
- `tests/bot/test_fsm.py`

Проверяется:

- `get_or_create_chat` возвращает UUID и использует cache;
- `send_message` корректно парсит SSE;
- `send_message` сохраняет пробелы и multiline-чанки;
- `clear_messages` отправляет DELETE;
- FSM `/ask` после выбора темы переходит в `waiting_for_question`;
- отмена темы очищает state.

Результат проверки:

- `tests/bot`: 6 passed
- общий прогон: 25 passed

## Результат

Telegram-бот реализован как тонкий клиент к backend chat-сервису:

- backend хранит историю;
- бот не знает про LLM;
- бот не хранит историю локально;
- обычные сообщения уходят в backend;
- ответы приходят через SSE;
- `/clear` чистит историю через backend;
- `/ask` реализован через FSM;
- тесты проходят.