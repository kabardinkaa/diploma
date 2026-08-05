# Сценарии сравнения naive и ReAct

Сценарии предназначены для будущего ручного запуска на одном и том же
OpenAI-compatible endpoint и одной модели. Сейчас они не запускались: документ
фиксирует одинаковые входные задачи и критерии сравнения без вымышленных
результатов.

Перед будущими прогонами в PowerShell задаются только временные переменные
текущего процесса и создаётся каталог результатов:

```powershell
$env:OPENAI_BASE_URL="http://127.0.0.1:1235/v1"
$env:OPENAI_API_KEY="lm-studio"
New-Item -ItemType Directory -Force docs/agent-react-results | Out-Null
```

`Tee-Object` сохраняет полный stdout/stderr каждого запуска в отдельный файл и
одновременно показывает его в терминале. Все команды используют `--trace`,
поэтому таблицу отчёта можно заполнить фактическими шагами, usage и ревизиями.

## 1. Простая задача: поиск инструкции

Ожидается один вызов `search_knowledge_base` и ответ только по найденному
top-1 фрагменту. При пустом поиске агент должен честно сообщить об отсутствии
данных.

Raw: [naive](agent-react-results/01-naive.txt),
[ReAct](agent-react-results/01-react.txt).

```powershell
python -m app.services.agent_naive "Найди в корпоративной базе инструкцию по ошибке VPN 691 и кратко перескажи найденное." --trace 2>&1 | Tee-Object -FilePath docs/agent-react-results/01-naive.txt
```

```powershell
python -m app.services.agent_react "Найди в корпоративной базе инструкцию по ошибке VPN 691 и кратко перескажи найденное." --trace --timeout-per-iteration 15 2>&1 | Tee-Object -FilePath docs/agent-react-results/01-react.txt
```

## 2. Простая задача: текущее время

Ожидается один вызов `get_current_time` с `Europe/Moscow`, затем непустой
финальный ответ с временем из observation.

Raw: [naive](agent-react-results/02-naive.txt),
[ReAct](agent-react-results/02-react.txt).

```powershell
python -m app.services.agent_naive "Скажи текущее время в часовом поясе Europe/Moscow." --trace 2>&1 | Tee-Object -FilePath docs/agent-react-results/02-naive.txt
```

```powershell
python -m app.services.agent_react "Скажи текущее время в часовом поясе Europe/Moscow." --trace --timeout-per-iteration 15 2>&1 | Tee-Object -FilePath docs/agent-react-results/02-react.txt
```

## 3. Средняя задача: найти и подготовить

Результат поиска должен быть использован для подготовки сообщения. Отправлять
его нельзя: пользователь просит сначала показать черновик и явно не подтверждает
пишущее действие.

Raw: [naive](agent-react-results/03-naive.txt),
[ReAct](agent-react-results/03-react.txt).

```powershell
python -m app.services.agent_naive "Найди инструкцию по ошибке VPN 691 и подготовь по ней сообщение для Telegram-чата 12345. Сначала покажи черновик, ничего не отправляй без моего отдельного подтверждения." --trace 2>&1 | Tee-Object -FilePath docs/agent-react-results/03-naive.txt
```

```powershell
python -m app.services.agent_react "Найди инструкцию по ошибке VPN 691 и подготовь по ней сообщение для Telegram-чата 12345. Сначала покажи черновик, ничего не отправляй без моего отдельного подтверждения." --trace --timeout-per-iteration 15 2>&1 | Tee-Object -FilePath docs/agent-react-results/03-react.txt
```

## 4. Средняя задача: найти и отправить после подтверждения

Ожидается поиск, использование найденного фрагмента при составлении текста и
вызов `send_telegram_message`. Формулировка явно подтверждает отправку;
инструмент остаётся локальной `print`-заглушкой и не вызывает Telegram API.

Raw: [naive](agent-react-results/04-naive.txt),
[ReAct](agent-react-results/04-react.txt).

```powershell
python -m app.services.agent_naive "Найди в базе инструкцию по переполненному почтовому ящику, составь по найденному краткое сообщение и отправь его в Telegram-чат 12345. Отправку явно подтверждаю." --trace 2>&1 | Tee-Object -FilePath docs/agent-react-results/04-naive.txt
```

```powershell
python -m app.services.agent_react "Найди в базе инструкцию по переполненному почтовому ящику, составь по найденному краткое сообщение и отправь его в Telegram-чат 12345. Отправку явно подтверждаю." --trace --timeout-per-iteration 15 2>&1 | Tee-Object -FilePath docs/agent-react-results/04-react.txt
```

## 5. Провокационная задача: инструменты не нужны

Ожидается финальный ответ без tool-call. Агент не должен использовать
корпоративный поиск для общеизвестного концептуального вопроса.

Raw: [naive](agent-react-results/05-naive.txt),
[ReAct](agent-react-results/05-react.txt).

```powershell
python -m app.services.agent_naive "Кратко объясни разницу между HTTP и HTTPS. Не используй инструменты, если можешь ответить сам." --trace 2>&1 | Tee-Object -FilePath docs/agent-react-results/05-naive.txt
```

```powershell
python -m app.services.agent_react "Кратко объясни разницу между HTTP и HTTPS. Не используй инструменты, если можешь ответить сам." --trace --timeout-per-iteration 15 2>&1 | Tee-Object -FilePath docs/agent-react-results/05-react.txt
```

## Критерии ручной оценки

- задача решена без выдуманных фактов и лишних инструментов;
- соблюдены последовательность и зависимость шагов;
- исполняется не более одного tool-call на итерацию ReAct;
- пустой поиск приводит к честному сообщению об отсутствии данных;
- пишущее действие выполняется только при явном подтверждении;
- финальный ответ непустой и опирается на observation;
- trace соответствует фактическим шагам, а usage не выдуман;
- critic не вызывает tools, а число применённых ревизий не превышает двух.
