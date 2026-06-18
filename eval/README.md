# Evaluation

Этот каталог содержит отдельный evaluation-слой для LLM-сервиса дипломного проекта.

Evaluation не является обычным pytest-тестом: быстрые unit-тесты запускаются часто и без сети, а evaluation делает реальные LLM-вызовы, занимает больше времени и может стоить денег.

## Состав каталога

- `golden_dataset.json` — golden dataset с вопросами, эталонными ожиданиями, категориями, сложностью и источниками кейсов.
- `run_evaluation.py` — CLI-прогон golden dataset через production-модель и judge-модель.
- `check_thresholds.py` — проверка результатов evaluation по минимальным порогам качества.
- `thresholds.yaml` — конфигурация порогов качества.
- `runs/` — сохранённые результаты evaluation-прогонов.

## Golden dataset

Файл `golden_dataset.json` содержит:

- `version` — версия набора данных;
- `items` — список кейсов.

Каждый item содержит:

- `id` — стабильный идентификатор кейса;
- `question` — вопрос пользователя;
- `expected_answer` — описание ожидаемого ответа;
- `expected_keywords` — ключевые слова и синонимы, которые желательно увидеть в ответе;
- `must_not_contain` — запрещённые фразы или небезопасные рекомендации;
- `category` — категория кейса;
- `difficulty` — сложность;
- `source` — источник кейса.

Набор нужен как контракт качества: при смене модели или промпта можно проверить, стало ли поведение ассистента лучше или хуже.

## Быстрые unit-тесты

Unit-тесты живут отдельно в `tests/` и запускаются без API-ключей и сетевых вызовов.

```bash
python -m pytest tests/test_pii.py tests/unit -m "not llm"
```

Эти тесты проверяют код вокруг LLM:

- сборку сообщений для промпта;
- экранирование пользовательского ввода;
- парсинг JSON из обычного текста и markdown-fence;
- обработку malformed JSON;
- парсинг tool_calls;
- Pydantic-валидацию;
- скрытие PII в repr;
- расчёт стоимости по usage;
- cache hit без вызова OpenAI-клиента.

## Evaluation-прогон

Evaluation запускается отдельно, вручную:

```bash
python eval/run_evaluation.py --golden eval/golden_dataset.json --judge gpt-5.2 --out eval/runs/2026-06-18.json
```

Для учебной среды можно указать доступную judge-модель:

```bash
python eval/run_evaluation.py --golden eval/golden_dataset.json --judge openrouter/free --out eval/runs/2026-06-18.json
```

Скрипт делает следующее:

1. Загружает `eval/golden_dataset.json`.
2. Для каждого кейса отправляет вопрос в production-модель с `temperature=0`.
3. Передаёт вопрос, эталон и ответ в judge-модель.
4. Judge оценивает ответ в стиле G-Eval: сначала reasoning, затем score.
5. Сохраняет результат в `eval/runs/<date>.json`.

## Формат результата

Файл результата содержит:

- `run_id`;
- `timestamp`;
- `model_under_test`;
- `judge_model`;
- `golden_version`;
- `items`;
- `aggregates`.

В каждом item сохраняются:

- `id`;
- `question`;
- `answer`;
- `scores`;
- `reasoning`;
- `explanation`.

Агрегаты:

- `relevance_avg`;
- `correctness_avg`;
- `completeness_avg`;
- `min_correctness`.

Пример проверки через jq:

```bash
jq ".aggregates.correctness_avg" eval/runs/2026-06-18.json
```

## Проверка порогов

Пороги лежат в `eval/thresholds.yaml`.

Запуск проверки:

```bash
python eval/check_thresholds.py eval/runs/2026-06-18.json
```

Если путь к run-файлу не передать, скрипт возьмёт последний JSON из `eval/runs/`:

```bash
python eval/check_thresholds.py
```

Минимальные учебные пороги:

- `correctness_avg >= 3.0`;
- `min_correctness >= 1.0`;
- `relevance_avg >= 3.0`;
- `completeness_avg >= 3.0`.

Для production-гейта пороги стоит поднять, например до `correctness_avg >= 4.0` и `min_correctness >= 2.0`.

## Почему eval отдельно от tests

`tests/` — быстрые проверки для разработки. Они должны быть зелёными на каждое сохранение и не должны зависеть от сети.

`eval/` — продуктовая проверка качества. Она запускается при смене модели, промпта или периодически, например раз в неделю. Такой прогон может быть медленным, дорогим и иногда падать не из-за ошибки кода, а из-за просадки качества модели.

@'

## Результат учебного прогона

Для учебного прогона использовались бесплатные OpenRouter-модели, поэтому качество может быть нестабильным: модель под капотом может меняться между запросами, а judge иногда попадает в rate-limit.

Пример сохранённого отчёта:

```bash
eval/runs/2026-06-18.json