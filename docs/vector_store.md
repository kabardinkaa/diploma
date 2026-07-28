# Vector Store - Qdrant

## Почему Qdrant

Qdrant выбран как self-hosted слой retrieval: он использует HNSW, поддерживает
типизированные payload filters, разворачивается одним Docker-сервисом и оставляет
прямой путь к hybrid dense+sparse search в следующих блоках диплома.

## Конфигурация

| Параметр | Значение |
|---|---|
| Embedding model | `intfloat/multilingual-e5-base` |
| Dimension | 768 |
| Production collection | `documents` |
| Distance | `COSINE` |
| Client | `AsyncQdrantClient` |

URL, API key, collection и dimension читаются из `app/core/config.py`. API key
хранится как optional `SecretStr`.

## Corpus

`data/support_kb.json` содержит 128 уникальных учебных документов внутренней
техподдержки в 16 категориях. Это не реальные корпоративные документы. Корпус
искусственно подготовлен для дипломного проекта и предметно расширяет пять
исходных статей про VPN, CRM, почту, рабочее место и обращения.

В корпусе есть 16 архивных версий и смесь дат от 2023 до июля 2026 года.
Payload каждой точки содержит `source`, `title`, `text`, `created_at`,
`category`, `department`, `is_archived` и `chunk_index`. Вектор в payload не
дублируется.

## Идемпотентность

ID точки вычисляется как UUIDv5 от `source + chunk_index` в фиксированном
namespace. Повторный запуск для того же business key перезаписывает точку, а не
создаёт новую.

In-memory smoke на реальных E5-векторах:

| Прогон | points_count |
|---|---:|
| Первый upsert | 128 |
| Повторный upsert | 128 |

## Метрика

Для эксперимента одни и те же нормализованные векторы загружались во временные
коллекции `documents_cosine` и `documents_dot`. После сравнения обе коллекции
удалялись.

| Query | COSINE top-5 | DOT top-5 | Same ranking |
|---|---|---|---|
| Не могу подключиться к рабочему VPN | `08e604de-8264-588c-8084-77b1ab78ae57`, `97c4e477-9cf5-5bcc-ae47-cfa6726768c8`, `ff3f566b-9a59-51bf-a051-5b423757ede2`, `0c1fbbea-ac6f-59d2-87a9-7ccb3687c73f`, `40059726-97e3-552e-8476-3cc8834bd3d9` | `08e604de-8264-588c-8084-77b1ab78ae57`, `97c4e477-9cf5-5bcc-ae47-cfa6726768c8`, `ff3f566b-9a59-51bf-a051-5b423757ede2`, `0c1fbbea-ac6f-59d2-87a9-7ccb3687c73f`, `40059726-97e3-552e-8476-3cc8834bd3d9` | yes |
| Как разблокировать учётную запись в CRM? | `3a86b9af-34c3-537d-990e-b6dff94de3b0`, `7870b3c9-9668-5f9c-8e44-1eedb0c1561d`, `2914dc48-55c2-5acf-8389-a2ac2ac50ea7`, `e6444b6f-e5a6-5e2b-992b-e9ae3ff6dcef`, `a5543a6d-de5c-5ba2-a9bf-739591bc0b99` | `3a86b9af-34c3-537d-990e-b6dff94de3b0`, `7870b3c9-9668-5f9c-8e44-1eedb0c1561d`, `2914dc48-55c2-5acf-8389-a2ac2ac50ea7`, `e6444b6f-e5a6-5e2b-992b-e9ae3ff6dcef`, `a5543a6d-de5c-5ba2-a9bf-739591bc0b99` | yes |
| Письма из корпоративной почты не отправляются | `6420d1f5-d5e7-5c15-9490-48bc1e8ada1a`, `2914dc48-55c2-5acf-8389-a2ac2ac50ea7`, `7d0d276e-525d-5260-825b-ffd13b347731`, `97f22866-cdc6-5d5f-9a9e-3646c9e4d488`, `77ec2b3b-ddd8-5f19-862f-0c654d98cc92` | `6420d1f5-d5e7-5c15-9490-48bc1e8ada1a`, `2914dc48-55c2-5acf-8389-a2ac2ac50ea7`, `7d0d276e-525d-5260-825b-ffd13b347731`, `97f22866-cdc6-5d5f-9a9e-3646c9e4d488`, `77ec2b3b-ddd8-5f19-862f-0c654d98cc92` | yes |
| В гарнитуре не работает микрофон | `32796f7b-72b8-5d71-bb2f-bb68f55bc72c`, `5b5d1a31-9d43-51f5-be9f-c0f516fcd4a2`, `bf03705e-bfb2-54a0-b01b-43bcf21b6d51`, `c97b030d-9db3-5577-bf85-980ad48f5e0b`, `bd3a528f-3816-5e7b-9ed0-d5a6f687ad58` | `32796f7b-72b8-5d71-bb2f-bb68f55bc72c`, `5b5d1a31-9d43-51f5-be9f-c0f516fcd4a2`, `bf03705e-bfb2-54a0-b01b-43bcf21b6d51`, `c97b030d-9db3-5577-bf85-980ad48f5e0b`, `bd3a528f-3816-5e7b-9ed0-d5a6f687ad58` | yes |
| Как запросить доступ к внутренней системе? | `e4f3cca2-e2e8-580b-8ccf-a744f8a521d4`, `f51049e7-43db-5b75-a8a0-57dc2e51be02`, `97c4e477-9cf5-5bcc-ae47-cfa6726768c8`, `40059726-97e3-552e-8476-3cc8834bd3d9`, `08e604de-8264-588c-8084-77b1ab78ae57` | `e4f3cca2-e2e8-580b-8ccf-a744f8a521d4`, `f51049e7-43db-5b75-a8a0-57dc2e51be02`, `97c4e477-9cf5-5bcc-ae47-cfa6726768c8`, `40059726-97e3-552e-8476-3cc8834bd3d9`, `08e604de-8264-588c-8084-77b1ab78ae57` | yes |

Итог: 5 из 5 ранжирований совпали. В production остаётся COSINE, потому что эта
метрика явно выражает similarity-контракт Sentence Transformers/E5. На
L2-нормализованных векторах DOT даёт тот же порядок, а доказанной причины менять
production contract нет.

## Metadata filters

Результаты ниже получены реальным запуском `scripts/vector_store_smoke.py
--local` на 128 E5-векторах. Local Qdrant применяет фильтры, но не строит
payload-индексы; серверный Qdrant создаёт все пять индексов в
`ensure_collection()`.

### MATCH: category = vpn

```python
Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="vpn"),
        )
    ]
)
```

| Rank | Source | Title | Score |
|---:|---|---|---:|
| 1 | `support/vpn/003` | VPN не подключается из домашней сети | 0.872973 |
| 2 | `support/vpn/006` | Частые разрывы VPN-соединения | 0.853997 |
| 3 | `support/vpn/005` | VPN подключён, но внутренние сайты недоступны | 0.849746 |

### DATETIME RANGE: последние 30 дней

```python
cutoff = datetime.now(timezone.utc) - timedelta(days=30)
Filter(
    must=[
        FieldCondition(
            key="created_at",
            range=DatetimeRange(gte=cutoff),
        )
    ]
)
```

Запрос: «Как импортировать vpn-office-2024.conf после обновления VPN-клиента?».
Фактический cutoff прогона: `2026-06-28T17:11:36.780724+00:00`.

Без фильтра:

| Rank | Source | Title | Score |
|---:|---|---|---:|
| 1 | `support/vpn/001` | Архивная настройка VPN-клиента 2024 | 0.917241 |
| 2 | `support/vpn/002` | Настройка VPN после обновления клиента | 0.874026 |
| 3 | `support/passwords/004` | Пароль истёк во время отпуска | 0.836987 |

С фильтром:

| Rank | Source | Title | Score |
|---:|---|---|---:|
| 1 | `support/vpn/002` | Настройка VPN после обновления клиента | 0.874026 |
| 2 | `support/passwords/002` | Забытый пароль Windows вне офиса | 0.827971 |
| 3 | `support/vpn/003` | VPN не подключается из домашней сети | 0.826615 |

Фильтр исключил точную, но архивную инструкцию 2024 года и поднял актуальную.

### COMPOSITE: department + must_not archived

```python
Filter(
    must=[
        FieldCondition(
            key="department",
            match=MatchValue(value="contact_center"),
        )
    ],
    must_not=[
        FieldCondition(
            key="is_archived",
            match=MatchValue(value=True),
        )
    ],
)
```

| Rank | Source | Title | Score |
|---:|---|---|---:|
| 1 | `support/tickets/001` | Как создать обращение в техподдержку | 0.860162 |
| 2 | `support/tickets/004` | Как дополнить уже созданное обращение | 0.841474 |
| 3 | `support/email/001` | Корпоративная почта не отправляет письма | 0.837153 |

## HNSW

Коллекция создаётся с `m=16` и `ef_construct=100`. Это консервативные параметры
Qdrant для небольшого корпуса; измерений recall/latency, оправдывающих более
дорогую настройку, пока нет.

Payload indexes:

- `source`: KEYWORD
- `created_at`: DATETIME
- `category`: KEYWORD
- `department`: KEYWORD
- `is_archived`: BOOL

## Проверка

```powershell
docker compose up -d qdrant
docker compose ps
python scripts/load_to_qdrant.py
python scripts/load_to_qdrant.py
python scripts/vector_store_smoke.py
```

Fallback без Docker:

```powershell
python scripts/vector_store_smoke.py --local
```

В среде выполнения Docker daemon был недоступен: отсутствовал named pipe
`dockerDesktopLinuxEngine`. Поэтому dashboard и серверные counts не проверены.
Фактические search/filter/COSINE-vs-DOT результаты выше получены в local
in-memory mode `qdrant-client`; серверные payload indexes покрыты unit-тестами
через async fake client.
