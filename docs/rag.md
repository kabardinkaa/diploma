# RAG architecture - LlamaIndex

## Dependencies

Фактически установленные и проверенные версии:

| Package | Version |
| --- | --- |
| `llama-index` | `0.14.23` |
| `llama-index-core` | `0.14.23` |
| `llama-index-vector-stores-qdrant` | `0.10.2` |
| `llama-index-readers-file` | `0.6.0` |
| `llama-index-embeddings-huggingface` | `0.7.0` |
| `llama-index-llms-openai` | `0.7.10` |
| `llama-index-llms-openai-like` | `0.7.2` |

Стандартный `llama_index.llms.openai.OpenAI` был сначала проверен на текущей
модели `openrouter/free` и завершился ошибкой `Unknown model 'openrouter/free'`
при вычислении context window. Поэтому для настроенного OpenRouter/OpenAI-
compatible `base_url` используется `OpenAILike` с явным context window. Для
прямого OpenAI без custom `base_url` остается стандартный `OpenAI`.

## Architecture

RAG разделен на три фазы:

```text
offline ingestion -> online retrieval -> generation
```

На старте приложения `RAGService.build()` либо индексирует пустую коллекцию,
либо подключается к готовой через `VectorStoreIndex.from_vector_store()`.
Retriever и QueryEngine создаются один раз. Endpoint выполняет retrieval,
проверяет score и вызывает `aquery()` только при достаточной релевантности.

## Corpus

`data/rag-block-03/` содержит ровно 10 учебных документов, собранных как
подмножество и адаптация синтетического корпуса Б5.2:

1. VPN и внутренние ресурсы.
2. CRM.
3. Корпоративная почта.
4. Пароли, SSO и MFA.
5. Доступ к внутренним системам.
6. Рабочее место.
7. Сеть и Wi-Fi.
8. Гарнитура и микрофон.
9. Создание обращения.
10. Уход за офисными растениями, намеренно нерелевантный документ.

В корпусе нет HR-процедур и правил оформления отпуска.

## Collections

| Collection | Purpose | Real points count |
| --- | --- | ---: |
| `documents` | Б5.2, плоский payload qdrant-client | 128, не изменялась |
| `rag_block_03` | LlamaIndex Node payload с `_node_content` | 10 |
| `rag_block_03_baremetal` | Ручной payload `text/source/chunk_index` | 10 |

Коллекция `documents` не используется LlamaIndex: ее плоский payload не
содержит сериализованный Node, необходимый для `source_nodes`, metadata и
цитирования. Раздельные RAG-коллекции также не смешивают разные форматы
payload и позволяют честно сравнивать orchestration и ручную реализацию.

Повторный build дал `10 -> 10` точек для обеих RAG-коллекций. LlamaIndex
подключился через `from_vector_store`, bare-metal не выполнял повторный upsert.

## Configuration

| Parameter | Value |
| --- | --- |
| Embedding model | `intfloat/multilingual-e5-base` |
| Dimension | 768 |
| Distance | COSINE |
| Chunk size | 512 |
| Chunk overlap | 64 |
| Similarity top-K | 3 |
| Minimum score | 0.82 |
| LLM provider | OpenRouter / OpenAI-compatible |
| Live model | `openrouter/free` |
| Temperature | 0.0 |

LlamaIndex `HuggingFaceEmbedding` настроен с `query_instruction="query: "`,
`text_instruction="passage: "` и `normalize=True`. Реальный sanity-прогон:
dimension `768`, query norm `1.0`, VPN score `0.906308`, нерелевантный документ
про растения `0.754444`.

Все значения, пути и имена коллекций читаются из `app/core/config.py` и env.
Секреты в документации и репозитории отсутствуют.

## LlamaIndex pipeline

```text
SimpleDirectoryReader
-> SentenceSplitter(512, 64)
-> QdrantVectorStore(rag_block_03)
-> VectorStoreIndex
-> Retriever(top_k=3)
-> QueryEngine(compact)
-> OpenAILike/OpenAI LLM
```

Metadata каждого Document содержит `file_name` и `source`. Готовая коллекция
дополнительно проверяется на dimension, COSINE и наличие `_node_content`.

## Bare-metal pipeline

```text
read sorted files
-> deterministic word chunks
-> EmbeddingService.embed_documents()
-> UUIDv5 upsert into rag_block_03_baremetal
-> EmbeddingService.embed_query()
-> qdrant-client query_points()
-> manual context/system/question prompt
-> AsyncOpenAI chat completion
```

В generation передается полный retrieved payload; в API preview источника
ограничен первыми 300 символами. Старый deprecated `search()` не используется.

## LlamaIndex vs bare-metal

LOC посчитан AST-скриптом по методам ingestion/retrieval/generation классов,
без imports, `__init__`, CLI и cleanup. В LlamaIndex-число вошли дополнительные
проверки совместимости готовой коллекции.

| Criterion | LlamaIndex | Bare-metal |
| --- | --- | --- |
| Строк кода ingestion + query, без imports | 178 | 110 |
| Поддержка форматов из коробки | Reader для TXT/MD/PDF/DOCX и других форматов | Сейчас TXT/MD |
| Что нужно для PDF/DOCX | Пакет readers-file и доступные parser dependencies | Явный parser и нормализация metadata |
| Batch ingestion / async | Framework batching, async retriever/query engine | Ручной batch embeddings/upsert и `to_thread` |
| Debug top_score / sources | `source_nodes` с Node metadata | Прозрачный `ScoredPoint.payload` |
| Подмена re-ranker / chunker | Компоненты QueryEngine/transformations | Ручная замена и обновление orchestration |

LlamaIndex остается основной реализацией диплома: он дает стандартные Node,
reader, retriever, QueryEngine и расширяемые компоненты. В этой небольшой
задаче его код длиннее из-за явной production-проверки коллекции, а не из-за
самого pipeline. Bare-metal остается учебной reference implementation: он
проще для отладки конкретного Qdrant-запроса, но потребует больше собственного
кода при добавлении форматов и компонентов.

## Fallback

Стартовый порог `0.5` был реально проверен и оказался слишком низким для
нормализованных multilingual E5 embeddings. Top scores четырех вопросов с
ответом лежат в диапазоне `0.869-0.909`, а вопрос про отпуск получил `0.782`.
Порог откалиброван до `0.82`, между этими группами.

При пустом retrieval или score ниже `RAG_MIN_SCORE` LLM не вызывается. Обе
реализации возвращают:

```text
В базе знаний не нашлось информации для ответа на этот вопрос.
```

Low-score candidates остаются в `sources` для диагностики.

## Прогон 5 вопросов

Фактический прогон: 3 good / 1 medium / 1 out-of-base.

| Type | Question | LlamaIndex top-1 / score | Bare-metal top-1 / score | Result |
| --- | --- | --- | --- | --- |
| good | Что проверить, если VPN подключился, но внутренние ресурсы не открываются? | `01_vpn.md` / 0.909 | `01_vpn.md` / 0.909 | Релевантно, ответ про профиль, DNS и переподключение; fallback no |
| good | Как разблокировать учетную запись в CRM? | `02_crm.md` / 0.907 | `02_crm.md` / 0.908 | Retrieval релевантен; bare-metal ответ корректен, финальный LlamaIndex вызов получил технический ответ free-provider; fallback no |
| good | Что делать, если в гарнитуре не работает микрофон? | `08_headset_microphone.md` / 0.888 | `08_headset_microphone.md` / 0.889 | Retrieval релевантен; bare-metal ответ корректен, финальный LlamaIndex вызов получил технический ответ free-provider; fallback no |
| medium | После смены пароля перестал работать вход сразу в несколько внутренних систем. Что проверить? | `04_passwords_sso_mfa.md` / 0.869 | `04_passwords_sso_mfa.md` / 0.871 | Релевантно; top-3 также содержит почту, поэтому возможен синтез; fallback no |
| out-of-base | Как оформить ежегодный отпуск на две недели? | `05_internal_access.md` / 0.782 | `10_office_plants.txt` / 0.782 | Нерелевантные weak candidates, fallback yes |

Top-1 совпал в обеих реализациях для всех четырех вопросов с ответом. Для
out-of-base top-1 различается, но обе версии отсекают запрос по одному score.
Гипотеза: общие слова «оформить» и «две недели» дают слабую семантическую
близость к заявкам и инструкциям, но score заметно ниже релевантной группы.

`openrouter/free` нестабилен: несколько live good-запросов вернули техническую
строку `User Safety: safe`, а другие запуски с теми же retrieval results дали
корректные ответы. Финальный smoke получил эту строку для CRM и микрофона;
VPN и medium были сгенерированы нормально. Это ограничение бесплатного
маршрутизатора, а не retrieval; для production следует закрепить конкретную
поддерживаемую модель.

## API

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Как разблокировать учетную запись в CRM?\"}"
```

Реальная проверка FastAPI: `/health` вернул `ok`, `/docs` - HTTP 200,
good query - HTTP 200, `top_score=0.909`, 3 sources; out-of-base - HTTP 200,
`top_score=0.782`, 3 debug sources и честный fallback.

## Commands

```powershell
pip install -r requirements.txt
docker compose up -d qdrant
docker compose ps
python scripts/rag_embedding_sanity.py
python -m app.services.rag
python -m app.services.rag_baremetal
python scripts/rag_smoke.py
python -m uvicorn app.main:app --reload --port 8000
pytest -q
python -m compileall app bot scripts
git diff --check
docker compose config --quiet
```
