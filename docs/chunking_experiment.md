# Chunking & Retrieval Optimization

## Corpus

Эксперимент использует неизмененный учебный корпус Б5.3:
`data/rag-block-03/`.

- Documents: 10.
- Embedding model: `intfloat/multilingual-e5-base`.
- Dimension: 768.
- Vectors: L2-normalized.
- Qdrant distance: COSINE.
- Qdrant server: `v1.18.0`.

Document IDs:

```text
01_vpn.md
02_crm.md
03_email.md
04_passwords_sso_mfa.md
05_internal_access.md
06_workplace.md
07_network_wifi.md
08_headset_microphone.md
09_support_ticket.md
10_office_plants.txt
```

`10_office_plants.txt` присутствует в индексе как шум, но не используется как
relevant document ни для одного domain-вопроса.

## Golden dataset

- Path: `tests/eval/retrieval_dataset.json`.
- Questions: 24.
- Single-document questions: 13.
- Multi-document questions: 11.
- Questions with three relevant documents: 3.
- SHA256:
  `f06eb067b41cd27ddd9fe8a6e25eaf948356cbed62e95ddccdcfc4715a227eae`.

Dataset и labels были сформированы и проверены по исходным файлам до первого
retrieval experiment. После просмотра метрик labels не менялись. Метрики
считаются на уровне уникальных `doc_id`; повторные chunks одного документа не
увеличивают Hit Rate или Recall.

## Strategies

**Fixed:** `TokenTextSplitter`, baseline `chunk_size=512`,
`chunk_overlap=64`.

**Recursive:** `SentenceSplitter`, `paragraph_separator="\n\n"`,
детерминированный regex-tokenizer русских предложений, baseline `512/64`.

**Semantic:** `SemanticSplitterNodeParser`, `buffer_size=1`,
`breakpoint_percentile_threshold=95`, та же E5-модель.

Для стратегий созданы отдельные коллекции `docs_fixed`, `docs_recursive` и
`docs_semantic`. Перед каждым ingestion целевая collection удалялась и
создавалась заново. `points_count` после ingestion совпал с `total_chunks`.
Коллекции `documents`, `rag_block_03` и `rag_block_03_baremetal` не изменялись.

## Chunk statistics

Один tokenizer LlamaIndex использовался для статистики всех стратегий.

| Strategy | Documents | Total chunks | Avg chunks/doc | Min tokens | Max tokens | Median tokens | Avg chunk tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 10 | 10 | 1.000 | 133 | 303 | 253.5 | 241.40 |
| recursive | 10 | 10 | 1.000 | 133 | 303 | 253.5 | 241.40 |
| semantic | 10 | 20 | 2.000 | 25 | 223 | 109.5 | 117.75 |

Корпус состоит из коротких инструкций, поэтому fixed и recursive не делят
документы при размере 512. Semantic режет каждый документ в среднем на две
части и удваивает индекс.

## Retrieval quality

Перед latency measurement выполнялся один warmup query. В latency входят
query embedding и Qdrant retrieval; ingestion, загрузка моделей и создание
collection не входят.

| Strategy | Hit Rate@5 | MRR@10 | Recall@10 | Avg chunk tokens | Avg retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed | 1.000 | 1.000 | 1.000 | 241.40 | 4.46 |
| recursive | 1.000 | 1.000 | 1.000 | 241.40 | 4.32 |
| semantic | 1.000 | 1.000 | 1.000 | 117.75 | 4.09 |

P95 latency: fixed `4.88 ms`, recursive `4.95 ms`, semantic `4.61 ms`.

## Best strategy

Все три стратегии получили одинаковые quality metrics. Разница средней
latency между semantic и recursive составила только `0.23 ms`, что меньше
заданного tie tolerance `1 ms` и не является убедительным преимуществом.
Semantic при этом создал 20 points вместо 10.

Выбрана **recursive**: она сохраняет полные связные инструкции, имеет вдвое
меньший индекс относительно semantic и не уступает по Hit/MRR/Recall.
Fixed также дал полное качество, но был немного медленнее в этом прогоне и
не сохраняет смысловые границы на более длинных документах.

## Reranker

Live model: `BAAI/bge-reranker-v2-m3`, локальный `CrossEncoder`.
Dense retrieval передавал top-20 candidates; reranker возвращал top-10.
Raw cosine и cross-encoder scores сохраняются раздельно.

| Configuration | Hit Rate@5 | MRR@10 | Recall@10 | Avg ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| recursive, no reranker | 1.000 | 1.000 | 1.000 | 3.94 | 4.36 |
| recursive + BGE reranker | 1.000 | 1.000 | 1.000 | 1366.70 | 1488.31 |

Reranker изменял порядок непервых кандидатов, но не улучшил ни одну quality
metric: dense retrieval уже ставил relevant document первым для всех 24
вопросов. Средний overhead составил `1362.76 ms`, поэтому reranker в финальной
конфигурации выключен.

## Parameter tuning

Каждая строка меняет ровно один параметр относительно baseline. Для каждого
варианта использовалась отдельная временная collection, которая удалялась
после измерения.

| Experiment | Strategy | chunk_size | overlap | candidate top_k | Points | Hit@5 | MRR@10 | Recall@10 | Avg ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | recursive | 512 | 64 | 10 | 10 | 1.000 | 1.000 | 1.000 | 4.45 |
| smaller chunks | recursive | 256 | 64 | 10 | 16 | 1.000 | 0.979 | 1.000 | 4.25 |
| larger chunks | recursive | 1024 | 64 | 10 | 10 | 1.000 | 1.000 | 1.000 | 4.02 |
| less overlap | recursive | 512 | 32 | 10 | 10 | 1.000 | 1.000 | 1.000 | 3.99 |
| wider retrieval | recursive | 512 | 64 | 20 | 10 | 1.000 | 1.000 | 1.000 | 3.95 |

Smaller chunks ухудшили MRR: для вопроса о дополнении существующего обращения
`05_internal_access.md` оказался перед `09_support_ticket.md`. Остальные
варианты фактически создают те же 10 chunks на этом коротком корпусе, поэтому
не демонстрируют реального quality improvement. Небольшая разница latency
является шумом короткого локального прогона.

Baseline `512/64/top_k=10` оставлен как минимальная конфигурация без ухудшения
качества и без бессмысленного расширения candidate pool.

## Error analysis

На выбранной конфигурации нет полных misses: Hit@5, MRR@10 и Recall@10 равны
1.0. Ниже пять диагностических near-miss и перестановок, которые остаются
полезными для будущего более сложного корпуса.

1. **Smaller chunks, дополнение обращения.** Expected:
   `09_support_ticket.md`; top: `05_internal_access.md`,
   `09_support_ticket.md`, `04_passwords_sso_mfa.md`. Гипотеза: чанк 256
   отделил правило про комментарий от контекста обращения, а слово «созданное»
   усилило документ про заявку.
2. **Пароль, почта, SSO и системы.** Expected:
   `03_email.md`, `04_passwords_sso_mfa.md`, `05_internal_access.md`; dense
   ranks: 2, 1, 4. Гипотеза: VPN-документ на rank 3 содержит близкие термины
   про внутренние системы и сессию.
3. **Проблема у коллег и тикет.** Expected: `07_network_wifi.md`,
   `09_support_ticket.md`; dense ranks: 1 и 5. BGE поднял их на ranks 2 и 1.
   Гипотеза: dense embedding сильнее выделил сетевую часть составного вопроса.
4. **Wi-Fi и доменная запись.** Expected: `07_network_wifi.md`,
   `04_passwords_sso_mfa.md`; dense ranks: 1 и 2, после BGE: 1 и 4.
   Гипотеза: cross-encoder переоценил документы про VPN и доступ из-за
   корпоративной аутентификации.
5. **Неожиданный MFA и безопасный тикет.** Expected:
   `04_passwords_sso_mfa.md`, `09_support_ticket.md`; dense ranks: 1 и 3,
   после BGE: 1 и 2. Гипотеза: reranker лучше связал запрет передачи секретов
   с безопасным созданием обращения.

Эти наблюдения являются гипотезами по порядку выдачи, а не доказанными
причинами. Для подтверждения нужен более длинный и менее separable corpus.

## Final configuration

Выбираю стратегию **recursive** с:

```text
chunk_size=512
chunk_overlap=64
retrieval_top_k=10
reranker_enabled=false
reranker_model=BAAI/bge-reranker-v2-m3
rerank_top_n=10
```

Финальный Hit Rate@5: `1.000`. Конфигурация выбрана потому, что дает максимальные
Hit/MRR/Recall, не удваивает индекс и избегает reranker overhead без потери
качества.

`RAG_SIMILARITY_TOP_K=3` остается отдельным production generation limit Б5.3.
`RAG_RETRIEVAL_TOP_K=10` фиксирует измеренный retrieval/evaluation candidate
limit и не меняет существующий `/rag/query` автоматически.

## Reindexing note

Любое изменение chunking strategy, `chunk_size` или `chunk_overlap` требует
переиндексации соответствующей vector collection. Нельзя подключать новую
конфигурацию к старым points и считать результаты сопоставимыми.

Экспериментальные `docs_*` не назначаются production collections. Если
production RAG перейдет на измеренную recursive factory, `rag_block_03` нужно
явно переиндексировать, а не молча переиспользовать.

## Raw results

Machine-readable результаты находятся в
`tests/eval/chunking_results.json`. Файл содержит timestamp, dataset hash,
параметры, chunk stats, latency и retrieved IDs для каждого вопроса,
reranker before/after и все tuning experiments. Embeddings в JSON не
сохраняются.

## Commands

```powershell
docker compose up -d qdrant
docker compose ps
python scripts/chunking_experiment.py
python -m compileall app bot scripts
pytest -q
git diff --check
docker compose config --quiet
```
