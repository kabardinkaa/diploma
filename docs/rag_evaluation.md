# RAG Evaluation & Monitoring

## Configuration

Evaluation выполнен локальным LLM-as-judge из-за требования zero-cost
evaluation. Платные API и `openrouter/free` не использовались.

- Runtime: LM Studio.
- Endpoint: `http://127.0.0.1:1234/v1`.
- Judge: `local-qwen-judge`, модель `Qwen3.5-35B-A3B Q4_K_M`.
- LM Studio: context 16384, parallel 3, full GPU offload.
- Judge parameters: temperature 0.01, top-p 0.1, max tokens 4096.
- Eval answer generation: тот же `local-qwen-judge`.
- Evaluator embeddings: локальная `intfloat/multilingual-e5-base`.
- Production model не изменён: `openai/gpt-5.4-mini`.
- RAG embeddings: `intfloat/multilingual-e5-base`, 768 dimensions.
- Production collection: `corporate_rag`.
- Baseline: recursive chunking, `512/64`, top-K `10`, reranker disabled.
- Refusal threshold: `0.82`.

## Golden Dataset

RAGAS `TestsetGenerator` обработал 64 документа и создал 42 raw-пары:
`tests/eval/golden_dataset_raw.csv`. После построчной редакторской проверки
оставлены 30 русскоязычных corpus-grounded записей. Удалены или заменены 12
кандидатов: дубль, транслит/английский, искусственные multi-hop комбинации,
общие вопросы и неоднозначные references.

- Raw SHA256:
  `047f278d2400e07f0c9820bf9c16a5b967be8afd939a1b973e6ec74e3f807187`.
- Curated SHA256:
  `424facc7e935d7c88152e0ef3010f79515fe6c796680e707d515f3f6ea8fea92`.
- Audit: [golden_dataset_review.md](golden_dataset_review.md).

## Baseline

Artifact: `2026-07-28_220831_baseline.csv`, 30 rows, failed 0.

| Metric | Value |
| --- | ---: |
| Faithfulness | 0.970 |
| Answer relevancy | 0.916 |
| Context precision | 0.948 |
| Context recall | 1.000 |
| Has citation | 1.000 |
| Avg latency ms | 2825.202 |

Latency p50/p95: `2481.550 / 5079.242 ms`.

## Experiment A - Chunking

The isolated collection `corporate_rag_eval_chunk256` contains 128 points.
Only chunk size/index changed: recursive `256/64`, top-K `10`, reranker off.

| Variant | Faithfulness | Answer relevancy | Context precision | Context recall | Has citation | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 512 | 0.970 | 0.916 | 0.948 | 1.000 | 1.000 | 2825.202 |
| chunk 256 | 0.977 | 0.915 | 0.954 | 1.000 | 1.000 | 2222.274 |

Chunk-256 improved faithfulness, precision and latency, while answer relevancy
decreased by `0.001`. Artifact:
`2026-07-28_221727_chunk_256.csv`, 30 rows, failed 0.

## Experiment B - Top-K

The variant uses the unchanged `corporate_rag` index and changes only top-K.

| Variant | Faithfulness | Answer relevancy | Context precision | Context recall | Has citation | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top-K 10 | 0.970 | 0.916 | 0.948 | 1.000 | 1.000 | 2825.202 |
| top-K 5 | 0.980 | 0.917 | 0.964 | 1.000 | 1.000 | 2759.664 |

Top-K 5 improved all measured quality metrics without reindexing. Artifact:
`2026-07-28_222525_top_k_5.csv`, 30 rows, failed 0.

## Final Configuration

Беру вариант `top_k_5`, потому что он проходит все mandatory gates и показывает
лучшие Faithfulness (`0.980`), AnswerRelevancy (`0.917`) и ContextPrecision
(`0.964`) при ContextRecall и has_citation `1.000`. Exact evaluated config:
recursive `512/64`, top-K `5`, reranker disabled, collection `corporate_rag`.

По прямому ограничению текущей задачи production RAG не менялся: deployed
default остаётся top-K `10`. Top-K `5` зафиксирован как измеренный winner и
рекомендация для отдельного production change.

## Failure Analysis

Все пять кейсов взяты из winning CSV и отсортированы по Faithfulness ASC.
Высокий ContextRecall при более низком Faithfulness указывает на возможную
generation/prompt проблему; это гипотеза, а не доказанная причина.

1. **Обязательное обновление Windows**: Faithfulness `0.800`, relevancy `0.976`,
   precision/recall/citation `1.000`. Контексты описывают питание, сохранение
   документов и запрет выключения; ответ пересказывает их списком. Гипотеза:
   judge дробит составную инструкцию строже; сократить ответ до формулировок
   исходного chunk.
2. **Компьютер не получил сетевой адрес**: Faithfulness `0.875`, relevancy
   `0.860`, остальные метрики `1.000`. Ответ добавляет общую категорию заявки к
   точной диагностике кабеля/адаптера. Гипотеза: убрать boilerplate, не
   относящийся к главному вопросу.
3. **Новое обращение в поддержку**: Faithfulness `0.875`, relevancy `0.914`,
   остальные метрики `1.000`. Ответ объединяет сведения из нескольких
   retrieved chunks. Гипотеза: ограничить генерацию первым наиболее точным
   источником и не расширять список ожидаемым результатом без необходимости.
4. **Нет входящих звонков**: Faithfulness `0.909`, relevancy `0.889`, остальные
   метрики `1.000`. Точная проверка очереди смешана с общей политикой секретов.
   Гипотеза: отделить диагностику от универсального ticket footer.
5. **Ошибка подключения к API**: Faithfulness `0.933`, relevancy `0.889`,
   остальные метрики `1.000`. Ответ добавляет общие требования к скриншотам и
   начинает лишнее примечание. Гипотеза: завершать ответ после HTTP code,
   method и correlation id.

## Phoenix Tracing

- UI: `http://localhost:6006`, HTTP 200 и визуальная проверка выполнены.
- OTLP: gRPC `http://phoenix:4317`.
- OpenAI и LlamaIndex используют один tracer provider.
- Manual spans: `rag.query`, `rag.retrieve`, `rag.generate`.
- Docker smoke: 24 queries, 24 traces, 210 spans.
- Retrieval/LLM/embedding spans: `72 / 42 / 72`.
- Summary: `tests/eval/results/trace_summary.json`.

## Quality Gates

- Faithfulness > 0.7: **PASS**, winner `0.980`.
- Answer relevancy > 0.7: **PASS**, winner `0.917`.
- Has citation > 0.95: **PASS**, winner `1.000`.

## Known Limitations

- Local judge может отличаться по качеству оценки от коммерческого Claude/GPT
  judge; абсолютные значения нельзя напрямую сравнивать с другим judge.
- Для полностью бесплатного прогона eval answers также генерировались локальной
  Qwen. Production OpenRouter model не вызывалась и остаётся неизменной.
- RAGAS judge metrics недетерминированы даже при temperature `0.01`.
- Corpus учебный и содержит повторяющиеся инструкции в разных форматах.
- Phoenix HallucinationEvaluator остаётся optional и не реализован.

## Improvement Plan

1. Повторить evaluation тем же local judge на расширенном real-world наборе.
2. Упростить generation prompt, чтобы исключить generic ticket boilerplate.
3. Отдельным production change проверить top-K `5` на пользовательском трафике.
4. Сравнить локального judge с другим локальным Qwen checkpoint на фиксированном
   поднаборе, не смешивая результаты одного A/B запуска.

## Commands

```powershell
$lms = "$env:USERPROFILE\.lmstudio\bin\lms.exe"
& $lms server start
& $lms load "qwen/qwen3.5-35b-a3b" --gpu max --context-length 16384 --parallel 3 --identifier local-qwen-judge -y

pip install ".[eval,tracing]"
python scripts/verify_eval.py
python scripts/generate_testset.py --size 40
python scripts/run_eval.py --label local_smoke --limit 1
python scripts/run_eval.py --label baseline
python scripts/build_eval_index.py
python scripts/run_eval.py --label chunk_256 --collection corporate_rag_eval_chunk256 --top-k 10
python scripts/run_eval.py --label top_k_5 --collection corporate_rag --top-k 5
```
