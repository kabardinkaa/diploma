# Documentation map

Корневой [README](../README.md) — основная точка входа. Документы ниже разделены по назначению, чтобы historical/evaluation evidence не воспринималось как production runbook.

## Final product docs

- [architecture.md](architecture.md) — фактическая production topology, lifecycle и security boundaries.
- [rag.md](rag.md) — текущий `corporate_rag` ingestion/query contract.
- [chat.md](chat.md) — chat persistence, SSE и media flow.
- [agent-persistent-report.md](agent-persistent-report.md) — текущий persistent agent, roles и session isolation.
- [telegram-bot.md](telegram-bot.md) — текущий bot/backend contract.
- [security/README.md](security/README.md) — LLM security tests и сохранённые garak evidence.
- [observability/README.md](observability/README.md) — logging/tracing contract и screenshots.

## Operations / deployment

- [deployment.md](deployment.md) — TLS deployment, environment validation, health, data volumes, backup и restore.
- `docker-compose.yml` — local/demo stack.
- `docker-compose.prod.yml` — public overlay.
- `docker-compose.restore.yml` — isolated restore rehearsal overlay.

## Evaluation / evidence

Эти документы фиксируют измерения и эксперименты. Они не переопределяют production settings, указанные в README/`rag.md`.

- [rag_evaluation.md](rag_evaluation.md) — финальные RAG metrics и aggregate artifacts.
- [golden_dataset_review.md](golden_dataset_review.md) — review golden dataset.
- [data_inventory.md](data_inventory.md) — corpus inventory.
- [embedding-model-choice.md](embedding-model-choice.md) — выбор embedding model.
- [chunking_experiment.md](chunking_experiment.md) — chunking/reranking experiment.
- [vector_store.md](vector_store.md) — историческое исследование Qdrant collections.
- [multi-agent-report.md](multi-agent-report.md) — multi-agent comparison evidence.

Названия `documents`, `rag_block_03` и другие экспериментальные collections в evidence-файлах являются историческими и не используются вместо production `corporate_rag`.

## Course history / archive

[archive/README.md](archive/README.md) описывает сохранённые отчёты, traces, raw benchmark results и исходную course-by-course историю. Команды, test counts и архитектурные планы внутри архива относятся к моменту соответствующего учебного этапа и не должны использоваться для текущего запуска.
