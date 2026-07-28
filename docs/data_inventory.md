# Data inventory

Корпус является учебным предметным корпусом внутренней техподдержки
контактного центра. Он не содержит реальных внутренних документов,
персональных данных или секретов компании.

Инвентаризация выполнена для ingestion root `data/`. Учтены только расширения,
которые реально обрабатывает `IngestionService`; `support_kb.json`, TXT и
`*.failed` не входят в индекс.

| Format | Files | Size |
| --- | ---: | ---: |
| Markdown | 25 | 26,368 bytes |
| HTML | 16 | 20,735 bytes |
| DOCX | 16 | 594,793 bytes |
| PDF | 16 | 790,046 bytes |
| **Total** | **73** | **1,431,942 bytes** |

64 документа в `data/corporate/` детерминированно построены скриптом
`scripts/build_corporate_corpus.py` из активных записей
`data/support_kb.json`: по четыре документа на каждую из 16 категорий и по 16
файлов каждого формата. Ещё 9 Markdown-файлов сохранены от предыдущих RAG
домашних работ в `data/rag-block-03/`.

Категории:

`access`, `applications`, `audio`, `browser`, `crm`, `email`, `hardware`,
`network_wifi`, `passwords`, `printing`, `rag-block-03`, `software_updates`,
`sso_mfa`, `telephony`, `tickets`, `vpn`, `workplace`.

Поддерживаемые readers:

| Extension | Reader |
| --- | --- |
| `.pdf` | `PyMuPDFReader` |
| `.docx` | `DocxReader` |
| `.html`, `.htm` | `HTMLTagReader` |
| `.md` | `MarkdownReader` |

Фактическая parser-проверка всех 73 файлов завершилась без ошибок и создала
73 LlamaIndex Document объекта.
