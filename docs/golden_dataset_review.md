# Golden Dataset Review

- Automated raw target: 40 rows through RAGAS `TestsetGenerator`.
- Full local generation result: 42 rows from 64 parsed documents.
- Runtime: LM Studio, `local-qwen-judge` (`Qwen3.5-35B-A3B Q4_K_M`),
  OpenAI-compatible endpoint `http://127.0.0.1:1234/v1`.
- Raw artifact: `tests/eval/golden_dataset_raw.csv`.
- Raw SHA256:
  `047f278d2400e07f0c9820bf9c16a5b967be8afd939a1b973e6ec74e3f807187`.
- Final curated count: 30 rows.
- Removed or replaced after raw review: 12.
- Reasons: one exact duplicate, transliterated or English questions unsuitable
  for the Russian assistant, artificial multi-hop combinations of unrelated
  support topics, overly generic questions, and ambiguous references.
- References corrected: 30 references were written and checked directly against
  the committed corporate corpus.

The raw file was reviewed row by row on 2026-07-29. The generated questions were
used as editorial candidates, while the final set retains exactly 30 concise,
Russian, corpus-grounded cases. Each final pair was checked against its source
document in `data/corporate`; `reference_contexts` contains the supporting
passage rather than a filename.

Dataset SHA256:
`424facc7e935d7c88152e0ef3010f79515fe6c796680e707d515f3f6ea8fea92`.
