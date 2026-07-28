# Golden Dataset Review

- Automated raw target: 40 rows through RAGAS `TestsetGenerator`.
- Automated smoke: 3 real rows generated from the VPN subset.
- Full raw result: not produced. OpenRouter returned HTTP 402 while the fixed
  `openai/gpt-5.4-mini` model was processing the 64-document corpus.
- Final curated count: 30 rows.
- Removed from automated raw: not applicable because the full raw run did not
  complete and no partial artifact was presented as a successful raw dataset.
- References corrected: 30 references were written and checked directly against
  the committed corporate corpus.

Each final pair was reviewed against the source document represented in
`data/corporate`. Questions that were generic, ambiguous, unsupported, or
duplicates were not included. `reference_contexts` contains the actual source
passage rather than a filename.

Dataset SHA256:
`424facc7e935d7c88152e0ef3010f79515fe6c796680e707d515f3f6ea8fea92`.
