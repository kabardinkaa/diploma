# RAG Evaluation & Monitoring

## Configuration

- Production model: `openai/gpt-5.4-mini` through the existing OpenRouter client.
- Judge model: `openai/gpt-5.4-mini`, provider `openai`.
- Evaluator embedding model: `openai/text-embedding-3-small`.
- RAG embedding model: `intfloat/multilingual-e5-base`, 768 dimensions.
- Production collection: `corporate_rag`.
- Baseline: recursive chunking, `512/64`, top-K `10`, reranker disabled.
- Refusal threshold: `0.82`.

## Golden Dataset

The curated dataset contains 30 reviewed question/reference/context records.
Every reference was checked against `data/corporate`; full source passages are
stored in `reference_contexts`.

The required full 40-row `TestsetGenerator` run was attempted on all 64 source
documents. It stopped during knowledge-graph transforms when OpenRouter returned
HTTP 402 for the fixed judge model. The successful 3-row VPN smoke was kept as a
temporary smoke artifact and was not presented as the required raw dataset.
Details are in [golden_dataset_review.md](golden_dataset_review.md).

Dataset SHA256:
`424facc7e935d7c88152e0ef3010f79515fe6c796680e707d515f3f6ea8fea92`.

## Baseline

No aggregate is reported. The provider balance blocked the live RAGAS run, so
no baseline CSV or synthetic metric values were created.

| Metric | Value |
| --- | ---: |
| Faithfulness | N/A |
| Answer relevancy | N/A |
| Context precision | N/A |
| Context recall | N/A |
| Has citation | N/A |
| Avg latency ms | N/A |

## Experiment A - Chunking

The isolated variant is implemented as
`corporate_rag_eval_chunk256`, recursive `256/64`, top-K `10`, reranker off.
Only chunk size and the corresponding index/docstore differ from baseline.

| Variant | Faithfulness | Answer relevancy | Context precision | Context recall | Has citation | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 512 | N/A | N/A | N/A | N/A | N/A | N/A |
| chunk 256 | N/A | N/A | N/A | N/A | N/A | N/A |

## Experiment B - Top-K

The variant uses the production `corporate_rag` index and changes only top-K
from `10` to `5`.

| Variant | Faithfulness | Answer relevancy | Context precision | Context recall | Has citation | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top-K 10 | N/A | N/A | N/A | N/A | N/A | N/A |
| top-K 5 | N/A | N/A | N/A | N/A | N/A | N/A |

## Final Configuration

The baseline remains the production default because no completed A/B metrics
exist to justify a change. A winner is deliberately not claimed.

## Failure Analysis

Unavailable until a complete per-row CSV exists. Selecting three rows without
judge results would invent faithfulness ordering and violate the evaluation
protocol.

## Phoenix Tracing

- UI: `http://localhost:6006`.
- OTLP transport: gRPC at `http://phoenix:4317`.
- Instrumentation: OpenAI and LlamaIndex share one tracer provider.
- Compatibility: OpenInference LlamaIndex `4.4.x` is pinned because the
  requested `3.3.3` imports an agent module removed from LlamaIndex `0.14.23`.
- Manual orchestration spans: `rag.query`, `rag.retrieve`, `rag.generate`.
- Live Docker smoke on 2026-07-28 sent 24 requests and Phoenix returned 24
  corresponding traces with 210 spans: 72 retrieval, 72 embedding, and 42
  LLM/generation spans.
- Three requests returned HTTP 200. The other 21 still produced complete
  retrieval/embedding/error traces, but response generation failed because
  OpenRouter returned HTTP 402 (500 requested output tokens, balance sufficient
  for 354).
- Phoenix UI was opened at `http://localhost:6006`; project
  `diploma-fastapi` displayed 32 total persisted traces, including all 24
  traces from this smoke. The UI and trace table were visually checked; no
  screenshot file was persisted.
- Machine-readable smoke counts are in
  `tests/eval/results/trace_summary.json`.

## Quality Gates

- Faithfulness > 0.7: not evaluated.
- Answer relevancy > 0.7: not evaluated.
- Has citation > 0.95: not evaluated.

## Known Limitations

- OpenRouter returned HTTP 402 after the fixed-model smoke succeeded. The
  account could not fund the complete generator and three evaluation runs.
- The same credit limit made 21 of 24 trace-smoke HTTP responses fail, although
  Phoenix still captured all 24 request traces and their failed LLM spans.
- RAGAS judge metrics have inherent noise and should be rerun with the same
  model and dataset after credits are available.
- The optional Phoenix HallucinationEvaluator is not implemented.
- No fake CSV, aggregate, winner, or failure ranking is committed.

## Improvement Plan

1. Add provider credits without changing either model slug.
2. Generate and commit the 36-40 row raw artifact as its own commit.
3. Run the one-row smoke, then baseline, chunk-256, and top-K-5 exactly once.
4. Populate both tables and analyze the 3-5 lowest-faithfulness rows.
5. Change production defaults only if a measured variant wins.

## Commands

```powershell
pip install ".[eval,tracing]"
python scripts/verify_eval.py
python scripts/generate_testset.py --size 40
python scripts/run_eval.py --label baseline --dry-run
python scripts/build_eval_index.py
python scripts/run_eval.py --label smoke --limit 1
python scripts/run_eval.py --label baseline
python scripts/run_eval.py --label chunk_256 --collection corporate_rag_eval_chunk256 --top-k 10
python scripts/run_eval.py --label top_k_5 --collection corporate_rag --top-k 5
docker compose up -d --build
docker compose exec -e BACKEND_URL=http://app:8000 -e PHOENIX_HTTP_ENDPOINT=http://phoenix:6006 app python scripts/trace_smoke.py
```
