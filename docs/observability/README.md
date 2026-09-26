# Observability

## Runtime contract

App emits structured logs with request ID, method/path, status and latency. LLM completion logs include model, token counts when available, finish reason, latency and a prompt hash.

Phoenix receives OpenTelemetry/OpenInference traces for OpenAI-compatible and LlamaIndex operations when `RAG_TRACING_ENABLED=true`. Tracing initialization is optional: missing collector/tracing extras do not change API behavior.

## Production privacy

Production validation requires:

```env
RAG_TRACING_ENABLED=false
TRACING_CAPTURE_CONTENT=false
LOG_PROMPT_PREVIEW_ENABLED=false
```

Tracing is therefore disabled by default. If an operator deliberately enables it while keeping `TRACING_CAPTURE_CONTENT=false`, OpenInference hides inputs, outputs, messages, images/text, prompts, choices, embedding text and vectors.

Logs never include provider/admin/internal/session secrets. Prompt preview is disabled in public mode; only non-reversible prompt hash and operational metadata remain.

Phoenix's persistent volume does not have automatic age-based pruning in this project. Operators who enable tracing must define separate retention for historical Phoenix data; ordinary restarts must not delete the volume.

## Evidence screenshots

[phoenix-trace.png](phoenix-trace.png) and [json-log.png](json-log.png) are historical development screenshots from a content-visible local configuration. They demonstrate instrumentation and structured logging, not production privacy defaults.
