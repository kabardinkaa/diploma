# Observability

## Phoenix trace

Файл `phoenix-trace.png` содержит пример успешного трейса LLM-запроса в Phoenix.

На скриншоте видно:
- проект `diploma-fastapi`;
- span `ChatCompletion`;
- статус `OK`;
- latency LLM-вызова;
- модель, использованную для ответа;
- входные и выходные данные LLM-запроса.

## JSON log

Файл `json-log.png` содержит пример JSON-лога события `llm_request_completed`.

В логе видно:
- `request_id`;
- `model`;
- `input_tokens`;
- `output_tokens`;
- `latency_ms`;
- `finish_reason`;
- `prompt_hash`;
- `prompt_preview`.

Сырые PII в JSON-логи не сохраняются: для prompt используется `prompt_hash`, а короткое превью проходит через `redact_pii`.