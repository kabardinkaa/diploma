# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS builder

ARG INSTALL_EXTRAS=tracing

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:0.6.10@sha256:57da96c4557243fc0a732817854084e81af9393f64dc7d172f39c16465b5e2ba /uv /uvx /bin/

WORKDIR /app

RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt constraints.txt pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    --constraint constraints.txt \
    "torch==2.7.1+cpu"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --constraint constraints.txt -r requirements.txt

COPY app ./app
COPY bot ./bot
COPY scripts ./scripts
COPY data ./data
COPY tests/eval/golden_dataset.json ./tests/eval/golden_dataset.json

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$INSTALL_EXTRAS" ]; then \
      uv pip install --constraint constraints.txt ".[${INSTALL_EXTRAS}]"; \
    fi


FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app
RUN mkdir -p /app/.cache/embeddings /app/.cache/rag /app/data \
    && chown -R appuser:appuser /app/.cache /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request; opener=urllib.request.build_opener(urllib.request.ProxyHandler({})); opener.open('http://127.0.0.1:8000/health/live')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
