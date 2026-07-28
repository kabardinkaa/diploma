from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.observability import tracing


@pytest.fixture(autouse=True)
def reset_tracing_state():
    tracing.reset_tracing_for_tests()
    yield
    tracing.reset_tracing_for_tests()


def test_tracing_disabled_does_not_import_optional_instrumentor(monkeypatch) -> None:
    assert tracing.setup_tracing(enabled=False) is None


def test_tracing_setup_is_idempotent_and_reuses_provider(mocker) -> None:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    provider = Mock()
    register = mocker.patch("phoenix.otel.register", return_value=provider)
    openai_instrument = mocker.patch.object(OpenAIInstrumentor, "instrument")
    llama_instrument = mocker.patch.object(LlamaIndexInstrumentor, "instrument")

    first = tracing.setup_tracing(enabled=True)
    second = tracing.setup_tracing(enabled=True)

    assert first is provider
    assert second is provider
    register.assert_called_once()
    openai_instrument.assert_called_once_with(tracer_provider=provider)
    llama_instrument.assert_called_once_with(tracer_provider=provider)
