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
    openai_instrument.assert_called_once()
    llama_instrument.assert_called_once()
    assert openai_instrument.call_args.kwargs["tracer_provider"] is provider
    assert llama_instrument.call_args.kwargs["tracer_provider"] is provider
    assert openai_instrument.call_args.kwargs["config"].hide_inputs is False
    assert llama_instrument.call_args.kwargs["config"].hide_outputs is False


def test_tracing_privacy_config_hides_content(mocker) -> None:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    mocker.patch("phoenix.otel.register", return_value=Mock())
    openai_instrument = mocker.patch.object(OpenAIInstrumentor, "instrument")
    llama_instrument = mocker.patch.object(LlamaIndexInstrumentor, "instrument")

    tracing.setup_tracing(enabled=True, capture_content=False)

    openai_config = openai_instrument.call_args.kwargs["config"]
    llama_config = llama_instrument.call_args.kwargs["config"]
    for config in (openai_config, llama_config):
        assert config.hide_inputs is True
        assert config.hide_outputs is True
        assert config.hide_prompts is True
        assert config.hide_choices is True
        assert config.hide_embeddings_text is True


def test_tracing_flush_failure_is_non_fatal() -> None:
    provider = Mock()
    provider.force_flush.side_effect = RuntimeError("collector unavailable")
    tracing._tracer_provider = provider

    assert tracing.force_flush_tracing(timeout_millis=123) is False
    provider.force_flush.assert_called_once_with(timeout_millis=123)


def test_tracing_shutdown_failure_is_non_fatal() -> None:
    provider = Mock()
    provider.shutdown.side_effect = RuntimeError("collector unavailable")
    tracing._tracer_provider = provider

    assert tracing.shutdown_tracing() is False
    provider.shutdown.assert_called_once_with()
