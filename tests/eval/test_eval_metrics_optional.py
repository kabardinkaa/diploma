from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("ragas")

from app.core.config import Settings
from app.eval.metrics import CitationDecision, build_judge, make_has_citation


@pytest.mark.asyncio
async def test_has_citation_decorator_wiring() -> None:
    llm = SimpleNamespace(
        agenerate=AsyncMock(
            return_value=CitationDecision(value="yes", reason="Есть маркер [1]")
        )
    )
    metric = make_has_citation(llm)

    result = await metric.ascore(response="Ответ [1]")

    assert metric.name == "has_citation"
    assert metric.allowed_values == ["yes", "no"]
    assert result.value == "yes"
    assert llm.agenerate.await_args.kwargs["response_model"] is CitationDecision


def test_local_judge_uses_json_schema_and_configured_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import instructor
    import ragas.llms

    client = object()
    patched_client = object()
    captured: dict[str, object] = {}

    def fake_from_openai(value: object, *, mode: object) -> object:
        captured["source_client"] = value
        captured["mode"] = mode
        return patched_client

    def fake_instructor_llm(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(instructor, "from_openai", fake_from_openai)
    monkeypatch.setattr(ragas.llms, "InstructorLLM", fake_instructor_llm)

    judge = build_judge(Settings(_env_file=None), client)  # type: ignore[arg-type]

    assert captured["source_client"] is client
    assert captured["mode"] is instructor.Mode.JSON_SCHEMA
    assert judge.client is patched_client
    assert judge.model_args.max_tokens == 4096
