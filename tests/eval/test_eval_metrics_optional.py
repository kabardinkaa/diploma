from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("ragas")

from app.eval.metrics import CitationDecision, make_has_citation


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
