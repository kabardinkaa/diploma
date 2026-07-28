from unittest.mock import Mock

from app.services.reranker import Reranker


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.predict = Mock(return_value=[0.2, 0.9, -0.1])


def candidates() -> list[dict]:
    return [
        {
            "text": "first",
            "doc_id": "doc_a",
            "metadata": {"source": "a.md"},
            "dense_score": 0.95,
        },
        {
            "text": "second",
            "doc_id": "doc_b",
            "metadata": {"source": "b.md"},
            "dense_score": 0.70,
        },
        {
            "text": "third",
            "doc_id": "doc_c",
            "metadata": {"source": "c.md"},
            "dense_score": 0.60,
        },
    ]


def test_reranker_loads_model_lazily_and_once() -> None:
    model = FakeCrossEncoder()
    factory = Mock(return_value=model)
    reranker = Reranker("fake-model", model_factory=factory)

    assert factory.call_count == 0
    reranker.rerank("query", candidates(), top_n=2)
    reranker.rerank("query", candidates(), top_n=2)

    factory.assert_called_once_with("fake-model")


def test_reranker_sorts_scores_preserves_metadata_and_top_n() -> None:
    model = FakeCrossEncoder()
    reranker = Reranker(
        "fake-model",
        model_factory=lambda _: model,
        batch_size=4,
    )

    result = reranker.rerank("query", candidates(), top_n=2)

    assert [item["doc_id"] for item in result] == ["doc_b", "doc_a"]
    assert [item["reranker_score"] for item in result] == [0.9, 0.2]
    assert result[0]["metadata"] == {"source": "b.md"}
    assert result[0]["dense_score"] == 0.70
    model.predict.assert_called_once_with(
        [("query", "first"), ("query", "second"), ("query", "third")],
        batch_size=4,
        show_progress_bar=False,
    )


def test_empty_candidates_do_not_load_model() -> None:
    factory = Mock()
    reranker = Reranker("fake-model", model_factory=factory)

    assert reranker.rerank("query", [], top_n=10) == []
    factory.assert_not_called()
