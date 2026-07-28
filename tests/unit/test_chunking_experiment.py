from pathlib import Path

import pytest

from scripts.chunking_experiment import (
    COLLECTIONS,
    TUNING_CONFIGS,
    choose_best_strategy,
    choose_best_tuning,
    dataset_sha256,
    load_dataset,
    validate_dataset,
)


def test_collection_mapping_is_stable() -> None:
    assert COLLECTIONS == {
        "fixed": "docs_fixed",
        "recursive": "docs_recursive",
        "semantic": "docs_semantic",
    }


def test_dataset_hash_is_sha256() -> None:
    digest = dataset_sha256()

    assert len(digest) == 64
    int(digest, 16)


def test_dataset_validation_rejects_too_few_questions() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        validate_dataset(
            [{"question": "q", "relevant_doc_ids": ["01_vpn.md"]}],
            {"01_vpn.md"},
        )


def test_dataset_validation_rejects_unknown_doc_id() -> None:
    dataset = [
        {"question": f"question {index}", "relevant_doc_ids": ["01_vpn.md"]}
        for index in range(20)
    ]
    dataset[4]["relevant_doc_ids"] = ["missing.md"]

    with pytest.raises(ValueError, match="unknown doc ids.*missing.md"):
        validate_dataset(dataset, {"01_vpn.md"})


def test_real_dataset_validates_against_corpus() -> None:
    corpus_ids = {
        path.name
        for path in Path("data/rag-block-03").iterdir()
        if path.is_file()
    }

    validate_dataset(load_dataset(), corpus_ids)


def test_tuning_changes_one_parameter_from_baseline() -> None:
    baseline = TUNING_CONFIGS[0]

    assert len(TUNING_CONFIGS) == 5
    for experiment in TUNING_CONFIGS[1:]:
        changed = {
            key
            for key in ("chunk_size", "chunk_overlap", "candidate_top_k")
            if experiment[key] != baseline[key]
        }
        assert len(changed) == 1


def test_best_strategy_uses_smaller_index_for_near_equal_latency() -> None:
    def result(latency: float, points: int) -> dict:
        return {
            "points_count": points,
            "metrics": {
                "hit_rate_at_5": 1.0,
                "mrr_at_10": 1.0,
                "recall_at_10": 1.0,
                "avg_retrieval_ms": latency,
            },
        }

    assert choose_best_strategy(
        {
            "recursive": result(4.26, 10),
            "semantic": result(4.18, 20),
        }
    ) == "recursive"


def test_best_tuning_prefers_baseline_when_quality_is_equal() -> None:
    results = []
    for index, config in enumerate(TUNING_CONFIGS):
        results.append(
            {
                "experiment": config["experiment"],
                "points_count": 10 if index != 1 else 16,
                "metrics": {
                    "hit_rate_at_5": 1.0,
                    "mrr_at_10": 0.97 if index == 1 else 1.0,
                    "recall_at_10": 1.0,
                    "avg_retrieval_ms": 4.0 - index * 0.01,
                },
            }
        )

    assert choose_best_tuning(results)["experiment"] == "baseline"
