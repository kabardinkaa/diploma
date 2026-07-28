from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
)

from app.services.vector_store import (
    PAYLOAD_INDEXES,
    VectorStore,
    VectorStoreConfigurationError,
)
from scripts.load_to_qdrant import (
    build_points,
    load_documents,
    stable_point_id,
    validate_embedding_dimension,
)


def settings(dim: int = 768) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant_url="http://qdrant.test:6333",
        qdrant_api_key=None,
        qdrant_collection="documents",
        embedding_dim=dim,
    )


def collection_info(
    *,
    dim: int = 768,
    distance: Distance = Distance.COSINE,
    payload_schema: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=dim, distance=distance)
            )
        ),
        payload_schema=payload_schema or {},
        points_count=0,
    )


def fake_client(mocker) -> SimpleNamespace:
    return SimpleNamespace(
        get_collections=mocker.AsyncMock(),
        create_collection=mocker.AsyncMock(),
        get_collection=mocker.AsyncMock(),
        create_payload_index=mocker.AsyncMock(),
        upsert=mocker.AsyncMock(),
        query_points=mocker.AsyncMock(),
        close=mocker.AsyncMock(),
    )


@pytest.mark.asyncio
async def test_ensure_collection_creates_expected_dimension(mocker) -> None:
    client = fake_client(mocker)
    client.get_collections.return_value = SimpleNamespace(collections=[])
    client.get_collection.return_value = collection_info()
    store = VectorStore(settings(), client=client)

    await store.ensure_collection()

    client.create_collection.assert_awaited_once()
    kwargs = client.create_collection.await_args.kwargs
    assert kwargs["collection_name"] == "documents"
    assert kwargs["vectors_config"].size == 768
    assert kwargs["vectors_config"].distance == Distance.COSINE
    assert kwargs["hnsw_config"].m == 16
    assert kwargs["hnsw_config"].ef_construct == 100


@pytest.mark.asyncio
async def test_existing_collection_with_matching_contract_passes(mocker) -> None:
    client = fake_client(mocker)
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="documents")]
    )
    client.get_collection.return_value = collection_info(
        payload_schema={name: {} for name in PAYLOAD_INDEXES}
    )
    store = VectorStore(settings(), client=client)

    await store.ensure_collection()

    client.create_collection.assert_not_awaited()
    client.create_payload_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_dimension_mismatch_has_clear_error(mocker) -> None:
    client = fake_client(mocker)
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="documents")]
    )
    client.get_collection.return_value = collection_info(dim=384)
    store = VectorStore(settings(), client=client)

    with pytest.raises(
        VectorStoreConfigurationError,
        match=r"has 384 dimensions.*EMBEDDING_DIM=768",
    ):
        await store.ensure_collection()


@pytest.mark.asyncio
async def test_distance_mismatch_has_clear_error(mocker) -> None:
    client = fake_client(mocker)
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="documents")]
    )
    client.get_collection.return_value = collection_info(distance=Distance.DOT)
    store = VectorStore(settings(), client=client)

    with pytest.raises(VectorStoreConfigurationError, match="Distance mismatch"):
        await store.ensure_collection()


@pytest.mark.asyncio
async def test_ensure_collection_creates_all_payload_indexes(mocker) -> None:
    client = fake_client(mocker)
    client.get_collections.return_value = SimpleNamespace(collections=[])
    client.get_collection.return_value = collection_info()
    store = VectorStore(settings(), client=client)

    await store.ensure_collection()

    created = {
        (
            call.kwargs["field_name"],
            call.kwargs["field_schema"],
        )
        for call in client.create_payload_index.await_args_list
    }
    assert created == {
        ("source", PayloadSchemaType.KEYWORD),
        ("created_at", PayloadSchemaType.DATETIME),
        ("category", PayloadSchemaType.KEYWORD),
        ("department", PayloadSchemaType.KEYWORD),
        ("is_archived", PayloadSchemaType.BOOL),
    }


@pytest.mark.asyncio
async def test_upsert_splits_points_into_bounded_batches(mocker) -> None:
    client = fake_client(mocker)
    store = VectorStore(settings(), client=client)
    points = [
        PointStruct(id=index, vector=[1.0, 0.0], payload={})
        for index in range(600)
    ]

    await store.upsert(points, batch_size=256)

    calls = client.upsert.await_args_list
    assert [len(call.kwargs["points"]) for call in calls] == [256, 256, 88]
    assert [call.kwargs["wait"] for call in calls] == [False, False, True]


@pytest.mark.asyncio
async def test_search_uses_query_points_and_passes_filter(mocker) -> None:
    client = fake_client(mocker)
    expected = ScoredPoint(
        id=1,
        version=0,
        score=0.91,
        payload={"source": "support/vpn/001"},
    )
    client.query_points.return_value = SimpleNamespace(points=[expected])
    store = VectorStore(settings(), client=client)
    query_filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="vpn"),
            )
        ]
    )

    result = await store.search(
        [0.1, 0.2],
        top_k=3,
        query_filter=query_filter,
    )

    assert result == [expected]
    assert isinstance(result[0], ScoredPoint)
    client.query_points.assert_awaited_once_with(
        collection_name="documents",
        query=[0.1, 0.2],
        query_filter=query_filter,
        limit=3,
        with_payload=True,
    )


@pytest.mark.asyncio
async def test_client_is_created_once_and_reused(mocker) -> None:
    client = fake_client(mocker)
    client.query_points.return_value = SimpleNamespace(points=[])
    client_factory = mocker.patch(
        "app.services.vector_store.AsyncQdrantClient",
        return_value=client,
    )
    store = VectorStore(settings())

    await store.search([0.1], top_k=1)
    await store.search([0.2], top_k=1)

    client_factory.assert_called_once_with(
        url="http://qdrant.test:6333",
        api_key=None,
    )
    assert client.query_points.await_count == 2


def test_stable_point_id_is_deterministic() -> None:
    first = stable_point_id("support/vpn/connection", 0)
    second = stable_point_id("support/vpn/connection", 0)

    assert first == second
    assert first != stable_point_id("support/vpn/connection", 1)


def test_reindexing_generates_the_same_ids() -> None:
    documents = load_documents()
    sample = documents[:2]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    first = build_points(sample, vectors, expected_dim=2)
    second = build_points(sample, vectors, expected_dim=2)

    assert [point.id for point in first] == [point.id for point in second]


def test_support_corpus_is_unique_and_filterable() -> None:
    documents = load_documents()
    required = {
        "source",
        "title",
        "text",
        "created_at",
        "category",
        "department",
        "is_archived",
    }

    assert len(documents) == 128
    assert len({document["source"] for document in documents}) == len(documents)
    assert len({document["text"] for document in documents}) == len(documents)
    assert all(required <= set(document) for document in documents)
    assert any(document["is_archived"] for document in documents)
    assert any(not document["is_archived"] for document in documents)


def test_embedding_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="model produced 3 dimensions"):
        validate_embedding_dimension([0.1, 0.2, 0.3], expected_dim=768)


def test_corpus_path_is_inside_project() -> None:
    documents = load_documents(Path("data/support_kb.json"))
    assert len(documents) >= 100
