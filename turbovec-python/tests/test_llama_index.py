from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from turbovec import IdMapIndex
from turbovec.llama_index import TurboQuantVectorStore


def _unit_vec(seed: int, dim: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def _make_node(text: str, seed: int, dim: int = 64, metadata: dict | None = None,
               ref_doc_id: str | None = None) -> TextNode:
    node = TextNode(text=text, metadata=metadata or {}, embedding=_unit_vec(seed, dim))
    if ref_doc_id is not None:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref_doc_id)
    return node


def test_from_params_creates_index():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    assert store._index.dim == 64
    assert store._index.bit_width == 4
    assert store.stores_text is True
    assert store.is_embedding_query is True


# ---- Lazy index construction --------------------------------------------

def test_constructor_no_index_is_lazy():
    # No-arg construction yields a lazy-uncommitted IdMapIndex.
    store = TurboQuantVectorStore()
    assert store._index.dim is None
    # Query before any add returns an empty result.
    result = store.query(
        VectorStoreQuery(query_embedding=_unit_vec(0, 64), similarity_top_k=3)
    )
    assert result.nodes == []
    assert result.similarities == []
    assert result.ids == []


def test_lazy_dim_locked_on_first_add():
    store = TurboQuantVectorStore(bit_width=2)
    store.add([_make_node("x", seed=0)])
    assert store._index.dim == 64
    assert store._index.bit_width == 2


def test_from_params_without_dim_is_lazy():
    store = TurboQuantVectorStore.from_params(bit_width=4)
    assert store._index.dim is None


def test_persist_and_load_lazy_uncommitted_store(tmp_path):
    # A store that's never seen an add must round-trip through persist
    # without committing a dim or losing its bit_width.
    store = TurboQuantVectorStore(bit_width=2)
    persist_dir = tmp_path / "lazy_store"
    store.persist(str(persist_dir))
    loaded = TurboQuantVectorStore.from_persist_path(
        str(persist_dir), allow_dangerous_deserialization=True
    )
    assert loaded._index.dim is None
    assert loaded._index.bit_width == 2
    loaded.add([_make_node("post-load", seed=0)])
    assert loaded._index.dim == 64


def test_add_and_query_returns_nodes():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [_make_node(f"doc {i}", seed=i) for i in range(5)]
    ids = store.add(nodes)
    assert len(ids) == 5
    assert set(ids) == {n.node_id for n in nodes}

    query = VectorStoreQuery(query_embedding=_unit_vec(0, 64), similarity_top_k=3)
    result = store.query(query)
    assert len(result.nodes) == 3
    assert len(result.similarities) == 3
    assert len(result.ids) == 3
    assert all(isinstance(n, TextNode) for n in result.nodes)


def test_metadata_and_text_roundtrip():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("hello world", seed=1, metadata={"source": "a", "page": 7}),
        _make_node("goodbye world", seed=2, metadata={"source": "b", "page": 12}),
    ]
    store.add(nodes)

    result = store.query(VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=2))
    texts = {n.get_content() for n in result.nodes}
    assert texts == {"hello world", "goodbye world"}
    sources = {n.metadata["source"] for n in result.nodes}
    assert sources == {"a", "b"}


def test_ref_doc_id_preserved_through_query():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    node = _make_node("child text", seed=3, ref_doc_id="parent-doc-123")
    store.add([node])

    result = store.query(VectorStoreQuery(query_embedding=_unit_vec(3, 64), similarity_top_k=1))
    returned = result.nodes[0]
    assert returned.ref_doc_id == "parent-doc-123"


def test_empty_query_returns_empty():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    result = store.query(VectorStoreQuery(query_embedding=_unit_vec(0, 64), similarity_top_k=5))
    assert result.nodes == []
    assert result.similarities == []
    assert result.ids == []


def test_k_larger_than_ntotal_is_clamped():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("a", seed=1), _make_node("b", seed=2)])
    result = store.query(VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=100))
    assert len(result.nodes) == 2


def test_query_without_embedding_raises():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("a", seed=1)])
    with pytest.raises(ValueError, match="query_embedding"):
        store.query(VectorStoreQuery(query_embedding=None, similarity_top_k=1))


def test_mismatched_dim_raises():
    store = TurboQuantVectorStore(index=IdMapIndex(32, 4))
    with pytest.raises(ValueError, match="embedding dim"):
        store.add([_make_node("x", seed=1, dim=64)])


def test_persist_and_from_persist_path_roundtrip(tmp_path):
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("one", seed=1, metadata={"n": 1}),
        _make_node("two", seed=2, metadata={"n": 2}),
        _make_node("three", seed=3, metadata={"n": 3}),
    ]
    store.add(nodes)
    store.persist(str(tmp_path))

    loaded = TurboQuantVectorStore.from_persist_path(
        str(tmp_path), allow_dangerous_deserialization=True
    )
    result = loaded.query(VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=3))
    assert {n.get_content() for n in result.nodes} == {"one", "two", "three"}


def test_from_persist_path_refuses_without_flag(tmp_path):
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("x", seed=1)])
    store.persist(str(tmp_path))
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
        TurboQuantVectorStore.from_persist_path(str(tmp_path))


def test_delete_by_ref_doc_id_removes_every_matching_node():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("a1", seed=1, ref_doc_id="parent-1"),
        _make_node("a2", seed=2, ref_doc_id="parent-1"),
        _make_node("b1", seed=3, ref_doc_id="parent-2"),
    ]
    store.add(nodes)

    store.delete("parent-1")
    # Only the parent-2 node survives.
    result = store.query(VectorStoreQuery(query_embedding=_unit_vec(3, 64), similarity_top_k=5))
    assert {n.get_content() for n in result.nodes} == {"b1"}
    assert len(store._index) == 1


def test_delete_by_missing_ref_doc_id_is_noop():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("a", seed=1, ref_doc_id="parent-1")])
    store.delete("does-not-exist")
    assert len(store._index) == 1


def test_delete_nodes_by_node_id():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("a", seed=1),
        _make_node("b", seed=2),
        _make_node("c", seed=3),
    ]
    ids = store.add(nodes)
    store.delete_nodes([ids[0], ids[2]])
    assert len(store._index) == 1
    result = store.query(VectorStoreQuery(query_embedding=_unit_vec(2, 64), similarity_top_k=3))
    assert {n.get_content() for n in result.nodes} == {"b"}


def test_add_upsert_replaces_same_node_id():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    # Build two nodes with the same node_id but different content/embeddings.
    first = TextNode(text="v1", embedding=_unit_vec(1, 64))
    second = TextNode(text="v2", id_=first.node_id, embedding=_unit_vec(2, 64))
    store.add([first])
    store.add([second])
    assert len(store._index) == 1
    result = store.query(VectorStoreQuery(query_embedding=_unit_vec(2, 64), similarity_top_k=1))
    assert result.nodes[0].get_content() == "v2"


# ------------------- Filtered query -------------------

def _store_with_tiered_nodes() -> TurboQuantVectorStore:
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node(f"doc {i}", seed=i, metadata={"tier": tier, "idx": i})
        for i, tier in enumerate(["free", "pro", "free", "pro", "enterprise"])
    ]
    store.add(nodes)
    return store


def test_query_with_eq_filter():
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    result = store.query(q)
    assert len(result.nodes) == 2
    assert all(n.metadata["tier"] == "pro" for n in result.nodes)


def test_query_with_in_filter():
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tier", value=["pro", "enterprise"], operator=FilterOperator.IN
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    result = store.query(q)
    assert len(result.nodes) == 3
    assert all(n.metadata["tier"] in {"pro", "enterprise"} for n in result.nodes)


def test_query_with_and_filter():
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ),
            MetadataFilter(key="idx", value=2, operator=FilterOperator.GT),
        ],
        condition=FilterCondition.AND,
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    result = store.query(q)
    # tier=="pro" matches idx 1, 3; combined with idx > 2 leaves only idx=3.
    assert len(result.nodes) == 1
    assert result.nodes[0].metadata["idx"] == 3


def test_query_with_or_filter():
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tier", value="enterprise", operator=FilterOperator.EQ),
            MetadataFilter(key="idx", value=0, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    result = store.query(q)
    # enterprise = idx 4, OR idx==0 → 2 nodes total.
    assert len(result.nodes) == 2
    idxs = {n.metadata["idx"] for n in result.nodes}
    assert idxs == {0, 4}


def test_query_filter_no_matches_returns_empty():
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value="nonexistent", operator=FilterOperator.EQ)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=5, filters=filters
    )
    result = store.query(q)
    assert result.nodes == []
    assert result.similarities == []
    assert result.ids == []


def test_query_filter_selective_returns_top_k_from_matches():
    # 50 nodes, filter selects 3 — we must return all 3 even when top_k=3
    # and the matching nodes wouldn't be in the unfiltered top-3.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node(f"doc {i}", seed=i, metadata={"tag": "needle" if i in (7, 23, 41) else "hay"})
        for i in range(50)
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tag", value="needle", operator=FilterOperator.EQ)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=3, filters=filters
    )
    result = store.query(q)
    assert len(result.nodes) == 3
    assert all(n.metadata["tag"] == "needle" for n in result.nodes)


def test_query_with_node_ids_filter():
    # `node_ids` restricts to specific node_ids — matches SimpleVectorStore's
    # canonical behaviour.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [_make_node(f"doc {i}", seed=i) for i in range(5)]
    store.add(nodes)
    keep = [nodes[1].node_id, nodes[3].node_id]
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=5, node_ids=keep
    )
    result = store.query(q)
    assert len(result.nodes) == 2
    assert {n.node_id for n in result.nodes} == set(keep)


def test_query_with_doc_ids_filter_matches_ref_doc_id_only():
    # `doc_ids` filters by `ref_doc_id` (source document), not node_id.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node(f"chunk {i}", seed=i, ref_doc_id=f"src-{i // 2}")
        for i in range(6)
    ]
    store.add(nodes)
    # Two source docs: src-0 (chunks 0, 1) and src-1 (chunks 2, 3).
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64),
        similarity_top_k=10,
        doc_ids=["src-0", "src-1"],
    )
    result = store.query(q)
    assert len(result.nodes) == 4
    # A bare node_id passed via doc_ids does NOT match; that's what node_ids
    # is for.
    q2 = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64),
        similarity_top_k=10,
        doc_ids=[nodes[0].node_id],
    )
    assert store.query(q2).nodes == []


def test_query_with_node_ids_and_filters_intersect():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node(f"doc {i}", seed=i, metadata={"tier": "pro" if i % 2 else "free"})
        for i in range(6)
    ]
    store.add(nodes)
    keep = [nodes[i].node_id for i in (0, 1, 2, 3)]  # narrow to first four
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64),
        similarity_top_k=10,
        node_ids=keep,
        filters=filters,
    )
    result = store.query(q)
    # tier=="pro" → odd indices (1, 3, 5). Intersect with {0,1,2,3} → {1,3}.
    assert {n.node_id for n in result.nodes} == {nodes[1].node_id, nodes[3].node_id}


def test_query_ne_filter_treats_missing_key_as_no_match():
    # Matches SimpleVectorStore reference: NE on a missing key returns False.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("with", seed=0, metadata={"tier": "free"}),
        _make_node("without", seed=1, metadata={}),  # no `tier` key
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value="pro", operator=FilterOperator.NE)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    result = store.query(q)
    # Only the doc with `tier=="free"` matches (free != pro). The doc with
    # no `tier` key is NOT a match — its key is missing.
    assert len(result.nodes) == 1
    assert result.nodes[0].metadata.get("tier") == "free"


def test_query_text_match_is_case_insensitive():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("a", seed=0, metadata={"title": "The Lord of the Rings"}),
        _make_node("b", seed=1, metadata={"title": "Lord of Light"}),
        _make_node("c", seed=2, metadata={"title": "The Hobbit"}),
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="title", value="LORD", operator=FilterOperator.TEXT_MATCH)
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    result = store.query(q)
    titles = {n.metadata["title"] for n in result.nodes}
    assert titles == {"The Lord of the Rings", "Lord of Light"}


def test_query_unsupported_filter_operator_raises():
    store = _store_with_tiered_nodes()
    # ANY is intentionally not in our supported list.
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value=["pro"], operator=FilterOperator.ANY)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=3, filters=filters
    )
    with pytest.raises(NotImplementedError):
        store.query(q)
