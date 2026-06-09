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
    persist_path = tmp_path / "lazy_store.json"
    store.persist(str(persist_path))
    loaded = TurboQuantVectorStore.from_persist_path(str(persist_path))
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
    persist_path = tmp_path / "store.json"
    store.persist(str(persist_path))

    loaded = TurboQuantVectorStore.from_persist_path(str(persist_path))
    result = loaded.query(VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=3))
    assert {n.get_content() for n in result.nodes} == {"one", "two", "three"}


def test_from_persist_dir_loads_default_namespace(tmp_path):
    # StorageContext-style: a directory containing a namespaced filename.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [_make_node(f"doc {i}", seed=i) for i in range(3)]
    store.add(nodes)
    # Mimic StorageContext.persist's filename layout.
    persist_path = tmp_path / "default__vector_store.json"
    store.persist(str(persist_path))

    loaded = TurboQuantVectorStore.from_persist_dir(str(tmp_path))
    result = loaded.query(
        VectorStoreQuery(query_embedding=_unit_vec(0, 64), similarity_top_k=3)
    )
    assert len(result.nodes) == 3


def test_from_persist_dir_with_custom_namespace(tmp_path):
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("ns-doc", seed=0)])
    persist_path = tmp_path / "custom-ns__vector_store.json"
    store.persist(str(persist_path))

    loaded = TurboQuantVectorStore.from_persist_dir(
        str(tmp_path), namespace="custom-ns"
    )
    assert len(loaded._nodes) == 1


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


def test_add_upsert_dim_mismatch_preserves_existing_node():
    # An upsert whose new embedding fails validation must not destroy the
    # existing node: the delete is deferred until the add succeeds.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    first = TextNode(text="v1", embedding=_unit_vec(1, 64))
    bad = TextNode(text="v2", id_=first.node_id, embedding=_unit_vec(2, 32))
    store.add([first])
    with pytest.raises(ValueError):
        store.add([bad])  # dim 32 != index dim 64

    assert len(store._index) == 1
    assert first.node_id in store._node_id_to_u64
    result = store.query(
        VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=1)
    )
    assert result.nodes[0].get_content() == "v1"


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


def test_query_text_match_is_case_sensitive():
    # Matches the reference (`utils.py:138-144`): TEXT_MATCH is a case-
    # SENSITIVE substring check. An earlier turbovec impl lowercased both
    # sides — we no longer do that (TEXT_MATCH_INSENSITIVE is the
    # opt-in case-folding variant; see below).
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("a", seed=0, metadata={"title": "The Lord of the Rings"}),
        _make_node("b", seed=1, metadata={"title": "Lord of Light"}),
        _make_node("c", seed=2, metadata={"title": "The Hobbit"}),
    ]
    store.add(nodes)

    # Case-correct query: matches the two "Lord" titles.
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="title", value="Lord", operator=FilterOperator.TEXT_MATCH)
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    titles = {n.metadata["title"] for n in store.query(q).nodes}
    assert titles == {"The Lord of the Rings", "Lord of Light"}

    # Wrong-case query: no match (used to match under the lowercasing bug).
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="title", value="LORD", operator=FilterOperator.TEXT_MATCH)
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    assert store.query(q).nodes == []


def test_query_text_match_insensitive_folds_case():
    # TEXT_MATCH_INSENSITIVE is the explicit opt-in for case-folding,
    # matching `utils.py:145-151`.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("a", seed=0, metadata={"title": "The Lord of the Rings"}),
        _make_node("b", seed=1, metadata={"title": "Lord of Light"}),
        _make_node("c", seed=2, metadata={"title": "The Hobbit"}),
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="title",
                value="LORD",
                operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    titles = {n.metadata["title"] for n in store.query(q).nodes}
    assert titles == {"The Lord of the Rings", "Lord of Light"}


def test_query_text_match_raises_type_error_on_non_string_operands():
    # Reference rejects non-string operands with a TypeError so the caller
    # can't silently fall through to string-coerced behaviour.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("a", seed=0, metadata={"page": 7})])
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="page", value="7", operator=FilterOperator.TEXT_MATCH)
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=1, filters=filters
    )
    with pytest.raises(TypeError, match="TEXT_MATCH"):
        store.query(q)


def test_query_all_operator_matches_set_subset():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("a", seed=0, metadata={"tags": ["python", "rust", "go"]}),
        _make_node("b", seed=1, metadata={"tags": ["python", "rust"]}),
        _make_node("c", seed=2, metadata={"tags": ["python"]}),
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tags", value=["python", "rust"], operator=FilterOperator.ALL
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    contents = {n.get_content() for n in store.query(q).nodes}
    assert contents == {"a", "b"}


def test_query_any_operator_matches_set_intersection():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node("a", seed=0, metadata={"tags": ["python"]}),
        _make_node("b", seed=1, metadata={"tags": ["rust"]}),
        _make_node("c", seed=2, metadata={"tags": ["go"]}),
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="tags", value=["python", "rust"], operator=FilterOperator.ANY
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    contents = {n.get_content() for n in store.query(q).nodes}
    assert contents == {"a", "b"}


def test_query_is_empty_treats_missing_key_as_match():
    # Reference (`utils.py:168-173`): IS_EMPTY matches when the key is
    # absent OR the value is empty string / empty list. Trickiest branch
    # because it's the only operator where a missing key is a hit.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([
        _make_node("a", seed=0, metadata={"note": "x"}),
        _make_node("b", seed=1, metadata={}),
        _make_node("c", seed=2, metadata={"note": ""}),
        _make_node("d", seed=3, metadata={"note": []}),
    ])
    filters = MetadataFilters(
        filters=[MetadataFilter(key="note", value=None, operator=FilterOperator.IS_EMPTY)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    contents = {n.get_content() for n in store.query(q).nodes}
    # All three "empty-ish" rows match; the populated one does not.
    assert contents == {"b", "c", "d"}


def test_query_contains_operator_matches_list_membership():
    # CONTAINS — the canonical "scalar value in list-valued metadata"
    # predicate. Distinct from ALL/ANY (which take list-valued targets).
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([
        _make_node("a", seed=0, metadata={"tags": ["python", "rust"]}),
        _make_node("b", seed=1, metadata={"tags": ["go"]}),
        _make_node("c", seed=2, metadata={"tags": ["python"]}),
    ])
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tags", value="python", operator=FilterOperator.CONTAINS)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    contents = {n.get_content() for n in store.query(q).nodes}
    assert contents == {"a", "c"}


def _store_with_scored_nodes() -> TurboQuantVectorStore:
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([
        _make_node(f"doc-{i}", seed=i, metadata={"score": i})
        for i in range(5)
    ])
    return store


def test_query_with_lt_filter():
    filters = MetadataFilters(
        filters=[MetadataFilter(key="score", value=2, operator=FilterOperator.LT)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    scores = {n.metadata["score"] for n in _store_with_scored_nodes().query(q).nodes}
    assert scores == {0, 1}


def test_query_with_lte_filter():
    # Boundary case: LTE must include the threshold value (where LT excludes it).
    filters = MetadataFilters(
        filters=[MetadataFilter(key="score", value=2, operator=FilterOperator.LTE)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    scores = {n.metadata["score"] for n in _store_with_scored_nodes().query(q).nodes}
    assert scores == {0, 1, 2}


def test_query_with_gte_filter():
    # Boundary case: GTE must include the threshold value (where GT excludes it).
    filters = MetadataFilters(
        filters=[MetadataFilter(key="score", value=3, operator=FilterOperator.GTE)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    scores = {n.metadata["score"] for n in _store_with_scored_nodes().query(q).nodes}
    assert scores == {3, 4}


def test_query_with_nin_filter():
    # NIN — values NOT in the supplied list. Distinct branch from IN.
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value=["pro"], operator=FilterOperator.NIN)]
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    tiers = {n.metadata["tier"] for n in store.query(q).nodes}
    assert tiers == {"free", "enterprise"}


def test_query_contradictive_same_key_and_returns_empty():
    # Two EQ filters on the same key with different values, joined by
    # AND, must return zero matches. Confirms AND is genuinely
    # conjunctive over same-key duplicates rather than accidentally
    # last-wins / OR.
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ),
            MetadataFilter(key="tier", value="free", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.AND,
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    assert store.query(q).nodes == []


def test_query_returns_results_sorted_by_similarity():
    # Top-1 of a self-query (query embedding == one of the stored
    # embeddings) must be that exact node. Quantization noise is small
    # enough that the self-match wins, and this is the only guard
    # against a regression that returns hits in insertion order.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [_make_node(f"doc-{i}", seed=i) for i in range(5)]
    ids = store.add(nodes)
    # Query with the embedding of node index 3.
    q = VectorStoreQuery(query_embedding=_unit_vec(3, 64), similarity_top_k=5)
    result = store.query(q)
    assert result.ids[0] == ids[3]
    # Similarities must be monotonically non-increasing (top-k contract).
    sims = result.similarities
    assert all(a >= b for a, b in zip(sims, sims[1:]))


def test_query_filter_condition_not_negates_inner_match():
    # Reference (`utils.py:187-189`): NOT matches when NONE of the inner
    # filters match. With a single inner EQ filter, NOT is just negation.
    store = _store_with_tiered_nodes()
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ)
        ],
        condition=FilterCondition.NOT,
    )
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64), similarity_top_k=10, filters=filters
    )
    tiers = {n.metadata["tier"] for n in store.query(q).nodes}
    assert "pro" not in tiers
    assert tiers == {"free", "enterprise"}


# ------------------- Tier 1: protocol completeness -----------------------

def test_get_raises_with_explanation():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [_make_node("a", seed=1)]
    ids = store.add(nodes)
    with pytest.raises(NotImplementedError, match="quantiz"):
        store.get(ids[0])


def test_get_nodes_by_node_ids():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [_make_node(f"doc {i}", seed=i) for i in range(3)]
    ids = store.add(nodes)
    fetched = store.get_nodes(node_ids=[ids[0], ids[2]])
    assert {n.node_id for n in fetched} == {ids[0], ids[2]}
    # Missing ids are silently skipped, matching SimpleVectorStore-ish convention.
    fetched = store.get_nodes(node_ids=[ids[0], "nonexistent"])
    assert {n.node_id for n in fetched} == {ids[0]}


def test_get_nodes_by_filters():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node(f"doc {i}", seed=i, metadata={"tier": "pro" if i % 2 else "free"})
        for i in range(4)
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ)]
    )
    fetched = store.get_nodes(filters=filters)
    assert len(fetched) == 2
    assert all(n.metadata["tier"] == "pro" for n in fetched)


def test_get_nodes_intersects_node_ids_and_filters():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node(f"doc {i}", seed=i, metadata={"tier": "pro" if i % 2 else "free"})
        for i in range(4)
    ]
    ids = store.add(nodes)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ)]
    )
    fetched = store.get_nodes(node_ids=[ids[0], ids[1]], filters=filters)
    # tier==pro is odd indices → {ids[1], ids[3]}. Intersect with {ids[0], ids[1]} → {ids[1]}.
    assert [n.node_id for n in fetched] == [ids[1]]


def test_get_nodes_empty_filter_returns_all():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [_make_node(f"doc {i}", seed=i) for i in range(3)]
    store.add(nodes)
    fetched = store.get_nodes()
    assert len(fetched) == 3


def test_clear_resets_store():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=2)
    store.add([_make_node(f"doc {i}", seed=i) for i in range(3)])
    assert len(store._nodes) == 3
    store.clear()
    assert len(store._nodes) == 0
    assert len(store._index) == 0
    # bit_width is preserved across clear; dim resets to lazy.
    assert store._index.bit_width == 2
    assert store._index.dim is None
    # The cleared store is still usable.
    store.add([_make_node("post-clear", seed=0)])
    assert len(store._nodes) == 1


def test_delete_nodes_with_filters():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    nodes = [
        _make_node(f"doc {i}", seed=i, metadata={"tier": "pro" if i % 2 else "free"})
        for i in range(4)
    ]
    store.add(nodes)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ)]
    )
    store.delete_nodes(filters=filters)
    assert len(store._nodes) == 2
    assert all(data["metadata"]["tier"] == "free" for data in store._nodes.values())


def test_delete_nodes_no_args_is_noop():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("a", seed=1)])
    store.delete_nodes()
    assert len(store._nodes) == 1


def test_to_dict_from_dict_roundtrip():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=2)
    cfg = store.to_dict()
    assert cfg["bit_width"] == 2
    assert cfg["dim"] == 64
    restored = TurboQuantVectorStore.from_dict(cfg)
    assert restored._index.dim == 64
    assert restored._index.bit_width == 2


def test_to_dict_from_dict_lazy_store():
    store = TurboQuantVectorStore(bit_width=2)
    cfg = store.to_dict()
    assert cfg["dim"] is None
    restored = TurboQuantVectorStore.from_dict(cfg)
    assert restored._index.dim is None
    assert restored._index.bit_width == 2


# ------------------- Tier 2: async overrides -----------------------------

def test_async_add_query_delete_clear_get():
    import asyncio

    async def runner():
        store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
        nodes = [_make_node(f"doc {i}", seed=i) for i in range(3)]
        await store.async_add(nodes)
        result = await store.aquery(
            VectorStoreQuery(query_embedding=_unit_vec(0, 64), similarity_top_k=3)
        )
        assert len(result.nodes) == 3
        fetched = await store.aget_nodes(node_ids=[n.node_id for n in nodes[:2]])
        assert len(fetched) == 2
        await store.adelete_nodes(node_ids=[nodes[0].node_id])
        assert len(store._nodes) == 2
        await store.aclear()
        assert len(store._nodes) == 0

    asyncio.run(runner())


def test_async_adelete_by_ref_doc_id():
    import asyncio

    async def runner():
        store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
        store.add([
            _make_node("a", seed=1, ref_doc_id="parent"),
            _make_node("b", seed=2, ref_doc_id="parent"),
            _make_node("c", seed=3, ref_doc_id="other"),
        ])
        await store.adelete("parent")
        assert len(store._nodes) == 1

    asyncio.run(runner())


# ------------------- End-to-end smoke tests: framework wiring -----------

def test_vector_store_index_from_vector_store_retrieve():
    # Smoke test: build a full VectorStoreIndex on top of our store and
    # retrieve through it. This is the canonical way users plug a vector
    # store into a LlamaIndex pipeline (RAG, query engine, etc.), so it
    # exercises the index/retriever glue that calls query() on us from
    # the framework side.
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode as LITextNode

    embed_model = MockEmbedding(embed_dim=64)
    store = TurboQuantVectorStore.from_params(bit_width=4)

    # Build nodes with embeddings (MockEmbedding is deterministic).
    nodes = []
    for i, text in enumerate(["alpha doc", "beta doc", "gamma doc"]):
        node = LITextNode(text=text, id_=f"n-{i}")
        node.embedding = embed_model.get_text_embedding(text)
        nodes.append(node)
    store.add(nodes)

    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=2)
    results = retriever.retrieve("alpha")
    assert len(results) == 2
    # Returned NodeWithScore objects wrap our TextNodes.
    assert all(r.score is not None for r in results)
    contents = {r.node.get_content() for r in results}
    assert contents.issubset({"alpha doc", "beta doc", "gamma doc"})


def test_storage_context_persist_and_load_roundtrip(tmp_path):
    # Smoke test: persist via StorageContext, reload via StorageContext.
    # This is the path real LlamaIndex users follow, and it depends on
    # our from_persist_dir + persist signature matching the framework's
    # expectations.
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode as LITextNode

    embed_model = MockEmbedding(embed_dim=64)
    persist_dir = tmp_path / "storage"

    # Build, persist.
    store = TurboQuantVectorStore.from_params(bit_width=4)
    nodes = []
    for i, text in enumerate(["one", "two", "three"]):
        node = LITextNode(text=text, id_=f"n-{i}")
        node.embedding = embed_model.get_text_embedding(text)
        nodes.append(node)
    store.add(nodes)

    storage_context = StorageContext.from_defaults(vector_store=store)
    storage_context.persist(persist_dir=str(persist_dir))

    # Reload via the from_persist_dir path.
    reloaded_store = TurboQuantVectorStore.from_persist_dir(str(persist_dir))
    assert len(reloaded_store._nodes) == 3
    # Original query still works after reload.
    storage_context2 = StorageContext.from_defaults(
        vector_store=reloaded_store, persist_dir=str(persist_dir),
    )
    index = VectorStoreIndex(
        nodes=[], storage_context=storage_context2, embed_model=embed_model,
    )
    retriever = index.as_retriever(similarity_top_k=2)
    results = retriever.retrieve("one")
    assert len(results) == 2


# ---- Full-node fidelity round-trip (covers the langchain-class bug
# pattern at the structural level: every field SimpleVectorStore would
# preserve through write -> query / get_nodes / persist+load must come
# back populated, not a bare {id, text, metadata} TextNode shell). ----

def _make_rich_node(
    text: str,
    seed: int,
    *,
    node_id: str | None = None,
    metadata: dict | None = None,
    excluded_embed_metadata_keys: list[str] | None = None,
    excluded_llm_metadata_keys: list[str] | None = None,
    relationships: dict | None = None,
    start_char_idx: int | None = None,
    end_char_idx: int | None = None,
    metadata_template: str | None = None,
    metadata_separator: str | None = None,
    text_template: str | None = None,
    mimetype: str | None = None,
) -> TextNode:
    """A TextNode with every framework field set to a non-default value so
    we can assert each one survives the write -> retrieve round-trip."""
    kwargs: dict = {
        "text": text,
        "metadata": metadata or {},
        "embedding": _unit_vec(seed, 64),
        "excluded_embed_metadata_keys": excluded_embed_metadata_keys or [],
        "excluded_llm_metadata_keys": excluded_llm_metadata_keys or [],
        "relationships": relationships or {},
    }
    if node_id is not None:
        kwargs["id_"] = node_id
    if start_char_idx is not None:
        kwargs["start_char_idx"] = start_char_idx
    if end_char_idx is not None:
        kwargs["end_char_idx"] = end_char_idx
    if metadata_template is not None:
        kwargs["metadata_template"] = metadata_template
    if metadata_separator is not None:
        kwargs["metadata_separator"] = metadata_separator
    if text_template is not None:
        kwargs["text_template"] = text_template
    if mimetype is not None:
        kwargs["mimetype"] = mimetype
    return TextNode(**kwargs)


def test_query_returns_node_with_full_field_fidelity():
    # Every field SimpleVectorStore preserves through query must survive
    # turbovec's write -> query round-trip. The previous narrow side-car
    # schema lost relationships (PREVIOUS/NEXT/PARENT/CHILD),
    # excluded_*_metadata_keys, template fields, char_idx, and mimetype.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    node = _make_rich_node(
        "chunk text",
        seed=1,
        node_id="node-rich-1",
        metadata={"page": 7, "section": "intro"},
        excluded_embed_metadata_keys=["section"],
        excluded_llm_metadata_keys=["page"],
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1"),
            NodeRelationship.PREVIOUS: RelatedNodeInfo(node_id="node-prev"),
            NodeRelationship.NEXT: RelatedNodeInfo(node_id="node-next"),
            NodeRelationship.PARENT: RelatedNodeInfo(node_id="node-parent"),
        },
        start_char_idx=100,
        end_char_idx=200,
        metadata_template="<<{key}::{value}>>",
        metadata_separator=" | ",
        text_template="META:{metadata_str}\nBODY:{content}",
        mimetype="text/markdown",
    )
    store.add([node])

    result = store.query(
        VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=1)
    )
    [returned] = result.nodes

    assert returned.node_id == "node-rich-1"
    assert returned.get_content() == "chunk text"
    assert returned.metadata == {"page": 7, "section": "intro"}
    assert returned.excluded_embed_metadata_keys == ["section"]
    assert returned.excluded_llm_metadata_keys == ["page"]
    assert returned.start_char_idx == 100
    assert returned.end_char_idx == 200
    assert returned.metadata_template == "<<{key}::{value}>>"
    assert returned.metadata_separator == " | "
    assert returned.text_template == "META:{metadata_str}\nBODY:{content}"
    assert returned.mimetype == "text/markdown"
    # All four relationships should be present, not just SOURCE.
    assert returned.relationships[NodeRelationship.SOURCE].node_id == "doc-1"
    assert returned.relationships[NodeRelationship.PREVIOUS].node_id == "node-prev"
    assert returned.relationships[NodeRelationship.NEXT].node_id == "node-next"
    assert returned.relationships[NodeRelationship.PARENT].node_id == "node-parent"


def test_get_nodes_returns_full_field_fidelity():
    # Same fidelity contract for get_nodes (not just query).
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    node = _make_rich_node(
        "chunk",
        seed=2,
        node_id="g-1",
        metadata={"k": "v"},
        excluded_embed_metadata_keys=["k"],
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc"),
            NodeRelationship.NEXT: RelatedNodeInfo(node_id="g-2"),
        },
        start_char_idx=0,
        end_char_idx=5,
        mimetype="application/json",
    )
    store.add([node])

    [fetched] = store.get_nodes(node_ids=["g-1"])
    assert fetched.relationships[NodeRelationship.NEXT].node_id == "g-2"
    assert fetched.excluded_embed_metadata_keys == ["k"]
    assert fetched.mimetype == "application/json"
    assert fetched.start_char_idx == 0
    assert fetched.end_char_idx == 5


def test_persist_round_trip_preserves_full_node_fidelity(tmp_path):
    # Save + load should not be lossier than query — the on-disk schema
    # must carry the full _node_content blob (v2+).
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    node = _make_rich_node(
        "persisted chunk",
        seed=3,
        node_id="p-1",
        metadata={"k": "v"},
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc"),
            NodeRelationship.PREVIOUS: RelatedNodeInfo(node_id="p-0"),
        },
        excluded_llm_metadata_keys=["k"],
        text_template="T:{content}",
    )
    store.add([node])
    persist_path = tmp_path / "store.json"
    store.persist(str(persist_path))

    loaded = TurboQuantVectorStore.from_persist_path(str(persist_path))
    [restored] = loaded.get_nodes(node_ids=["p-1"])
    assert restored.relationships[NodeRelationship.PREVIOUS].node_id == "p-0"
    assert restored.excluded_llm_metadata_keys == ["k"]
    assert restored.text_template == "T:{content}"


def test_persist_v1_schema_still_loads_with_narrow_fidelity(tmp_path):
    # A nodes.json written by an older turbovec (schema_version=1, narrow
    # `{text, metadata, ref_doc_id}` per entry) must still load — relationships
    # other than SOURCE will be missing (the data wasn't there), but the
    # text + metadata + SOURCE relationship round-trip as they always did.
    import json

    # Build a v1-shaped side-car by hand. We still need the binary index
    # file alongside it, so write a v2 store first and overwrite just
    # the JSON.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    node = TextNode(
        id_="v1-node",
        text="legacy text",
        metadata={"k": "v"},
        embedding=_unit_vec(0, 64),
        relationships={
            NodeRelationship.SOURCE: RelatedNodeInfo(node_id="legacy-doc"),
        },
    )
    store.add([node])
    persist_path = tmp_path / "store.json"
    store.persist(str(persist_path))

    nodes_path = persist_path.with_suffix(".nodes.json")
    with open(nodes_path) as f:
        state = json.load(f)
    state["schema_version"] = 1
    # Rewrite each entry in the v1 shape: {text, metadata, ref_doc_id} only.
    state["nodes"] = {
        nid: {
            "text": "legacy text",
            "metadata": {"k": "v"},
            "ref_doc_id": "legacy-doc",
        }
        for nid in state["nodes"]
    }
    with open(nodes_path, "w") as f:
        json.dump(state, f)

    loaded = TurboQuantVectorStore.from_persist_path(str(persist_path))
    [restored] = loaded.get_nodes(node_ids=["v1-node"])
    assert restored.get_content() == "legacy text"
    assert restored.metadata == {"k": "v"}
    assert restored.ref_doc_id == "legacy-doc"


def test_query_rejects_non_default_mode():
    # MMR / SVM / hybrid all need full-precision vectors which turbovec
    # discards. Raise loudly instead of silently treating as DEFAULT
    # (which the previous impl did, masking that the requested mode
    # wasn't honoured).
    from llama_index.core.vector_stores.types import VectorStoreQueryMode

    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("a", seed=0)])
    q = VectorStoreQuery(
        query_embedding=_unit_vec(0, 64),
        similarity_top_k=1,
        mode=VectorStoreQueryMode.MMR,
    )
    with pytest.raises(NotImplementedError, match="query mode"):
        store.query(q)


# ---- Tier-2 field-completeness tests. ----

def test_add_raises_on_intra_batch_duplicate_node_id():
    # If a single add() call contains two nodes with the same node_id,
    # the prior impl would leave the index with both vectors but only
    # the last node_id mapped back to one — orphaning the first handle
    # and silently returning the second node's payload attached to the
    # first node's vector on later queries. Now raises loudly.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    n1 = TextNode(id_="dup", text="first", embedding=_unit_vec(1, 64))
    n2 = TextNode(id_="dup", text="second", embedding=_unit_vec(2, 64))
    with pytest.raises(ValueError, match="duplicate node_id"):
        store.add([n1, n2])
    # And the store is left in a clean state — no half-written index.
    assert len(store._index) == 0
    assert store._nodes == {}


def test_query_round_trips_image_node_subtype():
    # PR #83 covered TextNode field fidelity exhaustively, but every
    # test uses TextNode. node_to_metadata_dict / metadata_dict_to_node
    # also handle ImageNode — pin that explicitly so a regression in
    # the helper dispatch (or a switch back to bare TextNode
    # reconstruction) gets caught.
    from llama_index.core.schema import ImageNode

    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    image_node = ImageNode(
        id_="img-1",
        text="caption",
        image_url="https://example.com/img.png",
        image_mimetype="image/png",
        embedding=_unit_vec(1, 64),
        metadata={"source": "test"},
    )
    store.add([image_node])

    result = store.query(
        VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=1)
    )
    [returned] = result.nodes
    assert isinstance(returned, ImageNode)
    assert returned.id_ == "img-1"
    assert returned.image_url == "https://example.com/img.png"
    assert returned.image_mimetype == "image/png"
    assert returned.metadata == {"source": "test"}


def test_query_round_trips_index_node_subtype():
    # Same fidelity check for IndexNode (used by composable indexes /
    # routers / sub-question pipelines).
    from llama_index.core.schema import IndexNode

    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    index_node = IndexNode(
        id_="idx-1",
        text="sub-index pointer",
        index_id="child-index-uuid",
        embedding=_unit_vec(1, 64),
        metadata={"router": "yes"},
    )
    store.add([index_node])

    result = store.query(
        VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=1)
    )
    [returned] = result.nodes
    assert isinstance(returned, IndexNode)
    assert returned.id_ == "idx-1"
    assert returned.index_id == "child-index-uuid"
    assert returned.metadata == {"router": "yes"}


def test_query_returned_node_always_has_none_embedding():
    # turbovec discards full-precision embeddings after quantization.
    # `get()` raises NotImplementedError to communicate this; `query`
    # and `get_nodes` honour the same contract by returning nodes with
    # `embedding=None` even when the input node had an embedding set.
    # Pin this so a future refactor doesn't silently start round-
    # tripping the raw float list and contradict the `get()` story.
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([
        _make_node("a", seed=1, metadata={"k": "v"}),
        _make_node("b", seed=2, metadata={"k": "w"}),
    ])

    qresult = store.query(
        VectorStoreQuery(query_embedding=_unit_vec(1, 64), similarity_top_k=2)
    )
    for node in qresult.nodes:
        assert node.embedding is None

    for node in store.get_nodes():
        assert node.embedding is None
