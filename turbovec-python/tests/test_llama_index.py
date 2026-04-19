from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from turbovec import TurboQuantIndex
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
    store = TurboQuantVectorStore(index=TurboQuantIndex(32, 4))
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


def test_delete_not_supported():
    store = TurboQuantVectorStore.from_params(dim=64, bit_width=4)
    store.add([_make_node("x", seed=1, ref_doc_id="parent-1")])
    with pytest.raises(NotImplementedError):
        store.delete("parent-1")
