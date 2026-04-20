from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from turbovec import IdMapIndex
from turbovec.langchain import TurboQuantVectorStore


class StubEmbeddings(Embeddings):
    """Deterministic text->vector function for tests.

    Hashes the input string to seed an RNG, producing a reproducible
    unit-norm vector. Similar strings do not map to similar vectors —
    that's fine for structural tests, and callers shouldn't rely on
    semantic ordering here.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v.tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_from_texts_infers_dim_and_indexes():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["apple", "banana", "cherry", "date"], emb, bit_width=4
    )
    assert len(store._str_to_u64) == 4
    assert store._index.dim == 64
    assert store._index.bit_width == 4


def test_similarity_search_returns_documents():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["a", "b", "c"], emb, bit_width=4)
    results = store.similarity_search("a", k=2)
    assert len(results) == 2
    assert all(isinstance(r, Document) for r in results)


def test_metadata_roundtrip():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["hello", "world"],
        emb,
        metadatas=[{"source": "a"}, {"source": "b"}],
        bit_width=4,
    )
    scored = store.similarity_search_with_score("hello", k=2)
    assert len(scored) == 2
    sources = {doc.metadata["source"] for doc, _ in scored}
    assert sources == {"a", "b"}


def test_add_texts_uses_provided_ids():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts([], emb, dim=64, bit_width=4)
    returned = store.add_texts(["x", "y"], ids=["id-x", "id-y"])
    assert returned == ["id-x", "id-y"]
    assert set(store._docs.keys()) == {"id-x", "id-y"}


def test_k_larger_than_ntotal_is_clamped():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["one", "two"], emb, bit_width=4)
    results = store.similarity_search("one", k=100)
    assert len(results) == 2


def test_empty_store_search_returns_empty():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts([], emb, dim=64, bit_width=4)
    assert store.similarity_search("anything", k=5) == []


def test_save_and_load_roundtrip(tmp_path):
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["one", "two", "three"],
        emb,
        metadatas=[{"n": 1}, {"n": 2}, {"n": 3}],
        bit_width=4,
    )
    store.save_local(tmp_path)

    loaded = TurboQuantVectorStore.load_local(
        tmp_path, emb, allow_dangerous_deserialization=True
    )
    assert len(loaded._docs) == 3
    results = loaded.similarity_search("one", k=3)
    assert {doc.page_content for doc in results} == {"one", "two", "three"}


def test_load_local_refuses_without_flag(tmp_path):
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["x"], emb, bit_width=4)
    store.save_local(tmp_path)
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
        TurboQuantVectorStore.load_local(tmp_path, emb)


def test_delete_removes_documents_and_returns_true():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["apple", "banana", "cherry"],
        emb,
        ids=["a", "b", "c"],
        bit_width=4,
    )
    assert store.delete(["b"]) is True
    assert set(store._docs.keys()) == {"a", "c"}
    assert len(store._index) == 2


def test_delete_returns_false_when_any_id_missing():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b"], emb, ids=["id-a", "id-b"], bit_width=4
    )
    assert store.delete(["id-a", "ghost"]) is False
    # The existing id was still removed.
    assert "id-a" not in store._docs


def test_delete_none_ids_raises():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["x"], emb, bit_width=4)
    with pytest.raises(ValueError, match="explicit list of ids"):
        store.delete(None)


def test_add_texts_upsert_replaces_existing_id():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["v1"], emb, ids=["same-id"], bit_width=4
    )
    # Re-add with the same id but different text.
    store.add_texts(["v2"], ids=["same-id"])
    assert len(store._docs) == 1
    assert store._docs["same-id"][0] == "v2"


def test_mismatched_dim_raises():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb, index=IdMapIndex(32, 4))
    with pytest.raises(ValueError, match="embedding dimension"):
        store.add_texts(["hi"])
