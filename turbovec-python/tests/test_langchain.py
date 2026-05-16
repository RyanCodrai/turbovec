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


def test_similarity_search_with_dict_filter():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        emb,
        metadatas=[
            {"tier": "free"},
            {"tier": "pro"},
            {"tier": "free"},
            {"tier": "pro"},
            {"tier": "pro"},
        ],
        bit_width=4,
    )
    results = store.similarity_search("alpha", k=10, filter={"tier": "pro"})
    assert len(results) == 3
    assert all(r.metadata["tier"] == "pro" for r in results)


def test_similarity_search_with_callable_filter():
    # Predicate receives a langchain_core Document (matching the in-tree
    # InMemoryVectorStore convention), not a bare metadata dict.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c", "d"],
        emb,
        metadatas=[{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}],
        bit_width=4,
    )
    results = store.similarity_search(
        "a", k=10, filter=lambda doc: doc.metadata.get("n", 0) > 2
    )
    assert {r.metadata["n"] for r in results} == {3, 4}


def test_similarity_search_callable_filter_can_use_page_content():
    # Document is passed to the predicate, so page_content is reachable.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["alpha", "beta", "alphabet"], emb, bit_width=4,
    )
    results = store.similarity_search(
        "alpha", k=10, filter=lambda doc: doc.page_content.startswith("alpha")
    )
    contents = {r.page_content for r in results}
    assert contents == {"alpha", "alphabet"}


def test_similarity_search_filter_with_scores():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c"],
        emb,
        metadatas=[{"k": 1}, {"k": 2}, {"k": 1}],
        bit_width=4,
    )
    results = store.similarity_search_with_score("a", k=10, filter={"k": 1})
    assert len(results) == 2
    for doc, score in results:
        assert doc.metadata["k"] == 1
        assert isinstance(score, float)


def test_similarity_search_filter_no_matches_returns_empty():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b"], emb, metadatas=[{"k": 1}, {"k": 2}], bit_width=4
    )
    assert store.similarity_search("a", k=5, filter={"k": 999}) == []


def test_similarity_search_filter_invalid_type_raises():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["a"], emb, bit_width=4)
    with pytest.raises(TypeError):
        store.similarity_search("a", k=1, filter=42)


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
    store = TurboQuantVectorStore.from_texts([], emb, bit_width=4)
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
    store = TurboQuantVectorStore.from_texts([], emb, bit_width=4)
    assert store.similarity_search("anything", k=5) == []


def test_dump_and_load_roundtrip(tmp_path):
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["one", "two", "three"],
        emb,
        metadatas=[{"n": 1}, {"n": 2}, {"n": 3}],
        bit_width=4,
    )
    store.dump(tmp_path)

    loaded = TurboQuantVectorStore.load(
        tmp_path, emb, allow_dangerous_deserialization=True
    )
    assert len(loaded._docs) == 3
    results = loaded.similarity_search("one", k=3)
    assert {doc.page_content for doc in results} == {"one", "two", "three"}


def test_load_refuses_without_flag(tmp_path):
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["x"], emb, bit_width=4)
    store.dump(tmp_path)
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
        TurboQuantVectorStore.load(tmp_path, emb)


def test_delete_removes_documents_returns_none():
    # Match InMemoryVectorStore convention: delete returns None.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["apple", "banana", "cherry"],
        emb,
        ids=["a", "b", "c"],
        bit_width=4,
    )
    result = store.delete(["b"])
    assert result is None
    assert set(store._docs.keys()) == {"a", "c"}
    assert len(store._index) == 2


def test_delete_missing_ids_silently_skips():
    # Match InMemoryVectorStore convention: missing ids are silently
    # skipped, no error.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b"], emb, ids=["id-a", "id-b"], bit_width=4
    )
    assert store.delete(["id-a", "ghost"]) is None
    assert "id-a" not in store._docs
    assert "id-b" in store._docs


def test_delete_none_ids_is_noop():
    # InMemoryVectorStore treats `delete(None)` as a no-op rather than
    # raising. Match that.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["x"], emb, bit_width=4)
    assert store.delete(None) is None
    assert "x" not in store._docs  # uuid-based ids, but the store has 1 doc
    assert len(store._docs) == 1


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


# ---- Lazy index construction (Tier 4) -------------------------------------

def test_constructor_no_index_is_lazy():
    # Without an `index`, the store can be constructed; the underlying
    # IdMapIndex is created on the first add.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb)
    assert store._index is None
    # Search before any add returns empty rather than raising.
    assert store.similarity_search("anything", k=3) == []


def test_lazy_index_created_on_first_add():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb, bit_width=2)
    store.add_texts(["hello"])
    assert store._index is not None
    assert store._index.dim == 64
    assert store._index.bit_width == 2


def test_from_texts_no_dim_arg_required():
    # Tier 4: dim is inferred from the embedding model, no explicit param.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["one", "two"], emb, bit_width=4
    )
    assert store._index.dim == 64


# ---- get_by_ids -----------------------------------------------------------

def test_get_by_ids_returns_documents():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c"], emb,
        metadatas=[{"n": 1}, {"n": 2}, {"n": 3}],
        ids=["id-a", "id-b", "id-c"],
        bit_width=4,
    )
    docs = store.get_by_ids(["id-a", "id-c"])
    assert {d.id for d in docs} == {"id-a", "id-c"}
    assert {d.metadata["n"] for d in docs} == {1, 3}


def test_get_by_ids_silently_skips_missing():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a"], emb, ids=["id-a"], bit_width=4
    )
    docs = store.get_by_ids(["id-a", "id-missing"])
    assert len(docs) == 1
    assert docs[0].id == "id-a"


# ---- Relevance score normalization ----------------------------------------

def test_select_relevance_score_fn_maps_to_unit_interval():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["hello"], emb, bit_width=4)
    fn = store._select_relevance_score_fn()
    # Cosine similarity in [-1, 1] → relevance in [0, 1].
    assert fn(-1.0) == 0.0
    assert fn(0.0) == 0.5
    assert fn(1.0) == 1.0


def test_similarity_search_with_relevance_scores_in_zero_one():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["one", "two", "three"], emb, bit_width=4
    )
    results = store.similarity_search_with_relevance_scores("one", k=3)
    assert len(results) == 3
    for _doc, score in results:
        assert 0.0 <= score <= 1.0


# ---- MMR raises with explanation ------------------------------------------

def test_max_marginal_relevance_search_raises_with_message():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["a", "b"], emb, bit_width=4)
    with pytest.raises(NotImplementedError, match="full-precision"):
        store.max_marginal_relevance_search("a", k=2)


def test_max_marginal_relevance_search_by_vector_raises():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["a", "b"], emb, bit_width=4)
    with pytest.raises(NotImplementedError, match="full-precision"):
        store.max_marginal_relevance_search_by_vector(emb._embed("a"), k=2)


# ---- add_documents partial-id support ------------------------------------

def test_add_documents_honors_partial_ids():
    # Tier 3: per-Document fallback — Documents with .id set keep their
    # id, others get a UUID. Base-class default (without override) would
    # drop all ids if any Document had .id=None.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb)
    docs = [
        Document(id="explicit-1", page_content="a"),
        Document(page_content="b"),  # no id → UUID
        Document(id="explicit-2", page_content="c"),
    ]
    returned_ids = store.add_documents(docs)
    assert "explicit-1" in returned_ids
    assert "explicit-2" in returned_ids
    # The UUID-generated id is some non-explicit string.
    uuid_id = [i for i in returned_ids if i not in ("explicit-1", "explicit-2")]
    assert len(uuid_id) == 1


# ---- Async surfaces -------------------------------------------------------

def test_async_add_search_delete():
    import asyncio

    async def runner():
        emb = StubEmbeddings(dim=64)
        store = TurboQuantVectorStore(emb)
        ids = await store.aadd_texts(["alpha", "beta", "gamma"])
        assert len(ids) == 3
        results = await store.asimilarity_search("alpha", k=2)
        assert len(results) == 2
        scored = await store.asimilarity_search_with_score("alpha", k=2)
        assert len(scored) == 2 and isinstance(scored[0][1], float)
        by_vec = await store.asimilarity_search_by_vector(emb._embed("alpha"), k=1)
        assert len(by_vec) == 1
        got = await store.aget_by_ids(ids)
        assert len(got) == 3
        await store.adelete([ids[0]])
        assert ids[0] not in store._docs

    asyncio.run(runner())


def test_async_add_documents_and_afrom_texts():
    import asyncio

    async def runner():
        emb = StubEmbeddings(dim=64)
        store = await TurboQuantVectorStore.afrom_texts(
            ["x", "y"], emb, bit_width=4
        )
        assert len(store._docs) == 2
        await store.aadd_documents([Document(page_content="z")])
        assert len(store._docs) == 3

    asyncio.run(runner())


def test_async_mmr_raises():
    import asyncio

    async def runner():
        emb = StubEmbeddings(dim=64)
        store = await TurboQuantVectorStore.afrom_texts(["x"], emb, bit_width=4)
        with pytest.raises(NotImplementedError, match="full-precision"):
            await store.amax_marginal_relevance_search("x", k=1)

    asyncio.run(runner())


# ---- Empty-store persistence round-trip (lazy index) ---------------------

def test_dump_and_load_empty_store(tmp_path):
    # When no documents have been added the underlying IdMapIndex doesn't
    # exist yet. dump/load must handle that without writing a phantom
    # index file or crashing on load.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb, bit_width=2)
    store.dump(tmp_path)
    loaded = TurboQuantVectorStore.load(
        tmp_path, emb, allow_dangerous_deserialization=True
    )
    assert loaded._index is None
    assert loaded._bit_width == 2
    # Subsequent search returns empty; subsequent add creates the index.
    assert loaded.similarity_search("anything", k=1) == []
    loaded.add_texts(["new"])
    assert loaded._index is not None
