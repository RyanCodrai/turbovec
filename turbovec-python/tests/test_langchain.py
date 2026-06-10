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


def test_similarity_search_results_carry_document_id():
    # Returned Documents must have `.id` populated — both the explicit ids
    # callers pass to `add_texts` and the UUIDs the store generates by
    # default. Matches the InMemoryVectorStore reference behaviour and what
    # downstream LangChain callers (retrievers, chains) expect.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c"], emb, bit_width=4, ids=["id-a", "id-b", "id-c"]
    )

    results = store.similarity_search("a", k=3)
    assert {r.id for r in results} == {"id-a", "id-b", "id-c"}

    scored = store.similarity_search_with_score("a", k=3)
    assert {doc.id for doc, _ in scored} == {"id-a", "id-b", "id-c"}

    by_vec = store.similarity_search_by_vector(emb._embed("a"), k=3)
    assert {r.id for r in by_vec} == {"id-a", "id-b", "id-c"}


def test_similarity_search_callable_filter_receives_document_id():
    # Predicate Document must carry `.id` so callers can filter on it,
    # not just on page_content / metadata.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c"], emb, bit_width=4, ids=["keep-1", "drop", "keep-2"]
    )
    results = store.similarity_search(
        "a", k=10, filter=lambda doc: doc.id.startswith("keep")
    )
    assert {r.id for r in results} == {"keep-1", "keep-2"}


# ---- Reference-parity tests against InMemoryVectorStore. Each pins a
# behaviour the langchain-core in-tree suite tests; the bug class is
# "drop-in regression that only shows up when users compare against
# InMemoryVectorStore". ----

def test_async_methods_await_aembed_functions():
    # If our async paths silently fall back to sync embedding, callers
    # with a real async embedder (e.g. OpenAI async client) end up
    # blocking the event loop. AsyncMock makes the side-effect visible:
    # if aembed_documents / aembed_query weren't awaited, await_count
    # stays zero.
    import asyncio
    from unittest.mock import AsyncMock

    class AsyncStub(StubEmbeddings):
        def __init__(self, dim: int = 64) -> None:
            super().__init__(dim)
            self.aembed_documents = AsyncMock(
                side_effect=lambda texts: [self._embed(t) for t in texts]
            )
            self.aembed_query = AsyncMock(
                side_effect=lambda text: self._embed(text)
            )

    emb = AsyncStub(dim=64)
    store = TurboQuantVectorStore(embedding=emb)

    async def run() -> None:
        await store.aadd_texts(["a", "b"])
        await store.asimilarity_search("a", k=1)

    asyncio.run(run())

    assert emb.aembed_documents.await_count >= 1
    assert emb.aembed_query.await_count >= 1


def test_add_documents_upsert_replaces_metadata():
    # Re-adding a Document with the same id and new metadata must let
    # the new metadata win — not silently retain the old one. Reference
    # `InMemoryVectorStore.add_documents` overwrites on duplicate id.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts([], emb, bit_width=4)
    store.add_documents([Document(id="x", page_content="v1", metadata={"tag": "old"})])
    store.add_documents([Document(id="x", page_content="v2", metadata={"tag": "new"})])

    [doc] = store.get_by_ids(["x"])
    assert doc.metadata == {"tag": "new"}
    assert doc.page_content == "v2"


def test_add_documents_does_not_mutate_inputs():
    # Caller-passed Documents must not be mutated by add_documents — no
    # in-place .id assignment, no metadata mutation. Reference behaviour;
    # easy regression target when refactoring the add path.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts([], emb, bit_width=4)
    docs = [
        Document(page_content="a", metadata={"k": 1}),
        Document(id="explicit", page_content="b", metadata={"k": 2}),
    ]
    original_meta_ids = [id(d.metadata) for d in docs]
    store.add_documents(docs)

    assert docs[0].id is None
    assert docs[1].id == "explicit"
    assert docs[0].metadata == {"k": 1}
    assert docs[1].metadata == {"k": 2}
    # Same dict objects — we didn't replace caller-provided metadata dicts.
    assert [id(d.metadata) for d in docs] == original_meta_ids


def test_add_documents_with_ids_is_idempotent():
    # Re-running ingestion on an unchanged corpus must not accrete
    # duplicates. Reference upserts on same id; size stays constant.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts([], emb, bit_width=4)
    docs = [
        Document(id="a", page_content="hello"),
        Document(id="b", page_content="world"),
    ]
    store.add_documents(docs)
    store.add_documents(docs)

    assert len(store._docs) == 2
    assert set(store._docs.keys()) == {"a", "b"}


def test_add_texts_intra_batch_duplicate_ids_keep_last():
    # Two rows sharing an id in a single call must not orphan a vector.
    # Reference InMemoryVectorStore overwrites on a repeated id (last wins);
    # we dedup the batch the same way so the index holds one vector per id.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts([], emb, bit_width=4)
    ret = store.add_texts(["alpha", "beta"], ids=["dup", "dup"])

    # Return value still mirrors the input (one entry per input text).
    assert ret == ["dup", "dup"]
    # No orphaned vector: index and id maps agree at one entry.
    assert len(store._index) == 1
    assert len(store._u64_to_str) == 1
    assert set(store._str_to_u64) == {"dup"}
    # Last occurrence wins.
    assert store._docs["dup"][0] == "beta"


def test_upsert_dim_mismatch_preserves_existing_data():
    # An upsert whose new embeddings fail validation must not destroy the
    # existing entry: the delete is deferred until the add succeeds.
    class VarDimEmbeddings(Embeddings):
        def __init__(self):
            self.call = 0

        def embed_documents(self, texts):
            self.call += 1
            dim = 64 if self.call == 1 else 32
            return [[float(i + 1)] * dim for i in range(len(texts))]

        def embed_query(self, t):
            return [1.0] * 64

    store = TurboQuantVectorStore(VarDimEmbeddings())
    store.add_texts(["hello"], ids=["my-id"])      # dim=64, OK
    with pytest.raises(ValueError):
        store.add_texts(["world"], ids=["my-id"])  # dim=32, must reject

    # Original data survives the failed upsert.
    assert "my-id" in store._docs
    assert store._docs["my-id"][0] == "hello"
    assert len(store._index) == 1
    assert len(store._u64_to_str) == 1


def test_get_by_ids_empty_input_and_order_preserved():
    # Two contract points the reference makes that our existing tests
    # don't pin: (1) empty input returns [] without erroring; (2) output
    # is in the order of input ids so callers can zip with parallel
    # arrays (e.g. scores from a separate retriever).
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c"], emb, bit_width=4, ids=["id-a", "id-b", "id-c"]
    )

    assert store.get_by_ids([]) == []

    docs = store.get_by_ids(["id-c", "id-a", "id-b"])
    assert [d.id for d in docs] == ["id-c", "id-a", "id-b"]


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

    loaded = TurboQuantVectorStore.load(tmp_path, emb)
    assert len(loaded._docs) == 3
    results = loaded.similarity_search("one", k=3)
    assert {doc.page_content for doc in results} == {"one", "two", "three"}


def test_dump_writes_json_sidecar(tmp_path):
    # Side-car is plain JSON. A reviewer auditing a turbovec-saved store
    # should be able to read it with a text editor.
    import json

    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["x"], emb, bit_width=4)
    store.dump(tmp_path)
    assert (tmp_path / "docstore.json").exists()
    assert not (tmp_path / "docstore.pkl").exists()
    with open(tmp_path / "docstore.json") as f:
        data = json.load(f)
    assert data["schema_version"] >= 1


def test_load_rejects_unknown_schema_version(tmp_path):
    import json

    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["x"], emb, bit_width=4)
    store.dump(tmp_path)
    with open(tmp_path / "docstore.json") as f:
        data = json.load(f)
    data["schema_version"] = 99
    with open(tmp_path / "docstore.json", "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="schema version"):
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
    # Without an `index`, the underlying IdMapIndex is constructed in its
    # lazy-uncommitted state — `dim` is None until the first add.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb)
    assert store._index.dim is None
    # Search before any add returns empty rather than raising.
    assert store.similarity_search("anything", k=3) == []


def test_lazy_index_dim_locked_on_first_add():
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb, bit_width=2)
    store.add_texts(["hello"])
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

# ---- End-to-end smoke tests: framework wiring ---------------------------

def test_as_retriever_invoke_returns_documents():
    # Smoke test: wire the store into LangChain's VectorStoreRetriever via
    # `as_retriever()` and run a query through the .invoke() interface.
    # This is the canonical way users plug a VectorStore into a Chain,
    # so it exercises the base-class wiring that calls similarity_search
    # on our store from the framework side.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["alpha", "beta", "gamma", "delta"],
        emb,
        metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "a"}, {"tag": "b"}],
        bit_width=4,
    )
    retriever = store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("alpha")
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)


def test_as_retriever_with_filter_kwarg():
    # The retriever passes search_kwargs (including `filter`) through to
    # similarity_search. This verifies the keyword reaches our store
    # without being dropped by the base class.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["alpha", "beta", "gamma"],
        emb,
        metadatas=[{"tag": "keep"}, {"tag": "drop"}, {"tag": "keep"}],
        bit_width=4,
    )
    retriever = store.as_retriever(
        search_kwargs={"k": 5, "filter": {"tag": "keep"}}
    )
    docs = retriever.invoke("alpha")
    assert len(docs) == 2
    assert all(d.metadata["tag"] == "keep" for d in docs)


def test_as_retriever_similarity_score_threshold():
    # `similarity_score_threshold` is the search_type that uses
    # similarity_search_with_relevance_scores under the hood, which
    # depends on our _select_relevance_score_fn override. If that's
    # missing or broken, this test fails with NotImplementedError.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["alpha", "beta", "gamma"], emb, bit_width=4
    )
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.0},
    )
    docs = retriever.invoke("alpha")
    # All scores should be >= threshold (relevance in [0, 1] >= 0).
    assert len(docs) >= 1


def test_dump_and_load_empty_store(tmp_path):
    # When no documents have been added the underlying IdMapIndex is in
    # its lazy-uncommitted state (dim=None). dump/load must round-trip
    # that without losing the bit_width or accidentally committing a dim.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb, bit_width=2)
    store.dump(tmp_path)
    loaded = TurboQuantVectorStore.load(tmp_path, emb)
    assert loaded._index.dim is None
    assert loaded._index.bit_width == 2
    # Subsequent search returns empty; subsequent add commits the dim.
    assert loaded.similarity_search("anything", k=1) == []
    loaded.add_texts(["new"])
    assert loaded._index.dim == 64


# ---- Tier-2 field-completeness tests. Each pins a value that a future
# refactor could silently drop (the #81 family: populated but
# unasserted). ----

def test_similarity_search_with_score_returns_descending_scores_and_self_match():
    # Pins the actual semantics of the float in (Document, float) — that
    # it's a real similarity score, that the self-match wins, and that
    # results are monotonically non-increasing. Without this, the tuple
    # position 1 could silently regress to a constant or get swapped
    # with `handle` and only `isinstance(score, float)` would still pass.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["alpha", "beta", "gamma"], emb, bit_width=4
    )
    scored = store.similarity_search_with_score("alpha", k=3)
    assert scored[0][0].page_content == "alpha"
    scores = [s for _, s in scored]
    assert all(a >= b for a, b in zip(scores, scores[1:]))


def test_load_then_add_assigns_fresh_handles_without_collision(tmp_path):
    # `_next_u64` is persisted across dump/load; if it were silently
    # dropped (back to 0 on load), a fresh add would reuse a handle
    # that's still mapped to an old doc, corrupting search results.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c"], emb, bit_width=4, ids=["id-a", "id-b", "id-c"]
    )
    store.dump(tmp_path)
    loaded = TurboQuantVectorStore.load(tmp_path, emb)
    loaded.add_texts(["d"], ids=["id-d"])

    # All four ids reachable; all four handles distinct.
    docs = loaded.get_by_ids(["id-a", "id-b", "id-c", "id-d"])
    assert [d.id for d in docs] == ["id-a", "id-b", "id-c", "id-d"]
    handles = list(loaded._str_to_u64.values())
    assert len(set(handles)) == len(handles)


def test_embeddings_property_returns_supplied_embedder():
    # Pins the `embeddings` property override. A refactor dropping it
    # would silently make the base class return None, breaking
    # `similarity_search_with_relevance_scores` discovery and some
    # retriever wiring.
    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore(emb)
    assert store.embeddings is emb


def test_aget_by_ids_preserves_order_and_returns_documents_with_id():
    # Async mirror of `test_get_by_ids_empty_input_and_order_preserved`.
    import asyncio

    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(
        ["a", "b", "c"], emb, bit_width=4, ids=["id-a", "id-b", "id-c"]
    )

    async def run() -> list[Document]:
        empty = await store.aget_by_ids([])
        assert empty == []
        return await store.aget_by_ids(["id-c", "id-a", "id-b"])

    docs = asyncio.run(run())
    assert [d.id for d in docs] == ["id-c", "id-a", "id-b"]


def test_load_rejects_side_car_desynced_from_index(tmp_path):
    # A side-car whose handle map doesn't match the .tvim index must fail
    # cleanly at load, not with a KeyError deep inside a later query.
    import json

    emb = StubEmbeddings(dim=64)
    store = TurboQuantVectorStore.from_texts(["a", "b", "c", "d"], emb, bit_width=4)
    store.dump(tmp_path)

    # Clean reload works.
    TurboQuantVectorStore.load(tmp_path, emb)

    with open(tmp_path / "docstore.json") as f:
        state = json.load(f)
    # Drop one id->handle mapping so the side-car holds fewer handles than
    # the index.
    state["str_to_u64"].pop(next(iter(state["str_to_u64"])))
    with open(tmp_path / "docstore.json", "w") as f:
        json.dump(state, f)

    with pytest.raises(ValueError):
        TurboQuantVectorStore.load(tmp_path, emb)
