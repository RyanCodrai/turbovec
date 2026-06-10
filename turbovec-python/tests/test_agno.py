"""Tests for the Agno VectorDb integration."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("agno")

from agno.knowledge.document import Document
from agno.vectordb.distance import Distance
from agno.vectordb.search import SearchType

from turbovec.agno import TurboQuantVectorDb


DIM = 64


class StubEmbedder:
    """Deterministic Agno-style embedder for tests.

    Implements the minimal embedder surface the integration uses:
    ``dimensions`` attribute, ``get_embedding(text)`` method,
    ``enable_batch`` attribute (False — we don't exercise batch).
    """

    enable_batch = False

    def __init__(self, dim: int = DIM) -> None:
        self.dimensions = dim

    def _embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dimensions).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v.tolist()

    def get_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def async_get_embedding(self, text: str) -> list[float]:
        return self._embed(text)


def _doc(content: str, *, doc_id: str | None = None, name: str | None = None,
         meta_data: dict | None = None, content_id: str | None = None,
         pre_embed: bool = True) -> Document:
    embedder = StubEmbedder(DIM)
    return Document(
        id=doc_id,
        name=name,
        content=content,
        meta_data=meta_data or {},
        content_id=content_id,
        embedding=embedder._embed(content) if pre_embed else None,
    )


class BatchEmbedder(StubEmbedder):
    """Embedder that advertises ``enable_batch`` and exposes both sync
    and async batch methods. Used to verify the integration takes the
    batch path when it's available."""

    enable_batch = True

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(dim)
        self.sync_batch_calls = 0
        self.async_batch_calls = 0

    def get_embeddings_batch_and_usage(self, texts):
        self.sync_batch_calls += 1
        return [self._embed(t) for t in texts], [{"tokens": 1} for _ in texts]

    async def async_get_embeddings_batch_and_usage(self, texts):
        self.async_batch_calls += 1
        return [self._embed(t) for t in texts], [{"tokens": 1} for _ in texts]


from agno.knowledge.reranker.base import Reranker as _AgnoReranker


class ReverseReranker(_AgnoReranker):
    """Trivial reranker that returns documents in reversed order. Used
    to verify the integration calls .rerank() on the result list."""

    calls: int = 0

    def rerank(self, query: str, documents):
        type(self).calls += 1
        return list(reversed(documents))


# ---- Constructor validation -----------------------------------------------


def test_search_results_carry_embedder():
    # Match LanceDb._build_search_results — returned Documents must have
    # `embedder` set to the store's embedder so downstream code can call
    # `doc.embed()` / `doc.async_embed()` on a retrieved hit. Without
    # the field set, `Document.embed` raises "No embedder provided".
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [_doc("hello", doc_id="d-1")])
    [result] = db.search(query="hello", limit=1)
    assert result.embedder is embedder


async def _async_search_embedder_body():
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [_doc("hello", doc_id="d-1")])
    [result] = await db.async_search(query="hello", limit=1)
    assert result.embedder is embedder


def test_async_search_results_also_carry_embedder():
    import asyncio
    asyncio.run(_async_search_embedder_body())


# ---- Reference-parity tests against agno's LanceDb test suite. Each
# pins behaviour the in-tree LanceDb unit tests exercise. ----

def test_update_metadata_updates_all_docs_sharing_content_id():
    # `content_id` is intentionally many-to-one — `update_metadata(cid, m)`
    # must update every doc with that cid, not just the first match. The
    # LanceDb reference covers this; our existing test only inserts one
    # doc per content_id and wouldn't have caught a "first-match only" bug.
    # Identify docs by a `label` in meta_data since agno's _derive_doc_id
    # rewrites doc.id at insert time.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [
        _doc("a", content_id="shared", meta_data={"label": "first"}),
        _doc("b", content_id="shared", meta_data={"label": "second"}),
        _doc("c", content_id="other", meta_data={"label": "third"}),
    ])
    db.update_metadata("shared", {"updated": True})

    results = db.search("a", limit=10)
    by_label = {d.meta_data["label"]: d for d in results}
    assert by_label["first"].meta_data.get("updated") is True
    assert by_label["second"].meta_data.get("updated") is True
    assert by_label["third"].meta_data.get("updated") is None


def test_update_metadata_preserves_quantized_codes():
    # `update_metadata` must not re-derive codes — codes are precious
    # (cost an embed + a quantise) and a silent re-embed would also
    # break determinism. Behaviour-level check: search results are
    # bit-identical before and after a metadata update.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [
        _doc(f"doc-{i}", doc_id=f"d-{i}", content_id=f"cid-{i}") for i in range(3)
    ])

    before = db.search("doc-0", limit=3)
    db.update_metadata("cid-0", {"updated": True})
    after = db.search("doc-0", limit=3)
    assert [d.id for d in before] == [d.id for d in after]


def test_insert_reembeds_document_with_empty_list_embedding():
    # In agno's async pipeline, failed embeds surface as `embedding=[]`
    # (an empty list, distinct path from None). The integration must
    # re-embed those rather than silently skipping or shipping the
    # zero-length vector to the kernel. Uses BatchEmbedder so we have
    # an embedder with the batch path the re-embed code prefers.
    embedder = BatchEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    doc = Document(id="d-1", content="hello", embedding=[], meta_data={})
    db.insert("h", [doc])
    assert db.get_count() == 1


def test_insert_empty_document_list_is_noop():
    # Drop-in safety: callers passing an empty list (e.g. a Knowledge
    # batch where every doc was filtered out upstream) must not raise
    # and must not change store state.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [])
    assert db.get_count() == 0


def test_search_with_empty_query_returns_empty():
    # LanceDb short-circuits `search("")` to []. Without this, a hash-
    # derived embedding of "" would return arbitrary near-match garbage.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [_doc("a", doc_id="d-1")])
    assert db.search("", limit=5) == []
    # None also short-circuits (defensive against upstream un-init).
    assert db.search(None, limit=5) == []  # type: ignore[arg-type]


def test_delete_by_metadata_handles_non_string_values():
    # `delete_by_metadata` matches on equality across heterogeneous
    # value types — bools, floats, ints. JSON round-trip during persist
    # could coerce types (`True` → `"true"`); the in-memory path must
    # at minimum work natively. Identify docs by `label` since agno's
    # _derive_doc_id rewrites doc.id at insert time.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [
        _doc("a", meta_data={"active": True, "score": 0.9, "label": "x"}),
        _doc("b", meta_data={"active": False, "score": 0.9, "label": "y"}),
        _doc("c", meta_data={"active": True, "score": 0.5, "label": "z"}),
    ])
    db.delete_by_metadata({"active": True})
    labels = {d.meta_data["label"] for d in db.search("a", limit=10)}
    assert labels == {"y"}


def test_update_metadata_with_empty_dict_is_noop():
    # `update_metadata(cid, {})` must not raise and must not strip the
    # existing metadata — empty dict is a real edge case (e.g. a caller
    # built `metadata` from filtered keys and ended up with no updates).
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [_doc("a", doc_id="d-1", content_id="cid-1", meta_data={"k": "v"})])
    db.update_metadata("cid-1", {})

    results = db.search("a", limit=1)
    assert results[0].meta_data == {"k": "v"}


def test_search_does_not_dedupe_distinct_documents_with_identical_content():
    # LanceDb dedupes search results by content string; turbovec
    # intentionally does NOT — each insert produces a distinct quantized
    # vector + id, and we return both as separate hits so callers can
    # tell them apart via content_id. Pin this deliberate divergence so
    # a future refactor doesn't silently start matching LanceDb.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [
        _doc("identical text", content_id="cid-1"),
        _doc("identical text", content_id="cid-2"),
    ])
    results = db.search("identical text", limit=10)
    assert len(results) == 2
    assert {d.content_id for d in results} == {"cid-1", "cid-2"}


def test_upsert_replaces_previous_generation():
    # A valid upsert under the same content_hash replaces the previous
    # generation entirely (deferred handle-based removal after the add).
    db = TurboQuantVectorDb(embedder=StubEmbedder(DIM))
    db.create()
    db.upsert("h1", [_doc("v1", content_id="cid-1")])
    db.upsert("h1", [_doc("v2", content_id="cid-1")])

    assert len(db._u64_to_doc) == 1
    contents = [r.content for r in db.search("v2", limit=10)]
    assert "v2" in contents
    assert "v1" not in contents


def test_upsert_dim_mismatch_preserves_existing():
    # An upsert whose new embeddings fail validation must not destroy the
    # data being replaced: the old generation is removed only after the
    # insert succeeds (issue #89).
    db = TurboQuantVectorDb(embedder=StubEmbedder(DIM))
    db.create()
    db.upsert("h1", [_doc("orig", content_id="cid-1")])

    bad = Document(content="new", meta_data={}, content_id="cid-1",
                   embedding=[0.1] * 32)
    with pytest.raises(ValueError):
        db.upsert("h1", [bad])  # dim 32 != index dim 64

    results = db.search("orig", limit=5)
    assert len(results) == 1
    assert results[0].content == "orig"
    assert len(db._u64_to_doc) == 1


def test_constructor_requires_embedder():
    with pytest.raises(ValueError, match="embedder.*required"):
        TurboQuantVectorDb()


def test_constructor_rejects_embedder_without_dimensions():
    class NoDimEmbedder:
        dimensions = None
        enable_batch = False
        def get_embedding(self, t): return [0.0] * 64

    with pytest.raises(ValueError, match="dimensions"):
        TurboQuantVectorDb(embedder=NoDimEmbedder())


def test_constructor_rejects_keyword_search_type():
    with pytest.raises(ValueError, match="search_type"):
        TurboQuantVectorDb(embedder=StubEmbedder(), search_type=SearchType.keyword)


def test_constructor_rejects_non_cosine_distance():
    with pytest.raises(ValueError, match="distance"):
        TurboQuantVectorDb(embedder=StubEmbedder(), distance=Distance.l2)


def test_constructor_rejects_invalid_bit_width():
    with pytest.raises(ValueError, match="bit_width"):
        TurboQuantVectorDb(embedder=StubEmbedder(), bit_width=8)


def test_dim_inferred_from_embedder():
    db = TurboQuantVectorDb(embedder=StubEmbedder(dim=128))
    assert db.dimensions == 128
    # The underlying index isn't constructed until create() (LanceDb's
    # contract: store object exists, but the "table" doesn't until you
    # ask for it). dim is on the embedder regardless.
    assert db._index is None
    db.create()
    assert db._index.dim == 128


# ---- Lifecycle (create / exists / drop / delete / get_count) -------------


def test_exists_false_until_create():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    assert db.exists() is False
    db.create()
    assert db.exists() is True


def test_create_is_idempotent():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    first_index = db._index
    db.create()
    # Second call doesn't blow away the existing index.
    assert db._index is first_index


def test_drop_returns_to_uncreated_state():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a")])
    assert db.exists() is True
    db.drop()
    assert db.exists() is False
    assert db._index is None
    # Re-create works and gives a fresh store.
    db.create()
    assert db.exists() is True
    assert db.get_count() == 0


def test_delete_returns_false_per_lancedb_contract():
    # LanceDb's delete() unconditionally returns False — actual removal
    # is via drop(). Mirror that exactly.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a")])
    assert db.delete() is False
    # delete() is a no-op; the index is still there.
    assert db.exists() is True


def test_get_count():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    assert db.get_count() == 0  # before create
    db.create()
    assert db.get_count() == 0  # empty after create
    db.insert("h", [_doc("a"), _doc("b"), _doc("c")])
    assert db.get_count() == 3


def test_optimize_is_noop():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    # Must not raise NotImplementedError (which the base class does).
    db.optimize()


def test_insert_before_create_raises():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    with pytest.raises(RuntimeError, match="not initialized"):
        db.insert("h", [_doc("a")])


def test_search_before_create_returns_empty():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    assert db.search("anything", limit=5) == []


def test_delete_methods_before_create_return_false():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    assert db.delete_by_id("x") is False
    assert db.delete_by_name("x") is False
    assert db.delete_by_metadata({"a": 1}) is False
    assert db.delete_by_content_id("x") is False


# ---- Basic insert / search ------------------------------------------------


def test_create_initializes_store():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    assert db.exists() is False
    db.create()
    assert db.exists() is True
    assert db._index.dim == DIM
    assert db.get_count() == 0


def test_insert_and_search_returns_documents():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("hash-1", [_doc("alpha"), _doc("beta"), _doc("gamma")])
    assert db.exists() is True
    results = db.search("alpha", limit=2)
    assert len(results) == 2
    assert all(isinstance(r, Document) for r in results)


def test_insert_raises_on_document_without_embedding():
    # When the embedder can't produce an embedding (returns None), insert
    # must raise rather than silently dropping the document.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()

    class FailingEmbedder:
        dimensions = DIM
        enable_batch = False
        # Document.embed() actually goes through get_embedding_and_usage.
        def get_embedding(self, t): return None
        def get_embedding_and_usage(self, t): return None, None

    db.embedder = FailingEmbedder()
    no_emb = _doc("x", pre_embed=False)
    no_emb.embedding = None
    with pytest.raises(ValueError, match="failed to embed"):
        db.insert("h", [no_emb])


def test_insert_batches_into_single_add_call():
    # Exercising a batch larger than one. We can't directly observe that the
    # underlying add_with_ids was called once, but we can confirm the result
    # is correct and the index size is right.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    docs = [_doc(f"text-{i}", doc_id=f"d-{i}") for i in range(8)]
    db.insert("hash-batch", docs)
    assert len(db._index) == 8


# ---- Existence checks -----------------------------------------------------


def test_name_exists():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("content", name="my-file.pdf")])
    assert db.name_exists("my-file.pdf") is True
    assert db.name_exists("nope.pdf") is False


def test_id_exists_uses_derived_id():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("content", doc_id="base-id")])
    # id_exists must use the *derived* id (md5(base_id + content_hash)),
    # not the input id directly.
    derived = next(iter(db._str_to_u64.keys()))
    assert db.id_exists(derived) is True
    assert db.id_exists("base-id") is False


def test_content_hash_exists_o1():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("unique-hash", [_doc("x")])
    assert db.content_hash_exists("unique-hash") is True
    assert db.content_hash_exists("other-hash") is False


# ---- Filters (kernel allowlist) -------------------------------------------


def test_search_with_metadata_filter():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    docs = [
        _doc("alpha", doc_id="a", meta_data={"tag": "keep"}),
        _doc("beta",  doc_id="b", meta_data={"tag": "drop"}),
        _doc("gamma", doc_id="g", meta_data={"tag": "keep"}),
    ]
    db.insert("h", docs)
    results = db.search("alpha", limit=10, filters={"tag": "keep"})
    assert len(results) == 2
    assert all(r.meta_data["tag"] == "keep" for r in results)


def test_search_with_selective_filter_returns_top_k():
    # Regression for the over-fetch / post-filter recall hit: filter that
    # matches 3 of 50 docs with limit=3 must return all 3, even when those
    # docs aren't in the unfiltered top-3 by raw score.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    docs = []
    for i in range(50):
        meta = {"tag": "needle"} if i in (7, 23, 41) else {"tag": "hay"}
        docs.append(_doc(f"text-{i}", doc_id=f"d-{i}", meta_data=meta))
    db.insert("h", docs)
    results = db.search("text-0", limit=3, filters={"tag": "needle"})
    assert len(results) == 3
    assert all(r.meta_data["tag"] == "needle" for r in results)


def test_search_with_no_matching_filter_returns_empty():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("alpha", meta_data={"tag": "a"})])
    results = db.search("alpha", limit=5, filters={"tag": "nonexistent"})
    assert results == []


def test_search_list_filter_silently_ignored():
    # Match LanceDb behavior: list-of-FilterExpr filters are ignored.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a"), _doc("b")])
    results = db.search("a", limit=2, filters=["something"])
    assert len(results) == 2  # filter ignored, full search


# ---- Similarity threshold -------------------------------------------------


def test_similarity_threshold_filters_low_scores():
    # similarity_threshold drops results whose scaled cosine is below
    # the threshold. Use a generous margin: the StubEmbedder produces
    # vectors that hash random text to roughly orthogonal directions
    # (raw cosine near 0 -> scaled ~0.5), so threshold=0.9 reliably
    # excludes the unrelated doc while letting the self-match through.
    db = TurboQuantVectorDb(embedder=StubEmbedder(), similarity_threshold=0.9)
    db.create()
    db.insert("h", [_doc("alpha"), _doc("very different content here")])
    results = db.search("alpha", limit=5)
    # Self-match scales to ~1.0; the unrelated doc to ~0.5. Threshold
    # filters out the unrelated one.
    assert len(results) >= 1
    assert all(r.content == "alpha" for r in results)


# ---- Upsert ---------------------------------------------------------------


def test_upsert_replaces_entire_batch_under_content_hash():
    # LanceDb's contract: upsert deletes EVERY existing document under
    # the given content_hash before inserting the new batch. The unit of
    # replacement is the content_hash, not the doc_id.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h-v1", [_doc("a"), _doc("b"), _doc("c")])
    assert db.get_count() == 3
    # Re-upsert under the same content_hash with a smaller batch — the
    # original 3 docs go away and only the new ones remain.
    db.upsert("h-v1", [_doc("z")])
    assert db.get_count() == 1


def test_upsert_distinct_content_hashes_keep_separate_entries():
    # The doc_id is derived from (base_id, content_hash). Different
    # content_hashes don't collide, so upserting under a new hash leaves
    # existing entries alone.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.upsert("hash-A", [_doc("x", doc_id="same-base")])
    db.upsert("hash-B", [_doc("x", doc_id="same-base")])
    assert db.get_count() == 2


# ---- Delete ---------------------------------------------------------------


def test_delete_by_id_returns_bool():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a", doc_id="x")])
    derived = next(iter(db._str_to_u64.keys()))
    assert db.delete_by_id(derived) is True
    assert db.delete_by_id("nonexistent") is False
    assert len(db._index) == 0


def test_delete_by_name_removes_all_matching():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [
        _doc("a", name="paper.pdf"),
        _doc("b", name="paper.pdf"),
        _doc("c", name="other.pdf"),
    ])
    assert db.delete_by_name("paper.pdf") is True
    assert db.delete_by_name("paper.pdf") is False  # already gone
    assert len(db._index) == 1


def test_delete_by_metadata_uses_and_semantics():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [
        _doc("a", meta_data={"tag": "x", "src": "web"}),
        _doc("b", meta_data={"tag": "x", "src": "pdf"}),
        _doc("c", meta_data={"tag": "y", "src": "web"}),
    ])
    assert db.delete_by_metadata({"tag": "x", "src": "web"}) is True
    assert len(db._index) == 2


def test_delete_by_content_id():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [
        _doc("a", content_id="cid-1"),
        _doc("b", content_id="cid-2"),
    ])
    assert db.delete_by_content_id("cid-1") is True
    assert db.delete_by_content_id("cid-1") is False
    assert len(db._index) == 1


def test_drop_clears_all_state():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a", name="foo")])
    db.drop()
    # drop() releases the underlying index; exists() goes back to False.
    assert db._index is None
    assert db._str_to_u64 == {}
    assert db._u64_to_doc == {}
    assert db._content_hashes == set()
    assert db._name_to_ids == {}


# ---- update_metadata ------------------------------------------------------


def test_update_metadata_merges_by_content_id():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a", content_id="cid", meta_data={"old": 1})])
    db.update_metadata("cid", {"new": 2})
    docs = list(db._u64_to_doc.values())
    assert docs[0]["meta_data"] == {"old": 1, "new": 2}


def test_update_metadata_writes_filters_field():
    # LanceDb's update_metadata writes to BOTH `meta_data` and a separate
    # `filters` payload field. Mirror that — drop-in callers reading the
    # filters field after an update expect to find it.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a", content_id="cid", meta_data={"old": 1})])
    db.update_metadata("cid", {"new": 2})
    docs = list(db._u64_to_doc.values())
    assert docs[0]["filters"] == {"new": 2}


# ---- Persistence ---------------------------------------------------------


def test_save_writes_json_sidecar(tmp_path):
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a", doc_id="x"), _doc("b", doc_id="y")])
    db.save(str(tmp_path))
    assert (tmp_path / "index.tvim").exists()
    assert (tmp_path / "docstore.json").exists()
    assert not (tmp_path / "docstore.pkl").exists()
    with open(tmp_path / "docstore.json") as f:
        data = json.load(f)
    assert data["schema_version"] == 1
    assert data["dimensions"] == DIM


def test_save_and_load_via_path_param(tmp_path):
    embedder = StubEmbedder()
    db = TurboQuantVectorDb(embedder=embedder, path=str(tmp_path))
    db.create()
    db.insert("h", [_doc("a"), _doc("b"), _doc("c")])
    db.save()

    # Fresh store with same path should load on create().
    db2 = TurboQuantVectorDb(embedder=embedder, path=str(tmp_path))
    db2.create()
    assert len(db2._index) == 3
    # Query through the loaded store still works.
    results = db2.search("a", limit=3)
    assert len(results) == 3


def test_save_requires_path():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    with pytest.raises(ValueError, match="path"):
        db.save()


def test_load_rejects_unknown_schema_version(tmp_path):
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a")])
    db.save(str(tmp_path))
    with open(tmp_path / "docstore.json") as f:
        data = json.load(f)
    data["schema_version"] = 99
    with open(tmp_path / "docstore.json", "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="schema_version"):
        TurboQuantVectorDb(embedder=StubEmbedder(), path=str(tmp_path)).create()


def test_load_rejects_dimension_mismatch(tmp_path):
    db = TurboQuantVectorDb(embedder=StubEmbedder(dim=64))
    db.create()
    db.insert("h", [_doc("a")])
    db.save(str(tmp_path))
    # New store with a different-dim embedder must refuse to load.
    with pytest.raises(ValueError, match="dimensions"):
        TurboQuantVectorDb(embedder=StubEmbedder(dim=128), path=str(tmp_path)).create()


# ---- Protocol coverage ----------------------------------------------------


def test_supported_search_types():
    # Mirror LanceDb's return shape: a list of SearchType enum members
    # (not their `.value` strings). Drop-in callers iterating this list
    # would unwrap enum members, so the return type matters.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    types = db.get_supported_search_types()
    assert types == [SearchType.vector]
    assert isinstance(types[0], SearchType)


def test_upsert_available():
    assert TurboQuantVectorDb(embedder=StubEmbedder()).upsert_available() is True


# ---- Async coverage -------------------------------------------------------


def test_async_round_trip():
    import asyncio

    async def runner():
        db = TurboQuantVectorDb(embedder=StubEmbedder())
        await db.async_create()
        await db.async_insert("h", [_doc("a"), _doc("b"), _doc("c")])
        assert await db.async_exists() is True
        results = await db.async_search("a", limit=2)
        assert len(results) == 2
        await db.async_drop()
        assert await db.async_exists() is False

    asyncio.run(runner())


# ---- End-to-end smoke test: framework wiring -----------------------------


def test_knowledge_search_routes_through_vector_db():
    # Smoke test: build an Agno Knowledge with our vector_db and call
    # Knowledge.search() — the framework's top-level retrieval API.
    # Exercises the wiring between Knowledge and VectorDb.search.
    from agno.knowledge import Knowledge

    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    # Seed the underlying vector_db directly — Knowledge.add_content's
    # real path goes through readers, which we don't need to exercise
    # here. The smoke test target is the search-routing surface.
    db.insert(
        "seed",
        [
            _doc("alpha", doc_id="d1", meta_data={"category": "a"}),
            _doc("beta",  doc_id="d2", meta_data={"category": "b"}),
            _doc("gamma", doc_id="d3", meta_data={"category": "a"}),
        ],
    )

    knowledge = Knowledge(vector_db=db)
    results = knowledge.search("alpha", max_results=3)
    assert len(results) == 3
    assert all(isinstance(r, Document) for r in results)


def test_knowledge_search_with_filter_routes_through_kernel_allowlist():
    # Verify the filter kwarg reaches our search() through Knowledge's
    # wrapper, and that the kernel-level allowlist path is used (not
    # post-filtering).
    from agno.knowledge import Knowledge

    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert(
        "seed",
        [
            _doc(f"text-{i}", doc_id=f"d-{i}",
                 meta_data={"tag": "needle" if i in (7, 23, 41) else "hay"})
            for i in range(50)
        ],
    )

    knowledge = Knowledge(vector_db=db)
    results = knowledge.search(
        "text-0",
        max_results=3,
        filters={"tag": "needle"},
    )
    # Selective filter: 3 of 50 match; max_results=3 must return all 3.
    # Post-filter+over-fetch would have returned <3 results.
    assert len(results) == 3
    assert all(r.meta_data["tag"] == "needle" for r in results)


# ---- Reranker integration -------------------------------------------------


def test_reranker_called_on_search_results():
    # Verify the constructor's `reranker` is actually invoked on the
    # search() result list and that its return value replaces the
    # unranked order.
    ReverseReranker.calls = 0
    reranker = ReverseReranker()
    db = TurboQuantVectorDb(embedder=StubEmbedder(), reranker=reranker)
    db.create()
    db.insert("h", [_doc("a"), _doc("b"), _doc("c")])
    results = db.search("a", limit=3)
    assert ReverseReranker.calls == 1
    # ReverseReranker reverses, so the natural top-result (the self-match
    # 'a') ends up last after rerank.
    assert results[-1].content == "a"


def test_reranker_called_on_async_search_results():
    import asyncio

    async def runner():
        ReverseReranker.calls = 0
        reranker = ReverseReranker()
        db = TurboQuantVectorDb(embedder=StubEmbedder(), reranker=reranker)
        await db.async_create()
        await db.async_insert("h", [_doc("a"), _doc("b"), _doc("c")])
        results = await db.async_search("a", limit=3)
        assert ReverseReranker.calls == 1
        assert results[-1].content == "a"

    asyncio.run(runner())


def test_reranker_not_called_on_empty_results():
    # Defensive: an empty results list shouldn't go to the reranker —
    # nothing to rank. Avoids unnecessary work.
    ReverseReranker.calls = 0
    reranker = ReverseReranker()
    db = TurboQuantVectorDb(embedder=StubEmbedder(), reranker=reranker)
    db.create()
    # No documents inserted -> empty search results.
    results = db.search("anything", limit=5)
    assert results == []
    assert ReverseReranker.calls == 0


# ---- insert(filters=) merges into meta_data -------------------------------


def test_insert_filters_kwarg_merges_into_doc_metadata():
    # The `filters` kwarg on insert should merge into each document's
    # meta_data — matches LanceDb's contract where these become part of
    # the doc's stored metadata and are searchable later.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    docs = [
        _doc("a", doc_id="d1", meta_data={"existing": 1}),
        _doc("b", doc_id="d2", meta_data={"existing": 2}),
    ]
    db.insert("h", docs, filters={"tenant": "acme", "tier": "pro"})
    for data in db._u64_to_doc.values():
        # Original meta_data preserved...
        assert "existing" in data["meta_data"]
        # ...and the filter kwargs merged in.
        assert data["meta_data"]["tenant"] == "acme"
        assert data["meta_data"]["tier"] == "pro"


def test_insert_filters_kwarg_is_searchable_after_insert():
    # The merged filter values should be visible to subsequent
    # search(..., filters={...}) calls — they're real metadata now.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h-a", [_doc("alpha")], filters={"tenant": "acme"})
    db.insert("h-b", [_doc("beta")],  filters={"tenant": "globex"})
    results = db.search("alpha", limit=5, filters={"tenant": "acme"})
    assert len(results) == 1
    assert results[0].content == "alpha"


# ---- Async-only coverage --------------------------------------------------


def test_async_name_exists():
    import asyncio

    async def runner():
        db = TurboQuantVectorDb(embedder=StubEmbedder())
        await db.async_create()
        await db.async_insert("h", [_doc("a", name="foo.pdf")])
        assert await db.async_name_exists("foo.pdf") is True
        assert await db.async_name_exists("missing.pdf") is False

    asyncio.run(runner())


def test_async_get_count():
    import asyncio

    async def runner():
        db = TurboQuantVectorDb(embedder=StubEmbedder())
        assert await db.async_get_count() == 0  # before create
        await db.async_create()
        await db.async_insert("h", [_doc("a"), _doc("b")])
        assert await db.async_get_count() == 2

    asyncio.run(runner())


def test_async_upsert_replaces_by_content_hash():
    # Same contract as sync upsert: replace everything under the
    # given content_hash. Exercises the async code path explicitly.
    import asyncio

    async def runner():
        db = TurboQuantVectorDb(embedder=StubEmbedder())
        await db.async_create()
        await db.async_insert("hv1", [_doc("a"), _doc("b"), _doc("c")])
        assert db.get_count() == 3
        await db.async_upsert("hv1", [_doc("z")])
        assert db.get_count() == 1

    asyncio.run(runner())


# ---- Batch embedder paths -------------------------------------------------


def test_insert_uses_sync_batch_embedder_path():
    # When embedder.enable_batch is True AND it exposes
    # get_embeddings_batch_and_usage, insert() should route through that
    # batch method instead of per-document embed() calls.
    emb = BatchEmbedder()
    db = TurboQuantVectorDb(embedder=emb)
    db.create()
    # Docs without embeddings → must be embedded by the integration.
    docs = [
        _doc("a", doc_id="d1", pre_embed=False),
        _doc("b", doc_id="d2", pre_embed=False),
        _doc("c", doc_id="d3", pre_embed=False),
    ]
    db.insert("h", docs)
    assert emb.sync_batch_calls == 1
    assert db.get_count() == 3


def test_async_insert_uses_async_batch_embedder_path():
    import asyncio

    async def runner():
        emb = BatchEmbedder()
        db = TurboQuantVectorDb(embedder=emb)
        await db.async_create()
        docs = [
            _doc("a", doc_id="d1", pre_embed=False),
            _doc("b", doc_id="d2", pre_embed=False),
        ]
        await db.async_insert("h", docs)
        assert emb.async_batch_calls == 1
        assert db.get_count() == 2

    asyncio.run(runner())


# ---- Defensive None-guard paths ------------------------------------------


def test_save_before_create_raises():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    with pytest.raises(RuntimeError, match="no index to save"):
        db.save("/tmp/nonexistent-turbovec-store")


def test_update_metadata_before_create_is_noop():
    # Defensive: calling update_metadata before create() shouldn't
    # raise; it just has no documents to touch.
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.update_metadata("any-content-id", {"key": "value"})  # no assertion: must not raise


def test_update_metadata_unknown_content_id_is_noop():
    db = TurboQuantVectorDb(embedder=StubEmbedder())
    db.create()
    db.insert("h", [_doc("a", content_id="cid-known", meta_data={"k": 1})])
    db.update_metadata("cid-not-in-store", {"k": 2})
    # Nothing changed on the known doc.
    docs = list(db._u64_to_doc.values())
    assert docs[0]["meta_data"] == {"k": 1}


# ---- Tier-2 field-completeness tests. Each pins behaviour around
# Document fields that are populated, omitted, or diverged from LanceDb.
# Without these, a refactor could silently change the contract. ----

def test_search_results_content_origin_divergence_from_lancedb():
    # `Document.content_origin` is an agno field set by some Reader
    # pipelines. LanceDb drops it on insert (stores `payload` as opaque
    # JSON of {id, name, meta_data, content_id, usage} only); turbovec
    # mirrors that. Pin the divergence so a future caller doesn't quietly
    # start relying on the field surviving — and so a future preservation
    # change (turbovec preserving it where LanceDb doesn't) is a
    # deliberate decision.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    doc = Document(
        id="d-1",
        content="hello",
        embedding=embedder._embed("hello"),
        content_origin="pdf",
    )
    db.insert("h", [doc])
    [result] = db.search("hello", limit=1)
    # Matches LanceDb behaviour: content_origin is not preserved.
    assert result.content_origin is None


def test_search_results_size_divergence_from_lancedb():
    # Same divergence pin for `Document.size`.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    doc = Document(
        id="d-1",
        content="hello",
        embedding=embedder._embed("hello"),
        size=42,
    )
    db.insert("h", [doc])
    [result] = db.search("hello", limit=1)
    assert result.size is None


def test_search_results_embedding_is_none_divergence_from_lancedb():
    # LanceDb's `_build_search_results` sets `embedding=item["vector"]`
    # so callers can read the original vector off a returned hit.
    # turbovec quantizes vectors to 2-4 bits per dim and discards the
    # full-precision form — the original vector is unrecoverable, so
    # we return `embedding=None`. Pin this so a caller who depends on
    # `result.embedding` after retrieval (e.g. for rerank-by-similarity)
    # gets a clear contract from the test suite, not a runtime surprise.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [_doc("a", doc_id="d-1")])
    [result] = db.search("a", limit=1)
    assert result.embedding is None


def test_reranker_output_documents_carry_reranking_score():
    # When the reranker sets `reranking_score`, it must survive the
    # post-rerank result list. Existing reranker tests only check the
    # order; a refactor that re-threads `embedder` could drop other
    # fields the reranker mutated.
    from agno.knowledge.reranker.base import Reranker as _AgnoReranker

    class ScoringReranker(_AgnoReranker):
        def rerank(self, query: str, documents):
            for i, d in enumerate(documents):
                d.reranking_score = float(len(documents) - i)
            return documents

    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder, reranker=ScoringReranker())
    db.create()
    db.insert("h", [_doc(f"doc-{i}") for i in range(3)])

    results = db.search("doc-0", limit=3)
    assert all(r.reranking_score is not None for r in results)
    assert all(isinstance(r.reranking_score, float) for r in results)


def test_delete_by_metadata_returns_false_when_no_match():
    # `delete_by_*` methods return bool — True when at least one doc
    # matched (and was deleted), False otherwise. Other delete tests
    # cover the True branch; this pins the False branch so a regression
    # always returning True can't silently mask "I expected to delete
    # something and nothing matched" upstream.
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder)
    db.create()
    db.insert("h", [_doc("a", meta_data={"tag": "x"})])
    assert db.delete_by_metadata({"tag": "no-such-value"}) is False
    # Original doc still present.
    assert db.get_count() == 1


# ---- Security/data-integrity regression (issue #104) ----------------------


def test_duplicate_doc_id_keeps_both_vectors_no_orphan():
    # agno's reference store (LanceDb) is append-only: two docs with the same
    # explicit id (hence same derived doc_id) are BOTH stored. Previously the
    # one-to-one _str_to_u64 map orphaned the first vector — counted and
    # searchable but undeletable. Both must now be reachable and deletable.
    db = TurboQuantVectorDb(embedder=StubEmbedder(DIM))
    db.create()
    db.insert("h", [_doc("alpha", doc_id="dup"), _doc("beta", doc_id="dup")])

    assert db.get_count() == 2
    [doc_id] = list(db._str_to_u64)            # both collapse to one derived id
    assert len(db._str_to_u64[doc_id]) == 2    # ...mapping to both handles
    assert len(db._u64_to_doc) == 2

    # Deleting that id removes BOTH vectors, leaving no orphan behind.
    assert db.delete_by_id(doc_id)
    assert db.get_count() == 0
    assert db._str_to_u64 == {}
    assert db._u64_to_doc == {}


def test_duplicate_doc_id_survives_persistence_roundtrip(tmp_path):
    embedder = StubEmbedder(DIM)
    db = TurboQuantVectorDb(embedder=embedder, path=str(tmp_path))
    db.create()
    db.insert("h", [_doc("alpha", doc_id="dup"), _doc("beta", doc_id="dup")])
    db.save()

    # Reload must rebuild the one-to-many id map, not drop a handle.
    db2 = TurboQuantVectorDb(embedder=embedder, path=str(tmp_path))
    db2.create()
    assert db2.get_count() == 2
    [doc_id] = list(db2._str_to_u64)
    assert len(db2._str_to_u64[doc_id]) == 2


def _doc_same_content(name=None, content_id=None, meta_data=None):
    # All share identical content + no explicit id -> identical derived doc_id,
    # differing only by name/content_id/metadata.
    return _doc("identical", name=name, content_id=content_id, meta_data=meta_data)


def test_delete_by_name_only_removes_matching_name_on_doc_id_collision():
    # name is not part of the derived doc_id, so two differently-named docs
    # with identical content collide. delete_by_name must remove only the
    # named doc, not its id-twin, and must not leave a stale name entry.
    db = TurboQuantVectorDb(embedder=StubEmbedder(DIM))
    db.create()
    db.insert("h", [_doc_same_content(name="A"), _doc_same_content(name="B")])
    assert db.get_count() == 2

    assert db.delete_by_name("A") is True
    assert db.get_count() == 1
    assert db.name_exists("A") is False  # no stale entry
    assert db.name_exists("B") is True


def test_delete_by_content_id_only_removes_matching_on_doc_id_collision():
    db = TurboQuantVectorDb(embedder=StubEmbedder(DIM))
    db.create()
    db.insert("h", [_doc_same_content(content_id="c1"), _doc_same_content(content_id="c2")])
    assert db.get_count() == 2

    assert db.delete_by_content_id("c1") is True
    assert db.get_count() == 1  # c2 survives


def test_delete_by_metadata_only_removes_matching_on_doc_id_collision():
    db = TurboQuantVectorDb(embedder=StubEmbedder(DIM))
    db.create()
    db.insert("h", [_doc_same_content(meta_data={"k": "x"}), _doc_same_content(meta_data={"k": "y"})])
    assert db.get_count() == 2

    assert db.delete_by_metadata({"k": "x"}) is True
    assert db.get_count() == 1  # the {"k": "y"} doc survives
