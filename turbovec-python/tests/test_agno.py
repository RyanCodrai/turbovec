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


# ---- Constructor validation -----------------------------------------------


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
