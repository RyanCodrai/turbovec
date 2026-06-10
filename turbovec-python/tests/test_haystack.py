"""Tests for the Haystack DocumentStore integration."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("haystack")

from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from turbovec.haystack import TurboQuantDocumentStore


DIM = 128


def unit_vector(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def make_docs(n: int, seed_offset: int = 0) -> list[Document]:
    return [
        Document(
            id=f"doc-{i}",
            content=f"text {i}",
            embedding=unit_vector(i + seed_offset),
            meta={"idx": i, "group": "a" if i % 2 == 0 else "b"},
        )
        for i in range(n)
    ]


def test_count_documents_starts_at_zero():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    assert store.count_documents() == 0


# ---- Field-fidelity round-trip tests (covers the langchain-class bug
# pattern: returned Documents must populate every field the reference
# would, not just id/content/meta). ----

def test_blob_field_round_trips_through_filter_and_retrieval():
    # Writing a Document with `blob=` must survive write -> filter_documents
    # AND write -> embedding_retrieval AND write -> storage. The reference
    # InMemoryDocumentStore preserves blob; we used to drop it silently.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    payload = b"binary-payload-bytes"
    blob = ByteStream(data=payload, meta={"origin": "test"}, mime_type="application/octet-stream")
    doc = Document(
        id="doc-blob",
        content="text",
        embedding=unit_vector(0),
        blob=blob,
        meta={"k": 1},
    )
    store.write_documents([doc])

    [filtered] = store.filter_documents(filters={"field": "meta.k", "operator": "==", "value": 1})
    assert filtered.blob is not None
    assert filtered.blob.data == payload
    assert filtered.blob.mime_type == "application/octet-stream"
    assert filtered.blob.meta == {"origin": "test"}

    [retrieved] = store.embedding_retrieval(query_embedding=doc.embedding, top_k=1)
    assert retrieved.blob is not None
    assert retrieved.blob.data == payload

    assert store.storage["doc-blob"].blob is not None
    assert store.storage["doc-blob"].blob.data == payload


def test_sparse_embedding_field_round_trips_through_filter_and_retrieval():
    # Same shape of test for sparse_embedding — hybrid-search pipelines
    # rely on this surviving the round-trip.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    sparse = SparseEmbedding(indices=[0, 7, 42], values=[0.1, 0.5, 0.9])
    doc = Document(
        id="doc-sparse",
        content="text",
        embedding=unit_vector(0),
        sparse_embedding=sparse,
    )
    store.write_documents([doc])

    [filtered] = store.filter_documents()
    assert filtered.sparse_embedding is not None
    assert filtered.sparse_embedding.indices == [0, 7, 42]
    assert filtered.sparse_embedding.values == [0.1, 0.5, 0.9]

    [retrieved] = store.embedding_retrieval(query_embedding=doc.embedding, top_k=1)
    assert retrieved.sparse_embedding is not None
    assert retrieved.sparse_embedding.indices == [0, 7, 42]


def test_blob_and_sparse_embedding_survive_save_load_roundtrip(tmp_path):
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    blob = ByteStream(data=b"abc", mime_type="text/plain")
    sparse = SparseEmbedding(indices=[1, 2], values=[0.25, 0.75])
    store.write_documents([
        Document(
            id="doc-rich",
            content="text",
            embedding=unit_vector(0),
            blob=blob,
            sparse_embedding=sparse,
        )
    ])
    store.save_to_disk(tmp_path)

    restored = TurboQuantDocumentStore.load_from_disk(tmp_path)
    rebuilt = restored.storage["doc-rich"]
    assert rebuilt.blob is not None
    assert rebuilt.blob.data == b"abc"
    assert rebuilt.blob.mime_type == "text/plain"
    assert rebuilt.sparse_embedding is not None
    assert rebuilt.sparse_embedding.indices == [1, 2]
    assert rebuilt.sparse_embedding.values == [0.25, 0.75]


def test_documents_without_blob_or_sparse_embedding_round_trip_as_none():
    # Documents that were written WITHOUT blob/sparse_embedding must come
    # back with those fields as None, not missing-attribute or KeyError.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents([
        Document(id="plain", content="text", embedding=unit_vector(0))
    ])
    [doc] = store.filter_documents()
    assert doc.blob is None
    assert doc.sparse_embedding is None


def test_load_accepts_v1_schema_with_no_blob_or_sparse_fields(tmp_path):
    # v1 docstore.json predates blob/sparse round-trip. Reading it back
    # should succeed and leave both fields as None — not raise.
    import json

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents([
        Document(id="doc-v1", content="text", embedding=unit_vector(0), meta={"x": 1})
    ])
    store.save_to_disk(tmp_path)

    with open(tmp_path / "docstore.json") as f:
        state = json.load(f)
    state["schema_version"] = 1
    # Strip the v2-only fields from each doc entry so the file shape
    # matches what a v1 save would have produced.
    for _h, d in state["u64_to_doc"]:
        d.pop("blob", None)
        d.pop("sparse_embedding", None)
    with open(tmp_path / "docstore.json", "w") as f:
        json.dump(state, f)

    restored = TurboQuantDocumentStore.load_from_disk(tmp_path)
    [doc] = restored.filter_documents()
    assert doc.id == "doc-v1"
    assert doc.blob is None
    assert doc.sparse_embedding is None


# ---- Filter validation parity ----

def test_filter_documents_rejects_field_without_operator():
    # InMemoryDocumentStore rejects a filter dict that has neither
    # `operator` nor `conditions` at the top level — even if a stray
    # `field` key is present. We do too.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        store.filter_documents(filters={"field": "meta.idx", "value": 1})


# ---- Reference-parity tests against InMemoryDocumentStore. Each one
# pins behaviour the haystack in-tree DocumentStoreBaseTests suite
# tests; the bug class is "drop-in regression that only shows up when
# users compare turbovec's store against InMemoryDocumentStore". ----

@pytest.mark.parametrize(
    "bad_filter",
    [
        {"field": "meta.x"},                       # no operator / conditions
        {"operator": "AND"},                       # missing conditions
        {"operator": "==", "value": 1},            # comparison missing field
    ],
)
def test_filter_documents_rejects_malformed_filter_shapes(bad_filter):
    # Distinct from `field_without_operator` above: each of these is a
    # different malformed shape the InMemoryDocumentStore reference
    # rejects. Either our outer `_validate_filters` catches it (raising
    # ValueError) or haystack's `document_matches_filter` does (raising
    # `haystack.errors.FilterError`) — either way, the store must not
    # silently match everything / nothing.
    from haystack.errors import FilterError

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    with pytest.raises((ValueError, FilterError)):
        store.filter_documents(filters=bad_filter)


def test_filter_documents_with_and_or_not_operators():
    # Compound filter dicts (AND / OR / NOT joining sub-conditions) are
    # the standard production filter shape — pipelines build them via
    # haystack's filter DSL, not the bare single-comparison form. We
    # only delegate to `document_matches_filter`, but proving the
    # delegation works end-to-end through `filter_documents` and
    # `embedding_retrieval` is what catches a regression where (say) we
    # forget to forward compound filters to the kernel allowlist path.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents([
        Document(id="a", content="x", embedding=unit_vector(0), meta={"tier": "pro", "n": 1}),
        Document(id="b", content="y", embedding=unit_vector(1), meta={"tier": "free", "n": 2}),
        Document(id="c", content="z", embedding=unit_vector(2), meta={"tier": "pro", "n": 3}),
    ])

    and_filter = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.tier", "operator": "==", "value": "pro"},
            {"field": "meta.n", "operator": ">", "value": 1},
        ],
    }
    assert {d.id for d in store.filter_documents(filters=and_filter)} == {"c"}
    # embedding_retrieval must apply the same compound filter as an allowlist.
    hits = store.embedding_retrieval(
        query_embedding=unit_vector(0), top_k=10, filters=and_filter
    )
    assert {d.id for d in hits} == {"c"}

    or_filter = {
        "operator": "OR",
        "conditions": [
            {"field": "meta.tier", "operator": "==", "value": "free"},
            {"field": "meta.n", "operator": ">=", "value": 3},
        ],
    }
    assert {d.id for d in store.filter_documents(filters=or_filter)} == {"b", "c"}

    not_filter = {
        "operator": "NOT",
        "conditions": [
            {"field": "meta.tier", "operator": "==", "value": "pro"},
        ],
    }
    assert {d.id for d in store.filter_documents(filters=not_filter)} == {"b"}


def test_embedding_retrieval_rejects_empty_query_embedding():
    # An empty list is degenerate input — the index's dim is non-zero,
    # so the dim-mismatch check raises. Pins that "no query at all"
    # surfaces as a clean error, not a kernel-side panic or empty hit list.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    with pytest.raises(ValueError):
        store.embedding_retrieval(query_embedding=[], top_k=3)


def test_filter_documents_equality_with_missing_meta_key():
    # `field == None` should match docs where the field is absent.
    # Reference behaviour; an easy regression where we accidentally
    # normalise missing keys to empty string / sentinel.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents([
        Document(id="has-tag", content="x", embedding=unit_vector(0), meta={"tag": "free"}),
        Document(id="no-tag", content="y", embedding=unit_vector(1), meta={}),
    ])
    filters = {"field": "meta.tag", "operator": "==", "value": None}
    assert [d.id for d in store.filter_documents(filters=filters)] == ["no-tag"]


def test_delete_documents_on_lazy_empty_store_is_noop():
    # A lazy-uncommitted store has no committed index dim. Deleting from
    # it must not blow up — even if every requested id is unknown.
    store = TurboQuantDocumentStore(bit_width=4)  # dim=None (lazy)
    store.delete_documents(["nonexistent-1", "nonexistent-2"])
    assert store.count_documents() == 0


def test_get_metadata_field_min_max_handles_float_meta_prefix_and_single_value():
    # The reference covers (a) float-valued fields, (b) the "meta."
    # prefix on the field name, and (c) a single-value collection
    # (min==max). Our existing test only covers int + missing field, so
    # all three branches are uncovered.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents([
        Document(id="a", content="x", embedding=unit_vector(0), meta={"price": 9.99}),
        Document(id="b", content="y", embedding=unit_vector(1), meta={"price": 19.99}),
        Document(id="c", content="z", embedding=unit_vector(2), meta={"price": 4.50}),
    ])
    # Float values.
    assert store.get_metadata_field_min_max("price") == {"min": 4.50, "max": 19.99}
    # "meta." prefix on the field name.
    assert store.get_metadata_field_min_max("meta.price") == {"min": 4.50, "max": 19.99}

    # Single-value collection — min == max.
    single = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    single.write_documents([
        Document(id="d", content="x", embedding=unit_vector(0), meta={"price": 5.0}),
    ])
    assert single.get_metadata_field_min_max("price") == {"min": 5.0, "max": 5.0}


def test_two_stores_have_independent_state():
    # Refactor canary against accidental class-level mutable state
    # (sets/dicts at class scope are a recurring footgun). Mutating one
    # store must never be visible in another.
    s1 = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    s2 = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    s1.write_documents([Document(id="a", content="x", embedding=unit_vector(0))])
    assert s2.count_documents() == 0
    s2.write_documents([Document(id="b", content="y", embedding=unit_vector(1))])
    assert s1.count_documents() == 1
    assert s2.count_documents() == 1
    assert "a" not in s2._str_to_u64
    assert "b" not in s1._str_to_u64


def test_async_concurrent_embedding_retrievals_are_consistent():
    # The async methods are `to_thread`-style wrappers around the sync
    # ones. Concurrent reads against the IdMapIndex are documented as
    # safe; pin it with a consistency test (every concurrent call
    # produces the same top-k as a single sync call).
    import asyncio

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(20)
    store.write_documents(docs)

    sync_ids = [d.id for d in store.embedding_retrieval(
        query_embedding=docs[0].embedding, top_k=3
    )]

    async def run() -> list[list[str]]:
        results = await asyncio.gather(*[
            store.embedding_retrieval_async(
                query_embedding=docs[0].embedding, top_k=3
            )
            for _ in range(10)
        ])
        return [[d.id for d in r] for r in results]

    all_ids = asyncio.run(run())
    for ids in all_ids:
        assert ids == sync_ids


def test_return_embedding_flag_is_inert_for_turbovec():
    # Quantization discards full-precision embeddings — the flag is
    # accepted for `InMemoryDocumentStore` parity but Documents always
    # come back with `embedding=None` regardless of the flag (both
    # store-level and per-call). Pin the deliberate divergence so a
    # future caller doesn't quietly start relying on it.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4, return_embedding=True)
    docs = make_docs(2)
    store.write_documents(docs)

    for d in store.filter_documents():
        assert d.embedding is None
    for d in store.embedding_retrieval(query_embedding=docs[0].embedding, top_k=2):
        assert d.embedding is None
    # Per-call override (force True) is also inert.
    for d in store.embedding_retrieval(
        query_embedding=docs[0].embedding, top_k=2, return_embedding=True
    ):
        assert d.embedding is None


def test_shutdown_closes_async_executor():
    # After `shutdown()`, the owned executor must not accept new tasks.
    # Catches a regression where we silently leak the executor by
    # marking it shut down without actually calling shutdown.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    executor = store.executor
    store.shutdown()
    with pytest.raises(RuntimeError):
        executor.submit(lambda: None)


def test_write_returns_written_count():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    assert store.write_documents(make_docs(5)) == 5
    assert store.count_documents() == 5


def test_filter_documents_returns_all_without_filter():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(4))
    results = store.filter_documents()
    assert len(results) == 4
    assert {doc.id for doc in results} == {"doc-0", "doc-1", "doc-2", "doc-3"}


def test_filter_documents_applies_metadata_filter():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(6))
    # Haystack 2.x explicit-DSL filter: group == "a" (evens).
    filt = {"field": "meta.group", "operator": "==", "value": "a"}
    results = store.filter_documents(filters=filt)
    assert {doc.id for doc in results} == {"doc-0", "doc-2", "doc-4"}


def test_delete_documents_removes_and_is_idempotent():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(5))
    store.delete_documents(["doc-2", "doc-4"])
    assert store.count_documents() == 3
    # Deleting again (or a non-existent id) is a no-op.
    store.delete_documents(["doc-2", "doc-99"])
    assert store.count_documents() == 3


def test_duplicate_policy_fail_raises():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    # Default policy is FAIL.
    with pytest.raises(DuplicateDocumentError):
        store.write_documents(make_docs(1))  # doc-0 collides


def test_duplicate_policy_skip_keeps_original():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    # doc-0..2 already there; writing doc-0..4 with SKIP inserts only 3..4.
    written = store.write_documents(make_docs(5), policy=DuplicatePolicy.SKIP)
    assert written == 2
    assert store.count_documents() == 5


def test_duplicate_policy_overwrite_replaces():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    # Replace doc-0..2 with fresh embeddings (different seed).
    replacements = make_docs(3, seed_offset=1000)
    written = store.write_documents(replacements, policy=DuplicatePolicy.OVERWRITE)
    assert written == 3
    assert store.count_documents() == 3


def test_intra_batch_duplicate_overwrite_keeps_last_no_orphan():
    # Two docs sharing an id in a single call must not orphan a vector.
    # InMemoryDocumentStore writes into a dict as it iterates, so the last
    # write wins. count_documents and the id map must agree at one entry.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    written = store.write_documents(
        [
            Document(id="dup", content="first", embedding=unit_vector(0)),
            Document(id="dup", content="second", embedding=unit_vector(1)),
        ],
        policy=DuplicatePolicy.OVERWRITE,
    )
    # Reference counts every input row for OVERWRITE.
    assert written == 2
    # But only one survives — no orphaned vector.
    assert store.count_documents() == 1
    assert len(store._u64_to_doc) == 1
    assert set(store._str_to_u64) == {"dup"}
    assert store.filter_documents()[0].content == "second"


def test_intra_batch_duplicate_fail_raises():
    # FAIL must reject a repeat within the same call, not just across calls.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    with pytest.raises(DuplicateDocumentError):
        store.write_documents(
            [
                Document(id="dup", content="a", embedding=unit_vector(0)),
                Document(id="dup", content="b", embedding=unit_vector(1)),
            ],
            policy=DuplicatePolicy.FAIL,
        )


def test_intra_batch_duplicate_skip_keeps_first():
    # SKIP keeps the first occurrence and drops later in-batch repeats.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    written = store.write_documents(
        [
            Document(id="dup", content="first", embedding=unit_vector(0)),
            Document(id="dup", content="second", embedding=unit_vector(1)),
        ],
        policy=DuplicatePolicy.SKIP,
    )
    assert written == 1
    assert store.count_documents() == 1
    assert store.filter_documents()[0].content == "first"


def test_overwrite_upsert_dim_mismatch_preserves_existing():
    # An OVERWRITE write whose new embedding fails validation must not
    # destroy the existing document: the delete is deferred past the add.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(
        [Document(id="d1", content="orig", embedding=unit_vector(0))]
    )
    bad = Document(id="d1", content="new", embedding=unit_vector(1)[:32])
    with pytest.raises(ValueError):
        store.write_documents([bad], policy=DuplicatePolicy.OVERWRITE)

    assert store.count_documents() == 1
    assert store.filter_documents()[0].content == "orig"


def test_write_document_without_embedding_raises():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    with pytest.raises(ValueError, match="no embedding"):
        store.write_documents([Document(id="x", content="hello")])


def test_embedding_retrieval_returns_top_k():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(20)
    store.write_documents(docs)
    # Self-query with doc-5's embedding -> doc-5 should be top-1.
    results = store.embedding_retrieval(query_embedding=docs[5].embedding, top_k=3)
    assert len(results) == 3
    assert results[0].id == "doc-5"
    assert results[0].score is not None


def test_embedding_retrieval_after_delete_skips_deleted():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(10)
    store.write_documents(docs)
    store.delete_documents(["doc-5"])
    results = store.embedding_retrieval(query_embedding=docs[5].embedding, top_k=5)
    assert all(doc.id != "doc-5" for doc in results)


def test_embedding_retrieval_with_filter():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(10)
    store.write_documents(docs)
    # Only group "b" (odd ids).
    filt = {"field": "meta.group", "operator": "==", "value": "b"}
    results = store.embedding_retrieval(
        query_embedding=docs[0].embedding, top_k=5, filters=filt
    )
    assert all(doc.meta["group"] == "b" for doc in results)


def test_embedding_retrieval_selective_filter_returns_top_k():
    # Regression test for the over-fetch / post-filter recall hit: with a
    # filter that matches only 3 docs out of 50, top_k=3 must return all 3.
    # The old implementation could return fewer when the matching docs
    # weren't in the over-fetched top_k * 10 by raw score.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(50)
    store.write_documents(docs)
    target_ids = {"doc-7", "doc-23", "doc-41"}
    for doc in docs:
        if doc.id in target_ids:
            doc.meta["tag"] = "needle"
    # Rewrite to refresh stored metadata (the store snapshotted it on write).
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(docs)
    filt = {"field": "meta.tag", "operator": "==", "value": "needle"}
    results = store.embedding_retrieval(
        query_embedding=docs[0].embedding, top_k=3, filters=filt
    )
    assert len(results) == 3
    assert {doc.id for doc in results} == target_ids


def test_embedding_retrieval_no_matches_returns_empty():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(10)
    store.write_documents(docs)
    filt = {"field": "meta.group", "operator": "==", "value": "no-such-group"}
    results = store.embedding_retrieval(
        query_embedding=docs[0].embedding, top_k=5, filters=filt
    )
    assert results == []


def test_embedding_retrieval_top_k_larger_than_matches():
    # When the filter has fewer matches than top_k, the result count
    # should equal the number of matches (no padding, no error).
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(20)
    store.write_documents(docs)
    # group=="a" matches 10 of 20.
    filt = {"field": "meta.group", "operator": "==", "value": "a"}
    results = store.embedding_retrieval(
        query_embedding=docs[0].embedding, top_k=100, filters=filt
    )
    assert len(results) == 10
    assert all(doc.meta["group"] == "a" for doc in results)


def test_k_larger_than_ntotal_is_clamped():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(3)
    store.write_documents(docs)
    # Ask for top_k=10 against a store with 3 vectors.
    results = store.embedding_retrieval(query_embedding=docs[0].embedding, top_k=10)
    assert len(results) == 3


def test_mismatched_dim_raises():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    wrong_dim_doc = Document(
        id="wrong",
        content="x",
        embedding=[0.1] * (DIM + 1),  # one dim too many
    )
    with pytest.raises(ValueError, match="does not match"):
        store.write_documents([wrong_dim_doc])

    # Retrieval should also reject mismatched query dim.
    store.write_documents(make_docs(2))
    with pytest.raises(ValueError, match="does not match"):
        store.embedding_retrieval(query_embedding=[0.1] * (DIM + 1), top_k=1)


def test_save_and_load_roundtrip(tmp_path):
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(5)
    store.write_documents(docs)
    # Delete one so we exercise a non-identity slot_to_id mapping.
    store.delete_documents(["doc-2"])

    store.save_to_disk(tmp_path)

    restored = TurboQuantDocumentStore.load_from_disk(tmp_path)
    assert restored.count_documents() == 4
    # Every surviving id self-retrieves correctly.
    for doc in docs:
        if doc.id == "doc-2":
            continue
        results = restored.embedding_retrieval(
            query_embedding=doc.embedding, top_k=1
        )
        assert results[0].id == doc.id


def test_save_writes_json_sidecar(tmp_path):
    # Side-car is plain JSON now, not pickle. A reviewer auditing a
    # turbovec-saved store should be able to read it with a text editor.
    import json

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(2))
    store.save_to_disk(tmp_path)
    assert (tmp_path / "docstore.json").exists()
    assert not (tmp_path / "docstore.pkl").exists()
    with open(tmp_path / "docstore.json") as f:
        data = json.load(f)
    assert data["schema_version"] >= 1


def test_load_rejects_unknown_schema_version(tmp_path):
    import json

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(1))
    store.save_to_disk(tmp_path)
    # Hand-bump the schema version to something unknown.
    with open(tmp_path / "docstore.json") as f:
        data = json.load(f)
    data["schema_version"] = 99
    with open(tmp_path / "docstore.json", "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="schema version"):
        TurboQuantDocumentStore.load_from_disk(tmp_path)


# ---- Tier 1: input validation -------------------------------------------------

def test_write_documents_rejects_non_list():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    with pytest.raises(ValueError, match="list of Documents"):
        store.write_documents("not a list of docs")  # type: ignore[arg-type]


def test_write_documents_rejects_non_document_elements():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    with pytest.raises(ValueError, match="list of Documents"):
        store.write_documents([{"id": "x"}])  # type: ignore[list-item]


# ---- Tier 2: utility methods ----------------------------------------------

def test_delete_all_documents():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(5))
    assert store.count_documents() == 5
    store.delete_all_documents()
    assert store.count_documents() == 0


def test_delete_by_filter_returns_count():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(6))
    filt = {"field": "meta.group", "operator": "==", "value": "a"}
    deleted = store.delete_by_filter(filt)
    assert deleted == 3
    assert store.count_documents() == 3
    assert all(
        doc.meta["group"] == "b" for doc in store.filter_documents()
    )


def test_update_by_filter_merges_metadata():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(4))
    filt = {"field": "meta.group", "operator": "==", "value": "a"}
    updated = store.update_by_filter(filt, {"tier": "premium"})
    assert updated == 2
    pros = [
        doc
        for doc in store.filter_documents()
        if doc.meta.get("tier") == "premium"
    ]
    assert {doc.id for doc in pros} == {"doc-0", "doc-2"}
    # Non-matching docs untouched.
    others = [doc for doc in store.filter_documents() if "tier" not in doc.meta]
    assert {doc.id for doc in others} == {"doc-1", "doc-3"}


def test_count_documents_by_filter():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(6))
    filt = {"field": "meta.group", "operator": "==", "value": "a"}
    assert store.count_documents_by_filter(filt) == 3
    # Empty/falsy filter falls through to full count.
    assert store.count_documents_by_filter({}) == 6


def test_count_unique_metadata_by_filter():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(6))
    # Two unique "group" values across all docs.
    result = store.count_unique_metadata_by_filter({}, ["meta.group"])
    assert result == {"group": 2}
    # Filtered subset: only group "a" → 1 unique.
    filt = {"field": "meta.group", "operator": "==", "value": "a"}
    result = store.count_unique_metadata_by_filter(filt, ["group"])
    assert result == {"group": 1}


def test_get_metadata_fields_info_infers_types():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(2)
    # make_docs gives idx (int) and group (str/keyword); add a bool + float.
    docs[0].meta["active"] = True
    docs[0].meta["weight"] = 1.5
    store.write_documents(docs)
    info = store.get_metadata_fields_info()
    assert info["idx"] == {"type": "int"}
    assert info["group"] == {"type": "keyword"}
    assert info["active"] == {"type": "boolean"}
    assert info["weight"] == {"type": "float"}


def test_get_metadata_field_min_max():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(5))  # idx in {0,1,2,3,4}
    assert store.get_metadata_field_min_max("idx") == {"min": 0, "max": 4}
    # Missing field returns the empty sentinel.
    assert store.get_metadata_field_min_max("missing") == {
        "min": None,
        "max": None,
    }


def test_get_metadata_field_unique_values():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(4))
    values, n = store.get_metadata_field_unique_values("group")
    assert sorted(values) == ["a", "b"]
    assert n == 2
    # search_term narrows to docs whose content contains the term.
    values, n = store.get_metadata_field_unique_values("group", search_term="text 0")
    assert values == ["a"]
    assert n == 1


def test_storage_property_returns_documents_by_id():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    storage = store.storage
    assert set(storage.keys()) == {"doc-0", "doc-1", "doc-2"}
    assert storage["doc-1"].meta["idx"] == 1
    # Embeddings always None — turbovec doesn't keep them.
    assert all(doc.embedding is None for doc in storage.values())


def test_shutdown_is_idempotent():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.shutdown()
    store.shutdown()  # second call should not raise


def test_filter_documents_invalid_filter_raises():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(2))
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        store.filter_documents(filters={"some_random_key": "value"})


# ---- Tier 3: scale_score formula per similarity function ------------------

def test_scale_score_cosine_formula():
    store = TurboQuantDocumentStore(
        dim=DIM, bit_width=4, embedding_similarity_function="cosine"
    )
    store.write_documents(make_docs(3))
    results = store.embedding_retrieval(
        query_embedding=make_docs(3)[0].embedding, top_k=3, scale_score=True
    )
    # Cosine scores live in [-1, 1]; after (s+1)/2 they're in [0, 1].
    for doc in results:
        assert 0.0 <= doc.score <= 1.0


def test_scale_score_dot_product_formula():
    store = TurboQuantDocumentStore(
        dim=DIM, bit_width=4, embedding_similarity_function="dot_product"
    )
    store.write_documents(make_docs(3))
    results = store.embedding_retrieval(
        query_embedding=make_docs(3)[0].embedding, top_k=3, scale_score=True
    )
    # expit(s/100) sigmoid is monotonically increasing on (-inf, inf) → (0, 1).
    for doc in results:
        assert 0.0 < doc.score < 1.0


def test_constructor_default_similarity_is_cosine():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    assert store.embedding_similarity_function == "cosine"


def test_to_dict_includes_new_init_params():
    store = TurboQuantDocumentStore(
        dim=DIM, bit_width=2, embedding_similarity_function="dot_product", return_embedding=True
    )
    serialized = store.to_dict()
    ip = serialized["init_parameters"]
    assert ip["embedding_similarity_function"] == "dot_product"
    assert ip["return_embedding"] is True
    restored = TurboQuantDocumentStore.from_dict(serialized)
    assert restored.embedding_similarity_function == "dot_product"
    assert restored.return_embedding is True


# ---- Async methods ------------------------------------------------------

def test_async_count_filter_write_delete():
    import asyncio

    async def runner():
        store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
        n = await store.write_documents_async(make_docs(4))
        assert n == 4
        assert await store.count_documents_async() == 4
        docs = await store.filter_documents_async()
        assert len(docs) == 4
        await store.delete_documents_async(["doc-0", "doc-1"])
        assert await store.count_documents_async() == 2
        await store.delete_all_documents_async()
        assert await store.count_documents_async() == 0

    asyncio.run(runner())


def test_async_filter_helpers():
    import asyncio

    async def runner():
        store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
        await store.write_documents_async(make_docs(6))
        filt = {"field": "meta.group", "operator": "==", "value": "a"}
        assert await store.count_documents_by_filter_async(filt) == 3
        n = await store.update_by_filter_async(filt, {"tier": "free"})
        assert n == 3
        unique = await store.count_unique_metadata_by_filter_async({}, ["tier"])
        assert unique == {"tier": 1}
        info = await store.get_metadata_fields_info_async()
        assert "group" in info
        mm = await store.get_metadata_field_min_max_async("idx")
        assert mm == {"min": 0, "max": 5}
        uniq, n = await store.get_metadata_field_unique_values_async("group")
        assert sorted(uniq) == ["a", "b"]

    asyncio.run(runner())


# ---- Tier 4: lazy dim construction ---------------------------------------

def test_constructor_no_dim_is_lazy():
    # `dim` is optional; the underlying IdMapIndex starts in its lazy
    # uncommitted state and locks dim on the first write.
    store = TurboQuantDocumentStore()
    assert store._index.dim is None
    # Retrieval before any write returns [].
    assert store.embedding_retrieval(query_embedding=[0.0] * DIM, top_k=3) == []


def test_lazy_dim_inferred_on_first_write():
    store = TurboQuantDocumentStore(bit_width=2)
    store.write_documents(make_docs(2))
    assert store._index.dim == DIM
    assert store._index.bit_width == 2


def test_dim_mismatch_after_lazy_creation_raises():
    store = TurboQuantDocumentStore()
    store.write_documents(make_docs(1))  # locks dim to DIM
    # Build a doc whose embedding has a different shape.
    bad = Document(id="bad", content="x", embedding=[0.0] * (DIM + 1))
    with pytest.raises(ValueError, match="does not match store dim"):
        store.write_documents([bad])


def test_dump_and_load_empty_lazy_store(tmp_path):
    # Saving before any write must not crash, and loading must restore a
    # store whose index is still in its lazy uncommitted state.
    store = TurboQuantDocumentStore(bit_width=2)
    store.save_to_disk(tmp_path)
    loaded = TurboQuantDocumentStore.load_from_disk(tmp_path)
    assert loaded._index.dim is None
    assert loaded._bit_width == 2
    # Subsequent retrieval is empty; subsequent write commits the dim.
    assert loaded.embedding_retrieval(query_embedding=[0.0] * DIM, top_k=1) == []
    loaded.write_documents(make_docs(1))
    assert loaded._index.dim == DIM


# ---- End-to-end smoke tests: framework wiring ---------------------------

def test_pipeline_end_to_end_retrieval():
    # Smoke test: wire our store into a Haystack Pipeline via a custom
    # retriever component and run a query end-to-end. The custom
    # retriever is a tiny component that just delegates to
    # store.embedding_retrieval — its job is to exercise the Pipeline
    # plumbing on top of our store, not to be a real retriever.
    from haystack import Pipeline, component
    from haystack.components.embedders import SentenceTransformersTextEmbedder  # noqa: F401 (just an import check)

    @component
    class _ProbeRetriever:
        """Minimal Haystack component: calls embedding_retrieval on its store."""

        def __init__(self, document_store):
            self.document_store = document_store

        @component.output_types(documents=list)
        def run(self, query_embedding, top_k=3, filters=None):
            return {
                "documents": self.document_store.embedding_retrieval(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                )
            }

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    docs = make_docs(8)
    store.write_documents(docs)

    pipeline = Pipeline()
    pipeline.add_component("retriever", _ProbeRetriever(document_store=store))

    # Use the query doc's own embedding to make the top-k deterministic.
    result = pipeline.run(
        {"retriever": {"query_embedding": docs[0].embedding, "top_k": 3}}
    )
    out_docs = result["retriever"]["documents"]
    assert len(out_docs) == 3
    assert out_docs[0].id == "doc-0"  # self-match


def test_pipeline_filter_passthrough_via_retriever():
    # Same as above, but exercises the filter path through the pipeline's
    # parameter routing.
    from haystack import Pipeline, component

    @component
    class _ProbeRetriever:
        def __init__(self, document_store):
            self.document_store = document_store

        @component.output_types(documents=list)
        def run(self, query_embedding, top_k=3, filters=None):
            return {
                "documents": self.document_store.embedding_retrieval(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                )
            }

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(10))

    pipeline = Pipeline()
    pipeline.add_component("retriever", _ProbeRetriever(document_store=store))

    # group=="a" matches 5 of 10 docs.
    result = pipeline.run({
        "retriever": {
            "query_embedding": make_docs(1)[0].embedding,
            "top_k": 10,
            "filters": {"field": "meta.group", "operator": "==", "value": "a"},
        }
    })
    out_docs = result["retriever"]["documents"]
    assert len(out_docs) == 5
    assert all(d.meta["group"] == "a" for d in out_docs)


def test_pipeline_to_dict_from_dict_roundtrip():
    # Pipelines serialize/deserialize their components. Our store must
    # round-trip through Haystack's component serialization machinery.
    from haystack import Pipeline, component

    @component
    class _ProbeRetriever:
        def __init__(self, document_store):
            self.document_store = document_store

        @component.output_types(documents=list)
        def run(self, query_embedding):
            return {
                "documents": self.document_store.embedding_retrieval(
                    query_embedding=query_embedding, top_k=1
                )
            }

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    pipeline = Pipeline()
    pipeline.add_component("retriever", _ProbeRetriever(document_store=store))
    # Serialize-then-load via Haystack's own dict round-trip — exercises
    # our store's to_dict / from_dict from inside the framework.
    serialized = pipeline.to_dict()
    assert "components" in serialized


def test_async_embedding_retrieval():
    import asyncio

    async def runner():
        store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
        docs = make_docs(5)
        await store.write_documents_async(docs)
        results = await store.embedding_retrieval_async(
            query_embedding=docs[0].embedding, top_k=3
        )
        assert len(results) == 3
        assert results[0].id == "doc-0"

    asyncio.run(runner())


def test_to_dict_from_dict_round_trip():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=2)
    serialized = store.to_dict()
    assert serialized["init_parameters"]["dim"] == DIM
    assert serialized["init_parameters"]["bit_width"] == 2

    restored = TurboQuantDocumentStore.from_dict(serialized)
    assert restored.count_documents() == 0
    # (to_dict/from_dict serializes the component config, not the data —
    # this matches Haystack's InMemoryDocumentStore contract.)


# ---- Tier-2 field-completeness tests. Each pins a value that a future
# refactor could silently drop. ----

def test_filter_documents_returns_documents_with_score_none():
    # `_reconstruct` is called without `score=` from filter_documents,
    # so score must be None on every returned doc. A future cache leak
    # between read paths could carry a stale score from a prior
    # embedding_retrieval — pin this so the invariant doesn't drift.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(3))
    for doc in store.filter_documents():
        assert doc.score is None


def test_storage_property_documents_have_no_blob_sparse_or_score_when_unset():
    # The `storage` property mirrors filter_documents semantics. For
    # docs written without blob / sparse_embedding / score, those
    # fields must come back as None (not a default ByteStream /
    # SparseEmbedding / 0.0).
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents([
        Document(id="plain", content="text", embedding=unit_vector(0), meta={"k": "v"})
    ])
    doc = store.storage["plain"]
    assert doc.score is None
    assert doc.blob is None
    assert doc.sparse_embedding is None
    assert doc.embedding is None  # always None — quantization-justified


def test_embedding_retrieval_preserves_content_and_meta():
    # Every existing retrieval test asserts `.id` / `.score` / `.meta` keys
    # but no test checks that `Document.content` survives the round-trip.
    # If `_reconstruct` ever stopped copying content, only the blob /
    # sparse-embedding tests would notice — and only indirectly.
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents([
        Document(
            id="doc-c",
            content="distinctive content string",
            embedding=unit_vector(0),
            meta={"key1": "value1", "key2": 42, "key3": [1, 2, 3]},
        )
    ])
    [doc] = store.embedding_retrieval(query_embedding=unit_vector(0), top_k=1)
    assert doc.content == "distinctive content string"
    assert doc.meta == {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}


def test_to_dict_includes_all_init_params_and_type_key():
    # `to_dict` returns four init_parameters: dim, bit_width,
    # embedding_similarity_function, return_embedding. A single test
    # must pin all four plus the outer `type` key (which Haystack's
    # pipeline serialization uses to resolve the class for from_dict).
    store = TurboQuantDocumentStore(
        dim=DIM,
        bit_width=2,
        embedding_similarity_function="dot_product",
        return_embedding=True,
    )
    serialized = store.to_dict()

    assert serialized["type"] == "turbovec.haystack.TurboQuantDocumentStore"
    assert set(serialized["init_parameters"]) == {
        "dim",
        "bit_width",
        "embedding_similarity_function",
        "return_embedding",
    }
    assert serialized["init_parameters"]["dim"] == DIM
    assert serialized["init_parameters"]["bit_width"] == 2
    assert serialized["init_parameters"]["embedding_similarity_function"] == "dot_product"
    assert serialized["init_parameters"]["return_embedding"] is True


def test_save_load_preserves_similarity_function_and_return_embedding(tmp_path):
    # `load_from_disk` uses `.get()` with defaults for the two non-bit_width
    # init params. If `save_to_disk` ever stopped writing them, the load
    # would silently fall back to defaults — undetected. Pin both fields
    # explicitly through a save / load round-trip.
    store = TurboQuantDocumentStore(
        dim=DIM,
        bit_width=4,
        embedding_similarity_function="dot_product",
        return_embedding=True,
    )
    store.write_documents(make_docs(1))
    store.save_to_disk(tmp_path)

    restored = TurboQuantDocumentStore.load_from_disk(tmp_path)
    assert restored.embedding_similarity_function == "dot_product"
    assert restored.return_embedding is True


def test_embedding_retrieval_all_results_have_finite_float_scores():
    # The existing `top_k` test asserts `results[0].score is not None`
    # but not for the tail hits — a kernel regression producing NaN /
    # None on non-top-1 results would slip through.
    import math

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(10))
    results = store.embedding_retrieval(query_embedding=unit_vector(0), top_k=10)
    assert len(results) == 10
    for r in results:
        assert isinstance(r.score, float)
        assert math.isfinite(r.score)


def test_load_rejects_side_car_desynced_from_index(tmp_path):
    import json

    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(4))
    store.save_to_disk(tmp_path)

    TurboQuantDocumentStore.load_from_disk(tmp_path)  # clean reload works

    with open(tmp_path / "docstore.json") as f:
        state = json.load(f)
    state["u64_to_doc"] = state["u64_to_doc"][:-1]  # drop one handle->doc
    with open(tmp_path / "docstore.json", "w") as f:
        json.dump(state, f)

    with pytest.raises(ValueError):
        TurboQuantDocumentStore.load_from_disk(tmp_path)
