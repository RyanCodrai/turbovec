"""Tests for the Haystack DocumentStore integration."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("haystack")

from haystack import Document
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

    store.save(tmp_path)

    restored = TurboQuantDocumentStore.load(
        tmp_path, allow_dangerous_deserialization=True
    )
    assert restored.count_documents() == 4
    # Every surviving id self-retrieves correctly.
    for doc in docs:
        if doc.id == "doc-2":
            continue
        results = restored.embedding_retrieval(
            query_embedding=doc.embedding, top_k=1
        )
        assert results[0].id == doc.id


def test_load_refuses_without_flag(tmp_path):
    store = TurboQuantDocumentStore(dim=DIM, bit_width=4)
    store.write_documents(make_docs(2))
    store.save(tmp_path)
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
        TurboQuantDocumentStore.load(tmp_path)


def test_to_dict_from_dict_round_trip():
    store = TurboQuantDocumentStore(dim=DIM, bit_width=2)
    serialized = store.to_dict()
    assert serialized["init_parameters"]["dim"] == DIM
    assert serialized["init_parameters"]["bit_width"] == 2

    restored = TurboQuantDocumentStore.from_dict(serialized)
    assert restored.count_documents() == 0
    # (to_dict/from_dict serializes the component config, not the data —
    # this matches Haystack's InMemoryDocumentStore contract.)
