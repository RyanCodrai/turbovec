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

    store.save_to_disk(tmp_path)

    restored = TurboQuantDocumentStore.load_from_disk(
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
    store.save_to_disk(tmp_path)
    with pytest.raises(ValueError, match="allow_dangerous_deserialization"):
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
    loaded = TurboQuantDocumentStore.load_from_disk(
        tmp_path, allow_dangerous_deserialization=True
    )
    assert loaded._index.dim is None
    assert loaded._bit_width == 2
    # Subsequent retrieval is empty; subsequent write commits the dim.
    assert loaded.embedding_retrieval(query_embedding=[0.0] * DIM, top_k=1) == []
    loaded.write_documents(make_docs(1))
    assert loaded._index.dim == DIM


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
