"""Pickle / copy semantics: the core in-memory serialization API
(`to_bytes` / `from_bytes`) and correct `pickle` / `copy.copy` /
`copy.deepcopy` behaviour on all four integration stores.

Covers issues #148 (LlamaIndex pickle silently returned an EMPTY store —
total data loss) and #149 (copy.copy shared the Rust index so mutations
bled across copies; deepcopy raised, so no safe copy existed).

The embedders here are deterministic *across processes* (hashlib-seeded,
not `hash()`-seeded, since spawn children get a different hash seed) so
the multiprocessing smoke tests can compare search results between
parent and child.
"""

from __future__ import annotations

import copy
import hashlib
import multiprocessing
import pickle
import threading

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex

DIM = 32


def _text_seed(text: str) -> int:
    return int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "little")


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _embed_text(text: str) -> list[float]:
    return _unit(_text_seed(text)).tolist()


def _vectors(n: int, seed: int = 7) -> np.ndarray:
    return np.stack([_unit(seed + i) for i in range(n)])


# ---------------------------------------------------------------------------
# Core binding: to_bytes / from_bytes
# ---------------------------------------------------------------------------


def test_index_to_bytes_round_trips_and_matches_file(tmp_path):
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_vectors(16))
    payload = idx.to_bytes()
    assert isinstance(payload, bytes)

    # Byte-identical to the .tv file write() produces.
    idx.write(str(tmp_path / "index.tv"))
    assert payload == (tmp_path / "index.tv").read_bytes()

    back = TurboQuantIndex.from_bytes(payload)
    assert len(back) == 16
    q = _vectors(2, seed=99)
    s0, i0 = idx.search(q, 5)
    s1, i1 = back.search(q, 5)
    np.testing.assert_array_equal(s0, s1)
    np.testing.assert_array_equal(i0, i1)


def test_id_map_to_bytes_round_trips_and_matches_file(tmp_path):
    idx = IdMapIndex(DIM, 4)
    ids = np.arange(1000, 1016, dtype=np.uint64)
    idx.add_with_ids(_vectors(16), ids)
    payload = idx.to_bytes()
    assert isinstance(payload, bytes)

    idx.write(str(tmp_path / "index.tvim"))
    assert payload == (tmp_path / "index.tvim").read_bytes()

    back = IdMapIndex.from_bytes(payload)
    assert len(back) == 16
    assert all(int(i) in back for i in ids)
    q = _vectors(2, seed=99)
    s0, i0 = idx.search(q, 5)
    s1, i1 = back.search(q, 5)
    np.testing.assert_array_equal(s0, s1)
    np.testing.assert_array_equal(i0, i1)


def test_from_bytes_accepts_bytearray():
    idx = IdMapIndex(DIM, 4)
    idx.add_with_ids(_vectors(3), np.arange(3, dtype=np.uint64))
    back = IdMapIndex.from_bytes(bytearray(idx.to_bytes()))
    assert len(back) == 3


def test_from_bytes_rejects_corrupt_and_wrong_type():
    idx = IdMapIndex(DIM, 4)
    idx.add_with_ids(_vectors(3), np.arange(3, dtype=np.uint64))
    payload = idx.to_bytes()

    with pytest.raises(ValueError, match="magic"):
        IdMapIndex.from_bytes(b"garbage-not-a-tvim-payload")
    with pytest.raises(ValueError, match="truncated"):
        IdMapIndex.from_bytes(payload[:-5])
    with pytest.raises(TypeError, match="bytes-like"):
        IdMapIndex.from_bytes("not bytes")
    with pytest.raises(ValueError, match="magic"):
        TurboQuantIndex.from_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    # A .tv payload is not a .tvim payload.
    with pytest.raises(ValueError, match="TVIM"):
        IdMapIndex.from_bytes(TurboQuantIndex(DIM, 4).to_bytes())


def test_lazy_index_bytes_round_trip():
    lazy = IdMapIndex(bit_width=3)
    back = IdMapIndex.from_bytes(lazy.to_bytes())
    assert back.dim is None
    assert back.bit_width == 3
    assert len(back) == 0


# ---------------------------------------------------------------------------
# Store fixtures
# ---------------------------------------------------------------------------


class LangchainEmb:
    """Deterministic, picklable, cross-process-stable embedder."""

    def embed_documents(self, texts):
        return [_embed_text(t) for t in texts]

    def embed_query(self, text):
        return _embed_text(text)


class AgnoEmb:
    dimensions = DIM

    def get_embedding(self, text):
        return _embed_text(text)


LC_TEXTS = ["alpha", "beta", "gamma"]
LC_IDS = ["id-a", "id-b", "id-c"]


def _make_langchain():
    pytest.importorskip("langchain_core")
    from turbovec.langchain import TurboQuantVectorStore

    store = TurboQuantVectorStore(embedding=LangchainEmb())
    store.add_texts(
        LC_TEXTS,
        metadatas=[{"n": i, "tag": "t"} for i in range(3)],
        ids=LC_IDS,
    )
    return store


def _lc_search_ids(store, text="alpha", k=3):
    return sorted(doc.id for doc in store.similarity_search(text, k=k))


def _make_haystack():
    pytest.importorskip("haystack")
    from haystack import Document

    from turbovec.haystack import TurboQuantDocumentStore

    store = TurboQuantDocumentStore(dim=DIM)
    store.write_documents(
        [
            Document(id=f"doc-{i}", content=f"content {i}", embedding=_unit(i).tolist(), meta={"n": i})
            for i in range(3)
        ]
    )
    return store


def _hs_search_ids(store, k=3):
    return sorted(d.id for d in store.embedding_retrieval(_unit(0).tolist(), top_k=k))


def _make_llama():
    pytest.importorskip("llama_index")
    from llama_index.core.schema import TextNode

    from turbovec.llama_index import TurboQuantVectorStore

    store = TurboQuantVectorStore(bit_width=4)
    store.add(
        [
            TextNode(
                id_=f"node-{i}",
                text=f"text {i}",
                embedding=_unit(i).tolist(),
                metadata={"n": i},
            )
            for i in range(3)
        ]
    )
    return store


def _llama_query(store, k=3):
    from llama_index.core.vector_stores.types import VectorStoreQuery

    return store.query(
        VectorStoreQuery(query_embedding=_unit(0).tolist(), similarity_top_k=k)
    )


def _make_agno():
    pytest.importorskip("agno")
    from agno.knowledge.document import Document

    from turbovec.agno import TurboQuantVectorDb

    db = TurboQuantVectorDb(embedder=AgnoEmb())
    db.create()
    db.insert(
        "hash-1",
        [
            Document(id=f"agno-{i}", content=f"agno content {i}", embedding=_unit(i).tolist())
            for i in range(3)
        ],
    )
    return db


def _agno_search_contents(db, k=3):
    return sorted(d.content for d in db.search("agno content 0", limit=k))


# ---------------------------------------------------------------------------
# Pickle round-trips REAL data (headline: on main the llama store came
# back silently EMPTY — issue #148)
# ---------------------------------------------------------------------------


def test_llama_pickle_round_trips_data():
    store = _make_llama()
    before = _llama_query(store)
    assert sorted(before.ids) == ["node-0", "node-1", "node-2"]

    restored = pickle.loads(pickle.dumps(store))
    after = _llama_query(restored)
    # The #148 failure mode: after.ids == [] with no error.
    assert sorted(after.ids) == sorted(before.ids)
    assert after.similarities == before.similarities
    # Full node payload survives (metadata, text).
    nodes = restored.get_nodes(node_ids=["node-1"])
    assert nodes[0].metadata == {"n": 1}
    assert "text 1" in nodes[0].get_content()
    # Persistence-affecting state survives: a post-restore add issues a
    # fresh handle (no collision with the restored ones) and is
    # immediately searchable.
    from llama_index.core.schema import TextNode

    restored.add([TextNode(id_="node-9", text="text 9", embedding=_unit(9).tolist())])
    assert sorted(_llama_query(restored, k=4).ids) == [
        "node-0", "node-1", "node-2", "node-9",
    ]


def test_langchain_pickle_round_trips_data():
    store = _make_langchain()
    restored = pickle.loads(pickle.dumps(store))
    assert _lc_search_ids(restored) == _lc_search_ids(store)
    docs = restored.get_by_ids(["id-b"])
    assert docs[0].page_content == "beta"
    assert docs[0].metadata == {"n": 1, "tag": "t"}
    # Score parity, not just membership.
    s0 = store.similarity_search_with_score("alpha", k=3)
    s1 = restored.similarity_search_with_score("alpha", k=3)
    assert [(d.id, s) for d, s in s0] == [(d.id, s) for d, s in s1]
    restored.add_texts(["delta"], ids=["id-d"])
    assert _lc_search_ids(restored, k=4) == sorted(LC_IDS + ["id-d"])


def test_haystack_pickle_round_trips_data():
    store = _make_haystack()
    restored = pickle.loads(pickle.dumps(store))
    assert restored.count_documents() == 3
    assert _hs_search_ids(restored) == _hs_search_ids(store)
    doc = restored.filter_documents(
        {"field": "meta.n", "operator": "==", "value": 1}
    )
    assert len(doc) == 1 and doc[0].id == "doc-1"
    # A restored store owns a working executor and lock: writing works.
    from haystack import Document

    restored.write_documents([Document(id="doc-9", content="c9", embedding=_unit(9).tolist())])
    assert restored.count_documents() == 4


def test_agno_pickle_round_trips_data():
    db = _make_agno()
    restored = pickle.loads(pickle.dumps(db))
    assert restored.get_count() == 3
    assert _agno_search_contents(restored) == _agno_search_contents(db)
    assert restored.content_hash_exists("hash-1")
    from agno.knowledge.document import Document

    restored.insert("hash-2", [Document(id="agno-9", content="agno content 9", embedding=_unit(9).tolist())])
    assert restored.get_count() == 4


def test_agno_uncreated_store_pickles():
    pytest.importorskip("agno")
    from turbovec.agno import TurboQuantVectorDb

    db = TurboQuantVectorDb(embedder=AgnoEmb())
    restored = pickle.loads(pickle.dumps(db))
    assert restored.exists() is False
    restored.create()
    assert restored.get_count() == 0


# ---------------------------------------------------------------------------
# deepcopy independence (the #149 repro), both directions
# ---------------------------------------------------------------------------


def test_langchain_deepcopy_is_independent():
    store = _make_langchain()
    clone = copy.deepcopy(store)
    # Mutate the copy: original untouched.
    clone.add_texts(["zeta"], ids=["id-z"])
    assert len(store._index) == 3
    assert "id-z" not in store._docs
    # Mutate the original: copy untouched.
    store.delete(["id-a"])
    assert len(clone._index) == 4
    assert "id-a" in clone._docs
    assert _lc_search_ids(clone, k=4) == sorted(LC_IDS + ["id-z"])


def test_haystack_deepcopy_is_independent():
    store = _make_haystack()
    clone = copy.deepcopy(store)
    from haystack import Document

    clone.write_documents([Document(id="doc-z", content="cz", embedding=_unit(42).tolist())])
    assert store.count_documents() == 3
    store.delete_documents(["doc-0"])
    assert clone.count_documents() == 4


def test_llama_deepcopy_is_independent():
    store = _make_llama()
    clone = copy.deepcopy(store)
    from llama_index.core.schema import TextNode

    clone.add([TextNode(id_="node-z", text="tz", embedding=_unit(42).tolist())])
    assert sorted(_llama_query(store, k=10).ids) == ["node-0", "node-1", "node-2"]
    store.delete_nodes(node_ids=["node-0"])
    assert sorted(_llama_query(clone, k=10).ids) == [
        "node-0", "node-1", "node-2", "node-z",
    ]


def test_agno_deepcopy_is_independent():
    db = _make_agno()
    clone = copy.deepcopy(db)
    from agno.knowledge.document import Document

    clone.insert("hash-z", [Document(id="agno-z", content="agno z", embedding=_unit(42).tolist())])
    assert db.get_count() == 3
    db.delete_by_id(next(iter(db._str_to_u64)))
    assert clone.get_count() == 4


# ---------------------------------------------------------------------------
# copy.copy no longer bleeds (it now equals deepcopy — there is no
# meaningful shallow copy of a store)
# ---------------------------------------------------------------------------


def test_langchain_copy_does_not_bleed():
    store = _make_langchain()
    clone = copy.copy(store)
    assert clone._index is not store._index
    assert clone._docs is not store._docs
    clone.add_texts(["zeta"], ids=["id-z"])
    assert len(store._index) == 3
    assert list(store._docs) == LC_IDS


def test_haystack_copy_does_not_bleed():
    store = _make_haystack()
    clone = copy.copy(store)
    from haystack import Document

    clone.write_documents([Document(id="doc-z", content="cz", embedding=_unit(42).tolist())])
    assert store.count_documents() == 3


def test_llama_copy_does_not_bleed():
    store = _make_llama()
    clone = copy.copy(store)
    from llama_index.core.schema import TextNode

    clone.add([TextNode(id_="node-z", text="tz", embedding=_unit(42).tolist())])
    assert sorted(_llama_query(store, k=10).ids) == ["node-0", "node-1", "node-2"]


def test_agno_copy_does_not_bleed():
    db = _make_agno()
    clone = copy.copy(db)
    from agno.knowledge.document import Document

    clone.insert("hash-z", [Document(id="agno-z", content="agno z", embedding=_unit(42).tolist())])
    assert db.get_count() == 3


# ---------------------------------------------------------------------------
# Similarity-mode round-trip: pickle and deepcopy must preserve the
# store's mode (cosine vs dot_product) with score parity. The mode lives
# outside the index bytes (a pydantic PrivateAttr on the llama store),
# so a __getstate__ that misses it silently resets a dot_product store
# to cosine on restore — same ranking, diverging scores, and future adds
# normalized when they shouldn't be. Embeddings here are deliberately
# NON-unit so cosine and dot_product scores differ.
# ---------------------------------------------------------------------------


def _scaled(seed: int) -> np.ndarray:
    """Deterministic non-unit vector (norm varies with the seed)."""
    return _unit(seed) * (2.0 + seed % 5)


def _embed_text_scaled(text: str) -> list[float]:
    return _scaled(_text_seed(text) % 1000).tolist()


class ScaledLangchainEmb:
    def embed_documents(self, texts):
        return [_embed_text_scaled(t) for t in texts]

    def embed_query(self, text):
        return _embed_text_scaled(text)


class ScaledAgnoEmb:
    dimensions = DIM

    def get_embedding(self, text):
        return _embed_text_scaled(text)


def _round_trips(store):
    """The two restore paths under test, labelled."""
    return [
        ("pickle", pickle.loads(pickle.dumps(store))),
        ("deepcopy", copy.deepcopy(store)),
    ]


@pytest.mark.parametrize("mode", ["cosine", "dot_product"])
def test_langchain_mode_survives_pickle_and_deepcopy(mode):
    pytest.importorskip("langchain_core")
    from turbovec.langchain import TurboQuantVectorStore

    store = TurboQuantVectorStore(embedding=ScaledLangchainEmb(), similarity=mode)
    store.add_texts(LC_TEXTS, ids=LC_IDS)
    before = store.similarity_search_with_score("alpha", k=3)
    for label, restored in _round_trips(store):
        assert restored.similarity == mode, label
        after = restored.similarity_search_with_score("alpha", k=3)
        assert [(d.id, s) for d, s in after] == [(d.id, s) for d, s in before], label


@pytest.mark.parametrize("mode", ["cosine", "dot_product"])
def test_haystack_mode_survives_pickle_and_deepcopy(mode):
    pytest.importorskip("haystack")
    from haystack import Document

    from turbovec.haystack import TurboQuantDocumentStore

    store = TurboQuantDocumentStore(dim=DIM, embedding_similarity_function=mode)
    store.write_documents(
        [
            Document(id=f"doc-{i}", content=f"content {i}", embedding=_scaled(i).tolist())
            for i in range(3)
        ]
    )
    q = _scaled(0).tolist()
    before = [(d.id, d.score) for d in store.embedding_retrieval(q, top_k=3)]
    for label, restored in _round_trips(store):
        assert restored.embedding_similarity_function == mode, label
        after = [(d.id, d.score) for d in restored.embedding_retrieval(q, top_k=3)]
        assert after == before, label


@pytest.mark.parametrize("mode", ["cosine", "dot_product"])
def test_llama_mode_survives_pickle_and_deepcopy(mode):
    pytest.importorskip("llama_index")
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery

    from turbovec.llama_index import TurboQuantVectorStore

    store = TurboQuantVectorStore(similarity=mode)
    store.add(
        [
            TextNode(id_=f"node-{i}", text=f"text {i}", embedding=_scaled(i).tolist())
            for i in range(3)
        ]
    )
    query = VectorStoreQuery(query_embedding=_scaled(0).tolist(), similarity_top_k=3)
    before = store.query(query)
    for label, restored in _round_trips(store):
        # The mode is a pydantic PrivateAttr — the regression here was
        # a dot_product store silently unpickling as cosine.
        assert restored.similarity == mode, label
        after = restored.query(query)
        assert after.ids == before.ids, label
        assert after.similarities == before.similarities, label


@pytest.mark.parametrize("mode_name", ["cosine", "max_inner_product"])
def test_agno_mode_survives_pickle_and_deepcopy(mode_name):
    pytest.importorskip("agno")
    from agno.knowledge.document import Document
    from agno.vectordb.distance import Distance

    from turbovec.agno import TurboQuantVectorDb

    mode = getattr(Distance, mode_name)
    db = TurboQuantVectorDb(embedder=ScaledAgnoEmb(), distance=mode)
    db.create()
    db.insert(
        "hash-1",
        [
            Document(id=f"agno-{i}", content=f"agno content {i}", embedding=_scaled(i).tolist())
            for i in range(3)
        ],
    )
    before = [d.id for d in db.search("agno content 0", limit=3)]
    for label, restored in _round_trips(db):
        assert restored.distance == mode, label
        after = [d.id for d in restored.search("agno content 0", limit=3)]
        assert after == before, label
        # The mode drives insert-side normalization: a post-restore
        # insert must go through the same path as on the original.
        restored.insert(
            "hash-9",
            [Document(id="agno-9", content="agno content 9", embedding=_scaled(9).tolist())],
        )
        assert restored.get_count() == 4, label


# ---------------------------------------------------------------------------
# Snapshot isolation: __getstate__'s captured state must be immune to
# in-place metadata mutations that land after it returns (pickle
# serializes the state outside the store lock, so a shared inner dict
# would tear the payload mid-serialization).
# ---------------------------------------------------------------------------


def test_agno_getstate_snapshot_isolated_from_update_metadata():
    pytest.importorskip("agno")
    from agno.knowledge.document import Document

    from turbovec.agno import TurboQuantVectorDb

    db = TurboQuantVectorDb(embedder=AgnoEmb())
    db.create()
    db.insert(
        "hash-1",
        [
            Document(
                id=f"agno-{i}",
                content=f"agno content {i}",
                embedding=_unit(i).tolist(),
                content_id="cid-1",
                meta_data={"version": 1},
            )
            for i in range(3)
        ],
    )
    state = db.__getstate__()
    # In-place mutation landing after the snapshot was taken (the
    # deterministic form of the update-during-pickle interleave).
    db.update_metadata("cid-1", {"version": 2})
    metas = [d["meta_data"] for d in state["_u64_to_doc"].values()]
    assert metas == [{"version": 1}] * 3, "snapshot tore: post-snapshot update leaked in"
    # And the live store did apply the update — the snapshot is old, not
    # the mutation lost.
    assert all(
        d["meta_data"] == {"version": 2} for d in db._u64_to_doc.values()
    )


def test_haystack_getstate_snapshot_isolated_from_update_by_filter():
    pytest.importorskip("haystack")
    from haystack import Document

    from turbovec.haystack import TurboQuantDocumentStore

    store = TurboQuantDocumentStore(dim=DIM)
    store.write_documents(
        [
            Document(id=f"doc-{i}", content=f"content {i}", embedding=_unit(i).tolist(), meta={"version": 1})
            for i in range(3)
        ]
    )
    state = store.__getstate__()
    store.update_by_filter(
        {"field": "meta.version", "operator": "==", "value": 1}, {"version": 2}
    )
    metas = [d["meta"] for d in state["_u64_to_doc"].values()]
    assert metas == [{"version": 1}] * 3, "snapshot tore: post-snapshot update leaked in"


# ---------------------------------------------------------------------------
# The restored lock still serializes mutations (quick stress)
# ---------------------------------------------------------------------------


def test_unpickled_langchain_store_lock_serializes_writes():
    restored = pickle.loads(pickle.dumps(_make_langchain()))
    n_threads, per_thread = 4, 25
    errors = []

    def add_batch(t: int) -> None:
        try:
            for i in range(per_thread):
                restored.add_texts([f"t{t}-{i}"], ids=[f"tid-{t}-{i}"])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=add_batch, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    # 3 restored + every concurrent add landed exactly once.
    assert len(restored._index) == 3 + n_threads * per_thread
    assert len(restored._docs) == 3 + n_threads * per_thread


# ---------------------------------------------------------------------------
# Multiprocessing spawn smoke: pickle a populated store into a spawned
# child and search there (spawn, not fork — #147 is parked)
# ---------------------------------------------------------------------------


def _llama_child(payload: bytes, out) -> None:  # pragma: no cover - child process
    import pickle as _pickle

    store = _pickle.loads(payload)
    res = _llama_query(store)
    out.put(sorted(res.ids))


def _langchain_child(payload: bytes, out) -> None:  # pragma: no cover - child process
    import pickle as _pickle

    store = _pickle.loads(payload)
    out.put(_lc_search_ids(store))


@pytest.mark.parametrize(
    "maker, child, expected",
    [
        (_make_llama, _llama_child, ["node-0", "node-1", "node-2"]),
        (_make_langchain, _langchain_child, sorted(LC_IDS)),
    ],
    ids=["llama", "langchain"],
)
def test_spawned_child_searches_pickled_store(maker, child, expected):
    store = maker()
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    proc = ctx.Process(target=child, args=(pickle.dumps(store), out))
    proc.start()
    try:
        result = out.get(timeout=120)
    finally:
        proc.join(timeout=120)
    assert proc.exitcode == 0
    assert result == expected
