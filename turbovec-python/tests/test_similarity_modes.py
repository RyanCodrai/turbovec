"""Per-store similarity modes (issue #114).

Every integration store supports two modes, fixed at construction:

- ``cosine`` (default): documents are L2-normalized at add time and
  queries at search time, so raw scores are true cosine similarity in
  [-1, 1], the (sim + 1) / 2 relevance mappings are correct, and
  threshold filtering works for embeddings of any magnitude.
- ``dot_product`` (opt-in): raw vectors, magnitude-aware ranking —
  exactly the pre-mode behavior.

Covered here:
- the three wave-8 threshold breakages as regression tests (non-unit
  embedders; fixed under cosine, pinned old-way under dot_product);
- reference-parity of cosine-mode ranking on non-unit embeddings;
- legacy stores written before the format v5 rotation break are refused
  on load with an actionable conversion error rather than silently
  mis-scoring (the pre-v5 `index.tvim` fixtures under
  tests/fixtures/legacy_pre_similarity can no longer be decoded, #206);
- mode round-trip through save/load and mode-conflict handling;
- zero-vector handling; and mode-parameter validation.

Geometry helper: documents are built with exact target cosines against
the query direction, with gaps far above 4-bit quantization noise, and
wildly varying magnitudes so dot-product ranking would disagree.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

DIM = 16
FIXTURES = Path(__file__).parent / "fixtures" / "legacy_pre_similarity"


def directed_docs(
    cosines: list[float],
    mags: list[float],
    dim: int = DIM,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(docs, query)`` where ``docs[i]`` has exact cosine
    ``cosines[i]`` against ``query`` and L2 norm ``mags[i]``; ``query``
    is a unit vector."""
    rng = np.random.default_rng(seed)
    qhat = rng.standard_normal(dim)
    qhat /= np.linalg.norm(qhat)
    docs = []
    for c, m in zip(cosines, mags):
        r = rng.standard_normal(dim)
        r -= (r @ qhat) * qhat
        r /= np.linalg.norm(r)
        docs.append(m * (c * qhat + np.sqrt(1.0 - c * c) * r))
    return np.asarray(docs, dtype=np.float32), qhat.astype(np.float32)


# All-negative geometry: every raw inner product is < -1 (magnitudes
# 12 / 6 / 3 times cosines -0.2 / -0.6 / -0.98 give -2.4 / -3.6 / -2.94),
# so the legacy (sim + 1) / 2 mapping clamps everything to 0.0. The true
# cosines map to relevances 0.4 / 0.2 / 0.01.
NEG_COS = [-0.2, -0.6, -0.98]
NEG_MAGS = [12.0, 6.0, 3.0]

# All-positive-large geometry: raw inner products 380 / 50 / 4 all clamp
# to relevance 1.0 (indistinguishable); true cosines 0.95 / 0.5 / 0.1
# map to 0.975 / 0.75 / 0.55.
POS_COS = [0.95, 0.5, 0.1]
POS_MAGS = [400.0, 100.0, 40.0]

# Ranking-parity geometry: cosine order is 0, 3, 2, 7, 5, 1, 6, 4 while
# raw-dot order is completely different (e.g. doc 2 with cosine 0.45 and
# magnitude 11 out-scores doc 0 with cosine 0.9 and magnitude 0.5).
PARITY_COS = [0.9, -0.3, 0.45, 0.68, -0.75, 0.0, -0.52, 0.22]
PARITY_MAGS = [0.5, 30.0, 11.0, 2.0, 25.0, 7.0, 18.0, 0.9]


# =====================================================================
# LangChain
# =====================================================================


def _lc_store(docs: np.ndarray, query: np.ndarray, **kwargs):
    langchain_core = pytest.importorskip("langchain_core")  # noqa: F841
    from langchain_core.embeddings import Embeddings

    from turbovec.langchain import TurboQuantVectorStore

    class E(Embeddings):
        def embed_documents(self, ts):
            return [docs[int(t)].tolist() for t in ts]

        def embed_query(self, q):
            return query.tolist()

    store = TurboQuantVectorStore(embedding=E(), **kwargs)
    store.add_texts(
        [str(i) for i in range(len(docs))],
        ids=[f"d{i}" for i in range(len(docs))],
    )
    return store


def test_langchain_score_threshold_retriever_works_under_cosine_default():
    # Wave-8 breakage 1: with non-unit embeddings and all-negative raw
    # inner products, the score_threshold retriever returned [] because
    # every relevance clamped to 0.0. Under the cosine default the
    # genuine best matches survive a small threshold, best first.
    docs, query = directed_docs(NEG_COS, NEG_MAGS)
    store = _lc_store(docs, query)
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.05, "k": 3},
    )
    got = [d.id for d in retriever.invoke("q")]
    # True relevances ~0.4 / 0.2 / 0.01: the first two pass 0.05.
    assert got == ["d0", "d1"]


def test_langchain_relevance_scores_distinct_under_cosine_default():
    # The all-positive-large case: raw IPs 380 / 50 / 4 used to clamp
    # every relevance to 1.0 (no threshold could separate them). True
    # cosines give distinct, ordered relevances.
    docs, query = directed_docs(POS_COS, POS_MAGS)
    store = _lc_store(docs, query)
    pairs = store.similarity_search_with_relevance_scores("q", k=3)
    ids = [d.id for d, _ in pairs]
    scores = [s for _, s in pairs]
    assert ids == ["d0", "d1", "d2"]
    assert scores[0] > scores[1] > scores[2]
    for got, want in zip(scores, [0.975, 0.75, 0.55]):
        assert got == pytest.approx(want, abs=0.02)


def test_langchain_dot_product_mode_pins_old_threshold_behavior():
    # Explicit dot_product opt-in keeps the raw-inner-product scoring:
    # all-negative IPs below -1 still clamp to relevance 0.0 and the
    # threshold retriever returns [] — the documented magnitude-aware
    # trade-off of that mode.
    docs, query = directed_docs(NEG_COS, NEG_MAGS)
    store = _lc_store(docs, query, similarity="dot_product")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no relevant docs" UserWarning
        pairs = store.similarity_search_with_relevance_scores(
            "q", k=3, score_threshold=0.01
        )
    assert pairs == []
    # And raw scores stay raw inner products (magnitude-aware).
    raw = store.similarity_search_with_score("q", k=3)
    assert all(s < -1.0 for _, s in raw)


def test_langchain_cosine_ranking_matches_inmemory_reference():
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import InMemoryVectorStore

    docs, query = directed_docs(PARITY_COS, PARITY_MAGS)

    class E(Embeddings):
        def embed_documents(self, ts):
            return [docs[int(t)].tolist() for t in ts]

        def embed_query(self, q):
            return query.tolist()

    ref = InMemoryVectorStore(embedding=E())
    ref.add_texts(
        [str(i) for i in range(len(docs))],
        ids=[f"d{i}" for i in range(len(docs))],
    )
    ref_ids = [d.id for d, _ in ref.similarity_search_with_score("q", k=6)]

    store = _lc_store(docs, query)
    got_ids = [d.id for d, _ in store.similarity_search_with_score("q", k=6)]
    assert got_ids == ref_ids


def test_langchain_mode_round_trips_through_dump_load(tmp_path):
    docs, query = directed_docs(POS_COS, POS_MAGS)
    for mode in ("cosine", "dot_product"):
        store = _lc_store(docs, query, similarity=mode)
        before = [(d.id, s) for d, s in store.similarity_search_with_score("q", k=3)]
        store.dump(tmp_path / mode)

        from turbovec.langchain import TurboQuantVectorStore

        loaded = TurboQuantVectorStore.load(
            tmp_path / mode, embedding=store.embeddings
        )
        assert loaded.similarity == mode
        after = [(d.id, s) for d, s in loaded.similarity_search_with_score("q", k=3)]
        assert after == before


def test_langchain_legacy_sidecar_is_refused_after_v5_rotation_break():
    # The pre-v5 fixture's `index.tvim` was encoded through the old (QR)
    # rotation. After the format v5 hard break (#206) it can no longer be
    # loaded — the quantized codes would decode against a different
    # rotation and silently return near-zero recall — so the store load
    # must fail loudly with a conversion hint. The original scores are
    # unrecoverable (the index is lossily quantized), so the legacy-score
    # parity assertion is retired in favour of the rejection contract.
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings

    from turbovec.langchain import TurboQuantVectorStore

    class E(Embeddings):
        def embed_documents(self, ts):
            raise AssertionError("not used")

        def embed_query(self, q):
            return [0.0] * DIM

    with pytest.raises(Exception) as ei:
        TurboQuantVectorStore.load(FIXTURES / "langchain", embedding=E())
    # v7-only: a pre-v7 fixture is refused and pointed at conversion.
    assert "convert" in str(ei.value).lower()


def test_langchain_zero_vectors_score_zero_under_cosine():
    pytest.importorskip("langchain_core")
    docs, query = directed_docs([0.5], [3.0])
    all_docs = np.vstack([docs, np.zeros((1, DIM), dtype=np.float32)])
    store = _lc_store(all_docs, query)
    got = store.similarity_search_with_score("q", k=2)
    scores = {d.id: s for d, s in got}
    # The recovered cosine sits within DIM=16 / 4-bit quantization noise of
    # the true 0.5 (the exact value depends on the rotation, an internal
    # detail); the tolerance checks "cosine mode ≈ true cosine, not the raw
    # dot 1.5", not a specific rotation's per-vector error.
    assert scores["d0"] == pytest.approx(0.5, abs=0.05)
    assert scores["d1"] == pytest.approx(0.0, abs=1e-6)


def test_langchain_zero_query_scores_zero_under_cosine():
    pytest.importorskip("langchain_core")
    docs, _ = directed_docs([0.5, -0.5], [3.0, 7.0])
    store = _lc_store(docs, np.zeros(DIM, dtype=np.float32))
    got = store.similarity_search_with_score("q", k=2)
    assert len(got) == 2
    assert all(s == pytest.approx(0.0, abs=1e-6) for _, s in got)


def test_langchain_rejects_unknown_similarity():
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings

    from turbovec.langchain import TurboQuantVectorStore

    class E(Embeddings):
        def embed_documents(self, ts):
            return [[0.0] * DIM for _ in ts]

        def embed_query(self, q):
            return [0.0] * DIM

    with pytest.raises(ValueError, match="similarity"):
        TurboQuantVectorStore(embedding=E(), similarity="euclidean")


def test_langchain_from_texts_forwards_similarity():
    docs, query = directed_docs([0.5], [3.0])
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings

    from turbovec.langchain import TurboQuantVectorStore

    class E(Embeddings):
        def embed_documents(self, ts):
            return [docs[0].tolist() for _ in ts]

        def embed_query(self, q):
            return query.tolist()

    store = TurboQuantVectorStore.from_texts(["x"], E(), similarity="dot_product")
    assert store.similarity == "dot_product"
    [(_, score)] = store.similarity_search_with_score("q", k=1)
    # Raw inner product = mag * cos = 1.5, not the cosine 0.5. Tolerance
    # covers DIM=16 / 4-bit quantization noise (rotation-independent).
    assert score == pytest.approx(1.5, abs=0.12)


# =====================================================================
# LlamaIndex
# =====================================================================


def _li_nodes(docs: np.ndarray):
    from llama_index.core.schema import TextNode

    return [
        TextNode(id_=f"n{i}", text=f"t{i}", embedding=docs[i].tolist())
        for i in range(len(docs))
    ]


def _li_query(query: np.ndarray, k: int):
    from llama_index.core.vector_stores.types import VectorStoreQuery

    return VectorStoreQuery(query_embedding=query.tolist(), similarity_top_k=k)


def test_llama_similarities_are_true_cosine_under_default():
    # LlamaIndex passes similarities straight through to consumers (e.g.
    # node postprocessors with similarity cutoffs). Under the cosine
    # default they are true cosines in [-1, 1] regardless of magnitude.
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    docs, query = directed_docs(POS_COS, POS_MAGS)
    store = TurboQuantVectorStore()
    store.add(_li_nodes(docs))
    res = store.query(_li_query(query, 3))
    assert res.ids == ["n0", "n1", "n2"]
    for got, want in zip(res.similarities, POS_COS):
        assert got == pytest.approx(want, abs=0.02)


def test_llama_dot_product_mode_passes_raw_inner_products():
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    docs, query = directed_docs(POS_COS, POS_MAGS)
    store = TurboQuantVectorStore(similarity="dot_product")
    store.add(_li_nodes(docs))
    res = store.query(_li_query(query, 3))
    # Length first: zip() truncates to the shortest input, so a short
    # (or empty) `similarities` would check nothing.
    assert len(res.similarities) == 3
    raw = [c * m for c, m in zip(POS_COS, POS_MAGS)]  # 380 / 50 / 4
    # Quantization noise on the raw inner product scales with the
    # document magnitude (~2% of ||v|| at 4 bits), so tolerate that.
    for got, want, mag in zip(res.similarities, raw, POS_MAGS):
        assert got == pytest.approx(want, abs=0.03 * mag)


def test_llama_cosine_ranking_matches_simple_reference():
    pytest.importorskip("llama_index.core")
    from llama_index.core.vector_stores.simple import SimpleVectorStore

    from turbovec.llama_index import TurboQuantVectorStore

    docs, query = directed_docs(PARITY_COS, PARITY_MAGS)
    ref = SimpleVectorStore()
    ref.add(_li_nodes(docs))
    ref_ids = list(ref.query(_li_query(query, 6)).ids)

    store = TurboQuantVectorStore()
    store.add(_li_nodes(docs))
    got_ids = list(store.query(_li_query(query, 6)).ids)
    assert got_ids == ref_ids


def test_llama_mode_round_trips_through_persist(tmp_path):
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    docs, query = directed_docs(POS_COS, POS_MAGS)
    for mode in ("cosine", "dot_product"):
        store = TurboQuantVectorStore(similarity=mode)
        store.add(_li_nodes(docs))
        before = store.query(_li_query(query, 3))
        store.persist(str(tmp_path / mode / "vector_store.json"))
        loaded = TurboQuantVectorStore.from_persist_path(
            str(tmp_path / mode / "vector_store.json")
        )
        assert loaded.similarity == mode
        after = loaded.query(_li_query(query, 3))
        assert after.ids == before.ids
        assert after.similarities == before.similarities


def test_llama_to_dict_from_dict_round_trips_similarity():
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    store = TurboQuantVectorStore(similarity="dot_product")
    config = store.to_dict()
    assert config["similarity"] == "dot_product"
    rebuilt = TurboQuantVectorStore.from_dict(config)
    assert rebuilt.similarity == "dot_product"


def test_llama_legacy_sidecar_is_refused_after_v5_rotation_break():
    # See the langchain counterpart: the pre-v5 index is refused with a
    # rebuild hint rather than silently mis-decoding under the v5 rotation.
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    with pytest.raises(Exception) as ei:
        TurboQuantVectorStore.from_persist_path(
            str(FIXTURES / "llama_index" / "store.json")
        )
    # v7-only: a pre-v7 fixture is refused and pointed at conversion.
    assert "convert" in str(ei.value).lower()


def test_llama_zero_vectors_score_zero_under_cosine():
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    docs, query = directed_docs([0.5], [3.0])
    all_docs = np.vstack([docs, np.zeros((1, DIM), dtype=np.float32)])
    store = TurboQuantVectorStore()
    store.add(_li_nodes(all_docs))
    res = store.query(_li_query(query, 2))
    scores = dict(zip(res.ids, res.similarities))
    # Within DIM=16 / 4-bit quantization noise of the true cosine 0.5
    # (rotation-independent tolerance; see the langchain counterpart).
    assert scores["n0"] == pytest.approx(0.5, abs=0.05)
    assert scores["n1"] == pytest.approx(0.0, abs=1e-6)


def test_llama_rejects_unknown_similarity():
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    with pytest.raises(ValueError, match="similarity"):
        TurboQuantVectorStore(similarity="l2")


# =====================================================================
# Haystack
# =====================================================================


def _hs_docs(docs: np.ndarray):
    from haystack import Document

    return [
        Document(id=f"h{i}", content=f"c{i}", embedding=docs[i].tolist())
        for i in range(len(docs))
    ]


def test_haystack_scale_score_distinct_and_ordered_under_cosine_default():
    # Wave-8 breakage 3: with non-unit embeddings the DEFAULT cosine
    # branch of scale_score collapsed distinct raw scores to 1.0,
    # violating the "same ranking, mapped into [0, 1]" contract. Under
    # the cosine mode the scaled scores are distinct and order-preserving.
    pytest.importorskip("haystack")
    from turbovec.haystack import TurboQuantDocumentStore

    docs, query = directed_docs(POS_COS, POS_MAGS)
    store = TurboQuantDocumentStore(dim=DIM)
    store.write_documents(_hs_docs(docs))
    out = store.embedding_retrieval(
        query_embedding=query.tolist(), top_k=3, scale_score=True
    )
    assert [d.id for d in out] == ["h0", "h1", "h2"]
    scores = [d.score for d in out]
    assert scores[0] > scores[1] > scores[2]
    for got, want in zip(scores, [0.975, 0.75, 0.55]):
        assert got == pytest.approx(want, abs=0.02)


def test_haystack_dot_product_mode_keeps_sigmoid_scale_and_raw_ranking():
    pytest.importorskip("haystack")
    import math

    from turbovec.haystack import TurboQuantDocumentStore

    docs, query = directed_docs(POS_COS, POS_MAGS)
    store = TurboQuantDocumentStore(
        dim=DIM, embedding_similarity_function="dot_product"
    )
    store.write_documents(_hs_docs(docs))
    raw = [
        d.score
        for d in store.embedding_retrieval(query_embedding=query.tolist(), top_k=3)
    ]
    # Raw inner products, magnitude-aware: ~380 / 50 / 4. Quantization
    # noise scales with the document magnitude (~2% of ||v|| at 4 bits).
    for got, want, mag in zip(raw, [380.0, 50.0, 4.0], POS_MAGS):
        assert got == pytest.approx(want, abs=0.03 * mag)
    scaled = [
        d.score
        for d in store.embedding_retrieval(
            query_embedding=query.tolist(), top_k=3, scale_score=True
        )
    ]
    for got, r in zip(scaled, raw):
        assert got == pytest.approx(1.0 / (1.0 + math.exp(-r / 100.0)), abs=1e-6)


def test_haystack_cosine_ranking_matches_inmemory_reference():
    pytest.importorskip("haystack")
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    from turbovec.haystack import TurboQuantDocumentStore

    docs, query = directed_docs(PARITY_COS, PARITY_MAGS)
    ref = InMemoryDocumentStore(embedding_similarity_function="cosine")
    ref.write_documents(_hs_docs(docs))
    ref_ids = [
        d.id
        for d in ref.embedding_retrieval(query_embedding=query.tolist(), top_k=6)
    ]

    store = TurboQuantDocumentStore(dim=DIM)
    store.write_documents(_hs_docs(docs))
    got_ids = [
        d.id
        for d in store.embedding_retrieval(query_embedding=query.tolist(), top_k=6)
    ]
    assert got_ids == ref_ids


def test_haystack_mode_round_trips_through_save_load(tmp_path):
    pytest.importorskip("haystack")
    from turbovec.haystack import TurboQuantDocumentStore

    docs, query = directed_docs(POS_COS, POS_MAGS)
    for mode in ("cosine", "dot_product"):
        store = TurboQuantDocumentStore(
            dim=DIM, embedding_similarity_function=mode
        )
        store.write_documents(_hs_docs(docs))
        before = [
            (d.id, d.score)
            for d in store.embedding_retrieval(query_embedding=query.tolist(), top_k=3)
        ]
        store.save_to_disk(tmp_path / mode)
        loaded = TurboQuantDocumentStore.load_from_disk(tmp_path / mode)
        assert loaded.embedding_similarity_function == mode
        assert loaded._vectors_normalized == (mode == "cosine")
        after = [
            (d.id, d.score)
            for d in loaded.embedding_retrieval(query_embedding=query.tolist(), top_k=3)
        ]
        assert after == before


def test_haystack_legacy_sidecar_is_refused_after_v5_rotation_break():
    # See the langchain counterpart. A pre-v5 store on disk is refused on
    # load with a rebuild hint; both the read and any follow-up write path
    # are gated behind that load, so there is nothing left to preserve.
    pytest.importorskip("haystack")
    from turbovec.haystack import TurboQuantDocumentStore

    with pytest.raises(Exception) as ei:
        TurboQuantDocumentStore.load_from_disk(FIXTURES / "haystack")
    # v7-only: a pre-v7 fixture is refused and pointed at conversion.
    assert "convert" in str(ei.value).lower()


def test_haystack_zero_vectors_score_zero_under_cosine():
    pytest.importorskip("haystack")
    from haystack import Document

    from turbovec.haystack import TurboQuantDocumentStore

    docs, query = directed_docs([0.5], [3.0])
    store = TurboQuantDocumentStore(dim=DIM)
    store.write_documents(_hs_docs(docs))
    store.write_documents(
        [Document(id="zero", content="z", embedding=[0.0] * DIM)]
    )
    out = store.embedding_retrieval(query_embedding=query.tolist(), top_k=2)
    scores = {d.id: d.score for d in out}
    # Within DIM=16 / 4-bit quantization noise of the true cosine 0.5
    # (rotation-independent tolerance; see the langchain counterpart).
    assert scores["h0"] == pytest.approx(0.5, abs=0.05)
    assert scores["zero"] == pytest.approx(0.0, abs=1e-6)


def test_haystack_rejects_unknown_similarity_function():
    pytest.importorskip("haystack")
    from turbovec.haystack import TurboQuantDocumentStore

    with pytest.raises(ValueError, match="embedding_similarity_function"):
        TurboQuantDocumentStore(dim=DIM, embedding_similarity_function="l2")


# =====================================================================
# Agno
# =====================================================================


def _agno_db(docs: np.ndarray, query: np.ndarray, **kwargs):
    pytest.importorskip("agno")
    from agno.knowledge.document import Document

    from turbovec.agno import TurboQuantVectorDb

    class E:
        enable_batch = False
        dimensions = docs.shape[1]

        def get_embedding(self, text):
            if text.startswith("doc"):
                return docs[int(text[3:])].tolist()
            return query.tolist()

        def get_embedding_and_usage(self, text):
            return self.get_embedding(text), None

    db = TurboQuantVectorDb(embedder=E(), **kwargs)
    db.create()
    db.insert(
        "hash-sim", [Document(content=f"doc{i}") for i in range(len(docs))]
    )
    return db


def test_agno_similarity_threshold_meaningful_under_cosine_default():
    # Wave-8 breakage 2 (fixed side): similarity_threshold compares
    # against (raw + 1) / 2, which under the cosine default is a true
    # [0, 1] relevance. A high threshold keeps only the genuinely close
    # match instead of admitting everything.
    docs, query = directed_docs(POS_COS, POS_MAGS)
    db = _agno_db(docs, query, similarity_threshold=0.9)
    got = [d.content for d in db.search("the query", limit=3)]
    assert got == ["doc0"]  # relevances ~0.975 / 0.75 / 0.55


def test_agno_similarity_threshold_discards_negatives_under_cosine():
    # agno defines the cosine score as the raw cosine — `normalize_cosine`
    # is `max(0, min(1, 1 - distance))` — so every negative cosine scores
    # 0 and any positive threshold rejects it. pgvector, the only other
    # agno store implementing this knob, enforces the same `cos >= t`.
    #
    # This used to keep doc0 and doc1, because the score was mapped
    # through the inner-product formula `(cos + 1) / 2`, which lifts
    # -0.2 and -0.6 to 0.4 and 0.2 and puts them over a 0.05 threshold.
    # That admitted documents agno's contract rejects (#503).
    docs, query = directed_docs(NEG_COS, NEG_MAGS)
    db = _agno_db(docs, query, similarity_threshold=0.05)
    assert [d.content for d in db.search("the query", limit=3)] == []

    # With no threshold the same negatives are still returned and still
    # ranked best-first — the mapping changes what the threshold means,
    # not the ordering.
    db_all = _agno_db(docs, query)
    assert [d.content for d in db_all.search("the query", limit=3)] == [
        "doc0",
        "doc1",
        "doc2",
    ]


def test_agno_similarity_threshold_inert_under_dot_product():
    # Explicit max_inner_product opt-in pins the old raw behavior, where
    # the threshold is effectively inert on non-unit embeddings:
    # all-negative IPs discard everything at threshold 0.01, and
    # all-positive-large IPs admit everything at threshold 0.99.
    pytest.importorskip("agno")
    from agno.vectordb.distance import Distance

    docs, query = directed_docs(NEG_COS, NEG_MAGS)
    db = _agno_db(
        docs,
        query,
        distance=Distance.max_inner_product,
        similarity_threshold=0.01,
    )
    assert db.search("the query", limit=3) == []

    docs, query = directed_docs(POS_COS, POS_MAGS)
    db = _agno_db(
        docs,
        query,
        distance=Distance.max_inner_product,
        similarity_threshold=0.99,
    )
    assert len(db.search("the query", limit=3)) == 3


def test_agno_cosine_ranking_matches_cosine_order():
    # agno's in-tree LanceDb is the drop-in reference, but as of agno
    # 2.8 its wrapper never forwards the distance metric to the query
    # builder, so an un-indexed LanceDb table searches by L2 regardless
    # of distance=Distance.cosine — it cannot serve as a cosine oracle.
    # Assert against the analytic cosine ranking instead, which the
    # LangChain / Haystack / LlamaIndex references (same geometry) agree
    # with in their parity tests above.
    docs, query = directed_docs(PARITY_COS, PARITY_MAGS)
    db = _agno_db(docs, query)
    got = [d.content for d in db.search("the query", limit=6)]
    want = [f"doc{i}" for i in np.argsort(-np.asarray(PARITY_COS))[:6]]
    assert got == want


def test_agno_mode_round_trips_through_save_create(tmp_path):
    pytest.importorskip("agno")
    from agno.vectordb.distance import Distance

    from turbovec.agno import TurboQuantVectorDb

    docs, query = directed_docs(POS_COS, POS_MAGS)
    for mode in (Distance.cosine, Distance.max_inner_product):
        path = tmp_path / mode.value
        db = _agno_db(docs, query, distance=mode, path=str(path))
        before = [d.content for d in db.search("the query", limit=3)]
        db.save()

        reloaded = TurboQuantVectorDb(
            embedder=db.embedder, distance=mode, path=str(path)
        )
        reloaded.create()
        assert reloaded.distance == mode
        after = [d.content for d in reloaded.search("the query", limit=3)]
        assert after == before


def test_agno_load_rejects_conflicting_recorded_mode(tmp_path):
    pytest.importorskip("agno")
    from agno.vectordb.distance import Distance

    from turbovec.agno import TurboQuantVectorDb

    docs, query = directed_docs(POS_COS, POS_MAGS)
    db = _agno_db(
        docs, query, distance=Distance.max_inner_product, path=str(tmp_path)
    )
    db.save()

    # The construct-then-create() shape means the store already has a
    # mode when the load runs: a conflicting recorded mode is an error,
    # not a silent adoption.
    other = TurboQuantVectorDb(
        embedder=db.embedder, distance=Distance.cosine, path=str(tmp_path)
    )
    with pytest.raises(ValueError, match="distance"):
        other.create()


def test_agno_legacy_sidecar_is_refused_after_v5_rotation_break():
    # See the langchain counterpart: the pre-v5 index is refused with a
    # rebuild hint (agno defers the actual index load to `create()`).
    pytest.importorskip("agno")

    from turbovec.agno import TurboQuantVectorDb

    # The agno fixture's persisted side-car records dim=8; the embedder
    # must match so `create()` gets past the dim check to the index load
    # (where the v5 rotation break rejects the pre-v5 codes).
    fixture_dim = 8

    class E:
        enable_batch = False
        dimensions = fixture_dim

        def get_embedding(self, text):
            return [0.0] * fixture_dim

        def get_embedding_and_usage(self, text):
            return self.get_embedding(text), None

    db = TurboQuantVectorDb(embedder=E(), path=str(FIXTURES / "agno"))
    with pytest.raises(Exception) as ei:
        db.create()
    # v7-only: a pre-v7 fixture is refused and pointed at conversion.
    assert "convert" in str(ei.value).lower()


def test_agno_zero_vectors_score_zero_under_cosine():
    pytest.importorskip("agno")
    docs, query = directed_docs([0.5], [3.0])
    all_docs = np.vstack([docs, np.zeros((1, DIM), dtype=np.float32)])
    db = _agno_db(all_docs, query)
    # No threshold: both docs come back, the zero vector ranking last on
    # a raw score of 0.
    got = [d.content for d in db.search("the query", limit=2)]
    assert got == ["doc0", "doc1"]
    # Under cosine the score *is* the raw cosine, so the zero vector
    # scores 0 and any positive threshold drops it. (It scored 0.5 when
    # the mapping was `(0 + 1) / 2`, so this needed a threshold above 0.5
    # to discriminate — see #503.)
    db2 = _agno_db(all_docs, query, similarity_threshold=0.1)
    assert [d.content for d in db2.search("the query", limit=2)] == ["doc0"]


def test_agno_rejects_l2_distance():
    pytest.importorskip("agno")
    from agno.vectordb.distance import Distance

    from turbovec.agno import TurboQuantVectorDb

    class E:
        dimensions = DIM

        def get_embedding(self, text):
            return [0.0] * DIM

    with pytest.raises(ValueError, match="distance"):
        TurboQuantVectorDb(embedder=E(), distance=Distance.l2)


def test_langchain_dot_product_relevance_is_unclamped_and_warns():
    # Issue #322: the relevance clamp mapped every raw inner product
    # >= 1.0 onto exactly 1.0, so a score_threshold retriever admitted
    # unrelated documents and LangChain's own out-of-range warning never
    # fired. In dot_product mode the mapping is now unclamped, and asking
    # for a relevance fn warns that the mode has no calibrated [0, 1]
    # relevance.
    docs, query = directed_docs(POS_COS, POS_MAGS)
    store = _lc_store(docs, query, similarity="dot_product")

    with pytest.warns(UserWarning, match="dot_product"):
        fn = store._select_relevance_score_fn()
    # Raw inner products well above 1 no longer saturate.
    assert fn(24.91) > 1.0
    assert fn(2.42) > 1.0
    assert fn(24.91) > fn(2.42)

    # The retriever repro: with the clamp, both d0 and d1 scored exactly
    # 1.0 and passed a 0.99 threshold. Unclamped they stay distinct, and
    # LangChain's base class surfaces the out-of-range scores itself.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pairs = store.similarity_search_with_relevance_scores("q", k=3)
    scores = [s for _, s in pairs]
    assert scores[0] > scores[1] > scores[2]
    assert len(set(scores)) == len(scores)
    assert any("0 and 1" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


def test_langchain_cosine_relevance_stays_clamped_and_silent():
    # The default mode is unaffected by the #322 change: still clamped,
    # still no warning.
    docs, query = directed_docs(POS_COS, POS_MAGS)
    store = _lc_store(docs, query)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn = store._select_relevance_score_fn()
    assert fn(1.0001) == 1.0
    assert fn(-1.0001) == 0.0
