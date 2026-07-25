"""Thread-safety tests for the four integration stores (issue #161).

Two halves:

1. **Stress matrix** (marked ``slow``): a port of the issue-#161 research
   harness. Every store runs 7 concurrency scenarios (add-vs-search,
   churn-vs-search, filtered-search-vs-churn, add-vs-add,
   delete-vs-delete, add-vs-delete, get-vs-churn) with plain threads,
   ``sys.setswitchinterval(1e-7)`` and batched adds (batch 64 widens the
   searchable-before-resolvable window). Each cell asserts zero
   exceptions in every role plus the final-state invariants
   ``len(index) == len(each side-car map)`` — and, for add-vs-add, an
   exact ``_next_u64`` (which is what catches lost increments /
   duplicate handles, the LlamaIndex data-loss race).

2. **Deterministic units**: mutation ordering (maps populated before the
   index add; index removal precedes map pops) observed via a spy index;
   preservation of the issue-#89 no-partial-mutation guarantee through
   the failure-unwind path; tolerant skip of a handle deleted mid-search
   (the reader-straddle); and the non-raising fallback when a filtered
   search's allowlist keeps going stale.
"""
from __future__ import annotations

import itertools
import re
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pytest

DIM = 32
POOL = 4096
_rng = np.random.default_rng(0)
VECPOOL = _rng.standard_normal((POOL, DIM)).astype(np.float32)
VECPOOL /= np.linalg.norm(VECPOOL, axis=1, keepdims=True) + 1e-9

BATCH = 64
N_READERS = 3
DUR = 1.5  # seconds per stress cell; every issue-#161 race reproduced <1 s


def vec_for(n: int) -> np.ndarray:
    return VECPOOL[n % POOL]


# ---------------------------------------------------------------------------
# Stress-harness core (threads + classified exceptions)
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    ops: Counter = field(default_factory=Counter)      # role -> completed ops
    errors: Counter = field(default_factory=Counter)   # (role, exc, msg) -> n
    first_tb: dict = field(default_factory=dict)


def _msg_head(exc: BaseException) -> str:
    s = str(exc).splitlines()[0] if str(exc) else ""
    return re.sub(r"\d+", "#", s)[:90]


def run_roles(roles, duration=DUR):
    """roles: list of (role_name, fn); each fn is called in a loop on its
    own thread until the duration elapses (or it returns StopIteration).
    Every exception is recorded, never raised."""
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-7)
    res = ScenarioResult()
    stop = threading.Event()
    lock = threading.Lock()  # protects the result counters only

    def wrap(role, fn):
        def runner():
            while not stop.is_set():
                try:
                    step = fn()
                    with lock:
                        res.ops[role] += 1
                    if step is StopIteration:
                        break
                except BaseException as exc:
                    key = (role, type(exc).__name__, _msg_head(exc))
                    with lock:
                        res.errors[key] += 1
                        tbkey = (role, type(exc).__name__)
                        if tbkey not in res.first_tb:
                            res.first_tb[tbkey] = traceback.format_exc()
        return runner

    threads = [threading.Thread(target=wrap(r, f), daemon=True) for r, f in roles]
    try:
        for t in threads:
            t.start()
        time.sleep(duration)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)
        sys.setswitchinterval(old_interval)
    return res


def assert_clean(res: ScenarioResult, counts: dict, extra: dict | None = None):
    """Zero exceptions in every role + all side-car maps agree with the
    index (+ any extra exact invariants)."""
    if res.errors:
        lines = [f"  {role}: {exc}: {msg!r} x{n}" for (role, exc, msg), n in res.errors.items()]
        tb = next(iter(res.first_tb.values()), "")
        pytest.fail(
            "concurrent ops raised:\n" + "\n".join(lines) + "\nfirst traceback:\n" + tb
        )
    index_len = counts.pop("index")
    for name, value in counts.items():
        assert value == index_len, (
            f"final-state desync: len(index)={index_len} but {name}={value}"
        )
    for name, (got, want) in (extra or {}).items():
        assert got == want, f"invariant {name}: got {got}, want {want}"


# ---------------------------------------------------------------------------
# Store adapters (uniform add/delete/search/... over the four stores)
# ---------------------------------------------------------------------------


class LangChainAdapter:
    name = "langchain"

    def __init__(self):
        pytest.importorskip("langchain_core")
        from langchain_core.embeddings import Embeddings

        class FakeEmb(Embeddings):
            def embed_documents(self, texts):
                return [vec_for(hash(t)).tolist() for t in texts]

            def embed_query(self, text):
                return vec_for(hash(text)).tolist()

        from turbovec.langchain import TurboQuantVectorStore

        self._emb = FakeEmb()
        self._cls = TurboQuantVectorStore

    def make(self):
        return self._cls(embedding=self._emb)

    def add(self, s, ids, texts, metas):
        s.add_texts(texts, metadatas=metas, ids=ids)

    def delete(self, s, ids):
        s.delete(list(ids))

    def search(self, s, q, k=10):
        return s.similarity_search_by_vector(list(map(float, q)), k=k)

    def search_filtered(self, s, q, k=10):
        return s._search_vector(np.asarray(q, dtype=np.float32), k, filter={"g": 1})

    def get(self, s, ids):
        return s.get_by_ids(list(ids))

    def counts(self, s):
        return {
            "index": len(s._index),
            "docs": len(s._docs),
            "str_to_u64": len(s._str_to_u64),
            "u64_to_str": len(s._u64_to_str),
        }

    def next_u64(self, s):
        return s._next_u64


class HaystackAdapter:
    name = "haystack"

    def __init__(self):
        pytest.importorskip("haystack")
        from haystack import Document
        from haystack.document_stores.types import DuplicatePolicy

        from turbovec.haystack import TurboQuantDocumentStore

        self.Document = Document
        self.OVERWRITE = DuplicatePolicy.OVERWRITE
        self._cls = TurboQuantDocumentStore

    def make(self):
        return self._cls(dim=DIM)

    def add(self, s, ids, texts, metas):
        docs = [
            self.Document(id=i, content=t, meta=m, embedding=vec_for(hash(t)).tolist())
            for i, t, m in zip(ids, texts, metas)
        ]
        s.write_documents(docs, policy=self.OVERWRITE)

    def delete(self, s, ids):
        s.delete_documents(list(ids))

    def search(self, s, q, k=10):
        return s.embedding_retrieval(list(map(float, q)), top_k=k)

    def search_filtered(self, s, q, k=10):
        return s.embedding_retrieval(
            list(map(float, q)),
            top_k=k,
            filters={"field": "meta.g", "operator": "==", "value": 1},
        )

    def get(self, s, ids):
        # Nearest read-by-id analogue: filter_documents full scan.
        return s.filter_documents()

    def counts(self, s):
        return {
            "index": len(s._index),
            "u64_to_doc": len(s._u64_to_doc),
            "str_to_u64": len(s._str_to_u64),
        }

    def next_u64(self, s):
        return s._next_u64


class LlamaIndexAdapter:
    name = "llama_index"

    def __init__(self):
        pytest.importorskip("llama_index.core")
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import (
            MetadataFilter,
            MetadataFilters,
            VectorStoreQuery,
        )

        from turbovec.llama_index import TurboQuantVectorStore

        self.TextNode = TextNode
        self.VectorStoreQuery = VectorStoreQuery
        self._filters = MetadataFilters(filters=[MetadataFilter(key="g", value=1)])
        self._cls = TurboQuantVectorStore

    def make(self):
        return self._cls()

    def add(self, s, ids, texts, metas):
        nodes = [
            self.TextNode(id_=i, text=t, metadata=m, embedding=vec_for(hash(t)).tolist())
            for i, t, m in zip(ids, texts, metas)
        ]
        s.add(nodes)

    def delete(self, s, ids):
        s.delete_nodes(node_ids=list(ids))

    def search(self, s, q, k=10):
        return s.query(
            self.VectorStoreQuery(query_embedding=list(map(float, q)), similarity_top_k=k)
        )

    def search_filtered(self, s, q, k=10):
        return s.query(
            self.VectorStoreQuery(
                query_embedding=list(map(float, q)),
                similarity_top_k=k,
                filters=self._filters,
            )
        )

    def get(self, s, ids):
        return s.get_nodes(node_ids=list(ids))

    def counts(self, s):
        return {
            "index": len(s._index),
            "nodes": len(s._nodes),
            "node_id_to_u64": len(s._node_id_to_u64),
            "u64_to_node_id": len(s._u64_to_node_id),
        }

    def next_u64(self, s):
        return s._next_u64


class AgnoAdapter:
    name = "agno"

    def __init__(self):
        pytest.importorskip("agno")
        from agno.knowledge.document import Document
        from agno.knowledge.embedder import Embedder

        class FakeAgnoEmb(Embedder):
            dimensions: int = DIM

            def get_embedding(self, text):
                return vec_for(hash(text)).tolist()

            def get_embedding_and_usage(self, text):
                return self.get_embedding(text), None

        from turbovec.agno import TurboQuantVectorDb

        self.Document = Document
        self._emb = FakeAgnoEmb()
        self._emb.dimensions = DIM
        self._cls = TurboQuantVectorDb

    def make(self):
        s = self._cls(embedder=self._emb)
        s.create()
        return s

    def add(self, s, ids, texts, metas):
        docs = [
            self.Document(
                id=i, name=f"name-{i}", content=t, meta_data=m,
                embedding=vec_for(hash(t)).tolist(),
            )
            for i, t, m in zip(ids, texts, metas)
        ]
        s.insert(content_hash=f"ch-{ids[0]}", documents=docs)

    def delete(self, s, ids):
        # agno keys deletes on the *derived* doc_id; callers sample live
        # derived ids via live_ids().
        for did in ids:
            s.delete_by_id(did)

    def live_ids(self, s, n):
        return list(s._str_to_u64.keys())[:n]

    def search(self, s, q, k=10):
        # agno search embeds a query string; rotate the text per call.
        return s.search(f"q-{int(q[0] * 1e6)}", limit=k)

    def search_filtered(self, s, q, k=10):
        return s.search(f"q-{int(q[0] * 1e6)}", limit=k, filters={"g": 1})

    def get(self, s, ids):
        return [s.id_exists(i) for i in ids]

    def counts(self, s):
        return {
            "index": len(s._index),
            "u64_to_doc": len(s._u64_to_doc),
            "str_to_u64_handles": sum(len(hs) for hs in s._str_to_u64.values()),
        }

    def next_u64(self, s):
        return s._next_u64


ADAPTERS = {
    "langchain": LangChainAdapter,
    "haystack": HaystackAdapter,
    "llama_index": LlamaIndexAdapter,
    "agno": AgnoAdapter,
}


@pytest.fixture(params=list(ADAPTERS), ids=list(ADAPTERS))
def adapter(request):
    return ADAPTERS[request.param]()


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


class IdGen:
    def __init__(self, prefix):
        self._c = itertools.count()
        self._lock = threading.Lock()
        self.prefix = prefix

    def batch(self, n):
        with self._lock:
            ks = [next(self._c) for _ in range(n)]
        return [f"{self.prefix}-{k}" for k in ks]


def _mk_batch(gen, tag):
    ids = gen.batch(BATCH)
    texts = [f"text-{tag}-{i}" for i in ids]
    metas = [{"g": j % 2, "t": tag} for j in range(len(ids))]
    return ids, texts, metas


# ---------------------------------------------------------------------------
# Stress matrix: 4 stores x 7 scenarios
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_add_vs_search(adapter):
    s = adapter.make()
    gen = IdGen("a")
    qit = itertools.count()

    def writer():
        ids, texts, metas = _mk_batch(gen, "w")
        adapter.add(s, ids, texts, metas)

    def reader():
        adapter.search(s, vec_for(next(qit)))

    roles = [("add", writer)] + [(f"search{i}", reader) for i in range(N_READERS)]
    res = run_roles(roles)
    assert res.ops["add"] > 0 and res.ops["search0"] > 0
    assert_clean(res, adapter.counts(s))


def _churn_cell(adapter, filtered):
    s = adapter.make()
    gen = IdGen("c")
    qit = itertools.count()

    def writer():
        ids, texts, metas = _mk_batch(gen, "w")
        adapter.add(s, ids, texts, metas)
        if isinstance(adapter, AgnoAdapter):
            adapter.delete(s, adapter.live_ids(s, BATCH))
        else:
            adapter.delete(s, ids)

    search = adapter.search_filtered if filtered else adapter.search

    def reader():
        search(s, vec_for(next(qit)))

    roles = [("churn", writer)] + [(f"search{i}", reader) for i in range(N_READERS)]
    res = run_roles(roles)
    assert res.ops["churn"] > 0 and res.ops["search0"] > 0
    assert_clean(res, adapter.counts(s))


@pytest.mark.slow
def test_churn_vs_search(adapter):
    _churn_cell(adapter, filtered=False)


@pytest.mark.slow
def test_filtered_search_vs_churn(adapter):
    _churn_cell(adapter, filtered=True)


@pytest.mark.slow
def test_add_vs_add(adapter):
    """Two concurrent writers; the exact `_next_u64` invariant is what
    catches lost increments / duplicate handles (the LlamaIndex
    data-loss race — batches rejected with 'id already present')."""
    s = adapter.make()
    gens = [IdGen("x"), IdGen("y")]
    total = [0, 0]

    def writer(slot):
        def step():
            ids, texts, metas = _mk_batch(gens[slot], f"w{slot}")
            adapter.add(s, ids, texts, metas)
            total[slot] += len(ids)
        return step

    res = run_roles([("add0", writer(0)), ("add1", writer(1))])
    docs_added = sum(total)
    assert docs_added > 0
    counts = adapter.counts(s)
    assert_clean(
        res,
        counts,
        extra={
            "index holds every added doc": (len(s._index), docs_added),
            "_next_u64 == docs added (no lost increments / dup handles)": (
                adapter.next_u64(s), docs_added,
            ),
        },
    )


@pytest.mark.slow
def test_delete_vs_delete(adapter):
    s = adapter.make()
    gen = IdGen("d")
    all_ids = []
    for _ in range(60):
        ids, texts, metas = _mk_batch(gen, "pre")
        adapter.add(s, ids, texts, metas)
        all_ids.extend(ids)
    if isinstance(adapter, AgnoAdapter):
        all_ids = list(s._str_to_u64.keys())
    orders = [list(all_ids), list(reversed(all_ids))]
    pos = [0, 0]

    def deleter(slot):
        def step():
            p = pos[slot]
            if p >= len(orders[slot]):
                return StopIteration
            adapter.delete(s, orders[slot][p : p + 16])
            pos[slot] = p + 16
        return step

    res = run_roles([("del0", deleter(0)), ("del1", deleter(1))])
    counts = adapter.counts(s)
    assert_clean(res, counts, extra={"store fully emptied": (len(s._index), 0)})


@pytest.mark.slow
def test_add_vs_delete(adapter):
    import collections

    s = adapter.make()
    gen = IdGen("m")
    q = collections.deque()
    qlock = threading.Lock()

    def writer():
        ids, texts, metas = _mk_batch(gen, "w")
        adapter.add(s, ids, texts, metas)
        with qlock:
            q.append(ids)

    def deleter():
        with qlock:
            ids = q.popleft() if q else None
        if ids is None:
            return
        if isinstance(adapter, AgnoAdapter):
            adapter.delete(s, adapter.live_ids(s, len(ids)))
        else:
            adapter.delete(s, ids)

    res = run_roles([("add", writer), ("delete", deleter)])
    assert res.ops["add"] > 0 and res.ops["delete"] > 0
    assert_clean(res, adapter.counts(s))


@pytest.mark.slow
def test_get_vs_churn(adapter):
    s = adapter.make()
    gen = IdGen("g")

    def writer():
        ids, texts, metas = _mk_batch(gen, "w")
        adapter.add(s, ids, texts, metas)
        if isinstance(adapter, AgnoAdapter):
            adapter.delete(s, adapter.live_ids(s, BATCH))
        else:
            adapter.delete(s, ids)

    kit = itertools.count()

    def getter():
        n = next(kit)
        ids = [f"g-{(n + d) % 2000}" for d in range(8)]
        if isinstance(adapter, AgnoAdapter):
            ids = adapter.live_ids(s, 8)
        adapter.get(s, ids)

    roles = [("churn", writer)] + [(f"get{i}", getter) for i in range(2)]
    res = run_roles(roles)
    assert res.ops["churn"] > 0 and res.ops["get0"] > 0
    assert_clean(res, adapter.counts(s))


# ---------------------------------------------------------------------------
# Deterministic units
# ---------------------------------------------------------------------------


class SpyIndex:
    """Wraps a store's IdMapIndex; hooks fire inside add_with_ids /
    remove / search so tests can observe (or perturb) side-car state at
    the exact moment the index mutates."""

    def __init__(self, inner):
        self._inner = inner
        self.on_add = None      # fn(handles) called BEFORE the real add
        self.on_remove = None   # fn(handle) called BEFORE the real remove
        self.on_search = None   # fn() called AFTER the real search returns
        self.raise_on_allowlist = False

    # --- passthrough surface the stores use ---
    @property
    def dim(self):
        return self._inner.dim

    @property
    def bit_width(self):
        return self._inner.bit_width

    def __len__(self):
        return len(self._inner)

    def add_with_ids(self, vectors, handles):
        if self.on_add is not None:
            self.on_add(handles)
        return self._inner.add_with_ids(vectors, handles)

    def remove(self, handle):
        if self.on_remove is not None:
            self.on_remove(handle)
        return self._inner.remove(handle)

    def search(self, qvec, k, allowlist=None):
        if allowlist is not None and self.raise_on_allowlist:
            raise KeyError("allowlist contains id(s) not present in index")
        if allowlist is not None:
            out = self._inner.search(qvec, k, allowlist=allowlist)
        else:
            out = self._inner.search(qvec, k)
        if self.on_search is not None:
            self.on_search()
        return out


def _install_spy(store, adapter):
    spy = SpyIndex(store._index)
    if isinstance(adapter, LlamaIndexAdapter):
        object.__setattr__(store, "_index", spy)
    else:
        store._index = spy
    return spy


def _seed(adapter, s, n=8, prefix="seed"):
    ids = [f"{prefix}-{j}" for j in range(n)]
    texts = [f"text-{prefix}-{j}" for j in range(n)]
    metas = [{"g": j % 2} for j in range(n)]
    adapter.add(s, ids, texts, metas)
    return ids


def test_ordering_maps_before_index_add(adapter):
    """Layer (b), add side: every issued handle must already resolve
    through the side-car maps at the moment the index add runs."""
    s = adapter.make()
    spy = _install_spy(s, adapter)
    resolved_at_add = []

    if isinstance(adapter, LangChainAdapter):
        resolve = lambda h: s._u64_to_str.get(int(h)) in s._docs
    elif isinstance(adapter, HaystackAdapter):
        resolve = lambda h: int(h) in s._u64_to_doc
    elif isinstance(adapter, LlamaIndexAdapter):
        resolve = lambda h: s._u64_to_node_id.get(int(h)) in s._nodes
    else:  # agno
        resolve = lambda h: int(h) in s._u64_to_doc

    spy.on_add = lambda handles: resolved_at_add.append(
        all(resolve(h) for h in handles)
    )
    _seed(adapter, s)
    assert resolved_at_add == [True], (
        "side-car maps must be populated before the index add — otherwise a "
        "concurrent search can surface a handle that does not resolve yet"
    )


def test_ordering_index_removed_before_map_pop(adapter):
    """Layer (b), delete side: at the moment the index removal runs, the
    side-car maps must still resolve the handle (index first, maps
    second)."""
    s = adapter.make()
    ids = _seed(adapter, s)
    spy = _install_spy(s, adapter)
    still_resolved = []

    if isinstance(adapter, LangChainAdapter):
        resolve = lambda h: s._u64_to_str.get(int(h)) in s._docs
    elif isinstance(adapter, HaystackAdapter):
        resolve = lambda h: int(h) in s._u64_to_doc
    elif isinstance(adapter, LlamaIndexAdapter):
        resolve = lambda h: s._u64_to_node_id.get(int(h)) in s._nodes
    else:  # agno
        resolve = lambda h: int(h) in s._u64_to_doc

    spy.on_remove = lambda handle: still_resolved.append(resolve(handle))

    if isinstance(adapter, AgnoAdapter):
        adapter.delete(s, adapter.live_ids(s, len(ids)))
    else:
        adapter.delete(s, ids)
    assert still_resolved and all(still_resolved), (
        "the index removal must precede the side-car map pops — otherwise a "
        "concurrent search can surface a handle whose entry is already gone"
    )
    assert len(s._index) == 0


def test_failed_add_unwinds_cleanly(adapter):
    """Issue-#89 preservation under the maps-first ordering: an add whose
    index insert fails (non-finite embedding) must leave every map and
    the index exactly as they were — including restoring the previous
    entry of an upserted id."""
    s = adapter.make()
    _seed(adapter, s, n=4)
    before = adapter.counts(s)
    before_next = adapter.next_u64(s)

    bad_ids = ["seed-0", "fresh-0"]  # one upsert collision + one new id
    texts = ["bad-a", "bad-b"]
    metas = [{"g": 0}, {"g": 1}]
    if isinstance(adapter, LangChainAdapter):
        bad = np.full((2, DIM), np.nan, dtype=np.float32)
        with pytest.raises(Exception):
            s._store_texts_and_vectors(texts, bad, metas, bad_ids)
        assert s._docs["seed-0"][0] == "text-seed-0"  # old payload restored
    elif isinstance(adapter, HaystackAdapter):
        docs = [
            adapter.Document(id=i, content=t, meta=m, embedding=[float("nan")] * DIM)
            for i, t, m in zip(bad_ids, texts, metas)
        ]
        with pytest.raises(Exception):
            s.write_documents(docs, policy=adapter.OVERWRITE)
        assert s._u64_to_doc[s._str_to_u64["seed-0"]]["content"] == "text-seed-0"
    elif isinstance(adapter, LlamaIndexAdapter):
        nodes = [
            adapter.TextNode(id_=i, text=t, metadata=m, embedding=[float("nan")] * DIM)
            for i, t, m in zip(bad_ids, texts, metas)
        ]
        with pytest.raises(Exception):
            s.add(nodes)
        assert s._nodes["seed-0"]["node_dict"]["_node_content"]  # old node kept
    else:  # agno: append semantics, no by-id upsert in insert
        docs = [
            adapter.Document(
                id=i, name=f"n-{i}", content=t, meta_data=m,
                embedding=[float("nan")] * DIM,
            )
            for i, t, m in zip(bad_ids, texts, metas)
        ]
        with pytest.raises(Exception):
            s.insert(content_hash="ch-bad", documents=docs)
        assert "ch-bad" not in s._content_hashes

    assert adapter.counts(s) == before, "failed add must not change any map"
    # Handles issued for the failed batch may stay consumed (gaps are
    # fine); the counter must never go backwards.
    assert adapter.next_u64(s) >= before_next


def test_search_skips_handle_deleted_mid_flight(adapter):
    """Layer (c): a delete that completes between the kernel search and
    result translation (the reader-straddle) must be skipped, not raise."""
    s = adapter.make()
    ids = _seed(adapter, s, n=6)
    spy = _install_spy(s, adapter)

    def straddle():
        # Runs after index.search returned its handles, before the store
        # translates them — delete one of the seeded entries.
        spy.on_search = None  # once
        if isinstance(adapter, AgnoAdapter):
            adapter.delete(s, adapter.live_ids(s, 1))
        else:
            adapter.delete(s, [ids[0]])

    spy.on_search = straddle
    out = adapter.search(s, vec_for(0), k=6)
    if isinstance(adapter, LlamaIndexAdapter):
        n = len(out.ids)
    else:
        n = len(out)
    assert n == 5, f"expected the deleted handle to be skipped, got {n} results"


def test_filtered_search_stale_allowlist_fallback(adapter):
    """Layer (c): when every allowlist attempt hits the kernel's
    'not present in index' KeyError (sustained churn), the filtered
    search must fall back to a non-raising post-filtered result instead
    of surfacing the misleading KeyError."""
    s = adapter.make()
    _seed(adapter, s, n=8)
    spy = _install_spy(s, adapter)
    spy.raise_on_allowlist = True

    out = adapter.search_filtered(s, vec_for(0), k=8)
    if isinstance(adapter, LlamaIndexAdapter):
        nodes = out.nodes
        assert len(nodes) == 4
        assert all(node.metadata["g"] == 1 for node in nodes)
    elif isinstance(adapter, LangChainAdapter):
        assert len(out) == 4
        assert all(doc.metadata["g"] == 1 for doc, _score in out)
    elif isinstance(adapter, HaystackAdapter):
        assert len(out) == 4
        assert all(doc.meta["g"] == 1 for doc in out)
    else:  # agno
        assert len(out) == 4
        assert all(doc.meta_data["g"] == 1 for doc in out)
