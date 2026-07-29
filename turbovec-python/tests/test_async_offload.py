"""Event-loop behaviour of the async integration surfaces (issue #342).

The langchain / llama_index / agno ``async``/``a*`` methods used to call
their sync bodies inline on the loop thread, so a large add blocked the
loop for its whole duration and ``asyncio.wait_for`` could not fire. They
now run the sync body on a worker thread.

Three properties are pinned here, all structural rather than timing-tight:

1. **Offload** — the sync body observably runs on a thread other than the
   one running the loop. Deterministic: no timing at all.
2. **Responsiveness** — a heartbeat task ticks while an async op is in
   flight. The op is made slow by an injected ``time.sleep`` (which
   releases the GIL exactly like the index's own hot loops do), and the
   threshold is a small fraction of the ticks a healthy loop produces.
3. **Cancellation** — ``asyncio.wait_for`` raises on a slow op instead of
   running it to completion, and the write it interrupted lands
   *all-or-nothing*. Both branches are real and neither is guaranteed:
   if the worker had already started, cancelling frees the awaiting
   caller but cannot abort work inside the Rust core, so the write
   commits in full; if the executor was saturated the call is cancelled
   before it ever starts and nothing is written. The contract a caller
   can rely on is "outcome unknown, never torn" — pinned here in that
   form, with a dedicated cell for the never-started branch, so the docs
   and the behaviour can't drift apart.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

DIM = 32
# Injected op duration for the responsiveness / cancellation cells. Long
# enough that a loaded machine only ever makes the margins wider.
SLOW = 1.0


def _unit(text: str, dim: int = DIM) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def _record_thread(monkeypatch, owner, name):
    """Patch ``owner.name`` to record the thread it runs on. Returns the
    (growing) list of thread idents."""
    seen: list[int] = []
    orig = getattr(owner, name)

    def wrapper(self, *args, **kwargs):
        seen.append(threading.get_ident())
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return seen


def _make_slow(monkeypatch, owner, name, delay=SLOW):
    """Patch ``owner.name`` to take ``delay`` seconds before doing its
    real work. ``time.sleep`` releases the GIL, which is the same thing
    the index does around its heavy kernels."""
    orig = getattr(owner, name)

    def wrapper(self, *args, **kwargs):
        time.sleep(delay)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)


def _assert_offloaded(seen, loop_thread):
    assert seen, "the sync body never ran — the test proved nothing"
    assert loop_thread not in seen, (
        "the sync body ran on the event-loop thread; the async method is "
        "blocking the loop for its full duration"
    )


def _run_capturing_loop_thread(coro_factory):
    """Run ``coro_factory(...)`` under asyncio.run and return the ident of
    the thread the loop ran on."""
    holder: list[int] = []

    async def main():
        holder.append(threading.get_ident())
        await coro_factory()

    asyncio.run(main())
    return holder[0]


# ---------------------------------------------------------------------------
# langchain
# ---------------------------------------------------------------------------


def _langchain_store():
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings

    from turbovec.langchain import TurboQuantVectorStore

    class StubEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [_unit(t) for t in texts]

        def embed_query(self, text):
            return _unit(text)

        # Native async overrides, deliberately. `Embeddings` inherits
        # `aembed_*` implementations that hop through
        # `run_in_executor(None, ...)`, i.e. the loop's default executor —
        # so a cell that saturates that executor would strand the
        # coroutine at the *embedding* step and never reach the index
        # offload under test. These keep the embedding step on the loop
        # so the only thread hop is the one being measured.
        async def aembed_documents(self, texts):
            return self.embed_documents(texts)

        async def aembed_query(self, text):
            return self.embed_query(text)

    return TurboQuantVectorStore(embedding=StubEmbeddings()), TurboQuantVectorStore


LANGCHAIN_CASES = {
    # async method -> (sync body it must offload, coroutine factory)
    "aadd_texts": (
        "_store_texts_and_vectors",
        lambda s: s.aadd_texts(["alpha", "beta"]),
    ),
    "asimilarity_search_with_score": (
        "_search_vector",
        lambda s: s.asimilarity_search_with_score("alpha", k=1),
    ),
    "asimilarity_search_by_vector": (
        "similarity_search_by_vector",
        lambda s: s.asimilarity_search_by_vector(_unit("alpha"), k=1),
    ),
    "asimilarity_search_with_score_by_vector": (
        "similarity_search_with_score_by_vector",
        lambda s: s.asimilarity_search_with_score_by_vector(_unit("alpha"), k=1),
    ),
    "aget_by_ids": ("get_by_ids", lambda s: s.aget_by_ids(["id-0"])),
    "adelete": ("delete", lambda s: s.adelete(["id-0"])),
}


@pytest.mark.parametrize("method", sorted(LANGCHAIN_CASES))
def test_langchain_async_runs_off_the_event_loop(method, monkeypatch):
    store, cls = _langchain_store()
    store.add_texts(["alpha", "beta"], ids=["id-0", "id-1"])
    sync_name, factory = LANGCHAIN_CASES[method]
    seen = _record_thread(monkeypatch, cls, sync_name)
    loop_thread = _run_capturing_loop_thread(lambda: factory(store))
    _assert_offloaded(seen, loop_thread)


def test_langchain_aadd_texts_leaves_the_loop_responsive(monkeypatch):
    store, cls = _langchain_store()
    _make_slow(monkeypatch, cls, "_store_texts_and_vectors")

    async def main():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await store.aadd_texts(["alpha", "beta"])
        beat.cancel()
        return ticks

    ticks = asyncio.run(main())
    # A healthy loop ticks ~100x over SLOW=1s. Require 5 — enough to prove
    # the loop ran (it ticked zero times when the add was inline), loose
    # enough to survive a badly loaded machine.
    assert ticks >= 5, f"loop ticked only {ticks} times during a {SLOW}s add"


def test_langchain_aadd_texts_is_cancellable_and_lands_all_or_nothing(monkeypatch):
    """`wait_for` must now fire, and the interrupted write must be
    all-or-nothing. Which of the two it is depends on whether the worker
    had started, so neither branch may be asserted on its own — only that
    the store is coherent afterwards."""
    store, cls = _langchain_store()
    _make_slow(monkeypatch, cls, "_store_texts_and_vectors")

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                store.aadd_texts(["alpha", "beta"], ids=["id-0", "id-1"]),
                timeout=SLOW / 20,
            )

    # asyncio.run shuts the default executor down on the way out, so any
    # worker that did start has finished by the time this returns.
    asyncio.run(main())
    assert len(store._docs) in (0, 2), "the cancelled write landed torn"
    assert len(store._docs) == len(store._str_to_u64) == len(store._index)


def test_langchain_cancelled_before_start_writes_nothing():
    """The other branch of the same contract: with the executor already
    saturated the offloaded call is cancelled before it ever runs, so the
    write never happens. This is why the docs say "outcome unknown"
    rather than "the write still commits"."""
    store, _cls = _langchain_store()
    release = threading.Event()

    # Trace the embedding step. Without this the cell could pass for the
    # wrong reason — a coroutine stranded at an earlier `run_in_executor`
    # also writes nothing — and would then pass against a build that has
    # no offload at all.
    embedded: list[int] = []
    inner = store._embedding.aembed_documents

    async def traced(texts):
        embedded.append(len(texts))
        return await inner(texts)

    store._embedding.aembed_documents = traced

    async def main():
        executor = ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(executor)
        # Occupy the only worker for the whole test.
        executor.submit(release.wait)
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    store.aadd_texts(["alpha", "beta"], ids=["id-0", "id-1"]),
                    timeout=SLOW / 20,
                )
        finally:
            release.set()
            executor.shutdown(wait=True)

    asyncio.run(main())
    assert embedded == [2], (
        "the coroutine never reached the offload — it was cancelled at an "
        "earlier suspension point, so this cell proves nothing about the write"
    )
    assert len(store._docs) == 0, "the add ran despite never being scheduled"


# ---------------------------------------------------------------------------
# llama_index
# ---------------------------------------------------------------------------


def _llama_store():
    pytest.importorskip("llama_index.core")
    from turbovec.llama_index import TurboQuantVectorStore

    return TurboQuantVectorStore(), TurboQuantVectorStore


def _llama_nodes(n=2):
    from llama_index.core.schema import TextNode

    return [
        TextNode(id_=f"n{i}", text=f"t{i}", embedding=_unit(f"t{i}"))
        for i in range(n)
    ]


def _llama_query(k=1):
    from llama_index.core.vector_stores.types import VectorStoreQuery

    return VectorStoreQuery(query_embedding=_unit("t0"), similarity_top_k=k)


LLAMA_CASES = {
    "async_add": ("add", lambda s: s.async_add(_llama_nodes())),
    "aquery": ("query", lambda s: s.aquery(_llama_query())),
    "aget_nodes": ("get_nodes", lambda s: s.aget_nodes(node_ids=["n0"])),
    "adelete": ("delete", lambda s: s.adelete("ref-0")),
    "adelete_nodes": ("delete_nodes", lambda s: s.adelete_nodes(node_ids=["n0"])),
    "aclear": ("clear", lambda s: s.aclear()),
}


@pytest.mark.parametrize("method", sorted(LLAMA_CASES))
def test_llama_index_async_runs_off_the_event_loop(method, monkeypatch):
    store, cls = _llama_store()
    store.add(_llama_nodes())
    sync_name, factory = LLAMA_CASES[method]
    seen = _record_thread(monkeypatch, cls, sync_name)
    loop_thread = _run_capturing_loop_thread(lambda: factory(store))
    _assert_offloaded(seen, loop_thread)


def test_llama_index_async_add_is_cancellable_and_lands_all_or_nothing(monkeypatch):
    store, cls = _llama_store()
    _make_slow(monkeypatch, cls, "add")

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                store.async_add(_llama_nodes()), timeout=SLOW / 20
            )

    asyncio.run(main())
    # See the langchain cell: whether the worker started is not
    # guaranteed, so only all-or-nothing may be asserted.
    assert len(store._nodes) in (0, 2), "the cancelled add landed torn"
    assert len(store._nodes) == len(store._index)


# ---------------------------------------------------------------------------
# agno
# ---------------------------------------------------------------------------


def _agno_db():
    pytest.importorskip("agno")
    from turbovec.agno import TurboQuantVectorDb

    class StubEmbedder:
        enable_batch = False

        def __init__(self) -> None:
            self.dimensions = DIM

        def get_embedding(self, text: str):
            return _unit(text)

        async def async_get_embedding(self, text: str):
            return _unit(text)

    return TurboQuantVectorDb(embedder=StubEmbedder()), TurboQuantVectorDb


def _agno_docs(n=2):
    from agno.knowledge.document import Document

    return [
        Document(id=f"d{i}", content=f"c{i}", meta_data={}, embedding=_unit(f"c{i}"))
        for i in range(n)
    ]


AGNO_CASES = {
    "async_create": ("create", lambda db: db.async_create()),
    "async_drop": ("drop", lambda db: db.async_drop()),
    "async_insert": ("insert", lambda db: db.async_insert("h2", _agno_docs())),
    "async_upsert": ("upsert", lambda db: db.async_upsert("h", _agno_docs())),
    # `_retrieve` rather than the `_search_and_rank` wrapper: it predates
    # this change, so the cell discriminates on behaviour, not on a name.
    "async_search": ("_retrieve", lambda db: db.async_search("c0", limit=1)),
}


@pytest.mark.parametrize("method", sorted(AGNO_CASES))
def test_agno_async_runs_off_the_event_loop(method, monkeypatch):
    db, cls = _agno_db()
    db.create()
    db.insert("h", _agno_docs())
    sync_name, factory = AGNO_CASES[method]
    seen = _record_thread(monkeypatch, cls, sync_name)
    loop_thread = _run_capturing_loop_thread(lambda: factory(db))
    _assert_offloaded(seen, loop_thread)


def test_agno_async_insert_is_cancellable_and_lands_all_or_nothing(monkeypatch):
    db, cls = _agno_db()
    db.create()
    _make_slow(monkeypatch, cls, "insert")

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                db.async_insert("h", _agno_docs()), timeout=SLOW / 20
            )

    asyncio.run(main())
    # See the langchain cell: whether the worker started is not
    # guaranteed, so only all-or-nothing may be asserted.
    assert db.get_count() in (0, 2), "the cancelled insert landed torn"
    assert db.get_count() == len(db._u64_to_doc)
