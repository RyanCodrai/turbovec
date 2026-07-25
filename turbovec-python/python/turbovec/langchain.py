"""LangChain VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[langchain]``.

The public surface mirrors langchain_core's in-tree ``InMemoryVectorStore``
so this store can be swapped in wherever the in-memory store is used.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ._dedup import DuplicatePolicy, resolve_duplicates
from ._persist import check_persisted_handles, check_sidecar_keysets
from ._similarity import COSINE, DOT_PRODUCT, l2_normalize_rows, validate_similarity
from ._turbovec import IdMapIndex
from ._persist import atomic_save  # isort:skip

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:
    raise ImportError(
        "langchain-core is required to use turbovec.langchain. "
        "Install with: pip install turbovec[langchain]"
    ) from exc


_INDEX_FILENAME = "index.tvim"
_STORE_FILENAME = "docstore.json"
# Bump when the docstore.json shape changes; loader accepts the current
# version plus any older versions whose missing fields we know how to
# reconstruct (v1 predates the `similarity` field — those stores hold
# raw, unnormalized vectors, so they load in "dot_product" mode, which
# is exactly the scoring they were written under).
_DOCSTORE_SCHEMA_VERSION = 2
_DOCSTORE_SCHEMA_COMPAT = (1, 2)


class TurboQuantVectorStore(VectorStore):
    """LangChain VectorStore backed by a :class:`IdMapIndex`.

    Vectors are quantized to 2–4 bits per dimension. A side-car dictionary
    holds the original text and metadata keyed by document id. Deletion
    is supported in O(1) per id via the underlying :class:`IdMapIndex`.

    **Similarity modes.** ``similarity="cosine"`` (default) L2-normalizes
    document vectors at add time and query vectors at search time, so
    scores are cosine similarity in ``[-1, 1]`` and ranking matches the
    ``InMemoryVectorStore`` reference for embeddings of any magnitude.
    ``similarity="dot_product"`` stores and queries raw vectors: scores
    are raw inner products and ranking is magnitude-aware; absolute
    relevance thresholds are dataset-relative in this mode. The mode is
    fixed at construction and recorded by :meth:`dump`; :meth:`load`
    restores it (stores persisted before the mode existed load as
    ``"dot_product"`` — the raw-vector scoring they were written under).
    This parameter is a turbovec extension — the reference
    ``InMemoryVectorStore`` computes cosine unconditionally.

    **Thread safety.** The store is safe for concurrent multi-threaded
    use. Reads (``similarity_search*``, ``get_by_ids``) run concurrently
    and scale across threads; writes (``add_*``, ``delete``, ``dump``)
    serialize on a per-store lock. A read that overlaps a write sees
    either the pre- or post-write state — never a torn one — and under
    heavy concurrent churn a search may transiently return fewer than
    ``k`` results. There is no cross-call atomicity: a caller-side
    check-then-act sequence (e.g. ``get_by_ids`` then ``delete``) can
    still interleave with other writers. Multi-process access is not
    supported.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index: IdMapIndex | None = None,
        *,
        bit_width: int = 4,
        similarity: str = "cosine",
        docs: dict[str, tuple[str, dict[str, Any]]] | None = None,
        str_to_u64: dict[str, int] | None = None,
        next_u64: int = 0,
    ) -> None:
        """
        :param embedding: LangChain ``Embeddings`` instance used to encode
            documents and queries.
        :param index: Optional pre-built :class:`IdMapIndex`. When omitted,
            a lazy ``IdMapIndex`` is created — it commits to a dim on the
            first add and lets us match the no-arg constructor pattern of
            langchain_core's ``InMemoryVectorStore``.
        :param bit_width: Quantization width (2, 3, or 4) used when the index
            is created from scratch. Ignored if ``index`` is supplied.
        :param similarity: ``"cosine"`` (default) or ``"dot_product"``.
            See the class docstring. Fixed for the lifetime of the store.
        """
        self._embedding = embedding
        self._similarity = validate_similarity(similarity)
        # IdMapIndex itself supports lazy construction now — no per-store
        # lazy wrapping needed. When `index` is None we create a lazy
        # IdMapIndex(dim=None, bit_width) and let it handle the rest.
        self._index = index if index is not None else IdMapIndex(bit_width=bit_width)
        self._docs: dict[str, tuple[str, dict[str, Any]]] = docs if docs is not None else {}
        self._str_to_u64: dict[str, int] = str_to_u64 if str_to_u64 is not None else {}
        # Reverse map (u64 handle → str id) kept in sync so search results
        # can translate handles back to LangChain document ids.
        self._u64_to_str: dict[int, str] = {
            handle: sid for sid, handle in self._str_to_u64.items()
        }
        self._next_u64: int = next_u64
        # Serializes every mutation (add / delete / dump). Readers do NOT
        # take this lock — search stays lock-free so concurrent reads keep
        # scaling across threads (#186); reader safety comes from mutation
        # ordering plus tolerant handle resolution instead. RLock because
        # the add path may nest into delete-style cleanup.
        self._write_lock = threading.RLock()

    def _issue_handle(self) -> int:
        self._next_u64 += 1
        return self._next_u64

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def similarity(self) -> str:
        """The store's similarity mode: ``"cosine"`` or ``"dot_product"``."""
        return self._similarity

    # ---- Relevance score normalization --------------------------------

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # Under the default cosine mode both sides are unit vectors, so
        # the engine's raw inner product is true cosine similarity in
        # [-1, 1]; (sim + 1) / 2 maps it onto LangChain's [0, 1]
        # relevance scale and the clamp only absorbs quantization noise.
        # Under dot_product mode scores are raw inner products with no
        # fixed range — the same mapping is applied for continuity with
        # earlier releases, but values beyond [-1, 1] saturate at the
        # clamp, so score_threshold filtering is only meaningful there
        # if the embeddings are unit-normalized upstream.
        return lambda sim: max(0.0, min(1.0, (sim + 1.0) / 2.0))

    # ---- Embedder-output validation -----------------------------------

    @staticmethod
    def _check_embedded_batch(vectors: np.ndarray, n_texts: int) -> None:
        """Validate the shape of an embedder's document-batch output.

        Only 2D outputs are inspected here — any other ndim falls through
        to ``_store_texts_and_vectors``, whose existing guard names the bad
        dimensionality directly. Without the row-count check, a misbehaving
        embedder that returns fewer vectors than texts surfaces downstream
        as an id-count mismatch (or an IndexError during intra-batch
        dedup) that never names the embedder as the cause.
        """
        if vectors.ndim != 2:
            return
        if vectors.shape[0] != n_texts:
            raise ValueError(
                f"embedder returned {vectors.shape[0]} vectors for "
                f"{n_texts} texts"
            )
        if vectors.shape[1] == 0:
            raise ValueError(
                f"embedder returned empty vectors (dim 0) for {n_texts} texts"
            )

    def _validate_query_embedding(self, embedded: Any) -> np.ndarray:
        """Coerce an embedder's query output to a 1D float32 vector,
        rejecting None and wrong-rank outputs with an error that names
        the embedder (the raw values otherwise surface as opaque errors
        from the index kernel)."""
        if embedded is None:
            raise ValueError("embedder returned None instead of a query embedding")
        qvec = np.asarray(embedded, dtype=np.float32)
        if qvec.ndim != 1:
            raise ValueError(
                f"embedder returned a {qvec.ndim}D query embedding; "
                "expected a single 1D vector"
            )
        return qvec

    # ---- Write path ---------------------------------------------------

    @staticmethod
    def _normalize_ids(ids: list[str]) -> list[str]:
        """Return a fresh ``list[str]`` copy of ``ids`` with per-entry
        ``None`` replaced by a generated UUID (matches the reference
        InMemoryVectorStore and keeps None out of the JSON side-car,
        where it would round-trip as the string ``"null"``).

        Any other non-``str`` id raises ``TypeError``. This is a
        deliberate deviation from the reference InMemoryVectorStore,
        which accepts e.g. an ``int`` id and then corrupts it: JSON
        persistence coerces every key to ``str``, so ``2`` and ``"2"``
        are two documents in memory but collapse to one on ``dump``/
        ``load`` — silent data loss plus an out-of-sync side-car.
        Rejecting at the add boundary (the declared contract is
        ``list[str]``) makes that state unrepresentable. ``bool`` is a
        subclass of ``int`` and is rejected like any other non-str type:
        only ``str`` instances (and ``None``) are accepted.
        """
        normalized: list[str] = []
        for pos, id_ in enumerate(ids):
            if id_ is None:
                normalized.append(str(uuid.uuid4()))
            elif isinstance(id_, str):
                normalized.append(id_)
            else:
                raise TypeError(
                    f"ids[{pos}] is {id_!r} of type {type(id_).__name__}; "
                    "ids must be str (or None for a generated UUID). "
                    "Non-str ids are rejected because JSON persistence "
                    "coerces keys to str, silently colliding with any "
                    "equal-looking str id (e.g. 2 vs '2') on dump/load."
                )
        return normalized

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **_: Any,
    ) -> list[str]:
        texts_list = list(texts)
        if not texts_list:
            return []
        if metadatas is None:
            metadatas = [{} for _ in texts_list]
        else:
            # Materialize once so a generator (or other one-shot iterable)
            # survives the length check and the storage loop below.
            metadatas = list(metadatas)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            # Fresh list[str] (whatever container the caller passed),
            # per-entry None -> UUID, any other non-str id -> TypeError.
            # Raised here, before embedding or any store mutation.
            ids = self._normalize_ids(list(ids))
        if len(metadatas) != len(texts_list) or len(ids) != len(texts_list):
            raise ValueError("texts, metadatas, and ids must all have the same length")

        vectors = np.asarray(self._embedding.embed_documents(texts_list), dtype=np.float32)
        self._check_embedded_batch(vectors, len(texts_list))
        return self._store_texts_and_vectors(texts_list, vectors, metadatas, ids)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **_: Any,
    ) -> list[str]:
        texts_list = list(texts)
        if not texts_list:
            return []
        if metadatas is None:
            metadatas = [{} for _ in texts_list]
        else:
            # See add_texts: materialize one-shot iterables once.
            metadatas = list(metadatas)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            # See add_texts: fresh list[str], None -> UUID, non-str -> TypeError.
            ids = self._normalize_ids(list(ids))
        if len(metadatas) != len(texts_list) or len(ids) != len(texts_list):
            raise ValueError("texts, metadatas, and ids must all have the same length")

        vectors = np.asarray(
            await self._embedding.aembed_documents(texts_list), dtype=np.float32
        )
        self._check_embedded_batch(vectors, len(texts_list))
        return self._store_texts_and_vectors(texts_list, vectors, metadatas, ids)

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        # Override the base class default which drops the entire `ids` array
        # if any Document has a None id. The reference InMemoryVectorStore
        # falls back per-document so partial ids are honoured.
        # Materialize once — `documents` is iterated up to three times
        # below, which would silently drain a generator input.
        documents = list(documents)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None:
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        # See add_documents: materialize one-shot iterables once.
        documents = list(documents)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None:
            ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        return await self.aadd_texts(
            texts=texts, metadatas=metadatas, ids=ids, **kwargs
        )

    def _store_texts_and_vectors(
        self,
        texts_list: list[str],
        vectors: np.ndarray,
        metadatas: list[dict],
        ids: list[str],
    ) -> list[str]:
        if vectors.ndim != 2:
            raise ValueError(f"expected 2D embedding batch, got {vectors.ndim}D")

        # Validate every metadata entry before touching any state. A bad
        # entry (e.g. None) previously blew up as `dict(None)` mid-loop,
        # *after* the vectors had been added to the index — leaving the
        # docstore and index desynced in memory (and dump() would persist
        # the corruption). The reference InMemoryVectorStore rejects bad
        # metadata with a named error; mirror that here.
        for i, meta in enumerate(metadatas):
            if not isinstance(meta, dict):
                raise TypeError(
                    f"metadatas[{i}] must be a dict, "
                    f"got {type(meta).__name__}"
                )

        # Dedup intra-batch duplicate ids, keeping the last occurrence —
        # matches InMemoryVectorStore, whose dict store silently overwrites
        # on a repeated id. Without this every row is added to the index but
        # _str_to_u64 keeps only the last handle per id, orphaning the
        # earlier vectors. The returned id list still mirrors the input
        # (one entry per input text), as the reference does.
        result_ids = ids
        keep = resolve_duplicates(ids, DuplicatePolicy.KEEP_LAST)
        if len(keep) != len(ids):
            ids = [ids[i] for i in keep]
            texts_list = [texts_list[i] for i in keep]
            metadatas = [metadatas[i] for i in keep]
            vectors = vectors[keep]

        # Cosine mode: L2-normalize outside the lock (pure computation,
        # like the embedding step) so the engine's raw inner product is
        # true cosine similarity. Zero rows pass through unchanged.
        if self._similarity == COSINE:
            vectors = l2_normalize_rows(vectors)

        with self._write_lock:
            # Validate before mutating any existing data. IdMapIndex.add_with_ids
            # handles both eager (dim must match) and lazy (locks dim on first
            # call) cases. Pre-check the eager case so we surface a clean
            # ValueError rather than a Rust panic.
            existing_dim = self._index.dim
            if existing_dim is not None and vectors.shape[1] != existing_dim:
                raise ValueError(
                    f"embedding dimension {vectors.shape[1]} does not match index dim {existing_dim}"
                )
            if not vectors.flags["C_CONTIGUOUS"]:
                vectors = np.ascontiguousarray(vectors)

            handles = np.array(
                [self._issue_handle() for _ in texts_list], dtype=np.uint64
            )

            # Capture the previous state of any upserted id BEFORE the maps
            # are overwritten, so a failed index add can restore it and the
            # old vectors can be dropped once the add succeeds.
            old = [
                (i, self._str_to_u64[i], self._docs[i])
                for i in ids
                if i in self._str_to_u64
            ]

            # Maps BEFORE the index add: a concurrent search can only learn
            # a handle from the index, so an entry that is resolvable but
            # not yet searchable is invisible to readers (safe) — the
            # reverse ordering let readers surface handles that did not
            # resolve yet (issue #161).
            for id_, text, meta, handle in zip(ids, texts_list, metadatas, handles):
                h = int(handle)
                self._str_to_u64[id_] = h
                self._u64_to_str[h] = id_
                self._docs[id_] = (text, dict(meta))
            try:
                self._index.add_with_ids(vectors, handles)
            except BaseException:
                # Unwind the pre-inserted map entries (and restore the
                # previous mapping of any upserted id) so a failed add
                # never destroys existing data — preserving the issue-#89
                # guarantee under the maps-first ordering.
                for id_, handle in zip(ids, handles):
                    h = int(handle)
                    self._u64_to_str.pop(h, None)
                    if self._str_to_u64.get(id_) == h:
                        self._str_to_u64.pop(id_, None)
                        self._docs.pop(id_, None)
                for id_, old_h, old_doc in old:
                    self._str_to_u64[id_] = old_h
                    self._u64_to_str[old_h] = id_
                    self._docs[id_] = old_doc
                raise

            # Upsert: drop the replaced vectors, index first so the old
            # handle stops being searchable before it stops resolving. The
            # forward maps already hold the new entries, so only the old
            # handle and its reverse-map entry remain to clean up.
            for _id, old_h, _old_doc in old:
                self._index.remove(old_h)
                self._u64_to_str.pop(old_h, None)
        return result_ids

    # ---- Read path (similarity search) --------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
        **_: Any,
    ) -> list[Document]:
        return [
            doc
            for doc, _score in self.similarity_search_with_score(query, k=k, filter=filter)
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
        **_: Any,
    ) -> list[Document]:
        return [
            doc
            for doc, _score in await self.asimilarity_search_with_score(
                query, k=k, filter=filter
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
        **_: Any,
    ) -> list[tuple[Document, float]]:
        qvec = self._validate_query_embedding(self._embedding.embed_query(query))
        return self._search_vector(qvec, k, filter=filter)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
        **_: Any,
    ) -> list[tuple[Document, float]]:
        qvec = self._validate_query_embedding(
            await self._embedding.aembed_query(query)
        )
        return self._search_vector(qvec, k, filter=filter)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
        **_: Any,
    ) -> list[Document]:
        qvec = np.asarray(embedding, dtype=np.float32)
        return [doc for doc, _score in self._search_vector(qvec, k, filter=filter)]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
        **_: Any,
    ) -> list[Document]:
        # The search itself is sync (no embedding step). Delegate.
        return self.similarity_search_by_vector(embedding, k=k, filter=filter)

    def _search_vector(
        self,
        qvec: np.ndarray,
        k: int,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
    ) -> list[tuple[Document, float]]:
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        # Cosine mode: normalize the query so the raw inner product
        # against unit document vectors is true cosine similarity.
        if self._similarity == COSINE:
            qvec = l2_normalize_rows(qvec)
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)
        # IdMapIndex handles the lazy-uncommitted case internally (returns
        # empty search results). A len-zero check covers both that and
        # the eager-but-empty case.
        if len(self._index) == 0:
            return []

        if filter is None:
            search_k = min(k, len(self._index))
            scores, handles = self._index.search(qvec, search_k)
        else:
            predicate = self._compile_filter(filter)
            for _attempt in range(8):
                # Snapshot the docstore before iterating: list() of the
                # items view is a single C-level operation, so a concurrent
                # writer can't invalidate the iteration mid-scan.
                snapshot = list(self._docs.items())
                allowed_handles = []
                for sid, (text, meta) in snapshot:
                    h = self._str_to_u64.get(sid)
                    if h is None:
                        # The id vanished between the snapshot and here
                        # (concurrent delete) — skip rather than raise.
                        continue
                    if predicate(Document(id=sid, page_content=text, metadata=dict(meta))):
                        allowed_handles.append(h)
                if not allowed_handles:
                    return []
                allowlist = np.asarray(allowed_handles, dtype=np.uint64)
                try:
                    scores, handles = self._index.search(qvec, k, allowlist=allowlist)
                    break
                except KeyError:
                    # The allowlist went stale: a delete landed between the
                    # snapshot above and the kernel's membership check.
                    # Rebuild the allowlist and retry.
                    continue
            else:
                # Sustained churn kept invalidating the allowlist. Fall
                # back to an unfiltered search plus a tolerant Python-side
                # post-filter, which cannot raise (a search overlapping
                # heavy churn may return fewer than k hits).
                search_k = min(max(k * 4, 32), len(self._index) or 1)
                scores, handles = self._index.search(qvec, search_k)
                results: list[tuple[Document, float]] = []
                for score, handle in zip(scores[0], handles[0]):
                    sid = self._u64_to_str.get(int(handle))
                    if sid is None:
                        continue
                    entry = self._docs.get(sid)
                    if entry is None:
                        continue
                    text, meta = entry
                    doc = Document(id=sid, page_content=text, metadata=dict(meta))
                    if predicate(doc):
                        results.append((doc, float(score)))
                    if len(results) >= k:
                        break
                return results

        results = []
        for score, handle in zip(scores[0], handles[0]):
            # Tolerant translation: a handle surfaced by the index can stop
            # resolving if a delete completes between the kernel search and
            # this loop (the reader-straddle). Skip it — the document is
            # gone either way — rather than raising KeyError mid-search.
            sid = self._u64_to_str.get(int(handle))
            if sid is None:
                continue
            entry = self._docs.get(sid)
            if entry is None:
                continue
            text, meta = entry
            results.append(
                (Document(id=sid, page_content=text, metadata=dict(meta)), float(score))
            )
        return results

    @staticmethod
    def _compile_filter(
        filter: dict[str, Any] | Callable[[Document], bool],
    ) -> Callable[[Document], bool]:
        # Match the in-tree InMemoryVectorStore convention: callable filters
        # receive a Document, not a metadata dict
        # (langchain_core/vectorstores/in_memory.py).
        if callable(filter):
            return filter
        if isinstance(filter, dict):
            items = list(filter.items())
            return lambda doc: all(doc.metadata.get(k) == v for k, v in items)
        raise TypeError(
            "filter must be a dict of metadata key/value pairs or a callable "
            f"taking a Document, got {type(filter).__name__}"
        )

    # ---- Max marginal relevance ---------------------------------------
    #
    # MMR requires the full-precision vector of every candidate to compute
    # pairwise diversity scores. turbovec discards full vectors after
    # quantization (that's the point), so we can't faithfully implement
    # MMR. Raise loudly with a useful message rather than silently fall
    # back to the base class's bare NotImplementedError.

    _MMR_MSG = (
        "TurboQuantVectorStore does not support max-marginal-relevance "
        "search because the underlying quantized index discards "
        "full-precision vectors after compression. MMR requires the "
        "original embedding for every candidate to compute pairwise "
        "diversity. Use `similarity_search` / `similarity_search_with_score` "
        "instead, or maintain a parallel store with full-precision "
        "embeddings if you need MMR specifically."
    )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(self._MMR_MSG)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: Callable[[Document], bool] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(self._MMR_MSG)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(self._MMR_MSG)

    # ---- Get / delete -------------------------------------------------

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Return Documents for the given ids. Missing ids are silently skipped
        (matches the InMemoryVectorStore reference)."""
        out: list[Document] = []
        for sid in ids:
            # Single .get instead of check-then-read: a concurrent delete
            # between `in` and `[]` would raise KeyError; a missing id is
            # skipped either way.
            entry = self._docs.get(sid)
            if entry is not None:
                text, meta = entry
                out.append(Document(id=sid, page_content=text, metadata=dict(meta)))
        return out

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return self.get_by_ids(ids)

    def delete(self, ids: list[str] | None = None, **_: Any) -> None:
        """Remove documents by id. Missing ids are silently skipped — matches
        the InMemoryVectorStore reference (which also accepts ``ids=None``
        as a no-op)."""
        if ids is None:
            return
        # See from_texts: materialize once and use len()-based emptiness —
        # a bare `if not ids:` is ambiguous for a multi-element numpy array.
        ids = list(ids)
        if len(ids) == 0:
            return
        with self._write_lock:
            for sid in ids:
                handle = self._str_to_u64.get(sid)
                if handle is None:
                    continue
                # Index first: the handle stops being searchable before it
                # stops resolving, so a concurrent search can never surface
                # a handle whose side-car entries are already gone.
                self._index.remove(handle)
                self._str_to_u64.pop(sid, None)
                self._u64_to_str.pop(handle, None)
                self._docs.pop(sid, None)

    async def adelete(self, ids: list[str] | None = None, **_: Any) -> None:
        self.delete(ids)

    # ---- Construction helpers -----------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        bit_width: int = 4,
        similarity: str = "cosine",
        ids: list[str] | None = None,
        **_: Any,
    ) -> "TurboQuantVectorStore":
        # The underlying index is created lazily on the first `add_texts`
        # call, picking up `dim` from the first batch of embeddings — same
        # no-`dim` ergonomics as InMemoryVectorStore.
        # Materialize once and test emptiness via len(): a bare `if texts:`
        # is ambiguous for a numpy array and drains a generator input.
        texts = list(texts)
        store = cls(embedding=embedding, bit_width=bit_width, similarity=similarity)
        if len(texts) > 0:
            store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        bit_width: int = 4,
        similarity: str = "cosine",
        ids: list[str] | None = None,
        **_: Any,
    ) -> "TurboQuantVectorStore":
        # See from_texts: materialize once, len()-based emptiness.
        texts = list(texts)
        store = cls(embedding=embedding, bit_width=bit_width, similarity=similarity)
        if len(texts) > 0:
            await store.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return store

    # ---- Persistence --------------------------------------------------
    #
    # Method names match the InMemoryVectorStore reference (`dump`/`load`),
    # but the on-disk layout is a folder containing the binary index file
    # plus a JSON side-car (we can't embed the binary Rust index in a
    # single JSON file the way the reference does with its raw-vector
    # store).

    def dump(self, folder_path: str | Path) -> None:
        """Persist the quantized index plus the side-car to disk.

        ``folder_path`` is a directory; turbovec writes ``index.tvim``
        and ``docstore.json`` inside it. Document metadata must be
        JSON-serializable (same constraint as ``InMemoryVectorStore``).
        A lazy uncommitted index encodes its state via the index file's
        own ``dim = 0`` sentinel; no special-case handling needed here.
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        # Serializes with writers: snapshotting the maps and the index
        # concurrently with a write would persist a torn store to disk.
        # Reads may proceed while a dump runs.
        with self._write_lock:
            # `_docs` stores tuples `(text, metadata)` — JSON would drop the
            # tuple-ness on round-trip, so serialize each entry as an explicit
            # `{"text": ..., "metadata": ...}` dict.
            docs_payload = {
                sid: {"text": text, "metadata": meta}
                for sid, (text, meta) in self._docs.items()
            }
            payload = {
                "schema_version": _DOCSTORE_SCHEMA_VERSION,
                "docs": docs_payload,
                "str_to_u64": dict(self._str_to_u64),
                "next_u64": self._next_u64,
                # Pull bit_width off the live index — same value whether
                # the index was constructed eagerly or lazily.
                "bit_width": self._index.bit_width,
                # Recorded so `load` restores the mode the vectors were
                # written under (v2+).
                "similarity": self._similarity,
            }
            # Atomic: serializes in memory first, then temp-file + replace,
            # so a failed dump can't destroy a previous store at this path.
            atomic_save(
                self._index,
                folder / _INDEX_FILENAME,
                payload,
                folder / _STORE_FILENAME,
            )

    @classmethod
    def load(
        cls,
        folder_path: str | Path,
        embedding: Embeddings,
    ) -> "TurboQuantVectorStore":
        """Reload a store from a folder previously written by :meth:`dump`.
        Safe to call on any path — the side-car is plain JSON, never
        pickle, so there's no deserialization-of-code risk.

        The persisted ``similarity`` mode is restored from the side-car.
        A v1 side-car (written before similarity modes existed) holds
        raw, unnormalized vectors, so it loads in ``"dot_product"`` mode
        and keeps exactly the scoring it was written under."""
        folder = Path(folder_path)
        with open(folder / _STORE_FILENAME) as f:
            state = json.load(f)
        version = state.get("schema_version", 0)
        if version not in _DOCSTORE_SCHEMA_COMPAT:
            raise ValueError(
                f"docstore.json has schema version {version}; "
                f"this turbovec accepts versions {list(_DOCSTORE_SCHEMA_COMPAT)}"
            )
        # IdMapIndex.load handles the dim=0 (lazy-uncommitted) sentinel
        # internally and reconstructs the index in the right state.
        index = IdMapIndex.load(str(folder / _INDEX_FILENAME))
        # Rehydrate `_docs` from the explicit `{"text", "metadata"}` form
        # back into the internal tuple representation.
        docs = {
            sid: (entry["text"], entry["metadata"])
            for sid, entry in state["docs"].items()
        }
        # JSON object keys are strings; the str_to_u64 values are already
        # ints in the payload, just need to confirm.
        str_to_u64 = {sid: int(h) for sid, h in state["str_to_u64"].items()}
        check_persisted_handles(index, str_to_u64.values(), what="document")
        # The side-car holds two maps keyed by document id (`docs` and
        # `str_to_u64`); they can desync independently of the index. A
        # `docs` entry missing for a mapped id would otherwise surface as
        # a KeyError deep inside a later query (issue #125).
        check_sidecar_keysets(
            str_to_u64.keys(),
            docs.keys(),
            what="document",
            mapping_name="str_to_u64",
            sidecar_name="docs",
        )
        return cls(
            embedding=embedding,
            index=index,
            bit_width=state.get("bit_width", 4),
            # v1 side-cars predate the mode field: their vectors are raw,
            # so dot_product is the mode they actually contain.
            similarity=state.get("similarity", DOT_PRODUCT),
            docs=docs,
            str_to_u64=str_to_u64,
            next_u64=int(state["next_u64"]),
        )


__all__ = ["TurboQuantVectorStore"]
