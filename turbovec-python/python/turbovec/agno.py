"""Agno VectorDb backed by turbovec's quantized index.

Install with: ``pip install turbovec[agno]``.

Implements Agno's ``VectorDb`` interface and matches the public surface
of ``agno.vectordb.lancedb.LanceDb`` (the closest in-tree single-machine
backend) so this can be swapped in wherever ``LanceDb`` is used.
"""

from __future__ import annotations

import json
import threading
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import numpy as np

from ._persist import check_persisted_handles
from ._similarity import l2_normalize_rows
from ._turbovec import IdMapIndex
from ._persist import atomic_save  # isort:skip

try:
    from agno.knowledge.document import Document
    from agno.knowledge.embedder import Embedder
    from agno.knowledge.reranker.base import Reranker
    from agno.vectordb.base import VectorDb
    from agno.vectordb.distance import Distance
    from agno.vectordb.search import SearchType
except ImportError as exc:
    raise ImportError(
        "agno is required to use turbovec.agno. "
        "Install with: pip install turbovec[agno]"
    ) from exc


_INDEX_FILENAME = "index.tvim"
_STORE_FILENAME = "docstore.json"
# Bump when docstore.json shape changes; loader accepts the current
# version plus any older versions whose missing fields we know how to
# reconstruct (v1 predates the `distance` field — those stores hold raw,
# unnormalized vectors, so they load as Distance.max_inner_product,
# which is exactly the scoring they were written under).
_DOCSTORE_SCHEMA_VERSION = 2
_DOCSTORE_SCHEMA_COMPAT = (1, 2)


class TurboQuantVectorDb(VectorDb):
    """Agno VectorDb backed by a :class:`IdMapIndex`.

    Vectors are quantized to 2-4 bits per dimension. The public surface
    mirrors ``agno.vectordb.lancedb.LanceDb`` so this is a drop-in
    replacement wherever a single-machine LanceDb is used.

    **Similarity modes.** ``distance=Distance.cosine`` (default)
    L2-normalizes document embeddings at insert time and query
    embeddings at search time, so the kernel's raw score is cosine
    similarity in ``[-1, 1]`` and ``similarity_threshold`` filtering
    (which maps the score to ``[0, 1]`` via ``(sim + 1) / 2``) is
    meaningful for embeddings of any magnitude.
    ``distance=Distance.max_inner_product`` stores and queries raw
    vectors: ranking is by raw inner product (magnitude-aware) and
    ``similarity_threshold`` values are dataset-relative — the mapped
    score saturates at 0/1 once raw scores leave ``[-1, 1]``, so
    calibrate thresholds against your own embedder or leave the
    threshold unset. ``Distance.l2`` is not supported. The mode is fixed
    at construction and recorded by :meth:`save`; loading a save whose
    recorded mode conflicts with the constructor's raises ``ValueError``
    (saves from before the mode existed load as raw
    ``max_inner_product`` — the scoring they were written under — and
    ``self.distance`` is updated to match).

    Search-time
    filtering is resolved to an allowlist *before* scoring (kernel-level)
    rather than via post-filtering, so selective filters return up to
    ``limit`` results from the filtered set instead of fewer. Search
    results are deduplicated by content (``md5(doc.content)``, first
    occurrence kept) as the final step, matching ``LanceDb.search`` —
    when duplicate-content hits exist, callers receive fewer than
    ``limit`` documents.

    **Thread safety.** The store is safe for concurrent multi-threaded
    use. Reads (``search``, existence checks, ``get_count``) run
    concurrently and scale across threads; writes (``insert``,
    ``upsert``, ``delete_by_*``, ``update_metadata``, ``drop``,
    ``save``) serialize on a per-store lock — the ``async_*`` variants
    delegate to the same locked bodies. A read that overlaps a write
    sees either the pre- or post-write state — never a torn one — and
    under heavy concurrent churn a search may transiently return fewer
    than ``limit`` results. There is no cross-call atomicity: a
    caller-side check-then-act sequence (e.g. ``id_exists`` then
    ``delete_by_id``) can still interleave with other writers.
    Embedders and rerankers are invoked outside the store's lock and
    must be thread-safe themselves. Multi-process access is not
    supported.

    Example::

        from agno.knowledge.embedder.openai import OpenAIEmbedder
        from turbovec.agno import TurboQuantVectorDb

        vector_db = TurboQuantVectorDb(embedder=OpenAIEmbedder())
        vector_db.create()
        # ... use as a normal Agno VectorDb ...
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        embedder: Optional[Embedder] = None,
        bit_width: int = 4,
        search_type: SearchType = SearchType.vector,
        distance: Distance = Distance.cosine,
        reranker: Optional[Reranker] = None,
        path: Optional[str] = None,
    ) -> None:
        """
        :param embedder: Required. Agno embedder used to encode documents
            and queries. ``embedder.dimensions`` must be set — it's the
            sole source of truth for the underlying quantized index's
            dimensionality.
        :param bit_width: Quantization width (2, 3, or 4).
        :param search_type: Only :class:`SearchType.vector` is supported;
            other values raise :class:`ValueError`. (Keyword/hybrid search
            would require an external BM25/lexical index.)
        :param distance: :class:`Distance.cosine` (default) or
            :class:`Distance.max_inner_product` — see the class
            docstring. :class:`Distance.l2` raises :class:`ValueError`.
            Fixed for the lifetime of the store.
        :param reranker: Optional Agno reranker applied to the result set
            after vector retrieval.
        :param path: Optional directory for save/load persistence. When
            given to the constructor, :meth:`create` loads existing data
            from this path if present.
        """
        super().__init__(
            id=id,
            name=name,
            description=description,
            similarity_threshold=similarity_threshold,
        )
        if embedder is None:
            raise ValueError(
                "`embedder` is required; turbovec needs the embedder's "
                "`dimensions` to size the underlying index."
            )
        if embedder.dimensions is None:
            raise ValueError("Embedder.dimensions must be set.")
        if bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width}")
        if search_type != SearchType.vector:
            raise ValueError(
                f"TurboQuantVectorDb only supports search_type=SearchType.vector; "
                f"got {search_type}. Use LanceDb / Chroma / etc. for keyword "
                f"or hybrid search."
            )
        if distance not in (Distance.cosine, Distance.max_inner_product):
            raise ValueError(
                f"TurboQuantVectorDb supports distance=Distance.cosine or "
                f"distance=Distance.max_inner_product; got {distance}. "
                f"L2 distance is not supported by the underlying "
                f"inner-product kernel."
            )

        self.embedder: Embedder = embedder
        self.dimensions: int = embedder.dimensions
        self.bit_width = bit_width
        self.search_type = search_type
        self.distance = distance
        self.reranker = reranker
        self.path: Optional[str] = path

        # Lazy: the underlying IdMapIndex is created by `create()`, not
        # in __init__. This matches LanceDb's `exists()` contract: a
        # freshly-constructed store doesn't "exist" until `create()` is
        # called, and `drop()` returns it to that state.
        self._index: Optional[IdMapIndex] = None
        # str doc_id -> set of u64 handles. One-to-many: agno's derived
        # doc_id is NOT unique (two documents with identical content, or a
        # repeated explicit doc.id within a batch, derive the same id), and
        # LanceDb keeps every such row. Mapping one doc_id to a single handle
        # silently orphaned the earlier vectors — present in search and the
        # index count but unreachable by id, so undeletable (issue #104).
        self._str_to_u64: Dict[str, Set[int]] = {}
        # u64 handle -> stored payload (mirrors LanceDb's "payload" shape)
        self._u64_to_doc: Dict[int, Dict[str, Any]] = {}
        # u64 handle assignment counter
        self._next_u64: int = 0
        # Auxiliary indexes for O(1) protocol queries
        self._content_hashes: Set[str] = set()
        self._name_to_ids: Dict[str, Set[str]] = {}
        # Serializes every mutation (insert / upsert / delete / update /
        # drop / save). Readers do NOT take this lock — search stays
        # lock-free so concurrent reads keep scaling across threads
        # (#186); reader safety comes from mutation ordering plus
        # tolerant handle resolution instead. RLock: upsert nests into
        # insert and _remove_handle.
        self._write_lock = threading.RLock()

    # ---- handle allocation ------------------------------------------------

    def _issue_handle(self) -> int:
        self._next_u64 += 1
        return self._next_u64

    # ---- VectorDb protocol: lifecycle ------------------------------------

    def create(self) -> None:
        """Create the underlying index if it doesn't already exist.
        Idempotent — calling on an already-created store is a no-op.

        If ``path`` was set on the constructor and a previous save exists
        under it, ``create()`` loads that save; otherwise it instantiates
        a fresh empty index sized to ``embedder.dimensions``.
        """
        with self._write_lock:
            if self._index is not None:
                return
            # Try loading from path first if one was set; fall through to a
            # fresh index if the path doesn't contain a previous save.
            if self.path is not None and Path(self.path).is_dir():
                try:
                    self._load_from(Path(self.path))
                    return
                except FileNotFoundError:
                    pass
            self._index = IdMapIndex(self.dimensions, self.bit_width)

    async def async_create(self) -> None:
        self.create()

    def drop(self) -> None:
        """Drop the underlying index. After this call ``exists()`` returns
        ``False`` until ``create()`` is called again — matches LanceDb's
        contract where ``drop()`` removes the table entirely."""
        # Remove the persisted artifacts too, so a later create() starts
        # fresh instead of reloading — and resurrecting — the dropped
        # data (issue #169). Matches LanceDb's drop_table, which deletes
        # the on-disk table.
        with self._write_lock:
            if self.path is not None:
                folder = Path(self.path)
                for filename in (_INDEX_FILENAME, _STORE_FILENAME):
                    (folder / filename).unlink(missing_ok=True)
            self._index = None
            self._str_to_u64.clear()
            self._u64_to_doc.clear()
            self._next_u64 = 0
            self._content_hashes.clear()
            self._name_to_ids.clear()

    async def async_drop(self) -> None:
        self.drop()

    def exists(self) -> bool:
        """True iff the underlying index has been created via ``create()``
        and not subsequently dropped. Matches LanceDb's
        "table-exists-in-connection" semantic; *does not* mean
        "has any documents" — call ``get_count()`` for that."""
        return self._index is not None

    async def async_exists(self) -> bool:
        return self.exists()

    def delete(self) -> bool:
        """Returns ``False``. The Agno protocol declares this abstract
        method but LanceDb (the drop-in reference) unconditionally
        returns False — actual destruction goes through ``drop()``."""
        return False

    def optimize(self) -> None:
        """No-op. The underlying quantized index doesn't have a
        post-write optimization step. Matches LanceDb's ``optimize()``
        which is also a no-op."""
        return None

    def get_count(self) -> int:
        """Number of documents currently stored."""
        if self._index is None:
            return 0
        return len(self._index)

    async def async_get_count(self) -> int:
        return self.get_count()

    # ---- VectorDb protocol: existence checks ------------------------------

    def name_exists(self, name: str) -> bool:
        if self._index is None:
            return False
        return name in self._name_to_ids

    async def async_name_exists(self, name: str) -> bool:
        # LanceDb raises NotImplementedError here; we have a trivial sync
        # backing call, so we return the real answer. Intentional deviation.
        return self.name_exists(name)

    def id_exists(self, id: str) -> bool:
        if self._index is None:
            return False
        return id in self._str_to_u64

    def content_hash_exists(self, content_hash: str) -> bool:
        if self._index is None:
            return False
        return content_hash in self._content_hashes

    # ---- VectorDb protocol: insert / upsert -------------------------------

    @staticmethod
    def _derive_doc_id(doc: Document, content_hash: str, cleaned_content: str) -> str:
        """Match LanceDb's id-derivation contract so the same doc with the
        same content_hash produces the same stable doc_id across stores."""
        base_id = doc.id or md5(cleaned_content.encode()).hexdigest()
        return md5(f"{base_id}_{content_hash}".encode()).hexdigest()

    def _embed_missing(self, documents: List[Document]) -> None:
        """Populate embeddings on any documents that don't have one. Uses
        the embedder's batch path when available."""
        to_embed = [
            doc
            for doc in documents
            if doc.embedding is None
            or (isinstance(doc.embedding, list) and len(doc.embedding) == 0)
        ]
        if not to_embed:
            return
        if (
            getattr(self.embedder, "enable_batch", False)
            and hasattr(self.embedder, "get_embeddings_batch_and_usage")
        ):
            contents = [doc.content for doc in to_embed]
            embeddings, usages = self.embedder.get_embeddings_batch_and_usage(contents)
            # A misbehaving embedder can return None (or another non-sized
            # object) in place of the embeddings/usages lists. Treat that as
            # "nothing embedded": the documents keep no embedding, and
            # insert()'s missing-embedding check raises the standard
            # "failed to embed N document(s)" error instead of an opaque
            # `len(None)` TypeError here.
            if embeddings is None or not hasattr(embeddings, "__len__"):
                embeddings = []
            if usages is None or not hasattr(usages, "__len__"):
                usages = []
            for j, doc in enumerate(to_embed):
                if j < len(embeddings):
                    doc.embedding = embeddings[j]
                    doc.usage = usages[j] if j < len(usages) else None
        else:
            for doc in to_embed:
                doc.embed(embedder=self.embedder)

    async def _embed_missing_async(self, documents: List[Document]) -> None:
        to_embed = [
            doc
            for doc in documents
            if doc.embedding is None
            or (isinstance(doc.embedding, list) and len(doc.embedding) == 0)
        ]
        if not to_embed:
            return
        if (
            getattr(self.embedder, "enable_batch", False)
            and hasattr(self.embedder, "async_get_embeddings_batch_and_usage")
        ):
            contents = [doc.content for doc in to_embed]
            embeddings, usages = await self.embedder.async_get_embeddings_batch_and_usage(
                contents
            )
            # See _embed_missing: a None / non-sized batch return means
            # "nothing embedded" — insert()'s missing-embedding check
            # raises the clear "failed to embed" error.
            if embeddings is None or not hasattr(embeddings, "__len__"):
                embeddings = []
            if usages is None or not hasattr(usages, "__len__"):
                usages = []
            for j, doc in enumerate(to_embed):
                if j < len(embeddings):
                    doc.embedding = embeddings[j]
                    doc.usage = usages[j] if j < len(usages) else None
        else:
            # Embedder has no async batch path — fall back to sync.
            self._embed_missing(to_embed)

    def insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not documents:
            return
        if self._index is None:
            # Match LanceDb's "table not initialized" handling: do not
            # silently auto-create. Callers must invoke create() first.
            raise RuntimeError(
                "TurboQuantVectorDb not initialized — call create() before insert()."
            )

        # Merge `filters` into each document's metadata (matches LanceDb).
        if filters:
            for doc in documents:
                meta = dict(doc.meta_data) if doc.meta_data else {}
                meta.update(filters)
                doc.meta_data = meta

        self._embed_missing(documents)

        # Raise on any document that still lacks an embedding rather than
        # silently dropping — silent drops mask data-pipeline bugs.
        # None/len check instead of truthiness: `not <ndarray>` raises the
        # numpy truth-value-ambiguous ValueError (issue #135).
        missing = [
            doc
            for doc in documents
            if doc.embedding is None or len(doc.embedding) == 0
        ]
        if missing:
            ids = [doc.id or "<no id>" for doc in missing]
            raise ValueError(
                f"failed to embed {len(missing)} document(s): {ids}"
            )

        # Batch the entire `documents` list into a single add_with_ids call.
        # Per-document inserts would invalidate the SIMD-blocked cache
        # between every doc.
        vectors = np.asarray([doc.embedding for doc in documents], dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"expected 2D embedding batch, got {vectors.ndim}D"
            )
        if vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"embedding dim {vectors.shape[1]} does not match "
                f"index dim {self.dimensions}"
            )
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)
        # Cosine mode: L2-normalize outside the lock (pure computation,
        # like the embedding step) so the kernel's raw score is true
        # cosine similarity. Zero rows pass through unchanged.
        if self.distance == Distance.cosine:
            vectors = l2_normalize_rows(vectors)

        # Build every side-car payload BEFORE mutating any state (pure
        # computation, no store reads).
        prepared = []
        for doc in documents:
            cleaned = doc.content.replace("\x00", "�") if doc.content else ""
            doc_id = self._derive_doc_id(doc, content_hash, cleaned)
            prepared.append(
                (
                    doc_id,
                    doc.name,
                    {
                        "id": doc_id,
                        "name": doc.name,
                        "content": cleaned,
                        "meta_data": dict(doc.meta_data) if doc.meta_data else {},
                        "usage": doc.usage,
                        "content_id": doc.content_id,
                        "content_hash": content_hash,
                    },
                )
            )

        with self._write_lock:
            # Re-check under the lock: a concurrent drop() may have nulled
            # the index between the check at the top and here.
            if self._index is None:
                raise RuntimeError(
                    "TurboQuantVectorDb not initialized — call create() before insert()."
                )
            handles = np.array(
                [self._issue_handle() for _ in documents], dtype=np.uint64
            )

            # Maps BEFORE the index add: a concurrent search can only learn
            # a handle from the index, so an entry that is resolvable but
            # not yet searchable is invisible to readers (safe) — the
            # reverse ordering let readers surface handles that did not
            # resolve yet (issue #161).
            for (doc_id, name, payload), handle in zip(prepared, handles):
                h = int(handle)
                self._str_to_u64.setdefault(doc_id, set()).add(h)
                self._u64_to_doc[h] = payload
                self._content_hashes.add(content_hash)
                if name:
                    self._name_to_ids.setdefault(name, set()).add(doc_id)
            try:
                self._index.add_with_ids(vectors, handles)
            except BaseException:
                # Unwind the pre-inserted entries so a failed add leaves
                # the store exactly as it was — preserving the issue-#89
                # guarantee under the maps-first ordering. `_unlink_payload`
                # drops the side-index entries only where no surviving
                # handle still needs them, so pre-existing docs sharing an
                # id / name / content_hash keep theirs.
                for (doc_id, _name, _payload), handle in zip(prepared, handles):
                    h = int(handle)
                    data = self._u64_to_doc.pop(h, None)
                    if data is not None:
                        self._unlink_payload(h, data)
                raise

    async def async_insert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not documents:
            return
        await self._embed_missing_async(documents)
        # Now every doc should have an embedding; insert delegates to sync.
        self.insert(content_hash, documents, filters)

    def upsert_available(self) -> bool:
        return True

    def upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Match LanceDb's semantic: replace all documents previously
        # stored under this content_hash with the incoming batch. Not
        # "replace by derived doc_id" — that's a different contract.
        #
        # Embed up front so the embedder (a user component) runs outside
        # the writer lock; insert() then finds nothing left to embed.
        if documents:
            self._embed_missing(documents)
        # Capture the existing generation's handles, run the insert, and
        # only then drop the old vectors — so a failed insert (dim
        # mismatch, non-finite embeddings) never destroys the data being
        # replaced (issue #89). We delete by captured handle rather than
        # re-querying by content_hash, because insert() re-derives ids
        # under the SAME content_hash and would otherwise clobber the
        # just-inserted rows. The lock spans capture + insert + remove so
        # concurrent upserts of the same content_hash stay
        # last-writer-wins with no stale generations (issue #146).
        with self._write_lock:
            old_handles = self._handles_for_content_hash(content_hash)
            self.insert(content_hash, documents, filters)
            for handle in old_handles:
                self._remove_handle(handle)

    async def async_upsert(
        self,
        content_hash: str,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Await ONLY the embedding step, then delegate to the sync upsert
        # so no suspension point splits the old-handle capture from the
        # insert+remove. Capturing before an await let concurrent upserts
        # of the same content_hash all snapshot the same pre-insert
        # handle set, so no task removed a sibling's rows and stale
        # generations accumulated (issue #146). Delegating preserves the
        # insert-before-delete ordering (issue #89) and makes concurrent
        # async upserts last-writer-wins — identical to sync semantics.
        await self._embed_missing_async(documents)
        self.upsert(content_hash, documents, filters)

    def _handles_for_content_hash(self, content_hash: str) -> List[int]:
        """Internal handles of every document currently stored under this
        content_hash. Used by upsert to defer removal of the previous
        generation until the replacement add has succeeded (issue #89)."""
        return [
            handle
            for handle, data in self._u64_to_doc.items()
            if data.get("content_hash") == content_hash
        ]

    def _remove_handle(self, handle: int) -> None:
        """Remove a single vector by its internal handle, leaving other
        handles intact — including ones that share this document's derived
        id (two distinct documents can map to the same doc_id, matching
        LanceDb). Cleans the id, name, and content_hash side-indexes only
        where no surviving handle still needs them.

        Callers hold the writer lock. Index first: the handle stops being
        searchable before it stops resolving, so a concurrent search can
        never surface a handle whose side-car entry is already gone."""
        if self._index is None:
            return
        data = self._u64_to_doc.get(handle)
        if data is None:
            return
        self._index.remove(handle)
        self._u64_to_doc.pop(handle, None)
        self._unlink_payload(handle, data)

    def _unlink_payload(self, handle: int, data: Dict[str, Any]) -> None:
        """Drop ``handle``'s entries from the id / name / content_hash
        side-indexes, keeping any entry a surviving handle still needs.
        ``data`` is the payload already popped from ``_u64_to_doc``."""
        doc_id = data.get("id")
        # Drop just this handle from the id's handle set; remove the id
        # entirely only once no handle remains under it.
        if doc_id is not None:
            handles = self._str_to_u64.get(doc_id)
            if handles is not None:
                handles.discard(handle)
                if not handles:
                    del self._str_to_u64[doc_id]
        # Drop the name->id link only if no surviving handle keeps that
        # (name, id) pair. The derived doc_id excludes `name`, so two docs
        # with different names can share an id — matching on id alone would
        # leave a stale name entry when the last handle for this name goes.
        name = data.get("name")
        if name and name in self._name_to_ids:
            if not any(
                d.get("id") == doc_id and d.get("name") == name
                for d in self._u64_to_doc.values()
            ):
                self._name_to_ids[name].discard(doc_id)
                if not self._name_to_ids[name]:
                    del self._name_to_ids[name]
        # Drop the content_hash only if no surviving doc carries it.
        ch = data.get("content_hash")
        if ch and not any(
            d.get("content_hash") == ch for d in self._u64_to_doc.values()
        ):
            self._content_hashes.discard(ch)

    # ---- VectorDb protocol: search ----------------------------------------

    @staticmethod
    def _meta_matches(data: Dict[str, Any], items: List[Any]) -> bool:
        """True iff the stored payload's ``meta_data`` has every filter key
        PRESENT and equal to the filter value (AND). Key presence matters:
        ``.get(k) == v`` coerces absence to ``None``, so a ``None``-valued
        filter would match every doc *missing* the key — leaking (or
        deleting) the whole collection (issue #144). LanceDb requires the
        key to be present, and skips docs whose ``meta_data`` is ``None``."""
        md = data.get("meta_data")
        if md is None:
            return False
        return all(k in md and md[k] == v for k, v in items)

    def _resolve_filter_to_handles(
        self, filters: Optional[Union[Dict[str, Any], List[Any]]]
    ) -> Optional[List[int]]:
        """Convert a dict filter into the list of internal u64 handles
        whose document's ``meta_data`` matches every key/value pair (AND).
        Returns ``None`` when no filter was supplied — caller should run
        an unfiltered search. Returns ``[]`` to mean "no matches".

        Matches LanceDb's dict-filter semantics (exact equality, AND of
        keys). ``FilterExpr``-style list filters are not yet supported
        in LanceDb itself, so we silently ignore them here too with a
        debug log.
        """
        if filters is None:
            return None
        if isinstance(filters, list):
            # LanceDb logs a warning and ignores. Mirror that — the
            # alternative is to error and break callers that pass an
            # accidental list.
            return None
        if not isinstance(filters, dict) or not filters:
            return None
        items = list(filters.items())
        # list() snapshot: this runs on the (lock-free) search path, so a
        # concurrent writer must not invalidate the iteration mid-scan.
        return [
            handle
            for handle, data in list(self._u64_to_doc.items())
            if self._meta_matches(data, items)
        ]

    def _scaled_similarity(self, raw: float) -> float:
        """Map the kernel's raw score to ``[0, 1]`` via ``(raw + 1) / 2``.

        Under the default ``Distance.cosine`` mode both sides are unit
        vectors, so ``raw`` is true cosine similarity in ``[-1, 1]`` and
        the clamp only absorbs quantization noise — the value compares
        meaningfully against ``similarity_threshold``. Under
        ``Distance.max_inner_product`` ``raw`` is an unbounded inner
        product: values outside ``[-1, 1]`` saturate at 0/1, so
        thresholds are dataset-relative there (see the class docstring).
        """
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    @staticmethod
    def _dedup_by_content(results: List[Document]) -> List[Document]:
        """Collapse duplicate-content hits, keeping the first occurrence.

        Matches LanceDb.search's final step: after filtering and rerank,
        the result list is deduplicated by ``md5(doc.content)``. LanceDb
        fetches ``limit`` results and *then* dedups, so callers can
        receive fewer than ``limit`` documents when duplicates exist —
        we deliberately replicate that rather than over-fetching to
        refill the result set (issue #136)."""
        seen_hashes: Set[str] = set()
        unique_results: List[Document] = []
        for doc in results:
            doc_hash = md5(doc.content.encode()).hexdigest()
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_results.append(doc)
        return unique_results

    def _build_results(
        self, scores: np.ndarray, handles: np.ndarray
    ) -> List[Document]:
        results: List[Document] = []
        threshold = self.similarity_threshold
        for raw_score, handle in zip(scores[0], handles[0]):
            doc_data = self._u64_to_doc.get(int(handle))
            if doc_data is None:
                continue
            similarity = self._scaled_similarity(float(raw_score))
            if threshold is not None and similarity < threshold:
                continue
            results.append(
                Document(
                    id=doc_data["id"],
                    name=doc_data.get("name"),
                    content=doc_data.get("content", ""),
                    meta_data=dict(doc_data.get("meta_data") or {}),
                    usage=doc_data.get("usage"),
                    content_id=doc_data.get("content_id"),
                    # Match LanceDb._build_search_results: thread the
                    # store's embedder through so downstream code can call
                    # `doc.embed()` / `doc.async_embed()` on a retrieved
                    # hit without explicitly passing the embedder back in.
                    # Without this, `doc.embed()` raises
                    # "No embedder provided".
                    embedder=self.embedder,
                )
            )
        return results

    def _retrieve(
        self,
        qvec: np.ndarray,
        limit: int,
        filters: Optional[Union[Dict[str, Any], List[Any]]],
    ) -> List[Document]:
        """Kernel search + tolerant result construction, shared by
        :meth:`search` and :meth:`async_search`. Runs lock-free."""
        # Local reference: a concurrent drop() nulls self._index, and a
        # lock-free reader must not trip over that mid-search.
        index = self._index
        if index is None:
            return []
        # Cosine mode: normalize the query so the kernel's raw score
        # against unit document vectors is true cosine similarity.
        if self.distance == Distance.cosine:
            qvec = l2_normalize_rows(qvec)
        allowed_handles = self._resolve_filter_to_handles(filters)
        if allowed_handles is None:
            # Unfiltered.
            k = min(limit, len(index))
            scores, handles = index.search(qvec, k)
            return self._build_results(scores, handles)

        for _attempt in range(8):
            if not allowed_handles:
                return []
            allowlist = np.asarray(allowed_handles, dtype=np.uint64)
            try:
                scores, handles = index.search(qvec, limit, allowlist=allowlist)
                return self._build_results(scores, handles)
            except KeyError:
                # The allowlist went stale: a delete landed between
                # resolving it and the kernel's membership check. Rebuild
                # the allowlist and retry.
                allowed_handles = self._resolve_filter_to_handles(filters)
                continue

        # Sustained churn kept invalidating the allowlist. Fall back to an
        # unfiltered search plus a tolerant Python-side post-filter, which
        # cannot raise (a search overlapping heavy churn may return fewer
        # than `limit` hits).
        items = list(filters.items())
        fetch_k = min(max(limit * 4, 32), len(index) or 1)
        scores, handles = index.search(qvec, fetch_k)
        keep_scores: List[float] = []
        keep_handles: List[int] = []
        for score, handle in zip(scores[0], handles[0]):
            data = self._u64_to_doc.get(int(handle))
            if data is None or not self._meta_matches(data, items):
                continue
            keep_scores.append(float(score))
            keep_handles.append(int(handle))
            if len(keep_handles) >= limit:
                break
        return self._build_results(
            np.asarray([keep_scores], dtype=np.float32),
            np.asarray([keep_handles], dtype=np.uint64),
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> List[Document]:
        # An empty query string usually indicates an upstream bug
        # (uninitialised variable, failed prompt construction). LanceDb
        # short-circuits this to [] rather than searching with a hash-
        # derived embedding of "", which would return arbitrary garbage.
        if not query:
            return []
        if self._index is None or len(self._index) == 0:
            return []

        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            return []
        qvec = np.asarray(query_embedding, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)

        results = self._retrieve(qvec, limit, filters)
        if self.reranker is not None and results:
            results = self.reranker.rerank(query=query, documents=results)
        # Dedup by content as the final step, after rerank — matching
        # LanceDb.search's ordering exactly (issue #136).
        return self._dedup_by_content(results)

    async def async_search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ) -> List[Document]:
        if not query:
            return []
        if self._index is None or len(self._index) == 0:
            return []

        if hasattr(self.embedder, "async_get_embedding"):
            query_embedding = await self.embedder.async_get_embedding(query)
        else:
            query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            return []
        qvec = np.asarray(query_embedding, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)

        results = self._retrieve(qvec, limit, filters)
        if self.reranker is not None and results:
            results = self.reranker.rerank(query=query, documents=results)
        # Dedup by content as the final step, after rerank — matching
        # LanceDb.search's ordering exactly (issue #136).
        return self._dedup_by_content(results)

    def get_supported_search_types(self) -> List[SearchType]:
        # Only vector. Keyword and hybrid would require an external BM25
        # / lexical index that turbovec doesn't ship. Return shape
        # mirrors LanceDb: a list of SearchType enum members (not their
        # `.value` strings).
        return [SearchType.vector]

    # ---- VectorDb protocol: delete ----------------------------------------

    def delete_by_id(self, id: str) -> bool:
        if self._index is None:
            return False
        with self._write_lock:
            handles = self._str_to_u64.get(id)
            if not handles:
                return False
            # Remove every vector sharing this id — a non-unique derived doc_id
            # can map to several handles. _remove_handle maintains the id, name,
            # and content_hash side-indexes per handle.
            for handle in list(handles):
                self._remove_handle(handle)
            return True

    def delete_by_name(self, name: str) -> bool:
        if self._index is None:
            return False
        # Docs stored without a name keep it as None, so a None argument
        # (off-contract, param typed str) would delete every unnamed doc.
        # Same guard as delete_by_content_id — no-op instead.
        if name is None:
            return False
        # Remove exactly the handles whose stored name matches. Delegating to
        # delete_by_id would key on the derived doc_id, which excludes `name`,
        # so it would also delete a differently-named doc that happens to
        # share the id. LanceDb deletes rows matching the predicate directly.
        with self._write_lock:
            handles = [h for h, d in self._u64_to_doc.items() if d.get("name") == name]
            for handle in handles:
                self._remove_handle(handle)
            return bool(handles)

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Delete every document whose ``meta_data`` has all of
        ``metadata``'s keys present and equal (AND). An empty dict is a
        vacuous AND and therefore matches — and deletes — every document,
        matching LanceDb's behavior for the same input."""
        if self._index is None:
            return False
        # None is off-contract (param typed dict); LanceDb returns False
        # rather than raising. Same guard family as delete_by_name /
        # delete_by_content_id.
        if metadata is None:
            return False
        items = list(metadata.items())
        # Remove the matching handles directly (see delete_by_name): the
        # derived doc_id ignores metadata, so delete_by_id would over-delete
        # distinct docs that collide on the id.
        with self._write_lock:
            handles = [
                h
                for h, data in self._u64_to_doc.items()
                if self._meta_matches(data, items)
            ]
            for handle in handles:
                self._remove_handle(handle)
            return bool(handles)

    def delete_by_content_id(self, content_id: str) -> bool:
        if self._index is None:
            return False
        # Docs stored without a content_id keep it as None, so a None
        # argument (off-contract, param typed str) would match — and
        # delete — every such doc. Treat it as a no-op (issue #169).
        if content_id is None:
            return False
        # Remove the matching handles directly (see delete_by_name): the
        # derived doc_id ignores content_id, so delete_by_id would over-delete
        # distinct docs that collide on the id.
        with self._write_lock:
            handles = [
                h
                for h, data in self._u64_to_doc.items()
                if data.get("content_id") == content_id
            ]
            for handle in handles:
                self._remove_handle(handle)
            return bool(handles)

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
        """Merge ``metadata`` into both ``meta_data`` and the ``filters``
        payload field of every document whose ``content_id`` matches.
        Mirrors LanceDb's update_metadata semantic which writes to both
        fields (used by callers that pass filter-style restrictions at
        retrieval time)."""
        if self._index is None:
            return
        # See delete_by_content_id: a None content_id would match every
        # doc stored without one — no-op instead (issue #169).
        if content_id is None:
            return
        with self._write_lock:
            for data in self._u64_to_doc.values():
                if data.get("content_id") == content_id:
                    meta = dict(data.get("meta_data") or {})
                    meta.update(metadata)
                    data["meta_data"] = meta
                    filters = data.get("filters")
                    if isinstance(filters, dict):
                        filters = dict(filters)
                        filters.update(metadata)
                        data["filters"] = filters
                    else:
                        data["filters"] = dict(metadata)

    # ---- Persistence (JSON side-car) --------------------------------------

    def save(self, folder_path: Optional[str] = None) -> None:
        """Persist the quantized index plus a JSON side-car to disk. Pass
        ``folder_path`` to override the constructor's ``path=``.

        Writes two files under ``folder_path``:
          - ``index.tvim`` — the :class:`IdMapIndex` payload.
          - ``docstore.json`` — JSON-encoded document text, metadata, and
            id maps. Side-car carries a ``schema_version`` field; loaders
            reject unknown versions rather than silently misinterpreting
            bytes.
        """
        path = folder_path if folder_path is not None else self.path
        if path is None:
            raise ValueError(
                "No path to save to. Pass `folder_path=` here or set "
                "`path=` on the constructor."
            )
        if self._index is None:
            raise RuntimeError(
                "TurboQuantVectorDb has no index to save — call create() first."
            )
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        # Serializes with writers: snapshotting the maps and the index
        # concurrently with a write would persist a torn store to disk.
        # Reads may proceed while a save runs.
        with self._write_lock:
            payload = {
                "schema_version": _DOCSTORE_SCHEMA_VERSION,
                # Round-trip int handles via list-of-pairs (JSON keys must be
                # strings, but our handles are ints).
                "u64_to_doc": [[h, d] for h, d in self._u64_to_doc.items()],
                "next_u64": self._next_u64,
                "bit_width": self.bit_width,
                "dimensions": self.dimensions,
                # Recorded so a later create()/load restores the mode the
                # vectors were written under (v2+).
                "distance": self.distance.value,
            }
            # Atomic: serializes in memory first, then temp-file + replace,
            # so a failed save can't destroy a previous store at this path.
            atomic_save(
                self._index,
                folder / _INDEX_FILENAME,
                payload,
                folder / _STORE_FILENAME,
            )

    def _load_from(self, folder: Path) -> None:
        side_car = folder / _STORE_FILENAME
        index_file = folder / _INDEX_FILENAME
        if not side_car.exists() or not index_file.exists():
            raise FileNotFoundError(
                f"missing one of {_STORE_FILENAME}/{_INDEX_FILENAME} under {folder}"
            )
        with open(side_car) as f:
            state = json.load(f)
        version = state.get("schema_version", 0)
        if version not in _DOCSTORE_SCHEMA_COMPAT:
            raise ValueError(
                f"{_STORE_FILENAME} has schema_version {version}; this "
                f"turbovec accepts versions {list(_DOCSTORE_SCHEMA_COMPAT)}"
            )
        if state.get("dimensions") != self.dimensions:
            raise ValueError(
                f"persisted dimensions={state.get('dimensions')} does not "
                f"match this store's embedder dimensions={self.dimensions}"
            )
        # Similarity mode of the persisted vectors. The construct-then-
        # create() API shape means the store already has a mode when the
        # load runs, so a *recorded* mode that conflicts with it is an
        # error — silently adopting either side would surprise someone.
        # A v1 side-car (written before the mode existed) records no
        # mode but holds raw, unnormalized vectors: adopt
        # Distance.max_inner_product — the scoring those stores were
        # written under — and update `self.distance` so introspection
        # reflects how the store actually scores.
        recorded = state.get("distance")
        if recorded is not None:
            if recorded != self.distance.value:
                raise ValueError(
                    f"persisted store at {folder} was saved with "
                    f"distance={recorded!r}, but this store was constructed "
                    f"with distance={self.distance.value!r}. Construct the "
                    f"store with the matching distance to load it."
                )
        else:
            self.distance = Distance.max_inner_product

        self._index = IdMapIndex.load(str(index_file))
        self._u64_to_doc = {int(h): d for h, d in state["u64_to_doc"]}
        self._next_u64 = int(state["next_u64"])

        # Rebuild reverse indexes from the loaded payload. doc_id is
        # non-unique, so accumulate handles into a set per id rather than a
        # dict comprehension (which would drop all but the last handle and
        # re-orphan the very vectors issue #104 fixed).
        self._str_to_u64 = {}
        for handle, data in self._u64_to_doc.items():
            self._str_to_u64.setdefault(data["id"], set()).add(handle)
        self._content_hashes = set()
        self._name_to_ids = {}
        for data in self._u64_to_doc.values():
            ch = data.get("content_hash")
            if ch:
                self._content_hashes.add(ch)
            name = data.get("name")
            if name:
                self._name_to_ids.setdefault(name, set()).add(data["id"])

        # Reject a side-car whose handle set desynced from the .tvim index
        # (partial copy, stale backup, hand edit) with a clean ValueError
        # here rather than misbehaving later — matching the other
        # integrations' load paths (issue #115).
        check_persisted_handles(self._index, self._u64_to_doc.keys(), what="document")


__all__ = ["TurboQuantVectorDb"]
