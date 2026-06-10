"""LangChain VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[langchain]``.

The public surface mirrors langchain_core's in-tree ``InMemoryVectorStore``
so this store can be swapped in wherever the in-memory store is used.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ._dedup import DuplicatePolicy, resolve_duplicates
from ._persist import check_persisted_handles
from ._turbovec import IdMapIndex

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
# Bump when the docstore.json shape changes; loader refuses to deserialize
# unknown versions.
_DOCSTORE_SCHEMA_VERSION = 1


class TurboQuantVectorStore(VectorStore):
    """LangChain VectorStore backed by a :class:`IdMapIndex`.

    Vectors are quantized to 2–4 bits per dimension. A side-car dictionary
    holds the original text and metadata keyed by document id. Deletion
    is supported in O(1) per id via the underlying :class:`IdMapIndex`.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index: IdMapIndex | None = None,
        *,
        bit_width: int = 4,
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
        :param bit_width: Quantization width (2 or 4) used when the index
            is created from scratch. Ignored if ``index`` is supplied.
        """
        self._embedding = embedding
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

    def _issue_handle(self) -> int:
        self._next_u64 += 1
        return self._next_u64

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ---- Relevance score normalization --------------------------------

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # turbovec returns the raw inner product of unit-normalized vectors —
        # ideally cosine similarity in [-1, 1]. Quantization noise can
        # push that very slightly outside the bounds, so clamp after
        # mapping to LangChain's [0, 1] relevance scale via (sim + 1) / 2.
        return lambda sim: max(0.0, min(1.0, (sim + 1.0) / 2.0))

    # ---- Write path ---------------------------------------------------

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
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        if len(metadatas) != len(texts_list) or len(ids) != len(texts_list):
            raise ValueError("texts, metadatas, and ids must all have the same length")

        vectors = np.asarray(self._embedding.embed_documents(texts_list), dtype=np.float32)
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
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        if len(metadatas) != len(texts_list) or len(ids) != len(texts_list):
            raise ValueError("texts, metadatas, and ids must all have the same length")

        vectors = np.asarray(
            await self._embedding.aembed_documents(texts_list), dtype=np.float32
        )
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
        # Add first; if encoding rejects the batch (e.g. non-finite values)
        # this raises before any existing data is touched. Only once the add
        # has succeeded do we remove the old vectors for colliding ids, so a
        # failed upsert never destroys existing data (issue #89). Handles are
        # freshly issued, so the old and new vectors coexist until the delete.
        self._index.add_with_ids(vectors, handles)

        # Upsert: any id that already existed is removed so the re-added
        # vector wins. Matches LangChain user expectation that `add_texts`
        # with an existing id updates in place.
        duplicates = [i for i in ids if i in self._str_to_u64]
        if duplicates:
            self.delete(duplicates)

        for id_, text, meta, handle in zip(ids, texts_list, metadatas, handles):
            h = int(handle)
            self._str_to_u64[id_] = h
            self._u64_to_str[h] = id_
            self._docs[id_] = (text, dict(meta))
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
        qvec = np.asarray(self._embedding.embed_query(query), dtype=np.float32)
        return self._search_vector(qvec, k, filter=filter)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | Callable[[Document], bool] | None = None,
        **_: Any,
    ) -> list[tuple[Document, float]]:
        qvec = np.asarray(
            await self._embedding.aembed_query(query), dtype=np.float32
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
            allowed_handles = [
                self._str_to_u64[sid]
                for sid, (text, meta) in self._docs.items()
                if predicate(Document(id=sid, page_content=text, metadata=dict(meta)))
            ]
            if not allowed_handles:
                return []
            allowlist = np.asarray(allowed_handles, dtype=np.uint64)
            scores, handles = self._index.search(qvec, k, allowlist=allowlist)

        results: list[tuple[Document, float]] = []
        for score, handle in zip(scores[0], handles[0]):
            sid = self._u64_to_str[int(handle)]
            text, meta = self._docs[sid]
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
            if sid in self._docs:
                text, meta = self._docs[sid]
                out.append(Document(id=sid, page_content=text, metadata=dict(meta)))
        return out

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return self.get_by_ids(ids)

    def delete(self, ids: list[str] | None = None, **_: Any) -> None:
        """Remove documents by id. Missing ids are silently skipped — matches
        the InMemoryVectorStore reference (which also accepts ``ids=None``
        as a no-op)."""
        if not ids:
            return
        for sid in ids:
            handle = self._str_to_u64.pop(sid, None)
            if handle is None:
                continue
            self._u64_to_str.pop(handle, None)
            self._docs.pop(sid, None)
            self._index.remove(handle)

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
        ids: list[str] | None = None,
        **_: Any,
    ) -> "TurboQuantVectorStore":
        # The underlying index is created lazily on the first `add_texts`
        # call, picking up `dim` from the first batch of embeddings — same
        # no-`dim` ergonomics as InMemoryVectorStore.
        store = cls(embedding=embedding, bit_width=bit_width)
        if texts:
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
        ids: list[str] | None = None,
        **_: Any,
    ) -> "TurboQuantVectorStore":
        store = cls(embedding=embedding, bit_width=bit_width)
        if texts:
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
        self._index.write(str(folder / _INDEX_FILENAME))
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
            "str_to_u64": self._str_to_u64,
            "next_u64": self._next_u64,
            # Pull bit_width off the live index — same value whether
            # the index was constructed eagerly or lazily.
            "bit_width": self._index.bit_width,
        }
        with open(folder / _STORE_FILENAME, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(
        cls,
        folder_path: str | Path,
        embedding: Embeddings,
    ) -> "TurboQuantVectorStore":
        """Reload a store from a folder previously written by :meth:`dump`.
        Safe to call on any path — the side-car is plain JSON, never
        pickle, so there's no deserialization-of-code risk."""
        folder = Path(folder_path)
        with open(folder / _STORE_FILENAME) as f:
            state = json.load(f)
        version = state.get("schema_version", 0)
        if version != _DOCSTORE_SCHEMA_VERSION:
            raise ValueError(
                f"docstore.json has schema version {version}; "
                f"this turbovec expects version {_DOCSTORE_SCHEMA_VERSION}"
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
        return cls(
            embedding=embedding,
            index=index,
            bit_width=state.get("bit_width", 4),
            docs=docs,
            str_to_u64=str_to_u64,
            next_u64=int(state["next_u64"]),
        )


__all__ = ["TurboQuantVectorStore"]
