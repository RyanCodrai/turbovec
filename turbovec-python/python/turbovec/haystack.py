"""Haystack DocumentStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[haystack]``.

Implements the Haystack 2.x ``DocumentStore`` protocol and matches the
public surface of ``InMemoryDocumentStore`` so this store can be swapped
in wherever the in-memory store is used. The quantized index discards
full-precision embeddings after compression — callers that rely on
``Document.embedding`` after retrieval will see ``None``.
"""

from __future__ import annotations

import asyncio
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from ._turbovec import IdMapIndex

try:
    from haystack import Document
    from haystack.document_stores.errors import DuplicateDocumentError
    from haystack.document_stores.types import DuplicatePolicy
    from haystack.utils.filters import document_matches_filter
except ImportError as exc:
    raise ImportError(
        "haystack-ai is required to use turbovec.haystack. "
        "Install with: pip install turbovec[haystack]"
    ) from exc


class TurboQuantDocumentStore:
    """Haystack DocumentStore backed by a :class:`~turbovec.IdMapIndex`.

    Vectors are quantized to 2–4 bits per dimension. Full-precision
    embeddings are dropped after quantization — callers requesting
    ``return_embedding=True`` on retrieval will see ``None`` on the
    returned documents' ``embedding`` field regardless of the flag.

    Example::

        from turbovec.haystack import TurboQuantDocumentStore
        from haystack import Document

        store = TurboQuantDocumentStore(dim=1536, bit_width=4)
        store.write_documents([
            Document(content="...", embedding=[...], meta={"source": "a"}),
            ...
        ])
        results = store.embedding_retrieval(query_embedding=[...], top_k=5)
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        bit_width: int = 4,
        *,
        embedding_similarity_function: Literal["dot_product", "cosine"] = "cosine",
        async_executor: Optional[ThreadPoolExecutor] = None,
        return_embedding: bool = False,
    ) -> None:
        """
        :param dim: Vector dimensionality. When omitted, the underlying
            quantized index is created lazily by ``IdMapIndex`` itself on
            the first ``write_documents`` call — matches the no-``dim``
            ergonomics of ``InMemoryDocumentStore``.
        :param bit_width: Quantization width per coordinate (2 or 4).
        :param embedding_similarity_function: ``"cosine"`` (default) or
            ``"dot_product"``. Used to choose the ``scale_score`` formula
            during retrieval. Defaults to ``"cosine"`` because turbovec
            stores unit-normalized vectors.
        :param async_executor: Optional executor for the ``*_async``
            methods. If omitted, a single-threaded executor is created
            and cleaned up on instance destruction.
        :param return_embedding: Whether retrieval methods should leave
            the ``embedding`` field populated on returned Documents.
            turbovec never has the full-precision embedding available, so
            this is always ``None`` either way; the flag is accepted for
            API parity with ``InMemoryDocumentStore``.
        """
        self._bit_width = bit_width
        self.embedding_similarity_function = embedding_similarity_function
        self.return_embedding = return_embedding
        # IdMapIndex itself supports lazy construction — pass dim through
        # and let it handle eager vs lazy. No per-store lazy wrapping.
        self._index = IdMapIndex(dim, bit_width)
        # Haystack doc_id (str) -> u64 handle
        self._str_to_u64: Dict[str, int] = {}
        # u64 handle -> stored doc data {id, content, meta}
        self._u64_to_doc: Dict[int, Dict[str, Any]] = {}
        # Counter for assigning u64 handles. Starts at 0; each new
        # handle is `_next_u64 + 1`, then we bump. Plain int so pickle
        # can round-trip it directly.
        self._next_u64: int = 0

        # Executor lifecycle mirrors InMemoryDocumentStore: own one when
        # the caller didn't pass one in, and shut it down in __del__.
        self._owns_executor = async_executor is None
        self.executor = async_executor or ThreadPoolExecutor(
            thread_name_prefix=f"async-turbovec-docstore-executor-{id(self)}",
            max_workers=1,
        )

    def __del__(self) -> None:
        if (
            hasattr(self, "_owns_executor")
            and self._owns_executor
            and hasattr(self, "executor")
        ):
            self.executor.shutdown(wait=True)

    def shutdown(self) -> None:
        """Explicitly shut down the async executor if this store owns it."""
        if self._owns_executor:
            self.executor.shutdown(wait=True)

    def _issue_handle(self) -> int:
        self._next_u64 += 1
        return self._next_u64

    @property
    def storage(self) -> Dict[str, Document]:
        """Map of ``doc_id -> Document`` for the currently stored documents.

        Documents are reconstructed on every access; the
        ``embedding`` field is always ``None``.
        """
        return {data["id"]: self._reconstruct(data) for data in self._u64_to_doc.values()}

    # ---- DocumentStore protocol ---------------------------------------

    def count_documents(self) -> int:
        return len(self._str_to_u64)

    def filter_documents(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        if filters:
            self._validate_filters(filters)
            docs = [
                self._reconstruct(data)
                for data in self._u64_to_doc.values()
                if document_matches_filter(filters=filters, document=self._reconstruct(data))
            ]
        else:
            docs = [self._reconstruct(data) for data in self._u64_to_doc.values()]
        # `return_embedding` is informational here — we never have the
        # full-precision embedding to begin with. Kept for parity.
        return docs

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        # Match InMemoryDocumentStore's input-shape validation rather
        # than letting a bad input AttributeError on `.embedding`.
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            raise ValueError("Please provide a list of Documents.")

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        # First pass: validate and resolve duplicates according to policy.
        to_write: List[Document] = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(
                    f"Document {doc.id!r} has no embedding. "
                    "TurboQuantDocumentStore only stores documents with precomputed "
                    "embeddings — run an embedder component before writing."
                )
            if doc.id in self._str_to_u64:
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(
                        f"ID '{doc.id}' already exists in the document store."
                    )
                if policy == DuplicatePolicy.SKIP:
                    continue
                if policy == DuplicatePolicy.OVERWRITE:
                    self._remove_one(doc.id)
                # fall through to add
            to_write.append(doc)

        if not to_write:
            return 0

        vectors = np.asarray(
            [doc.embedding for doc in to_write], dtype=np.float32
        )
        if vectors.ndim != 2:
            raise ValueError(
                f"expected 2D embedding batch, got {vectors.ndim}D"
            )
        # IdMapIndex.add_with_ids handles both eager (dim must match) and
        # lazy (locks dim on first call) cases. Surface its mismatch
        # panic as a clean ValueError for parity with previous behaviour.
        existing_dim = self._index.dim
        if existing_dim is not None and vectors.shape[1] != existing_dim:
            raise ValueError(
                f"embedding dim {vectors.shape[1]} does not match store dim {existing_dim}"
            )
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        handles = np.array(
            [self._issue_handle() for _ in to_write], dtype=np.uint64
        )
        self._index.add_with_ids(vectors, handles)

        for doc, handle in zip(to_write, handles):
            h = int(handle)
            self._str_to_u64[doc.id] = h
            self._u64_to_doc[h] = {
                "id": doc.id,
                "content": doc.content,
                "meta": dict(doc.meta),
            }
        return len(to_write)

    def delete_documents(self, document_ids: List[str]) -> None:
        # Haystack's protocol says silently ignore missing ids.
        for doc_id in document_ids:
            self._remove_one(doc_id)

    # ---- Utility methods (InMemoryDocumentStore parity) ---------------

    def delete_all_documents(self) -> None:
        """Delete every document in the store."""
        for doc_id in list(self._str_to_u64.keys()):
            self._remove_one(doc_id)

    def update_by_filter(
        self, filters: Dict[str, Any], meta: Dict[str, Any]
    ) -> int:
        """Update metadata on every document matching ``filters``.

        The new ``meta`` is merged into each matching document's existing
        metadata. Embeddings are not touched — we never had them at full
        precision anyway. Returns the number of documents updated.
        """
        self._validate_filters(filters)
        updated = 0
        for data in self._u64_to_doc.values():
            if document_matches_filter(filters=filters, document=self._reconstruct(data)):
                data["meta"].update(meta)
                updated += 1
        return updated

    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete every document matching ``filters``. Returns the count."""
        self._validate_filters(filters)
        matching_ids = [
            data["id"]
            for data in self._u64_to_doc.values()
            if document_matches_filter(filters=filters, document=self._reconstruct(data))
        ]
        for doc_id in matching_ids:
            self._remove_one(doc_id)
        return len(matching_ids)

    def count_documents_by_filter(self, filters: Dict[str, Any]) -> int:
        if filters:
            self._validate_filters(filters)
            return sum(
                1
                for data in self._u64_to_doc.values()
                if document_matches_filter(filters=filters, document=self._reconstruct(data))
            )
        return self.count_documents()

    def count_unique_metadata_by_filter(
        self, filters: Dict[str, Any], metadata_fields: List[str]
    ) -> Dict[str, int]:
        if filters:
            self._validate_filters(filters)
            docs_meta = [
                data["meta"]
                for data in self._u64_to_doc.values()
                if document_matches_filter(filters=filters, document=self._reconstruct(data))
            ]
        else:
            docs_meta = [data["meta"] for data in self._u64_to_doc.values()]

        result: Dict[str, int] = {}
        for field in metadata_fields:
            key = field.removeprefix("meta.") if field.startswith("meta.") else field
            values = {meta.get(key) for meta in docs_meta if key in meta and meta[key] is not None}
            result[key] = len(values)
        return result

    def get_metadata_fields_info(self) -> Dict[str, Dict[str, str]]:
        type_map: Dict[str, str] = {}
        for data in self._u64_to_doc.values():
            for key, value in data["meta"].items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    type_map[key] = "boolean"
                elif isinstance(value, int):
                    type_map[key] = "int"
                elif isinstance(value, float):
                    type_map[key] = "float"
                else:
                    type_map[key] = "keyword"
        return {k: {"type": v} for k, v in type_map.items()}

    def get_metadata_field_min_max(self, metadata_field: str) -> Dict[str, Any]:
        key = (
            metadata_field.removeprefix("meta.")
            if metadata_field.startswith("meta.")
            else metadata_field
        )
        values = [
            data["meta"][key]
            for data in self._u64_to_doc.values()
            if key in data["meta"]
            and data["meta"][key] is not None
            and isinstance(data["meta"][key], (int, float, str))
        ]
        if not values:
            return {"min": None, "max": None}
        try:
            return {"min": min(values), "max": max(values)}
        except TypeError:
            return {"min": None, "max": None}

    def get_metadata_field_unique_values(
        self, metadata_field: str, search_term: Optional[str] = None
    ) -> Tuple[List[str], int]:
        key = (
            metadata_field.removeprefix("meta.")
            if metadata_field.startswith("meta.")
            else metadata_field
        )
        if search_term:
            docs_data = [
                data
                for data in self._u64_to_doc.values()
                if data["content"] and search_term.lower() in data["content"].lower()
            ]
        else:
            docs_data = list(self._u64_to_doc.values())
        values = sorted(
            {
                str(data["meta"][key])
                for data in docs_data
                if key in data["meta"] and data["meta"][key] is not None
            },
            key=str,
        )
        return values, len(values)

    @staticmethod
    def _validate_filters(filters: Optional[Dict[str, Any]]) -> None:
        if (
            filters
            and "operator" not in filters
            and "conditions" not in filters
            and "field" not in filters
        ):
            raise ValueError(
                "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
            )

    # ---- Retrieval (not in core protocol but expected) ----------------

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        """Return the ``top_k`` documents most similar to ``query_embedding``.

        ``return_embedding=None`` (default) honours the store-level
        ``return_embedding`` set in the constructor. turbovec never has
        the full-precision embedding either way — the parameter is here
        for API parity with ``InMemoryDocumentStore``.

        ``filters`` are resolved to an allowlist before scoring, so the
        kernel never wastes work on non-matching documents and the result
        count is always ``min(top_k, n_matches)`` rather than ``< top_k``
        when the filter is selective.
        """
        # `return_embedding` is accepted but we never have the full
        # embedding to populate; left as-is for signature parity.
        _ = return_embedding  # noqa: F841

        if self.count_documents() == 0:
            return []

        qvec = np.asarray(query_embedding, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        # By this point n_documents > 0, so the index has a committed dim.
        expected_dim = self._index.dim
        if qvec.shape[1] != expected_dim:
            raise ValueError(
                f"query_embedding dim {qvec.shape[1]} does not match store dim {expected_dim}"
            )
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)

        if filters is None:
            fetch_k = min(top_k, self.count_documents())
            scores, handles = self._index.search(qvec, fetch_k)
        else:
            self._validate_filters(filters)
            # Resolve filter → handle allowlist by walking the in-memory
            # doc table once. This is the same O(N) cost as the old
            # post-filter pass, just moved upfront so the kernel can score
            # only matching vectors.
            allowed_handles = [
                handle
                for handle, data in self._u64_to_doc.items()
                if document_matches_filter(filters, self._reconstruct(data))
            ]
            if not allowed_handles:
                return []
            allowlist = np.asarray(allowed_handles, dtype=np.uint64)
            scores, handles = self._index.search(qvec, top_k, allowlist=allowlist)

        out: List[Document] = []
        for score, handle in zip(scores[0], handles[0]):
            data = self._u64_to_doc[int(handle)]
            out.append(self._reconstruct(data, score=float(score), scale_score=scale_score))
        return out

    # ---- Async variants ----------------------------------------------

    async def count_documents_async(self) -> int:
        return self.count_documents()

    async def filter_documents_async(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, lambda: self.filter_documents(filters=filters)
        )

    async def write_documents_async(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, lambda: self.write_documents(documents=documents, policy=policy)
        )

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        await asyncio.get_running_loop().run_in_executor(
            self.executor, lambda: self.delete_documents(document_ids=document_ids)
        )

    async def delete_all_documents_async(self) -> None:
        await asyncio.get_running_loop().run_in_executor(
            self.executor, self.delete_all_documents
        )

    async def update_by_filter_async(
        self, filters: Dict[str, Any], meta: Dict[str, Any]
    ) -> int:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, lambda: self.update_by_filter(filters=filters, meta=meta)
        )

    async def count_documents_by_filter_async(self, filters: Dict[str, Any]) -> int:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, lambda: self.count_documents_by_filter(filters=filters)
        )

    async def count_unique_metadata_by_filter_async(
        self, filters: Dict[str, Any], metadata_fields: List[str]
    ) -> Dict[str, int]:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor,
            lambda: self.count_unique_metadata_by_filter(
                filters=filters, metadata_fields=metadata_fields
            ),
        )

    async def get_metadata_fields_info_async(self) -> Dict[str, Dict[str, str]]:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, self.get_metadata_fields_info
        )

    async def get_metadata_field_min_max_async(
        self, metadata_field: str
    ) -> Dict[str, Any]:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor,
            lambda: self.get_metadata_field_min_max(metadata_field=metadata_field),
        )

    async def get_metadata_field_unique_values_async(
        self, metadata_field: str, search_term: Optional[str] = None
    ) -> Tuple[List[str], int]:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor,
            lambda: self.get_metadata_field_unique_values(
                metadata_field=metadata_field, search_term=search_term
            ),
        )

    async def embedding_retrieval_async(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor,
            lambda: self.embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                scale_score=scale_score,
                return_embedding=return_embedding,
            ),
        )

    # ---- Serialization (Pipeline to_dict / from_dict) -----------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "init_parameters": {
                # `_index.dim` is None on a lazy uncommitted store and an
                # int once an add has locked the dim — both round-trip cleanly.
                "dim": self._index.dim,
                "bit_width": self._bit_width,
                "embedding_similarity_function": self.embedding_similarity_function,
                "return_embedding": self.return_embedding,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurboQuantDocumentStore":
        params = data.get("init_parameters", {})
        return cls(**params)

    # ---- Persistence -------------------------------------------------

    def save_to_disk(self, folder_path: str | Path) -> None:
        """Persist the quantized index plus the Haystack side-car to disk.

        Writes into ``folder_path``:
          - ``index.tvim`` — the :class:`IdMapIndex` payload. On a lazy
            store that has never seen a write the file encodes the
            uncommitted state via a ``dim=0`` sentinel; no special-case
            handling needed here.
          - ``docstore.pkl`` — the str-id ↔ Document mapping and store
            init parameters.
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        self._index.write(str(folder / "index.tvim"))
        with open(folder / "docstore.pkl", "wb") as f:
            pickle.dump(
                {
                    "u64_to_doc": self._u64_to_doc,
                    "next_u64": self._next_u64,
                    "bit_width": self._bit_width,
                    "embedding_similarity_function": self.embedding_similarity_function,
                    "return_embedding": self.return_embedding,
                },
                f,
            )

    @classmethod
    def load_from_disk(
        cls,
        folder_path: str | Path,
        *,
        allow_dangerous_deserialization: bool = False,
    ) -> "TurboQuantDocumentStore":
        if not allow_dangerous_deserialization:
            raise ValueError(
                "load_from_disk uses pickle, which is unsafe with untrusted input. "
                "Pass allow_dangerous_deserialization=True to confirm you "
                "trust the source of folder_path."
            )
        folder = Path(folder_path)
        with open(folder / "docstore.pkl", "rb") as f:
            state = pickle.load(f)
        store = cls(
            bit_width=state["bit_width"],
            embedding_similarity_function=state.get(
                "embedding_similarity_function", "cosine"
            ),
            return_embedding=state.get("return_embedding", False),
        )
        # Reload the index — it carries dim internally (None for lazy
        # uncommitted, int otherwise).
        store._index = IdMapIndex.load(str(folder / "index.tvim"))
        store._u64_to_doc = state["u64_to_doc"]
        store._next_u64 = state["next_u64"]
        # Rebuild str_to_u64 from the reloaded doc table.
        store._str_to_u64 = {
            data["id"]: handle for handle, data in store._u64_to_doc.items()
        }
        return store

    # ---- Internals ----------------------------------------------------

    def _remove_one(self, doc_id: str) -> bool:
        handle = self._str_to_u64.pop(doc_id, None)
        if handle is None:
            return False
        del self._u64_to_doc[handle]
        self._index.remove(handle)
        return True

    def _reconstruct(
        self,
        data: Dict[str, Any],
        score: Optional[float] = None,
        scale_score: bool = False,
    ) -> Document:
        if score is not None and scale_score:
            # Match Haystack's InMemoryDocumentStore._compute_query_embedding_similarity_scores
            # (document_store.py:818-822): different formula per similarity
            # function. turbovec uses unit-normalized vectors so the cosine
            # branch is the natural default.
            if self.embedding_similarity_function == "dot_product":
                score = 1.0 / (1.0 + math.exp(-score / 100.0))
            elif self.embedding_similarity_function == "cosine":
                score = (score + 1.0) / 2.0
        return Document(
            id=data["id"],
            content=data["content"],
            meta=dict(data["meta"]),
            score=score,
        )


__all__ = ["TurboQuantDocumentStore"]
