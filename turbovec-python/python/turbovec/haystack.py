"""Haystack DocumentStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[haystack]``.

Implements the Haystack 2.x ``DocumentStore`` protocol:
``count_documents``, ``filter_documents``, ``write_documents``,
``delete_documents``, plus ``to_dict`` / ``from_dict`` for pipeline
serialization.

Adds ``embedding_retrieval`` with a signature matching
``InMemoryDocumentStore`` so it can back an
``InMemoryEmbeddingRetriever``-style pipeline.

Delete is O(1) via the inner :class:`~turbovec.IdMapIndex`, so this
store can be used in pipelines that mutate their document set over
time — unlike the LangChain / LlamaIndex integrations that require
rebuilding.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    returned documents' ``embedding`` field.

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

    def __init__(self, dim: int, bit_width: int = 4) -> None:
        self._dim = dim
        self._bit_width = bit_width
        self._index = IdMapIndex(dim, bit_width)
        # Haystack doc_id (str) -> u64 handle
        self._str_to_u64: Dict[str, int] = {}
        # u64 handle -> stored doc data {id, content, meta}
        self._u64_to_doc: Dict[int, Dict[str, Any]] = {}
        # Counter for assigning u64 handles. Starts at 0; each new
        # handle is `_next_u64 + 1`, then we bump. Plain int so pickle
        # can round-trip it directly.
        self._next_u64: int = 0

    def _issue_handle(self) -> int:
        self._next_u64 += 1
        return self._next_u64

    # ---- DocumentStore protocol ---------------------------------------

    def count_documents(self) -> int:
        return len(self._str_to_u64)

    def filter_documents(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        docs = [self._reconstruct(data) for data in self._u64_to_doc.values()]
        if filters is None:
            return docs
        return [doc for doc in docs if document_matches_filter(filters, doc)]

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
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
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"embedding dim {vectors.shape[1]} does not match store dim {self._dim}"
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

    # ---- Retrieval (not in core protocol but expected) ----------------

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
    ) -> List[Document]:
        """Return the ``top_k`` documents most similar to ``query_embedding``.

        ``return_embedding`` is accepted for signature compatibility but
        always returns ``None`` on the ``embedding`` field — full-precision
        embeddings are discarded after quantization.

        ``filters`` are applied post-retrieval, so results may be fewer
        than ``top_k`` when filtering is restrictive.
        """
        if return_embedding:
            # Signature-compatible — but we warn once, could cause regressions
            # in callers that expect embeddings. We keep silent rather than
            # raising so pipelines run.
            pass

        if self.count_documents() == 0:
            return []

        qvec = np.asarray(query_embedding, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        if qvec.shape[1] != self._dim:
            raise ValueError(
                f"query_embedding dim {qvec.shape[1]} does not match store dim {self._dim}"
            )
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)

        # Over-fetch if we're going to post-filter so we still have
        # ~top_k results after dropping non-matches.
        fetch_k = top_k if filters is None else min(top_k * 10, self.count_documents())
        fetch_k = min(fetch_k, self.count_documents())
        scores, handles = self._index.search(qvec, fetch_k)

        out: List[Document] = []
        for score, handle in zip(scores[0], handles[0]):
            data = self._u64_to_doc[int(handle)]
            doc = self._reconstruct(data, score=float(score), scale_score=scale_score)
            if filters is None or document_matches_filter(filters, doc):
                out.append(doc)
                if len(out) >= top_k:
                    break
        return out

    # ---- Serialization (Pipeline to_dict / from_dict) -----------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "init_parameters": {
                "dim": self._dim,
                "bit_width": self._bit_width,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurboQuantDocumentStore":
        params = data.get("init_parameters", {})
        return cls(**params)

    # ---- Persistence -------------------------------------------------

    def save(self, folder_path: str | Path) -> None:
        """Persist the quantized index plus the Haystack side-car to disk.

        Writes two files into ``folder_path``:
          - ``index.tvim`` — the :class:`IdMapIndex` payload.
          - ``docstore.pkl`` — the str-id ↔ Document mapping.
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        self._index.write(str(folder / "index.tvim"))
        with open(folder / "docstore.pkl", "wb") as f:
            pickle.dump(
                {
                    "u64_to_doc": self._u64_to_doc,
                    "next_u64": self._next_u64,
                    "dim": self._dim,
                    "bit_width": self._bit_width,
                },
                f,
            )

    @classmethod
    def load(
        cls,
        folder_path: str | Path,
        *,
        allow_dangerous_deserialization: bool = False,
    ) -> "TurboQuantDocumentStore":
        if not allow_dangerous_deserialization:
            raise ValueError(
                "load uses pickle, which is unsafe with untrusted input. "
                "Pass allow_dangerous_deserialization=True to confirm you "
                "trust the source of folder_path."
            )
        folder = Path(folder_path)
        with open(folder / "docstore.pkl", "rb") as f:
            state = pickle.load(f)
        store = cls(dim=state["dim"], bit_width=state["bit_width"])
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
            # Scale raw inner-product score to [0, 1] — cheap linear squash,
            # matches Haystack's InMemoryDocumentStore default behaviour.
            score = 1.0 / (1.0 + np.exp(-score))
        return Document(
            id=data["id"],
            content=data["content"],
            meta=dict(data["meta"]),
            score=score,
        )


__all__ = ["TurboQuantDocumentStore"]
