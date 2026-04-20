"""LangChain VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[langchain]``.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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
_STORE_FILENAME = "docstore.pkl"


class TurboQuantVectorStore(VectorStore):
    """LangChain VectorStore backed by a :class:`IdMapIndex`.

    Vectors are quantized to 2–4 bits per dimension. A side-car dictionary
    holds the original text and metadata keyed by document id. Deletion
    is supported in O(1) per id via the underlying :class:`IdMapIndex`.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index: IdMapIndex,
        *,
        docs: dict[str, tuple[str, dict[str, Any]]] | None = None,
        str_to_u64: dict[str, int] | None = None,
        next_u64: int = 0,
    ) -> None:
        self._embedding = embedding
        self._index = index
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

        # Upsert: any id that already exists is removed so the re-added
        # vector wins. Matches LangChain user expectation that `add_texts`
        # with an existing id updates in place.
        duplicates = [i for i in ids if i in self._str_to_u64]
        if duplicates:
            self.delete(duplicates)

        vectors = np.asarray(self._embedding.embed_documents(texts_list), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._index.dim:
            raise ValueError(
                f"embedding dimension {vectors.shape[1]} does not match index dim {self._index.dim}"
            )
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        handles = np.array(
            [self._issue_handle() for _ in texts_list], dtype=np.uint64
        )
        self._index.add_with_ids(vectors, handles)

        for id_, text, meta, handle in zip(ids, texts_list, metadatas, handles):
            h = int(handle)
            self._str_to_u64[id_] = h
            self._u64_to_str[h] = id_
            self._docs[id_] = (text, dict(meta))
        return ids

    def similarity_search(self, query: str, k: int = 4, **_: Any) -> list[Document]:
        return [doc for doc, _score in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **_: Any
    ) -> list[tuple[Document, float]]:
        qvec = np.asarray(self._embedding.embed_query(query), dtype=np.float32)
        return self._search_vector(qvec, k)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **_: Any
    ) -> list[Document]:
        qvec = np.asarray(embedding, dtype=np.float32)
        return [doc for doc, _score in self._search_vector(qvec, k)]

    def _search_vector(self, qvec: np.ndarray, k: int) -> list[tuple[Document, float]]:
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)
        k = min(k, len(self._index))
        if k == 0:
            return []
        scores, handles = self._index.search(qvec, k)
        results: list[tuple[Document, float]] = []
        for score, handle in zip(scores[0], handles[0]):
            sid = self._u64_to_str[int(handle)]
            text, meta = self._docs[sid]
            results.append((Document(page_content=text, metadata=dict(meta)), float(score)))
        return results

    def delete(self, ids: list[str] | None = None, **_: Any) -> bool | None:
        """Remove documents by id. Returns ``True`` if every given id was
        present and removed; ``False`` if any was missing."""
        if ids is None:
            raise ValueError("delete() requires an explicit list of ids")
        all_ok = True
        for sid in ids:
            handle = self._str_to_u64.pop(sid, None)
            if handle is None:
                all_ok = False
                continue
            self._u64_to_str.pop(handle, None)
            self._docs.pop(sid, None)
            self._index.remove(handle)
        return all_ok

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        dim: int | None = None,
        bit_width: int = 4,
        ids: list[str] | None = None,
        **_: Any,
    ) -> "TurboQuantVectorStore":
        if dim is None:
            probe_text = texts[0] if texts else "probe"
            probe = np.asarray(embedding.embed_documents([probe_text]), dtype=np.float32)
            dim = int(probe.shape[1])
        index = IdMapIndex(dim, bit_width)
        store = cls(embedding=embedding, index=index)
        if texts:
            store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def save_local(self, folder_path: str | Path) -> None:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        self._index.write(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "wb") as f:
            pickle.dump(
                {
                    "docs": self._docs,
                    "str_to_u64": self._str_to_u64,
                    "next_u64": self._next_u64,
                },
                f,
            )

    @classmethod
    def load_local(
        cls,
        folder_path: str | Path,
        embedding: Embeddings,
        *,
        allow_dangerous_deserialization: bool = False,
    ) -> "TurboQuantVectorStore":
        if not allow_dangerous_deserialization:
            raise ValueError(
                "load_local uses pickle to deserialize the document store, which is "
                "unsafe with untrusted input. Pass allow_dangerous_deserialization=True "
                "to confirm you trust the source of folder_path."
            )
        folder = Path(folder_path)
        index = IdMapIndex.load(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "rb") as f:
            state = pickle.load(f)
        return cls(
            embedding=embedding,
            index=index,
            docs=state["docs"],
            str_to_u64=state["str_to_u64"],
            next_u64=state["next_u64"],
        )


__all__ = ["TurboQuantVectorStore"]
