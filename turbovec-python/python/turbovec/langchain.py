"""LangChain VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[langchain]``.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._turbovec import TurboQuantIndex

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:
    raise ImportError(
        "langchain-core is required to use turbovec.langchain. "
        "Install with: pip install turbovec[langchain]"
    ) from exc


_INDEX_FILENAME = "index.tv"
_STORE_FILENAME = "docstore.pkl"


class TurboQuantVectorStore(VectorStore):
    """LangChain VectorStore backed by a :class:`TurboQuantIndex`.

    Vectors are quantized to 2–4 bits per dimension. A side-car dictionary
    holds the original text and metadata keyed by document id.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index: TurboQuantIndex,
        *,
        docs: dict[str, tuple[str, dict[str, Any]]] | None = None,
        idx_to_id: list[str] | None = None,
    ) -> None:
        self._embedding = embedding
        self._index = index
        self._docs: dict[str, tuple[str, dict[str, Any]]] = docs if docs is not None else {}
        self._idx_to_id: list[str] = idx_to_id if idx_to_id is not None else []

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

        vectors = np.asarray(self._embedding.embed_documents(texts_list), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._index.dim:
            raise ValueError(
                f"embedding dimension {vectors.shape[1]} does not match index dim {self._index.dim}"
            )
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)
        self._index.add(vectors)

        for id_, text, meta in zip(ids, texts_list, metadatas):
            self._idx_to_id.append(id_)
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
        scores, indices = self._index.search(qvec, k)
        results: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            id_ = self._idx_to_id[int(idx)]
            text, meta = self._docs[id_]
            results.append((Document(page_content=text, metadata=dict(meta)), float(score)))
        return results

    def delete(self, ids: list[str] | None = None, **_: Any) -> bool | None:
        raise NotImplementedError(
            "TurboQuantVectorStore does not support deletion. "
            "Rebuild the store from the remaining documents."
        )

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
        index = TurboQuantIndex(dim, bit_width)
        store = cls(embedding=embedding, index=index)
        if texts:
            store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def save_local(self, folder_path: str | Path) -> None:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        self._index.write(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "wb") as f:
            pickle.dump({"docs": self._docs, "idx_to_id": self._idx_to_id}, f)

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
        index = TurboQuantIndex.load(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "rb") as f:
            state = pickle.load(f)
        return cls(
            embedding=embedding,
            index=index,
            docs=state["docs"],
            idx_to_id=state["idx_to_id"],
        )


__all__ = ["TurboQuantVectorStore"]
