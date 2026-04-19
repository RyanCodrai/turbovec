"""LlamaIndex VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[llama-index]``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ._turbovec import TurboQuantIndex

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import (
        BaseNode,
        MetadataMode,
        NodeRelationship,
        RelatedNodeInfo,
        TextNode,
    )
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError as exc:
    raise ImportError(
        "llama-index-core is required to use turbovec.llama_index. "
        "Install with: pip install turbovec[llama-index]"
    ) from exc


_INDEX_FILENAME = "index.tv"
_STORE_FILENAME = "nodes.pkl"


class TurboQuantVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore backed by a :class:`TurboQuantIndex`.

    Vectors are quantized to 2–4 bits per dimension. A side-car dictionary
    holds node text and metadata keyed by ``node_id``.
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    _index: Any = PrivateAttr()
    _nodes: dict[str, dict[str, Any]] = PrivateAttr()
    _idx_to_node_id: list[str] = PrivateAttr()

    def __init__(self, index: TurboQuantIndex, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._index = index
        self._nodes = {}
        self._idx_to_node_id = []

    @classmethod
    def class_name(cls) -> str:
        return "TurboQuantVectorStore"

    @classmethod
    def from_params(cls, dim: int, bit_width: int = 4) -> "TurboQuantVectorStore":
        return cls(index=TurboQuantIndex(dim, bit_width))

    @property
    def client(self) -> TurboQuantIndex:
        return self._index

    def add(self, nodes: list[BaseNode], **_: Any) -> list[str]:
        if not nodes:
            return []
        embeddings = [node.get_embedding() for node in nodes]
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._index.dim:
            raise ValueError(
                f"node embedding dim {vectors.shape[1]} does not match index dim {self._index.dim}"
            )
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)
        self._index.add(vectors)

        ids: list[str] = []
        for node in nodes:
            nid = node.node_id
            self._idx_to_node_id.append(nid)
            self._nodes[nid] = {
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "metadata": dict(node.metadata),
                "ref_doc_id": node.ref_doc_id,
            }
            ids.append(nid)
        return ids

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        raise NotImplementedError(
            "TurboQuantVectorStore does not support deletion. "
            "Rebuild the store from the remaining nodes."
        )

    def query(self, query: VectorStoreQuery, **_: Any) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError(
                "TurboQuantVectorStore requires a pre-computed query_embedding "
                "(is_embedding_query=True)."
            )
        qvec = np.asarray(query.query_embedding, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        if not qvec.flags["C_CONTIGUOUS"]:
            qvec = np.ascontiguousarray(qvec)

        k = min(query.similarity_top_k, len(self._index))
        if k == 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        scores, indices = self._index.search(qvec, k)

        result_nodes: list[TextNode] = []
        similarities: list[float] = []
        ids: list[str] = []
        for score, idx in zip(scores[0], indices[0]):
            nid = self._idx_to_node_id[int(idx)]
            state = self._nodes[nid]
            node = TextNode(
                id_=nid,
                text=state["text"],
                metadata=dict(state["metadata"]),
            )
            if state["ref_doc_id"] is not None:
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=state["ref_doc_id"]
                )
            result_nodes.append(node)
            similarities.append(float(score))
            ids.append(nid)

        return VectorStoreQueryResult(nodes=result_nodes, similarities=similarities, ids=ids)

    def persist(self, persist_path: str, fs: Any = None) -> None:
        if fs is not None:
            raise NotImplementedError("fsspec filesystems are not supported yet; pass a local path.")
        folder = Path(persist_path)
        folder.mkdir(parents=True, exist_ok=True)
        self._index.write(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "wb") as f:
            pickle.dump({"nodes": self._nodes, "idx_to_node_id": self._idx_to_node_id}, f)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Any = None,
        *,
        allow_dangerous_deserialization: bool = False,
    ) -> "TurboQuantVectorStore":
        if fs is not None:
            raise NotImplementedError("fsspec filesystems are not supported yet; pass a local path.")
        if not allow_dangerous_deserialization:
            raise ValueError(
                "from_persist_path uses pickle to deserialize the node store, which is "
                "unsafe with untrusted input. Pass allow_dangerous_deserialization=True "
                "to confirm you trust the source of persist_path."
            )
        folder = Path(persist_path)
        index = TurboQuantIndex.load(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "rb") as f:
            state = pickle.load(f)
        store = cls(index=index)
        store._nodes = state["nodes"]
        store._idx_to_node_id = state["idx_to_node_id"]
        return store


__all__ = ["TurboQuantVectorStore"]
