"""LlamaIndex VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[llama-index]``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ._turbovec import IdMapIndex

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


_INDEX_FILENAME = "index.tvim"
_STORE_FILENAME = "nodes.pkl"


class TurboQuantVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore backed by a :class:`IdMapIndex`.

    Vectors are quantized to 2–4 bits per dimension. A side-car dictionary
    holds node text and metadata keyed by ``node_id``. Supports ``delete``
    (by ``ref_doc_id``, removing every node with that ref) and
    ``delete_nodes`` (by ``node_id``) — both O(1) per node.
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    _index: Any = PrivateAttr()
    _nodes: dict[str, dict[str, Any]] = PrivateAttr()
    _node_id_to_u64: dict[str, int] = PrivateAttr()
    _u64_to_node_id: dict[int, str] = PrivateAttr()
    _next_u64: int = PrivateAttr()

    def __init__(self, index: IdMapIndex, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._index = index
        self._nodes = {}
        self._node_id_to_u64 = {}
        self._u64_to_node_id = {}
        self._next_u64 = 0

    def _issue_handle(self) -> int:
        self._next_u64 += 1
        return self._next_u64

    @classmethod
    def class_name(cls) -> str:
        return "TurboQuantVectorStore"

    @classmethod
    def from_params(cls, dim: int, bit_width: int = 4) -> "TurboQuantVectorStore":
        return cls(index=IdMapIndex(dim, bit_width))

    @property
    def client(self) -> IdMapIndex:
        return self._index

    def add(self, nodes: list[BaseNode], **_: Any) -> list[str]:
        if not nodes:
            return []

        # Upsert-like: if a node_id is already present, remove the old
        # entry before re-adding so the new embedding wins.
        duplicates = [n.node_id for n in nodes if n.node_id in self._node_id_to_u64]
        for node_id in duplicates:
            self._remove_node_by_id(node_id)

        embeddings = [node.get_embedding() for node in nodes]
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._index.dim:
            raise ValueError(
                f"node embedding dim {vectors.shape[1]} does not match index dim {self._index.dim}"
            )
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        handles = np.array([self._issue_handle() for _ in nodes], dtype=np.uint64)
        self._index.add_with_ids(vectors, handles)

        ids: list[str] = []
        for node, handle in zip(nodes, handles):
            h = int(handle)
            nid = node.node_id
            self._node_id_to_u64[nid] = h
            self._u64_to_node_id[h] = nid
            self._nodes[nid] = {
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "metadata": dict(node.metadata),
                "ref_doc_id": node.ref_doc_id,
            }
            ids.append(nid)
        return ids

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        """Delete every node whose ``ref_doc_id`` matches."""
        matching = [
            nid for nid, data in self._nodes.items() if data.get("ref_doc_id") == ref_doc_id
        ]
        for nid in matching:
            self._remove_node_by_id(nid)

    def delete_nodes(
        self,
        node_ids: list[str],
        filters: Any = None,
        **_: Any,
    ) -> None:
        """Delete specific nodes by their ``node_id``. Missing ids are ignored."""
        if filters is not None:
            raise NotImplementedError(
                "TurboQuantVectorStore does not support metadata filtering on delete_nodes."
            )
        for nid in node_ids:
            self._remove_node_by_id(nid)

    def _remove_node_by_id(self, node_id: str) -> bool:
        handle = self._node_id_to_u64.pop(node_id, None)
        if handle is None:
            return False
        self._u64_to_node_id.pop(handle, None)
        self._nodes.pop(node_id, None)
        self._index.remove(handle)
        return True

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

        scores, handles = self._index.search(qvec, k)

        result_nodes: list[TextNode] = []
        similarities: list[float] = []
        ids: list[str] = []
        for score, handle in zip(scores[0], handles[0]):
            nid = self._u64_to_node_id[int(handle)]
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
            pickle.dump(
                {
                    "nodes": self._nodes,
                    "node_id_to_u64": self._node_id_to_u64,
                    "next_u64": self._next_u64,
                },
                f,
            )

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
        index = IdMapIndex.load(str(folder / _INDEX_FILENAME))
        with open(folder / _STORE_FILENAME, "rb") as f:
            state = pickle.load(f)
        store = cls(index=index)
        store._nodes = state["nodes"]
        store._node_id_to_u64 = state["node_id_to_u64"]
        store._u64_to_node_id = {h: nid for nid, h in state["node_id_to_u64"].items()}
        store._next_u64 = state["next_u64"]
        return store


__all__ = ["TurboQuantVectorStore"]
