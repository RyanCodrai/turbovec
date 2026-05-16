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
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
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

    def __init__(self, index: IdMapIndex | None = None, *, bit_width: int = 4, **kwargs: Any) -> None:
        """Construct the vector store.

        :param index: Optional pre-built :class:`IdMapIndex`. When omitted,
            a lazy ``IdMapIndex`` is created — it commits to a dim on the
            first add and lets callers use the no-arg construction pattern
            common to LlamaIndex's other vector stores (e.g. via
            ``StorageContext.from_defaults(vector_store=TurboQuantVectorStore())``).
        :param bit_width: Quantization width used when constructing the
            lazy index. Ignored if ``index`` is supplied.
        """
        super().__init__(**kwargs)
        # IdMapIndex itself supports lazy construction now — no per-store
        # lazy wrapping needed.
        self._index = index if index is not None else IdMapIndex(bit_width=bit_width)
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
    def from_params(cls, dim: int | None = None, bit_width: int = 4) -> "TurboQuantVectorStore":
        """Build a store with a known ``dim`` (eager) or lazy when ``dim``
        is omitted."""
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
        if vectors.ndim != 2:
            raise ValueError(
                f"expected 2D embedding batch, got {vectors.ndim}D"
            )
        # IdMapIndex.add_with_ids handles eager (dim must match) and lazy
        # (locks dim on first add) — pre-check the eager case so we
        # surface a clean ValueError rather than a Rust panic.
        existing_dim = self._index.dim
        if existing_dim is not None and vectors.shape[1] != existing_dim:
            raise ValueError(
                f"node embedding dim {vectors.shape[1]} does not match index dim {existing_dim}"
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

    def _resolve_allowed_handles(
        self,
        filters: MetadataFilters | None,
        node_ids: list[str] | None,
        doc_ids: list[str] | None,
    ) -> list[int]:
        """Resolve ``query.filters``, ``query.node_ids`` and ``query.doc_ids``
        to the list of internal u64 handles that satisfy the filter. Empty
        list means no node matches.

        Semantics (matching the SimpleVectorStore reference where applicable):
          - ``node_ids``: filter by node_id (set membership).
          - ``doc_ids``: filter by ``ref_doc_id`` only (source document id).
          - ``filters``: apply metadata filters.
        All three intersect when more than one is supplied.
        """
        candidates = self._nodes.items()

        if node_ids:
            node_id_set = set(node_ids)
            candidates = [(nid, data) for nid, data in candidates if nid in node_id_set]

        if doc_ids:
            doc_id_set = set(doc_ids)
            candidates = [
                (nid, data)
                for nid, data in candidates
                if data.get("ref_doc_id") in doc_id_set
            ]

        if filters is None:
            return [self._node_id_to_u64[nid] for nid, _ in candidates]

        return [
            self._node_id_to_u64[nid]
            for nid, data in candidates
            if self._filters_match(data["metadata"], filters)
        ]

    @classmethod
    def _filters_match(
        cls, metadata: dict[str, Any], filters: MetadataFilters
    ) -> bool:
        condition = getattr(filters, "condition", None) or FilterCondition.AND
        results: list[bool] = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                results.append(cls._filters_match(metadata, f))
            else:
                results.append(cls._single_filter_match(metadata, f))
        if condition == FilterCondition.AND:
            return all(results) if results else True
        if condition == FilterCondition.OR:
            return any(results) if results else True
        raise NotImplementedError(
            f"filter condition {condition!r} not supported by TurboQuantVectorStore "
            "(supported: AND, OR)"
        )

    @staticmethod
    def _single_filter_match(metadata: dict[str, Any], f: MetadataFilter) -> bool:
        # Semantics mirror SimpleVectorStore's _build_metadata_filter_fn
        # (llama_index/core/vector_stores/simple.py) so that filtered
        # results agree with the in-tree reference store.
        op = f.operator
        target = f.value
        value = metadata.get(f.key)

        # IS_EMPTY is the only operator that treats a missing key as a hit.
        if op == FilterOperator.IS_EMPTY:
            return value is None or value == "" or value == []

        # Every other operator returns False when the key is absent — this
        # matches the reference implementation (notably NE returns False on
        # missing, not True).
        if value is None:
            return False

        if op == FilterOperator.EQ:
            return value == target
        if op == FilterOperator.NE:
            return value != target
        if op == FilterOperator.GT:
            return value > target
        if op == FilterOperator.LT:
            return value < target
        if op == FilterOperator.GTE:
            return value >= target
        if op == FilterOperator.LTE:
            return value <= target
        if op == FilterOperator.IN:
            return value in target
        if op == FilterOperator.NIN:
            return value not in target
        if op == FilterOperator.TEXT_MATCH:
            # Case-insensitive, like the reference impl.
            return str(target).lower() in str(value).lower()
        if op == FilterOperator.CONTAINS:
            return target in value
        raise NotImplementedError(
            f"filter operator {op!r} not supported by TurboQuantVectorStore"
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

        if len(self._index) == 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        has_filters = (
            query.filters is not None
            or bool(query.node_ids)
            or bool(query.doc_ids)
        )
        if not has_filters:
            k = min(query.similarity_top_k, len(self._index))
            scores, handles = self._index.search(qvec, k)
        else:
            allowed_handles = self._resolve_allowed_handles(
                query.filters, query.node_ids, query.doc_ids
            )
            if not allowed_handles:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            allowlist = np.asarray(allowed_handles, dtype=np.uint64)
            scores, handles = self._index.search(
                qvec, query.similarity_top_k, allowlist=allowlist
            )

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
