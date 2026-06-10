"""LlamaIndex VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[llama-index]``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from ._dedup import DuplicatePolicy, resolve_duplicates
from ._persist import check_persisted_handles
from ._turbovec import IdMapIndex

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import (
        BaseNode,
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
        VectorStoreQueryMode,
        VectorStoreQueryResult,
    )
    from llama_index.core.vector_stores.utils import (
        metadata_dict_to_node,
        node_to_metadata_dict,
    )
except ImportError as exc:
    raise ImportError(
        "llama-index-core is required to use turbovec.llama_index. "
        "Install with: pip install turbovec[llama-index]"
    ) from exc


# Persistence layout: a single user-facing ``persist_path`` (the file the
# framework asks us to write) is split into two files by extension —
# ``{base}.tvim`` for the binary IdMapIndex and ``{base}.nodes.json`` for
# the node side-car. Both live next to each other so the layout fits the
# directory-of-namespaced-files pattern used by StorageContext.
_INDEX_EXT = ".tvim"
_STORE_EXT = ".nodes.json"
# Filename template used by SimpleVectorStore for namespace lookup —
# mirrored here so `from_persist_dir` works the same way.
_NAMESPACE_SEP = "__"
_DEFAULT_PERSIST_FNAME = "vector_store.json"
_DEFAULT_VECTOR_STORE = "default"
# Bump when the nodes.json shape changes; loader accepts the current
# version plus any older versions whose missing fields we know how to
# reconstruct (currently v1, written before full-node round-trip was
# added — v1 entries are reconstructed as bare TextNodes with only
# text + metadata + SOURCE relationship, matching the original
# lossy behaviour rather than failing to load).
_NODES_SCHEMA_VERSION = 2
_NODES_SCHEMA_COMPAT = (1, 2)


def _split_persist_base(persist_path: str | Path) -> Path:
    """Strip the framework-provided extension off `persist_path` so the
    binary index and JSON side-car can sit next to each other under a
    shared base. We then append our own extensions in persist / load."""
    p = Path(persist_path)
    # Use the path without its suffix so both .tvim and .nodes.json share
    # a base. If the input has no suffix (e.g. a bare folder-like name),
    # use it as-is.
    return p.with_suffix("") if p.suffix else p


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

        # Reject intra-batch duplicates loudly. Letting them through would
        # leave the index with N vectors but only the last node_id mapped
        # back to one of them — the earlier handles become orphans that
        # `query` later resolves through the duplicate node_id, returning
        # the second node's payload attached to the first node's vector.
        # Caller's job to deduplicate before calling add.
        node_ids = [n.node_id for n in nodes]
        try:
            resolve_duplicates(node_ids, DuplicatePolicy.REJECT)
        except ValueError:
            seen: set[str] = set()
            dup = next(nid for nid in node_ids if nid in seen or seen.add(nid))
            raise ValueError(
                f"duplicate node_id {dup!r} appears multiple times "
                "in the input batch; deduplicate before calling add()"
            ) from None

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
        # Add first; if validation above or encoding here (e.g. non-finite
        # values) rejects the batch, it raises before any existing data is
        # touched. Only after the add succeeds do we remove the old entries
        # for colliding node_ids, so a failed upsert never destroys existing
        # data (issue #89). Handles are freshly issued, so the old and new
        # vectors coexist until the delete.
        self._index.add_with_ids(vectors, handles)

        # Upsert-like: if a node_id is already present in the STORE, remove
        # the old entry so the new embedding wins.
        duplicates = [n.node_id for n in nodes if n.node_id in self._node_id_to_u64]
        for node_id in duplicates:
            self._remove_node_by_id(node_id)

        ids: list[str] = []
        for node, handle in zip(nodes, handles):
            h = int(handle)
            nid = node.node_id
            self._node_id_to_u64[nid] = h
            self._u64_to_node_id[h] = nid
            # `metadata` and `ref_doc_id` are kept at top level for fast
            # filter / doc-id lookup (queries hit these on every hit;
            # parsing _node_content per hit would be wasteful). `node_dict`
            # is the framework's canonical metadata representation
            # (`_node_content` + `_node_type` + original metadata keys),
            # which `metadata_dict_to_node` reconstructs into a full
            # BaseNode — preserving relationships (PREVIOUS / NEXT /
            # PARENT / CHILD), excluded_*_metadata_keys, template fields,
            # start/end_char_idx and mimetype on retrieval. The narrow
            # `{text, metadata, ref_doc_id}` schema we used to keep lost
            # all of those silently.
            self._nodes[nid] = {
                "metadata": dict(node.metadata),
                "ref_doc_id": node.ref_doc_id,
                "node_dict": node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=False
                ),
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
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **_: Any,
    ) -> None:
        """Delete every node matching ``node_ids`` and/or ``filters``. Both
        constraints intersect when supplied. Missing node_ids are ignored.
        Matches the signature and semantics of ``SimpleVectorStore.delete_nodes``.
        """
        if not node_ids and filters is None:
            return
        candidates = list(self._nodes.items())
        if node_ids is not None:
            node_id_set = set(node_ids)
            candidates = [(nid, data) for nid, data in candidates if nid in node_id_set]
        if filters is not None:
            candidates = [
                (nid, data)
                for nid, data in candidates
                if self._filters_match(data["metadata"], filters)
            ]
        for nid, _data in candidates:
            self._remove_node_by_id(nid)

    def clear(self) -> None:
        """Drop every node from the store and reset to a fresh lazy index.

        The new index keeps the same ``bit_width`` so subsequent adds
        commit a new ``dim`` lazily.
        """
        bw = self._index.bit_width
        self._index = IdMapIndex(bit_width=bw)
        self._nodes = {}
        self._node_id_to_u64 = {}
        self._u64_to_node_id = {}
        self._next_u64 = 0

    def get(self, text_id: str) -> List[float]:
        """LlamaIndex's protocol expects this to return the full-precision
        embedding for a given node id. turbovec discards full-precision
        embeddings after quantization, so we raise loudly with an
        explanation rather than return a lossy reconstruction or zeroes.
        """
        raise NotImplementedError(
            "TurboQuantVectorStore.get(text_id) cannot return the original "
            "embedding because turbovec quantizes vectors to 2-4 bits per "
            "dimension and discards full precision after encoding. Keep a "
            "parallel docstore if you need the raw embedding."
        )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Return the nodes matching ``node_ids`` and/or ``filters``. Both
        constraints intersect when supplied; missing node_ids are
        silently skipped.

        Unlike ``SimpleVectorStore`` (which raises NotImplementedError
        here because it doesn't store nodes), turbovec keeps node text
        and metadata in a side-car so this can return populated
        ``TextNode`` objects directly.
        """
        candidates = list(self._nodes.items())
        if node_ids is not None:
            node_id_set = set(node_ids)
            candidates = [(nid, data) for nid, data in candidates if nid in node_id_set]
        if filters is not None:
            candidates = [
                (nid, data)
                for nid, data in candidates
                if self._filters_match(data["metadata"], filters)
            ]
        return [self._reconstruct_node(nid, data) for nid, data in candidates]

    @staticmethod
    def _reconstruct_node(nid: str, data: dict[str, Any]) -> BaseNode:
        # v2 entries carry `node_dict` — round-trip via the framework's
        # own helper so we get the full BaseNode subclass back
        # (TextNode / IndexNode / ImageNode) with every field populated.
        if "node_dict" in data:
            return metadata_dict_to_node(data["node_dict"])
        # v1 fallback: stores that were persisted before the full-node
        # round-trip landed only have {text, metadata, ref_doc_id}.
        # Reconstruct the minimum-fidelity TextNode they used to produce
        # so old on-disk stores keep loading without manual migration.
        node = TextNode(
            id_=nid,
            text=data["text"],
            metadata=dict(data["metadata"]),
        )
        if data.get("ref_doc_id") is not None:
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=data["ref_doc_id"]
            )
        return node

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
        if condition == FilterCondition.NOT:
            # Reference semantics (`build_metadata_filter_fn`,
            # `utils.py:187-189`): NOT matches when none of the inner
            # filters match. Empty inner list trivially satisfies NOT.
            return not any(results)
        raise NotImplementedError(
            f"filter condition {condition!r} not supported by TurboQuantVectorStore"
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
        if op == FilterOperator.CONTAINS:
            return target in value
        if op == FilterOperator.TEXT_MATCH:
            # Reference (`utils.py:138-144`): case-SENSITIVE substring,
            # both sides must be strings. Previous turbovec impl
            # lowercased both sides — a silent semantic divergence that
            # caused our results to disagree with SimpleVectorStore on
            # mixed-case keys.
            if isinstance(target, str) and isinstance(value, str):
                return target in value
            raise TypeError(
                "Both metadata value and filter value must be strings "
                "for the TEXT_MATCH operator"
            )
        if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
            if isinstance(target, str) and isinstance(value, str):
                return target.lower() in value.lower()
            raise TypeError(
                "Both metadata value and filter value must be strings "
                "for the TEXT_MATCH_INSENSITIVE operator"
            )
        if op == FilterOperator.ALL:
            # Reference (`utils.py:152-153`): every element of `target`
            # must be present in the metadata value (which is typically
            # a list — tag-set matching).
            return all(t in value for t in target)
        if op == FilterOperator.ANY:
            return any(t in value for t in target)
        raise NotImplementedError(
            f"filter operator {op!r} not supported by TurboQuantVectorStore"
        )

    def query(self, query: VectorStoreQuery, **_: Any) -> VectorStoreQueryResult:
        # MMR / SVM / LINEAR_REGRESSION / HYBRID etc. all need access to
        # full-precision vectors (for pairwise diversity, learned scoring,
        # or sparse-dense fusion). turbovec discards full precision after
        # quantization, so any non-DEFAULT mode is unsupportable here.
        # Raise loudly instead of silently treating it as DEFAULT, which
        # the previous impl did and which let callers think they were
        # getting e.g. MMR diversity when they were not.
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(
                f"TurboQuantVectorStore does not support query mode "
                f"{query.mode!r}. Only VectorStoreQueryMode.DEFAULT is "
                "supported — MMR / SVM / hybrid modes need access to "
                "full-precision vectors which turbovec discards after "
                "quantization. Maintain a parallel store with full vectors "
                "if you need a non-default scoring mode."
            )
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
            data = self._nodes[nid]
            result_nodes.append(self._reconstruct_node(nid, data))
            similarities.append(float(score))
            ids.append(nid)

        return VectorStoreQueryResult(nodes=result_nodes, similarities=similarities, ids=ids)

    # ---- Async overrides --------------------------------------------------
    #
    # The base class provides default async impls that delegate to sync via
    # `return self.<sync>(...)`. We override them explicitly so the signature
    # is visible on the class and an autodoc tool / IDE doesn't make
    # callers chase the abstract base class for the documentation.

    async def async_add(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> List[str]:
        return self.add(list(nodes), **kwargs)

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self.delete(ref_doc_id, **kwargs)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> None:
        self.delete_nodes(node_ids=node_ids, filters=filters, **kwargs)

    async def aclear(self) -> None:
        self.clear()

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        return self.query(query, **kwargs)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        return self.get_nodes(node_ids=node_ids, filters=filters)

    # ---- Config serialization ---------------------------------------------

    def to_dict(self, **_: Any) -> dict[str, Any]:
        """Serialize the store's *configuration* (not its data) so a
        fresh instance can be reconstructed via ``from_dict``. Mirrors
        the contract of ``SimpleVectorStore.to_dict`` — config-only;
        node data round-trips through ``persist`` / ``from_persist_path``.
        """
        return {
            "bit_width": self._index.bit_width,
            "dim": self._index.dim,  # may be None (lazy uncommitted)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **_: Any) -> "TurboQuantVectorStore":
        """Construct an empty store from a config dict produced by
        ``to_dict``. To restore data, use ``from_persist_path``."""
        dim = data.get("dim")
        bit_width = data.get("bit_width", 4)
        return cls(index=IdMapIndex(dim, bit_width))

    def persist(self, persist_path: str, fs: Any = None) -> None:
        """Persist the store. ``persist_path`` is treated as a path *stem*:
        the binary index goes to ``{stem}.tvim`` and the node side-car to
        ``{stem}.nodes.json``. Any extension on ``persist_path`` (e.g.
        ``.json`` from a StorageContext default) is replaced.

        This matches the layout assumed by ``StorageContext.persist`` —
        which calls us with ``persist_path = {persist_dir}/{namespace}__vector_store.json`` —
        and lets multiple namespaced stores coexist in the same directory.

        Node metadata must be JSON-serializable (same constraint as
        ``SimpleVectorStore``). ``fs`` (fsspec) is not yet supported;
        pass a local path.
        """
        if fs is not None:
            raise NotImplementedError(
                "fsspec filesystems are not supported yet; pass a local path."
            )
        base = _split_persist_base(persist_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self._index.write(str(base.with_suffix(_INDEX_EXT)))
        payload = {
            "schema_version": _NODES_SCHEMA_VERSION,
            "nodes": self._nodes,
            # JSON object keys must be strings; round-trip int keys via
            # an explicit list of [node_id, handle] pairs to preserve
            # type fidelity.
            "node_id_to_u64": list(self._node_id_to_u64.items()),
            "next_u64": self._next_u64,
        }
        with open(base.with_suffix(_STORE_EXT), "w") as f:
            json.dump(payload, f)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Any = None,
    ) -> "TurboQuantVectorStore":
        """Load a previously-persisted store. ``persist_path`` is the same
        path that was passed to :meth:`persist` (extension is ignored;
        ``{stem}.tvim`` and ``{stem}.nodes.json`` are read).

        Safe to call on any path — the side-car is plain JSON, never
        pickle, so there's no deserialization-of-code risk.
        """
        if fs is not None:
            raise NotImplementedError(
                "fsspec filesystems are not supported yet; pass a local path."
            )
        base = _split_persist_base(persist_path)
        index = IdMapIndex.load(str(base.with_suffix(_INDEX_EXT)))
        with open(base.with_suffix(_STORE_EXT)) as f:
            state = json.load(f)
        version = state.get("schema_version", 0)
        if version not in _NODES_SCHEMA_COMPAT:
            raise ValueError(
                f"{_STORE_EXT.lstrip('.')} has schema version {version}; "
                f"this turbovec accepts versions {list(_NODES_SCHEMA_COMPAT)}"
            )
        store = cls(index=index)
        # v1 entries lack `node_dict` and reconstruct as narrow TextNodes;
        # v2 entries carry it and reconstruct with full BaseNode fidelity.
        # `_reconstruct_node` dispatches on shape, so we just load the
        # dict as-is.
        store._nodes = state["nodes"]
        # Reconstruct {node_id: int handle} from the list-of-pairs form.
        store._node_id_to_u64 = {nid: int(h) for nid, h in state["node_id_to_u64"]}
        store._u64_to_node_id = {h: nid for nid, h in store._node_id_to_u64.items()}
        store._next_u64 = int(state["next_u64"])
        check_persisted_handles(index, store._u64_to_node_id.keys(), what="node")
        return store

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        namespace: str = _DEFAULT_VECTOR_STORE,
        fs: Any = None,
    ) -> "TurboQuantVectorStore":
        """Load a store from a ``StorageContext``-style persist directory.

        Builds the namespaced filename
        ``{persist_dir}/{namespace}__vector_store.json`` and forwards to
        :meth:`from_persist_path`. The ``.json`` suffix is conventional —
        our actual on-disk files use ``.tvim`` and ``.nodes.json``
        extensions derived from the same stem.
        """
        persist_fname = f"{namespace}{_NAMESPACE_SEP}{_DEFAULT_PERSIST_FNAME}"
        persist_path = os.path.join(persist_dir, persist_fname)
        return cls.from_persist_path(persist_path, fs=fs)


__all__ = ["TurboQuantVectorStore"]
