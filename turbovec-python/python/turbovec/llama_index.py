"""LlamaIndex VectorStore backed by turbovec's quantized index.

Install with: ``pip install turbovec[llama-index]``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from ._dedup import DuplicatePolicy, resolve_duplicates
from ._persist import check_persisted_handles, check_sidecar_keysets
from ._turbovec import IdMapIndex
from ._persist import atomic_save  # isort:skip

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

# Enum members added after llama-index-core 0.11.0. Resolved with getattr
# so that comparing against them never raises AttributeError on older
# releases where the member does not exist — queries that don't use these
# operators must keep working there. On old versions the sentinel is None,
# which can never equal a real FilterOperator/FilterCondition member.
_TEXT_MATCH_INSENSITIVE = getattr(FilterOperator, "TEXT_MATCH_INSENSITIVE", None)
_CONDITION_NOT = getattr(FilterCondition, "NOT", None)


def _validate_namespace(namespace: str) -> str:
    """Reject a ``namespace`` that would escape the caller-chosen
    ``persist_dir`` when composed into a filename.

    ``from_persist_dir`` builds ``{persist_dir}/{namespace}__vector_store.json``.
    A ``namespace`` containing a path separator or ``..`` resolves outside
    ``persist_dir`` (path traversal), and an empty/``.`` namespace names the
    directory itself rather than a store file. We reject these loudly rather
    than silently basenaming them — silent rewriting could load a *different*
    store than the caller named. Any other namespace is accepted verbatim
    (alphanumerics, dash, underscore, and other non-separator characters).
    Note a dotted namespace is truncated at its first dot by the current
    persistence layout, so two dotted namespaces sharing a prefix (e.g.
    ``v1.2`` and ``v1.3``) collide in one ``persist_dir`` — see issue #200.

    This is a deliberate divergence from ``SimpleVectorStore``, which does
    not sanitize its namespace, in the safer direction.
    """
    if namespace == "" or namespace == ".":
        raise ValueError(
            f"namespace must be a non-empty name, got {namespace!r}"
        )
    if "/" in namespace or "\\" in namespace or os.sep in namespace or ".." in namespace:
        raise ValueError(
            "namespace must not contain path separators ('/' or '\\\\') or "
            f"'..'; got {namespace!r}. It names a store within persist_dir, "
            "not a path."
        )
    return namespace


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

    **Thread safety.** The store is safe for concurrent multi-threaded
    use. Reads (``query``, ``get_nodes``) run concurrently and scale
    across threads; writes (``add``, ``delete``, ``delete_nodes``,
    ``clear``, ``persist``) serialize on a per-store lock — the ``a*`` /
    ``async_*`` variants delegate to the same locked bodies. A read that
    overlaps a write sees either the pre- or post-write state — never a
    torn one — and under heavy concurrent churn a query may transiently
    return fewer than ``similarity_top_k`` results. There is no
    cross-call atomicity: a caller-side check-then-act sequence (e.g.
    ``get_nodes`` then ``delete_nodes``) can still interleave with other
    writers. Multi-process access is not supported.
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    _index: Any = PrivateAttr()
    _nodes: dict[str, dict[str, Any]] = PrivateAttr()
    _node_id_to_u64: dict[str, int] = PrivateAttr()
    _u64_to_node_id: dict[int, str] = PrivateAttr()
    _next_u64: int = PrivateAttr()
    _write_lock: Any = PrivateAttr()

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
        # Serializes every mutation (add / delete / clear / persist).
        # Readers do NOT take this lock — query stays lock-free so
        # concurrent reads keep scaling across threads (#186); reader
        # safety comes from mutation ordering plus tolerant handle
        # resolution instead. The lock also makes `_next_u64 += 1` atomic
        # w.r.t. other writers: pydantic's PrivateAttr machinery makes the
        # bare increment tearable, which issued duplicate handles (and
        # silently rejected whole batches) under concurrent add (#161).
        self._write_lock = threading.RLock()

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
        # Materialize once up front: the input is iterated multiple times
        # below, so a generator / one-shot iterable would silently drain on
        # the first pass (async_add already does this via list(nodes)).
        nodes = list(nodes)
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
        # A batch of empty per-node embeddings has shape (N, 0) — 2D, so
        # it passes the ndim guard, then dies deep in the index kernel
        # with an opaque buffer-length error. Name the real cause instead.
        if vectors.shape[1] == 0:
            raise ValueError(
                "nodes have empty embeddings (dim 0); check the embed "
                "model that produced them"
            )
        # Build every side-car payload BEFORE mutating any state, so a
        # payload failure (e.g. non-serializable metadata) leaves the
        # store untouched. `metadata` and `ref_doc_id` are kept at top
        # level for fast filter / doc-id lookup (queries hit these on
        # every hit; parsing _node_content per hit would be wasteful).
        # `node_dict` is the framework's canonical metadata representation
        # (`_node_content` + `_node_type` + original metadata keys),
        # which `metadata_dict_to_node` reconstructs into a full
        # BaseNode — preserving relationships (PREVIOUS / NEXT /
        # PARENT / CHILD), excluded_*_metadata_keys, template fields,
        # start/end_char_idx and mimetype on retrieval. The narrow
        # `{text, metadata, ref_doc_id}` schema we used to keep lost
        # all of those silently.
        payloads = [
            {
                "metadata": dict(node.metadata),
                "ref_doc_id": node.ref_doc_id,
                "node_dict": node_to_metadata_dict(
                    node, remove_text=False, flat_metadata=False
                ),
            }
            for node in nodes
        ]

        with self._write_lock:
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

            # Capture the previous state of any upserted node_id BEFORE the
            # maps are overwritten, so a failed index add can restore it and
            # the old vectors can be dropped once the add succeeds.
            old = [
                (nid, self._node_id_to_u64[nid], self._nodes[nid])
                for nid in node_ids
                if nid in self._node_id_to_u64
            ]

            # Maps BEFORE the index add: a concurrent query can only learn
            # a handle from the index, so an entry that is resolvable but
            # not yet searchable is invisible to readers (safe) — the
            # reverse ordering let readers surface handles that did not
            # resolve yet (issue #161).
            ids: list[str] = []
            for node, payload, handle in zip(nodes, payloads, handles):
                h = int(handle)
                nid = node.node_id
                self._node_id_to_u64[nid] = h
                self._u64_to_node_id[h] = nid
                self._nodes[nid] = payload
                ids.append(nid)
            try:
                self._index.add_with_ids(vectors, handles)
            except BaseException:
                # Unwind the pre-inserted map entries (and restore the
                # previous state of any upserted node_id) so a failed add
                # never destroys existing data — preserving the issue-#89
                # guarantee under the maps-first ordering.
                for node, handle in zip(nodes, handles):
                    h = int(handle)
                    nid = node.node_id
                    self._u64_to_node_id.pop(h, None)
                    if self._node_id_to_u64.get(nid) == h:
                        self._node_id_to_u64.pop(nid, None)
                        self._nodes.pop(nid, None)
                for nid, old_h, old_data in old:
                    self._node_id_to_u64[nid] = old_h
                    self._u64_to_node_id[old_h] = nid
                    self._nodes[nid] = old_data
                raise

            # Upsert-like: drop the replaced vectors, index first so each
            # old handle stops being searchable before it stops resolving.
            # The forward maps already hold the new entries, so only the
            # old handle and its reverse-map entry remain to clean up.
            for _nid, old_h, _old_data in old:
                self._index.remove(old_h)
                self._u64_to_node_id.pop(old_h, None)
        return ids

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        """Delete every node whose ``ref_doc_id`` matches."""
        with self._write_lock:
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
        with self._write_lock:
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
        with self._write_lock:
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

        When ``node_ids`` is given, results come back in the requested-id
        order (matching the LangChain integration's ``get_by_ids``);
        otherwise in storage order.
        """
        if node_ids is not None:
            # Single .get instead of check-then-read: a concurrent delete
            # between `in` and `[]` would raise KeyError; a missing id is
            # skipped either way.
            candidates = []
            for nid in node_ids:
                data = self._nodes.get(nid)
                if data is not None:
                    candidates.append((nid, data))
        else:
            # list() snapshot: a concurrent writer must not invalidate the
            # iteration mid-scan (single C-level op, no torn view).
            candidates = list(self._nodes.items())
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
        # Callers hold the writer lock. Index first: the handle stops
        # being searchable before it stops resolving, so a concurrent
        # query can never surface a handle whose side-car entries are
        # already gone.
        handle = self._node_id_to_u64.get(node_id)
        if handle is None:
            return False
        self._index.remove(handle)
        self._node_id_to_u64.pop(node_id, None)
        self._u64_to_node_id.pop(handle, None)
        self._nodes.pop(node_id, None)
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
        # list() snapshot: this runs on the (lock-free) query path, so a
        # concurrent writer must not invalidate the iteration mid-scan.
        candidates = list(self._nodes.items())

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

        out: list[int] = []
        for nid, data in candidates:
            if filters is not None and not self._filters_match(data["metadata"], filters):
                continue
            # Tolerant resolution: the node can vanish between the snapshot
            # above and here (concurrent delete) — skip rather than raise.
            handle = self._node_id_to_u64.get(nid)
            if handle is not None:
                out.append(handle)
        return out

    @classmethod
    def _filters_match(
        cls, metadata: dict[str, Any], filters: MetadataFilters
    ) -> bool:
        condition = getattr(filters, "condition", None) or FilterCondition.AND
        results: list[bool] = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                # Deliberate superset of the reference: SimpleVectorStore's
                # build_metadata_filter_fn raises ValueError on nested
                # MetadataFilters groups; turbovec recurses and evaluates
                # them. Per-operator semantics still match the reference
                # (see _single_filter_match).
                results.append(cls._filters_match(metadata, f))
            else:
                results.append(cls._single_filter_match(metadata, f))
        if condition == FilterCondition.AND:
            return all(results) if results else True
        if condition == FilterCondition.OR:
            return any(results) if results else True
        if _CONDITION_NOT is not None and condition == _CONDITION_NOT:
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

        # Missing key: the reference (`build_metadata_filter_fn`,
        # `utils.py`) treats an absent value as a MATCH for the negative
        # operators NE / NIN ("not equal to X" is trivially true when the
        # key isn't there) and a non-match for every other operator.
        if value is None:
            return op in (FilterOperator.NE, FilterOperator.NIN)

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
        if _TEXT_MATCH_INSENSITIVE is not None and op == _TEXT_MATCH_INSENSITIVE:
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
            for _attempt in range(8):
                allowed_handles = self._resolve_allowed_handles(
                    query.filters, query.node_ids, query.doc_ids
                )
                if not allowed_handles:
                    return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
                allowlist = np.asarray(allowed_handles, dtype=np.uint64)
                try:
                    scores, handles = self._index.search(
                        qvec, query.similarity_top_k, allowlist=allowlist
                    )
                    break
                except KeyError:
                    # The allowlist went stale: a delete landed between
                    # resolving it and the kernel's membership check.
                    # Rebuild the allowlist and retry.
                    continue
            else:
                # Sustained churn kept invalidating the allowlist. Fall
                # back to an unfiltered search plus a tolerant Python-side
                # post-filter, which cannot raise (a query overlapping
                # heavy churn may return fewer than similarity_top_k hits).
                k = query.similarity_top_k
                fetch_k = min(max(k * 4, 32), len(self._index) or 1)
                scores, handles = self._index.search(qvec, fetch_k)
                node_id_set = set(query.node_ids) if query.node_ids else None
                doc_id_set = set(query.doc_ids) if query.doc_ids else None
                result_nodes: list[TextNode] = []
                similarities: list[float] = []
                ids: list[str] = []
                for score, handle in zip(scores[0], handles[0]):
                    nid = self._u64_to_node_id.get(int(handle))
                    if nid is None:
                        continue
                    data = self._nodes.get(nid)
                    if data is None:
                        continue
                    if node_id_set is not None and nid not in node_id_set:
                        continue
                    if doc_id_set is not None and data.get("ref_doc_id") not in doc_id_set:
                        continue
                    if query.filters is not None and not self._filters_match(
                        data["metadata"], query.filters
                    ):
                        continue
                    result_nodes.append(self._reconstruct_node(nid, data))
                    similarities.append(float(score))
                    ids.append(nid)
                    if len(ids) >= k:
                        break
                return VectorStoreQueryResult(
                    nodes=result_nodes, similarities=similarities, ids=ids
                )

        result_nodes = []
        similarities = []
        ids = []
        for score, handle in zip(scores[0], handles[0]):
            # Tolerant translation: a handle surfaced by the index can stop
            # resolving if a delete completes between the kernel search and
            # this loop (the reader-straddle). Skip it — the node is gone
            # either way — rather than raising KeyError mid-query.
            nid = self._u64_to_node_id.get(int(handle))
            if nid is None:
                continue
            data = self._nodes.get(nid)
            if data is None:
                continue
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
        # Serializes with writers: snapshotting the maps and the index
        # concurrently with a write would persist a torn store to disk.
        # Reads may proceed while a persist runs.
        with self._write_lock:
            payload = {
                "schema_version": _NODES_SCHEMA_VERSION,
                "nodes": self._nodes,
                # JSON object keys must be strings; round-trip int keys via
                # an explicit list of [node_id, handle] pairs to preserve
                # type fidelity.
                "node_id_to_u64": list(self._node_id_to_u64.items()),
                "next_u64": self._next_u64,
            }
            # Atomic: serializes in memory first, then temp-file + replace,
            # so a failed persist can't destroy a previous store at this path.
            atomic_save(
                self._index,
                base.with_suffix(_INDEX_EXT),
                payload,
                base.with_suffix(_STORE_EXT),
            )

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
        # Two node ids sharing a handle would silently collapse in the
        # inverse map built above; require the id map to be 1:1 before
        # trusting either direction.
        if len(store._u64_to_node_id) != len(store._node_id_to_u64):
            raise ValueError(
                "persisted store is corrupt: duplicate node handles in the side-car"
            )
        # The side-car holds two structures keyed by node id (`nodes` and
        # `node_id_to_u64`); they can desync independently of the index. A
        # `nodes` entry missing for a mapped id would otherwise surface as
        # a KeyError deep inside a later query (issue #133).
        check_sidecar_keysets(
            store._node_id_to_u64.keys(),
            store._nodes.keys(),
            what="node",
            mapping_name="node_id_to_u64",
            sidecar_name="nodes",
        )
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

        ``namespace`` names a store *within* ``persist_dir``; it must not
        contain path separators or ``..`` and must be non-empty. A traversal
        namespace raises ``ValueError`` rather than reading outside
        ``persist_dir``.
        """
        _validate_namespace(namespace)
        persist_fname = f"{namespace}{_NAMESPACE_SEP}{_DEFAULT_PERSIST_FNAME}"
        persist_path = os.path.join(persist_dir, persist_fname)
        return cls.from_persist_path(persist_path, fs=fs)


__all__ = ["TurboQuantVectorStore"]
