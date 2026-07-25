# LlamaIndex integration

`turbovec.llama_index.TurboQuantVectorStore` is a LlamaIndex [`BasePydanticVectorStore`](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/) backed by an `IdMapIndex`. It implements the same public surface as `llama_index.core.vector_stores.simple.SimpleVectorStore` and can be used as a drop-in replacement wherever the simple in-memory store is used.

## Install

```bash
pip install turbovec[llama-index]
```

## Basic usage

```python
from llama_index.core import VectorStoreIndex, StorageContext
from turbovec.llama_index import TurboQuantVectorStore

vector_store = TurboQuantVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
retriever = index.as_retriever(similarity_top_k=5)
```

The vector dimensionality is inferred from the embedding model on the first `add()` call.

## Construction

```python
# No-arg: lazy. dim is inferred from the first add.
vector_store = TurboQuantVectorStore()

# from_params: same lazy behaviour, plus an explicit bit_width.
vector_store = TurboQuantVectorStore.from_params(bit_width=4)

# Pre-built index: bring your own IdMapIndex (e.g. one you loaded from disk).
from turbovec import IdMapIndex
vector_store = TurboQuantVectorStore(index=IdMapIndex(1536, 4))
```

`bit_width` is one of `{2, 3, 4}` and is fixed once the index is created.

## Similarity modes

The `similarity` keyword (on the constructor and `from_params`) selects how the `similarities` returned by `query` are computed. It is fixed for the lifetime of the store:

- **`"cosine"` (default).** Node embeddings are L2-normalized at add time and query embeddings at query time, so `result.similarities` are true cosine similarities in `[-1, 1]` and ranking matches `SimpleVectorStore` regardless of embedding magnitude — safe to feed into similarity-cutoff postprocessors. Zero vectors are kept as-is and score `0` against everything.
- **`"dot_product"`.** Vectors are stored and queried raw: `result.similarities` are raw inner products and ranking is magnitude-aware.

```python
vector_store = TurboQuantVectorStore(similarity="dot_product")
```

The `similarity` keyword is a turbovec extension: `SimpleVectorStore` computes cosine unconditionally, so code written against the reference behaves identically under the default.

## The two `delete` signatures

LlamaIndex's vector-store protocol has two distinct delete entry points:

### `delete(ref_doc_id: str)` — remove an entire source document

Removes **every node** whose `ref_doc_id` matches. Use this when you want to delete a whole parent document and its chunks in one call.

```python
vector_store.delete("my-source-document-123")
```

Missing `ref_doc_id`s are silently ignored.

### `delete_nodes(node_ids, filters)` — remove specific chunks

Removes nodes matching either `node_ids`, `filters`, or both (intersected). Missing `node_id`s are silently ignored.

```python
# By node_id
vector_store.delete_nodes(node_ids=["abc-123", "def-456"])

# By metadata filter
from llama_index.core.vector_stores.types import (
    MetadataFilter, MetadataFilters, FilterOperator,
)
filters = MetadataFilters(
    filters=[MetadataFilter(key="tier", value="archived", operator=FilterOperator.EQ)],
)
vector_store.delete_nodes(filters=filters)

# Both: intersect — delete only nodes in this list that ALSO match the filter
vector_store.delete_nodes(node_ids=["abc-123"], filters=filters)
```

### `clear()` — drop everything

```python
vector_store.clear()
```

Resets the store while preserving the configured `bit_width`. The cleared store is immediately usable for new adds; `dim` is inferred again from the next batch.

## Query

LlamaIndex calls `query(VectorStoreQuery)` internally. If you've gone through `VectorStoreIndex.from_documents(...)`, you won't call this directly — the retriever does. For direct use:

```python
from llama_index.core.vector_stores.types import VectorStoreQuery

result = vector_store.query(VectorStoreQuery(
    query_embedding=[...],
    similarity_top_k=5,
))
# result.nodes, result.similarities, result.ids
```

`query_embedding` is **required**. turbovec doesn't embed query text itself; the calling component (retriever / query engine) is responsible for that.

### Filtered query

`VectorStoreQuery` accepts `filters`, `node_ids`, and `doc_ids`. All three intersect when more than one is supplied:

```python
from llama_index.core.vector_stores.types import (
    MetadataFilter, MetadataFilters, FilterCondition, FilterOperator,
    VectorStoreQuery,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="tier", value="pro", operator=FilterOperator.EQ),
        MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
    ],
    condition=FilterCondition.AND,
)

result = vector_store.query(VectorStoreQuery(
    query_embedding=[...],
    similarity_top_k=5,
    filters=filters,
    node_ids=["chunk-1", "chunk-2", "chunk-3"],   # restrict to these chunks
    doc_ids=["src-doc-42"],                        # restrict to chunks of this source doc
))
```

Supported operators on `MetadataFilter`: `EQ`, `NE`, `GT`, `LT`, `GTE`, `LTE`, `IN`, `NIN`, `TEXT_MATCH`, `TEXT_MATCH_INSENSITIVE`, `CONTAINS`, `ANY`, `ALL`, `IS_EMPTY`. Conditions: `AND`, `OR`, `NOT`. Nested `MetadataFilters` work.

Filter semantics match `SimpleVectorStore`'s reference implementation — notably, every operator except `IS_EMPTY` returns `False` when the filter key is missing from the document's metadata, and `TEXT_MATCH` is case-sensitive (use `TEXT_MATCH_INSENSITIVE` for a case-insensitive substring match).

Filters are resolved to a handle allowlist **before** scoring. Selective filters return up to `similarity_top_k` matches from the filtered set; you never get fewer just because the filter happened to exclude the top-scoring candidates.

## Get nodes

```python
nodes = vector_store.get_nodes(node_ids=["chunk-1", "chunk-2"])
nodes = vector_store.get_nodes(filters=filters)
nodes = vector_store.get_nodes(node_ids=["chunk-1", "chunk-2"], filters=filters)  # intersect
```

Returns a `List[BaseNode]` reconstructed from the side-car. Missing `node_id`s are silently skipped.

## Upsert semantics

Calling `add()` with a node whose `node_id` already exists **replaces** the existing entry. Matches LlamaIndex user expectation when re-indexing the same chunks.

A `node_id` repeated **within a single `add()` batch** raises `ValueError` — deduplicate before calling. (This differs from the LangChain and Haystack stores, which silently keep the last occurrence; here it's a hard error so an accidental duplicate doesn't quietly drop a node.)

```python
node = TextNode(text="v1", embedding=[...])
vector_store.add([node])

# Same node_id, different text/embedding → replaces.
updated = TextNode(text="v2", id_=node.node_id, embedding=[...])
vector_store.add([updated])
assert len(vector_store._index) == 1
```

## Async

Every public method has an async counterpart, suitable for use in LlamaIndex's async retriever / query-engine paths:

```python
await vector_store.async_add(nodes)
result = await vector_store.aquery(VectorStoreQuery(...))
fetched = await vector_store.aget_nodes(node_ids=[...])
await vector_store.adelete("ref-doc-id")
await vector_store.adelete_nodes(node_ids=[...])
await vector_store.aclear()
```

## Persist / load

### Direct (file-stem) interface

```python
vector_store.persist("./store/vectors.json")
# ... later ...
vector_store = TurboQuantVectorStore.from_persist_path("./store/vectors.json")
```

`persist_path` is treated as a path *stem* — the binary index and JSON side-car are written next to each other as `{stem}.tvim` and `{stem}.nodes.json`. The extension on `persist_path` (e.g. `.json`, as LlamaIndex's StorageContext default uses) is replaced. Node metadata must be JSON-serializable. If the `{stem}.nodes.json` side-car is out of sync with its `{stem}.tvim` index (a partial copy, a stale backup, tampering), `from_persist_path` raises a `ValueError` immediately rather than failing later with a `KeyError` at query time.

`persist` is atomic with respect to the destination: both files are written to sibling temp files and moved into place, so a failed persist (e.g. non-JSON-serializable metadata) leaves a store previously persisted at the same stem intact.

The similarity mode is recorded in `{stem}.nodes.json` and restored by `from_persist_path`. A store persisted before the mode field existed holds raw, unnormalized vectors, so it loads in `"dot_product"` mode — exactly the scoring it was written under — with no migration needed.

### Via `StorageContext`

The store works with `StorageContext.from_defaults(persist_dir=...)` the same way `SimpleVectorStore` does:

```python
# Persist
storage_context.persist(persist_dir="./store")

# Load
vector_store = TurboQuantVectorStore.from_persist_dir(persist_dir="./store")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir="./store",
)
```

`from_persist_dir(persist_dir, namespace="default", fs=None)` constructs the namespaced filename (`{persist_dir}/{namespace}__vector_store.json`) and delegates to `from_persist_path`. Multiple namespaced stores can share a persist directory. `namespace` names a store *within* `persist_dir`, so it must be non-empty and must not contain path separators or `..`; a value that would resolve outside `persist_dir` raises `ValueError`. Any other string (alphanumerics, dash, underscore) is accepted; a dotted namespace is truncated at its first dot by the current persistence layout, so avoid dotted namespaces that share a prefix in one persist directory.

### Config-only round-trip

```python
config = vector_store.to_dict()                # {"bit_width": 4, "dim": 1536, "similarity": "cosine"}
fresh = TurboQuantVectorStore.from_dict(config)                   # empty store with the same config
```

`to_dict` / `from_dict` serialize only the store's configuration. Node data round-trips through `persist` / `from_persist_path`.

## Thread safety

The store is safe for concurrent multi-threaded use:

- **Reads run concurrently and scale.** `query` and `get_nodes` take no lock; the underlying index releases the GIL during scoring, so independent queries from multiple threads overlap and scale.
- **Writes serialize.** `add`, `delete`, `delete_nodes`, `clear`, and `persist` serialize on a per-store lock. The `async_add` / `a*` variants delegate to the same locked bodies, so concurrent adds issue unique handles — no batch is ever rejected or lost to a handle collision.
- **A read overlapping a write sees pre- or post-write state** — never a torn one. Under heavy concurrent churn a query may transiently return fewer than `similarity_top_k` results (hits deleted mid-query are skipped).

What the contract does *not* cover:

- **No cross-call atomicity.** A caller-side check-then-act sequence (`get_nodes` then `delete_nodes`) can interleave with other writers. Batch writes are not atomic with respect to readers: a query overlapping a re-`add` of an existing `node_id` can briefly see that id under both its old and new entry.
- **`persist` serializes with writes** (so it always snapshots a consistent store); reads may proceed during a persist.
- **Multi-process access is not supported.**

## Known limitations

- **MMR is not supported.** Max-marginal-relevance retrieval requires the full-precision embedding of each candidate to compute pairwise diversity; turbovec discards full-precision vectors after quantization.
- **`get(text_id)` raises** rather than returning a vector — same reason. The full-precision embedding is not recoverable.
- **`fsspec` filesystems are not supported.** `persist`, `from_persist_path`, and `from_persist_dir` accept a local path. Pass `fs=None` (the default).
- **JSON-serializable metadata only.** Node metadata is stored as JSON in the side-car. Non-JSON-serializable values fail at persist time — same constraint as `SimpleVectorStore.persist`.
- **`stores_text = True`.** Unlike `SimpleVectorStore`, we keep node text in the side-car so query results return populated `TextNode`s without depending on a separate docstore. If you're swapping this in for `SimpleVectorStore` and your pipeline expects text to live elsewhere, the difference is harmless — the framework treats `stores_text` as informational.
