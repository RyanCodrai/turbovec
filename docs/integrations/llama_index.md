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

`bit_width` is `2` or `4` and is fixed once the index is created.

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

`persist_path` is treated as a path *stem* — the binary index and JSON side-car are written next to each other as `{stem}.tvim` and `{stem}.nodes.json`. The extension on `persist_path` (e.g. `.json`, as LlamaIndex's StorageContext default uses) is replaced. Node metadata must be JSON-serializable.

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

`from_persist_dir(persist_dir, namespace="default", fs=None)` constructs the namespaced filename (`{persist_dir}/{namespace}__vector_store.json`) and delegates to `from_persist_path`. Multiple namespaced stores can share a persist directory.

### Config-only round-trip

```python
config = vector_store.to_dict()                                   # {"bit_width": 4, "dim": 1536}
fresh = TurboQuantVectorStore.from_dict(config)                   # empty store with the same config
```

`to_dict` / `from_dict` serialize only the store's configuration. Node data round-trips through `persist` / `from_persist_path`.

## Known limitations

- **MMR is not supported.** Max-marginal-relevance retrieval requires the full-precision embedding of each candidate to compute pairwise diversity; turbovec discards full-precision vectors after quantization.
- **`get(text_id)` raises** rather than returning a vector — same reason. The full-precision embedding is not recoverable.
- **`fsspec` filesystems are not supported.** `persist`, `from_persist_path`, and `from_persist_dir` accept a local path. Pass `fs=None` (the default).
- **JSON-serializable metadata only.** Node metadata is stored as JSON in the side-car. Non-JSON-serializable values fail at persist time — same constraint as `SimpleVectorStore.persist`.
- **`stores_text = True`.** Unlike `SimpleVectorStore`, we keep node text in the side-car so query results return populated `TextNode`s without depending on a separate docstore. If you're swapping this in for `SimpleVectorStore` and your pipeline expects text to live elsewhere, the difference is harmless — the framework treats `stores_text` as informational.
