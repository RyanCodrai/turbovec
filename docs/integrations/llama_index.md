# LlamaIndex integration

`turbovec.llama_index.TurboQuantVectorStore` is a [LlamaIndex `BasePydanticVectorStore`](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/) backed by an `IdMapIndex`. Supports O(1) delete (both `delete(ref_doc_id)` and `delete_nodes(node_ids)`) and full `persist`/`from_persist_path`.

## Install

```bash
pip install turbovec[llama-index]
```

## Basic usage

```python
from llama_index.core import VectorStoreIndex, StorageContext
from turbovec.llama_index import TurboQuantVectorStore

vector_store = TurboQuantVectorStore.from_params(dim=768, bit_width=4)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
retriever = index.as_retriever(similarity_top_k=5)
```

## The two `delete` signatures

LlamaIndex's vector-store protocol has two distinct delete entry points:

### `delete(ref_doc_id: str)` — remove an entire source document

Removes **every node** whose `ref_doc_id` matches. Use this when you want to delete a whole parent document and its chunks in one call.

```python
vector_store.delete("my-source-document-123")
```

Missing `ref_doc_id`s are silently ignored (per LlamaIndex's convention).

### `delete_nodes(node_ids: list[str])` — remove specific chunks

Removes specific nodes by `node_id`. Use this when you've identified individual chunks you want to evict (e.g. after a rerank filter).

```python
vector_store.delete_nodes(["abc-123", "def-456"])
```

Missing `node_id`s are silently ignored. `filters=` is not supported and raises `NotImplementedError`.

## Persist / load

```python
vector_store.persist("./my-store")
# ... later ...
vector_store = TurboQuantVectorStore.from_persist_path(
    "./my-store",
    allow_dangerous_deserialization=True,
)
```

Produces two files:
- `index.tvim` — the `IdMapIndex` payload
- `nodes.pkl` — pickled node data (text, metadata, `ref_doc_id`) plus id maps

The `allow_dangerous_deserialization` flag is required because unpickling untrusted data is unsafe.

`fs=` (fsspec filesystem objects) is not supported yet; pass a local path.

## Upsert semantics

Calling `add()` with a node whose `node_id` already exists **replaces** the existing entry. Matches LlamaIndex user expectation when re-indexing the same chunks.

```python
node = TextNode(text="v1", embedding=[...])
vector_store.add([node])

# Same node_id, different text/embedding → replaces.
updated = TextNode(text="v2", id_=node.node_id, embedding=[...])
vector_store.add([updated])
assert len(vector_store._index) == 1
```

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

## Known limitations

- **Embeddings are dropped after quantization.** Query results return `TextNode`s with text, metadata, and `ref_doc_id`, but not the original embedding.
- **No `fsspec` support in persist**. Local paths only.
- **Side-car is pickle.** `allow_dangerous_deserialization=True` required on load.
