# Haystack integration

`turbovec.haystack.TurboQuantDocumentStore` is a Haystack 2.x [`DocumentStore`](https://docs.haystack.deepset.ai/docs/document-store) backed by an `IdMapIndex`. It implements the same public surface as `haystack.document_stores.in_memory.InMemoryDocumentStore` and can be used as a drop-in replacement wherever the in-memory store is used.

## Install

```bash
pip install turbovec[haystack]
```

## Basic usage

```python
from haystack import Document
from turbovec.haystack import TurboQuantDocumentStore

store = TurboQuantDocumentStore()
store.write_documents([
    Document(content="...", embedding=[...], meta={"source": "a"}),
    Document(content="...", embedding=[...], meta={"source": "b"}),
])

results = store.embedding_retrieval(query_embedding=[...], top_k=5)
```

Documents must have pre-computed embeddings — `TurboQuantDocumentStore` doesn't invoke an embedder. Pipe a Haystack embedder component upstream if your documents arrive without embeddings.

## Constructor

```python
TurboQuantDocumentStore(
    dim: Optional[int] = None,
    bit_width: int = 4,
    *,
    embedding_similarity_function: Literal["dot_product", "cosine"] = "cosine",
    async_executor: Optional[ThreadPoolExecutor] = None,
    return_embedding: bool = False,
)
```

| Parameter | Notes |
|---|---|
| `dim` | Optional. When omitted the vector dimensionality is inferred from the first `write_documents` call. |
| `bit_width` | Quantization width per coordinate; one of `{2, 4}`. |
| `embedding_similarity_function` | Drives the `scale_score=True` formula on retrieval. Defaults to `"cosine"` (right for unit-normalized embeddings); `"dot_product"` uses Haystack's `expit(s / 100)` formula. |
| `async_executor` | Optional `ThreadPoolExecutor` for the `*_async` methods. If omitted, a single-threaded executor is created and cleaned up with the store. |
| `return_embedding` | Accepted for API parity with `InMemoryDocumentStore`. The full-precision embedding is never available (quantized away), so `Document.embedding` on retrieved docs is always `None` regardless of the flag. |

## `DuplicatePolicy`

`write_documents` takes a `policy` argument controlling how id collisions are handled:

```python
from haystack.document_stores.types import DuplicatePolicy

store.write_documents(docs, policy=DuplicatePolicy.FAIL)      # raise if any id collides
store.write_documents(docs, policy=DuplicatePolicy.SKIP)      # silently skip colliding ids
store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE) # remove-then-re-add colliding ids
# DuplicatePolicy.NONE is treated as FAIL.
```

Returns the number of documents actually written (so `SKIP` may return less than `len(docs)`).

## Delete

```python
store.delete_documents(["id-1", "id-2"])     # by id; missing ids are silently ignored
store.delete_by_filter(filters)               # by filter; returns count
store.delete_all_documents()                  # clear everything
```

`delete_documents` and `delete_by_filter` are O(1) per matching document via the inner `IdMapIndex`.

## Filters

`filter_documents(filters)`, `embedding_retrieval(..., filters=...)`, and the other filter-aware helpers accept the full [Haystack filter DSL](https://docs.haystack.deepset.ai/docs/metadata-filtering):

```python
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.source", "operator": "==", "value": "manual"},
        {"field": "meta.version", "operator": ">=", "value": 2},
    ],
}

# All docs matching the filter (no vector search):
docs = store.filter_documents(filters=filters)

# Top-k nearest to a query, filtered:
results = store.embedding_retrieval(
    query_embedding=[...],
    top_k=5,
    filters=filters,
)
```

Filter evaluation is delegated to `haystack.utils.filters.document_matches_filter` — anything Haystack's own stores support, we support.

For `embedding_retrieval`, filters are resolved to an allowlist **before** scoring rather than via post-filtering. Selective filters return up to `top_k` matches from the filtered set; you never get fewer than `top_k` results just because the filter happened to exclude the top-scoring candidates.

## Metadata helpers

```python
store.count_documents_by_filter(filters)                          # int
store.count_unique_metadata_by_filter(filters, ["source", "tag"]) # dict[str, int]
store.update_by_filter(filters, {"reviewed": True})               # bulk metadata update; returns count

store.get_metadata_fields_info()
# {"source": {"type": "keyword"}, "version": {"type": "int"}, ...}

store.get_metadata_field_min_max("version")     # {"min": 1, "max": 5}
store.get_metadata_field_unique_values("source")
# (["a", "b", "c"], 3)
```

`update_by_filter` updates metadata only — embeddings are quantized at write time and not re-encoded.

## Async

Every public method has an `*_async` variant:

```python
await store.write_documents_async(docs)
results = await store.embedding_retrieval_async(query_embedding=q, top_k=5)
await store.delete_documents_async(["id-1"])
```

By default they run on a single-threaded executor owned by the store. Pass an `async_executor=` to the constructor to share an executor across stores (or to use more workers).

## Save / load

```python
store.save_to_disk("./my-store")
# ... later ...
store = TurboQuantDocumentStore.load_from_disk("./my-store")
```

Writes two files under the given folder path:
- `index.tvim` — the `IdMapIndex` payload (quantized vectors + id maps).
- `docstore.json` — JSON-encoded document text, metadata, and id maps.

Document metadata must be JSON-serializable — the same constraint `InMemoryDocumentStore.save_to_disk` imposes. If the `docstore.json` side-car is out of sync with its `index.tvim` (a partial copy, a stale backup, tampering), `load_from_disk` raises a `ValueError` immediately rather than failing later with a `KeyError` at query time.

## Using in a Haystack Pipeline

`TurboQuantDocumentStore` implements `to_dict` / `from_dict` so it can be serialized as part of a Haystack `Pipeline`. `to_dict` captures the component *config* (`dim`, `bit_width`, `embedding_similarity_function`, `return_embedding`); persisting the stored documents is the job of `save_to_disk` / `load_from_disk`.

Plug into a standard RAG pipeline the same way you'd use `InMemoryDocumentStore`:

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

store = TurboQuantDocumentStore()                 # dim inferred from first batch
indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
))
indexing.add_component("writer", DocumentWriter(document_store=store))
indexing.connect("embedder.documents", "writer.documents")

indexing.run({"embedder": {"documents": my_docs}})
```

## Known limitations

- **Embeddings are not retained.** `embedding_retrieval(..., return_embedding=True)` is accepted for signature compatibility but `Document.embedding` is always `None` on retrieved docs — turbovec discards the full-precision vector after quantization.
- **JSON-serializable metadata only.** Document metadata is stored as JSON in the side-car. Non-JSON-serializable values (custom objects, sets, etc.) fail at save time — the same constraint `InMemoryDocumentStore.save_to_disk` imposes.
- **`dim` is locked on the first add.** Subsequent calls with a different shape raise `ValueError`. If you need to change `dim`, construct a fresh store.
