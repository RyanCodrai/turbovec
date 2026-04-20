# Haystack integration

`turbovec.haystack.TurboQuantDocumentStore` is a [Haystack 2.x `DocumentStore`](https://docs.haystack.deepset.ai/docs/document-store) backed by an `IdMapIndex`. Supports the full mutation lifecycle (`write_documents`, `delete_documents`) and integrates with `InMemoryEmbeddingRetriever`-style pipelines via `embedding_retrieval`.

## Install

```bash
pip install turbovec[haystack]
```

## Basic usage

```python
from haystack import Document
from turbovec.haystack import TurboQuantDocumentStore

store = TurboQuantDocumentStore(dim=1536, bit_width=4)
store.write_documents([
    Document(content="...", embedding=[...], meta={"source": "a"}),
    Document(content="...", embedding=[...], meta={"source": "b"}),
])

results = store.embedding_retrieval(query_embedding=[...], top_k=5)
```

Documents must have pre-computed embeddings — `TurboQuantDocumentStore` doesn't invoke an embedder. Pipe a Haystack embedder component upstream if your documents arrive without embeddings.

## `DuplicatePolicy`

`write_documents` takes a `policy` argument controlling how id collisions are handled:

```python
from haystack.document_stores.types import DuplicatePolicy

store.write_documents(docs, policy=DuplicatePolicy.FAIL)      # default — raise if any id collides
store.write_documents(docs, policy=DuplicatePolicy.SKIP)      # silently skip colliding ids
store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE) # remove-then-re-add colliding ids
# DuplicatePolicy.NONE is treated as FAIL.
```

Returns the number of documents actually written (so `SKIP` may return less than `len(docs)`).

## Delete

```python
store.delete_documents(["id-1", "id-2"])
```

O(1) per id. Missing ids are silently ignored (per Haystack convention).

## Filters

`filter_documents(filters)` and `embedding_retrieval(..., filters=...)` accept the full [Haystack filter DSL](https://docs.haystack.deepset.ai/docs/metadata-filtering):

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

On `embedding_retrieval`, filters are applied **after** vector search. We over-fetch internally to keep the returned top-k at its requested size even with restrictive filters.

## Save / load

```python
store.save("./my-store")
# ... later ...
store = TurboQuantDocumentStore.load(
    "./my-store",
    allow_dangerous_deserialization=True,
)
```

Produces two files:
- `index.tvim` — the `IdMapIndex` payload
- `docstore.pkl` — pickled document text/metadata plus id maps

The `allow_dangerous_deserialization` flag is required because unpickling untrusted data is unsafe.

## Using in a Haystack Pipeline

`TurboQuantDocumentStore` implements `to_dict` / `from_dict` so it can be serialized as part of a Haystack `Pipeline`. `to_dict` captures the component *config* (`dim`, `bit_width`); persisting the stored documents is the job of `save`/`load`.

Plug into a standard RAG pipeline the same way you'd use `InMemoryDocumentStore`:

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

store = TurboQuantDocumentStore(dim=384, bit_width=4)

indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
))
indexing.add_component("writer", DocumentWriter(document_store=store))
indexing.connect("embedder.documents", "writer.documents")

indexing.run({"embedder": {"documents": my_docs}})
```

## Known limitations

- **Embeddings are dropped after quantization.** `embedding_retrieval(..., return_embedding=True)` is accepted for signature compatibility but returns `None` on the `embedding` field of each returned document.
- **`scale_score=True`** uses a sigmoid squash of the inner-product score. If you need `(score + 1) / 2` (Haystack's `InMemoryDocumentStore` default for cosine), post-process on the caller side.
- **Side-car is pickle.** `allow_dangerous_deserialization=True` required on load.
