# LangChain integration

`turbovec.langchain.TurboQuantVectorStore` is a [LangChain `VectorStore`](https://python.langchain.com/docs/integrations/vectorstores/) backed by an `IdMapIndex`. It implements the same public surface as `langchain_core.vectorstores.in_memory.InMemoryVectorStore` and can be used as a drop-in replacement wherever the in-memory store is used.

## Install

```bash
pip install turbovec[langchain]
```

## Basic usage

```python
from langchain_huggingface import HuggingFaceEmbeddings
from turbovec.langchain import TurboQuantVectorStore

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

store = TurboQuantVectorStore.from_texts(
    texts=["Document 1...", "Document 2...", "Document 3..."],
    embedding=embeddings,
    bit_width=4,
)

retriever = store.as_retriever(search_kwargs={"k": 5})
```

The dimensionality of the underlying quantized index is inferred from the embedding model on the first `add_*` call — no need to specify it up front.

## Construction

```python
# No-arg: lazy. dim is inferred from the first add.
store = TurboQuantVectorStore(embeddings)

# from_texts: same lazy behaviour, plus immediate ingest.
store = TurboQuantVectorStore.from_texts(texts, embeddings, bit_width=4)

# Pre-built index: bring your own IdMapIndex (e.g. one loaded from disk).
from turbovec import IdMapIndex
store = TurboQuantVectorStore(embeddings, index=IdMapIndex(1536, 4))
```

`bit_width` is `2` or `4` and is fixed once the index is created.

## Adding with explicit ids

```python
store.add_texts(
    texts=["a", "b", "c"],
    ids=["doc-a", "doc-b", "doc-c"],
    metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
)

# add_documents honours per-Document.id, falling back to a UUID per
# document if .id is missing — partial ids are not dropped wholesale.
store.add_documents([
    Document(id="explicit", page_content="..."),
    Document(page_content="..."),                  # gets a UUID
])
```

If an id is already present, `add_texts` **upserts** — the existing entry is removed and the new one added with the same id. This matches the typical user expectation that re-indexing a document with the same id should replace it, not duplicate it.

Async equivalents (`aadd_texts`, `aadd_documents`) use the embedding model's `aembed_documents` so they benefit from concurrent embedding generation when the model supports it.

## Search

```python
# By string query (uses the embedding function)
docs = store.similarity_search("what is turbovec?", k=5)

# With scores
docs_and_scores = store.similarity_search_with_score("...", k=5)

# By raw vector
import numpy as np
qvec = np.random.randn(768).astype(np.float32)
qvec /= np.linalg.norm(qvec)
docs = store.similarity_search_by_vector(qvec.tolist(), k=5)
```

Scores are raw inner products. Because vectors are L2-normalized on insert, inner product equals cosine similarity — higher is better, range `[-1, 1]`.

`similarity_search_with_relevance_scores` and `as_retriever(search_type="similarity_score_threshold")` work: the raw cosine is mapped to `[0, 1]` via `(sim + 1) / 2` (clamped to absorb the tiny overshoot caused by quantization noise).

Async equivalents (`asimilarity_search`, `asimilarity_search_with_score`, `asimilarity_search_by_vector`, `aget_by_ids`) are all implemented.

## Filters

`similarity_search`, `similarity_search_with_score`, and `similarity_search_by_vector` all accept a `filter` keyword:

```python
# Dict — AND of exact equality on Document.metadata.
docs = store.similarity_search(
    "query", k=5, filter={"source": "manual", "version": 2},
)

# Callable — predicate over the Document.
docs = store.similarity_search(
    "query", k=5, filter=lambda doc: doc.metadata.get("score", 0) > 0.8,
)
```

The callable form matches the `Callable[[Document], bool]` convention used by `InMemoryVectorStore`, so predicates ported from there work unchanged.

Filters are resolved to an id allowlist **before** scoring; the kernel only ever inserts allowed documents into the per-query heap. You get up to `k` results from the filtered set, never fewer than `k` because the filter happened to exclude the top-scoring candidates.

## Document retrieval by id

```python
docs = store.get_by_ids(["doc-a", "doc-c"])
# Missing ids are silently skipped.
```

`aget_by_ids` is also available.

## Delete

```python
store.delete(["doc-a", "doc-b"])  # missing ids silently skipped, returns None
```

Delete is O(1) per id. `delete(None)` is a no-op (matches the `InMemoryVectorStore` contract).

## Save / load

```python
store.dump("./my-store")
# ... later ...
store = TurboQuantVectorStore.load("./my-store", embedding=embeddings)
```

Writes two files under the given folder path:
- `index.tvim` — the `IdMapIndex` payload (see [api.md](../api.md#tvim--idmapindex)).
- `docstore.json` — JSON-encoded document text, metadata, and id maps.

Document metadata must be JSON-serializable — the same constraint `InMemoryVectorStore.dump` imposes. If the `docstore.json` side-car is out of sync with its `index.tvim` (a partial copy, a stale backup, tampering), `load` raises a `ValueError` immediately rather than failing later with a `KeyError` at query time.

## Known limitations

- **Max-marginal-relevance search is not supported.** `max_marginal_relevance_search` and its variants raise `NotImplementedError` with an explanation. MMR requires the full-precision embedding of each candidate to compute pairwise diversity; turbovec discards full-precision vectors after quantization. If you need MMR, keep a parallel store with the raw embeddings and run MMR over that.
- **Embeddings are not retained.** `search` returns `Document` objects with `page_content` and `metadata`, but the original embedding is not recoverable.
- **JSON-serializable metadata only.** Non-JSON-serializable values (custom objects, sets, etc.) fail at save time — same constraint as the in-tree reference store.
