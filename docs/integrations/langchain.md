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

`bit_width` is one of `{2, 3, 4}` and is fixed once the index is created.

## Similarity modes

The `similarity` keyword (on the constructor and `from_texts`/`afrom_texts`) selects how scores are computed. It is fixed for the lifetime of the store:

- **`"cosine"` (default).** Document vectors are L2-normalized before they reach the quantized index and query vectors are normalized before search, so scores are true cosine similarity in `[-1, 1]` and ranking matches `InMemoryVectorStore` regardless of embedding magnitude. Zero vectors are kept as-is and score `0` against everything (matching the reference's behavior).
- **`"dot_product"`.** Vectors are stored and queried raw: scores are raw inner products and ranking is magnitude-aware. The `(sim + 1) / 2` relevance mapping still applies for continuity, but values outside `[-1, 1]` saturate at the clamp — so `score_threshold` retrieval is only meaningful in this mode if your embeddings are unit-normalized upstream.

```python
store = TurboQuantVectorStore(embeddings, similarity="dot_product")
```

The `similarity` keyword is a turbovec extension: `InMemoryVectorStore` computes cosine unconditionally, so code written against the reference behaves identically under the default.

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

Under the default `similarity="cosine"` mode, scores are cosine similarity — higher is better, range `[-1, 1]` — for embeddings of any magnitude (see [Similarity modes](#similarity-modes)).

`similarity_search_with_relevance_scores` and `as_retriever(search_type="similarity_score_threshold")` work: the cosine is mapped to `[0, 1]` via `(sim + 1) / 2` (clamped to absorb the tiny overshoot caused by quantization noise).

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

The similarity mode is recorded in `docstore.json` and restored by `load`. A store folder written before the mode field existed holds raw, unnormalized vectors, so it loads in `"dot_product"` mode — exactly the scoring it was written under — with no migration needed.

Document metadata must be JSON-serializable — the same constraint `InMemoryVectorStore.dump` imposes. If the `docstore.json` side-car is out of sync with its `index.tvim` (a partial copy, a stale backup, tampering), `load` raises a `ValueError` immediately rather than failing later with a `KeyError` at query time.

`dump` is atomic with respect to the destination: both files are written to sibling temp files and moved into place, so a failed dump (e.g. non-JSON-serializable metadata) leaves a store previously saved at the same path intact.

## Thread safety

The store is safe for concurrent multi-threaded use:

- **Reads run concurrently and scale.** `similarity_search*` and `get_by_ids` take no lock; the underlying index releases the GIL during scoring, so independent searches from multiple threads overlap and scale.
- **Writes serialize.** `add_texts` / `add_documents`, `delete`, and `dump` (and their async counterparts) serialize on a per-store lock.
- **A read overlapping a write sees pre- or post-write state** — never a torn one. Under heavy concurrent churn a search may transiently return fewer than `k` results (hits deleted mid-search are skipped).

What the contract does *not* cover:

- **No cross-call atomicity.** A caller-side check-then-act sequence (`get_by_ids` then `delete`, a count then a search) can interleave with other writers. Batch writes are not atomic with respect to readers: a search overlapping an upsert can briefly see a document id under both its old and new entry.
- **`dump` serializes with writes** (so it always snapshots a consistent store); reads may proceed during a dump.
- **The embedder is invoked outside the store's lock** and must be thread-safe itself.
- **Multi-process access is not supported.**

## Known limitations

- **Max-marginal-relevance search is not supported.** `max_marginal_relevance_search` and its variants raise `NotImplementedError` with an explanation. MMR requires the full-precision embedding of each candidate to compute pairwise diversity; turbovec discards full-precision vectors after quantization. If you need MMR, keep a parallel store with the raw embeddings and run MMR over that.
- **Embeddings are not retained.** `search` returns `Document` objects with `page_content` and `metadata`, but the original embedding is not recoverable.
- **JSON-serializable metadata only.** Non-JSON-serializable values (custom objects, sets, etc.) fail at save time — same constraint as the in-tree reference store.
