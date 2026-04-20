# LangChain integration

`turbovec.langchain.TurboQuantVectorStore` is a [LangChain `VectorStore`](https://python.langchain.com/docs/integrations/vectorstores/) backed by an `IdMapIndex`, so it supports O(1) delete by document id and full save/load.

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

`bit_width` is `2` or `4`. `dim` is inferred from the first embedding by default; pass `dim=...` explicitly if starting from an empty store.

## Adding with explicit ids

```python
store.add_texts(
    texts=["a", "b", "c"],
    ids=["doc-a", "doc-b", "doc-c"],
    metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
)
```

If an id is already present, `add_texts` **upserts** — the existing entry is removed and the new one added with the same id. This matches the typical user expectation that re-indexing a document with the same id should replace it, not duplicate it.

## Delete

```python
store.delete(["doc-a", "doc-b"])  # True if all ids were present, False if any was missing
```

Delete is O(1) per id. Passing `ids=None` raises `ValueError` — the LangChain protocol allows `None` to mean "delete all", but we require an explicit list to avoid accidental wipes.

## Save / load

```python
store.save_local("./my-store")
# ... later ...
store = TurboQuantVectorStore.load_local(
    "./my-store",
    embedding=embeddings,
    allow_dangerous_deserialization=True,  # required — side-car is pickle
)
```

Produces two files:
- `index.tvim` — the `IdMapIndex` payload (see [api.md](../api.md#tvim--idmapindex))
- `docstore.pkl` — a pickled dictionary of document text, metadata, and id maps

The `allow_dangerous_deserialization` flag is required because unpickling untrusted data is unsafe. Only load files you produced yourself or trust the source of.

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

## Known limitations

- **No first-class metadata filtering.** LangChain's `VectorStore` interface doesn't standardize filter predicates. If you need filtered search, use the Haystack integration (which exposes a filter DSL).
- **Embeddings are dropped after quantization.** `search` returns `Document` objects with `page_content` and `metadata`, but not the original embedding.
- **Side-car is pickle.** We may migrate to a safer format in a future major version; for now, treat `allow_dangerous_deserialization=True` as mandatory reading.
