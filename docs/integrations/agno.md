# Agno integration

`turbovec.agno.TurboQuantVectorDb` is an [Agno](https://github.com/agno-agi/agno) `VectorDb` backed by an `IdMapIndex`. It implements the same public surface as `agno.vectordb.lancedb.LanceDb` (the closest in-tree single-machine backend) so this can be swapped in wherever LanceDb is used.

## Install

```bash
pip install turbovec[agno]
```

## Basic usage

```python
from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from turbovec.agno import TurboQuantVectorDb

vector_db = TurboQuantVectorDb(embedder=OpenAIEmbedder())

knowledge = Knowledge(vector_db=vector_db)
knowledge.add_content(text_content="Turbovec compresses vectors to 4 bits per dimension.")

agent = Agent(knowledge=knowledge)
agent.print_response("What does turbovec do?")
```

## Constructor

```python
TurboQuantVectorDb(
    *,
    id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    embedder: Embedder,                              # required
    bit_width: int = 4,
    search_type: SearchType = SearchType.vector,
    distance: Distance = Distance.cosine,
    reranker: Optional[Reranker] = None,
    path: Optional[str] = None,
)
```

| Parameter | Notes |
|---|---|
| `embedder` | **Required.** Source of truth for the embedding dimension — `embedder.dimensions` sizes the underlying quantized index. |
| `bit_width` | Quantization width per coordinate; one of `{2, 3, 4}`. |
| `search_type` | Only `SearchType.vector` is supported. Constructing with `keyword` or `hybrid` raises `ValueError` (keyword/hybrid would require an external BM25/lexical index that turbovec doesn't ship). |
| `distance` | `Distance.cosine` (default) or `Distance.max_inner_product` — see [Similarity modes](#similarity-modes). `Distance.l2` raises `ValueError`. |
| `similarity_threshold` | Optional. Scores are mapped to relevance `[0, 1]` via `(s + 1) / 2`; results below the threshold are dropped. Meaningful under `Distance.cosine`; dataset-relative under `Distance.max_inner_product` (see below). |
| `reranker` | Optional Agno reranker applied to the result set after vector retrieval. |
| `path` | Optional directory for save/load persistence. When given, `create()` loads existing data from this path if present. |

## Similarity modes

The `distance` parameter selects how scores are computed. It is fixed for the lifetime of the store:

- **`Distance.cosine` (default).** Document embeddings are L2-normalized at insert time and query embeddings at search time, so the kernel's raw score is cosine similarity in `[-1, 1]` for embeddings of any magnitude, and `similarity_threshold` (which compares against `(s + 1) / 2`) behaves as a true `[0, 1]` relevance cutoff. Zero vectors are kept as-is and score `0` against everything.
- **`Distance.max_inner_product`.** Vectors are stored and queried raw: ranking is by raw inner product (magnitude-aware). The `(s + 1) / 2` mapping saturates at 0/1 once raw scores leave `[-1, 1]`, so `similarity_threshold` values are dataset-relative in this mode — calibrate against your embedder, or leave the threshold unset.

## Insert / upsert

`insert` and `upsert` follow the same `(content_hash, documents, filters)` signature as `LanceDb`. The internal `doc_id` is derived as `md5(f"{base_id}_{content_hash}")` where `base_id` is `doc.id` (or `md5(content)` when missing). The contract: the same `(base_id, content_hash)` pair always produces the same internal id, and the same `base_id` with a *different* `content_hash` is treated as a new entry — letting you keep content versions side-by-side.

Because `doc_id` is derived from `base_id` + `content_hash` (not from `name`, `content_id`, or metadata), two documents can collide on the same `doc_id` — a repeated explicit `doc.id`, or two documents with identical content and no id. When that happens **both are stored and both remain individually deletable** — keep-all, matching `LanceDb`'s append-only behavior. (This differs from the LangChain store, which keeps the last write per id.) At query time, duplicate-content hits collapse to a single search result — see [Filtered search](#filtered-search).

```python
from agno.knowledge.document import Document

docs = [Document(id="doc-1", name="paper.pdf", content="...", meta_data={"source": "arxiv"})]
vector_db.insert(content_hash="v1", documents=docs)

# Same doc with a new content_hash → new stored entry.
vector_db.insert(content_hash="v2", documents=docs)
```

Documents without embeddings are embedded via `self.embedder` before insertion. If embedding fails (`get_embedding` returns `None`) the call raises `ValueError` rather than silently dropping the document.

## Filtered search

Filters are resolved to an allowlist **before** scoring — the kernel only ever inserts allowed candidates into the per-query heap. You always get up to `limit` results from the filtered set; no over-fetching, no recall hit on selective filters.

```python
results = vector_db.search(
    "quantum computing applications",
    limit=5,
    filters={"source": "arxiv", "year": 2024},      # AND of exact equality
)
```

Dict filters use AND-of-exact-equality on `Document.meta_data`. List-style `FilterExpr` filters (Agno's structured filter type) are silently ignored, matching `LanceDb`'s behaviour.

Search results are deduplicated by content: after filtering (and reranking, when a reranker is set), hits with identical `content` (keyed by `md5` of the text) collapse to the first occurrence, matching `LanceDb.search`. When duplicate-content documents are stored — e.g. the same text inserted under two `content_hash`es — a search can therefore return fewer than `limit` results; there is no over-fetch to refill the list.

## Existence checks

```python
vector_db.name_exists("paper.pdf")          # bool — by Document.name
vector_db.id_exists("derived-md5-id")        # bool — by the internally-derived id
vector_db.content_hash_exists("v1")          # O(1) — set lookup, not a scan
```

## Delete

```python
vector_db.delete_by_id(derived_id)               # by internal id
vector_db.delete_by_name("paper.pdf")             # by Document.name
vector_db.delete_by_metadata({"source": "web"})   # AND-of-equality on meta_data
vector_db.delete_by_content_id("cid-42")          # by Document.content_id
vector_db.drop()                                  # clear all
vector_db.delete()                                # alias for drop(), returns True
```

Each `delete_by_*` returns `True` iff at least one document was removed. `delete_by_name` / `delete_by_content_id` / `delete_by_metadata` remove only the documents matching that exact predicate, even when other stored documents share the same derived `doc_id`. `delete_by_id` removes every document under that internal id.

## update_metadata

```python
vector_db.update_metadata("cid-42", {"reviewed": True})
```

Merges the given metadata into `meta_data` of every document with the matching `content_id`. Overrides the base class's no-op warning.

## Save / load

```python
vector_db = TurboQuantVectorDb(embedder=embedder, path="./my-store")
vector_db.create()                          # loads from path if existing

# ... insert documents ...

vector_db.save()                            # persists to path
```

Writes two files under the given folder path:
- `index.tvim` — the `IdMapIndex` payload.
- `docstore.json` — JSON-encoded document text, metadata, and id maps.

Document metadata must be JSON-serializable — same constraint Agno's `LanceDb` imposes on its payload column. The side-car carries a `schema_version` field; loaders refuse to deserialize unknown versions, and validate that the side-car's id maps are consistent with the loaded `index.tvim` (a mismatched or out-of-sync pair raises at load rather than failing later at query time).

The similarity mode is recorded in `docstore.json`. Because the store is constructed (with a `distance`) before `create()` loads the files, a recorded mode that conflicts with the constructor's raises `ValueError` — construct with the matching `distance` to load. A save written before the mode field existed holds raw, unnormalized vectors: it loads as `Distance.max_inner_product` — exactly the scoring it was written under — and `self.distance` is updated to reflect that.

`save` is atomic with respect to the destination: both files are written to sibling temp files and moved into place, so a failed save (e.g. non-JSON-serializable metadata) leaves a store previously saved at the same path intact.

The store also supports `pickle` (e.g. for `multiprocessing` workers, provided the embedder/reranker are picklable) and `copy.copy` / `copy.deepcopy` — both copies return a fully independent store (there is no shallow copy that shares the underlying index).

## Async

The lifecycle, write, and read methods have async counterparts: `async_create`, `async_drop`, `async_exists`, `async_name_exists`, `async_get_count`, `async_insert`, `async_upsert`, `async_search`. The remaining methods (the `delete_by_*` family, `update_metadata`, `save`, `id_exists`, `content_hash_exists`, `optimize`) are sync-only. When the embedder exposes `async_get_embedding` / `async_get_embeddings_batch_and_usage`, the async paths use it for genuine async embedding generation.

## Thread safety

The store is safe for concurrent multi-threaded use:

- **Reads run concurrently and scale.** `search`, the existence checks, and `get_count` take no lock; the underlying index releases the GIL during scoring, so independent searches from multiple threads overlap and scale.
- **Writes serialize.** `insert`, `upsert`, the `delete_by_*` family, `update_metadata`, `drop`, and `save` serialize on a per-store lock. The `async_*` variants delegate to the same locked bodies.
- **A read overlapping a write sees pre- or post-write state** — never a torn one. Under heavy concurrent churn a search may transiently return fewer than `limit` results (hits deleted mid-search are skipped).

What the contract does *not* cover:

- **No cross-call atomicity.** A caller-side check-then-act sequence (`id_exists` then `delete_by_id`) can interleave with other writers. Batch writes are not atomic with respect to readers: a search overlapping an `upsert` can briefly see both the old and new generation of a `content_hash`.
- **`save` serializes with writes** (so it always snapshots a consistent store); reads may proceed during a save.
- **The embedder and reranker are invoked outside the store's lock** and must be thread-safe themselves.
- **Multi-process access is not supported.**

## Known limitations

- **Vector search only.** `search_type=SearchType.keyword` and `SearchType.hybrid` are not supported (would require an external BM25 / lexical index). Constructor raises `ValueError` on those.
- **No L2 distance.** `Distance.cosine` and `Distance.max_inner_product` are the supported metrics; `Distance.l2` raises `ValueError` (the underlying kernel scores by inner product).
- **Embeddings are not retained after quantization.** Stored vectors are the quantized form; the original full-precision embedding can't be recovered.
- **JSON-serializable metadata only.** Non-JSON-serializable values fail at `save()` time.
