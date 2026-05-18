# Changelog

All notable changes to turbovec are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Rust crate (`turbovec` on crates.io) and the Python distribution
(`turbovec` on PyPI) version independently. Each release section below
is split by surface — a single feature can affect both, and its bullet
appears under each surface it touches.

## [Unreleased]

## turbovec 0.5.0 (Python package) + turbovec 0.4.0 (Rust crate) — 2026-05-18

> **BREAKING** — on-disk file format version bumped from 1 to 2.
> Existing `.tv` and `.tvim` files written by turbovec ≤ 0.4.3 cannot
> be loaded by 0.5.0+. **Reindex from source vectors to migrate;**
> no in-place migration is provided.

### Migration

If you have indexes built with 0.4.3 or earlier, re-encode them:

```python
import numpy as np
from turbovec import TurboQuantIndex

# Source vectors (the f32 inputs your old index was built from).
vectors = np.load("my_vectors.npy")  # shape (n, dim)

# Build a fresh 0.5.0 index. Same API, same recall guarantees, but with
# the new length-renormalization correction applied.
index = TurboQuantIndex(dim=vectors.shape[1], bit_width=4)
index.add(vectors)
index.write("my_index_v2.tv")
```

If you load an old file under 0.5.0+, you will see:

```
this .tv file was written by turbovec ≤ 0.4.3 (format version 1).
It is incompatible with turbovec 0.4.4+ because the per-vector scalar's
meaning changed. Rebuild this index from the source vectors using
turbovec 0.4.4 or later.
```

### turbovec — Rust crate (current: 0.3.0 → next: 0.4.0)

#### Added

- **Length-renormalized scoring.** The per-vector scalar stored in
  `TurboQuantIndex` is now `||v|| / <u_rot, x̂>` instead of `||v||`,
  giving an unbiased estimator of the inner product. The SIMD kernel
  multiplies by this value at the same site it previously used the
  norm — no change to kernel speed, storage layout, or public API.

#### Changed

- **On-disk format version bumped to 2** for both `.tv` and `.tvim`.
  `.tv` now starts with a 4-byte magic `"TVPI"` + 1-byte version
  prefix; `.tvim` keeps its existing magic with version bumped from 1
  to 2. Loading a v1 file returns `io::Error` of kind `InvalidData`
  with an upgrade-hint message; no in-place migration is provided.
- **`TurboQuantIndex::norms` field renamed to `scales`.** Internal
  rename to match the value's new meaning. The SIMD kernel parameter
  is `vec_scales` (to disambiguate from the per-query LUT calibration
  `scales` parameter inside the same functions).

### turbovec — Python package (current: 0.4.3 → next: 0.5.0)

#### Added

- **Length-renormalized scoring.** Replaces the per-vector `||v||`
  scalar with a RaBitQ-style correction `||v|| / <u_rot, x̂>` that
  removes the systematic bias of the inner-product estimator. The
  SIMD kernel is byte-for-byte unchanged — it multiplies by the new
  scalar at the same site it previously used the norm. Recall@1
  gains across published benchmarks:
  - GloVe-200 2-bit:   0.5053 → 0.5524 (+4.7pp)
  - GloVe-200 4-bit:   0.8115 → 0.8440 (+3.3pp)
  - OpenAI-1536 2-bit: 0.8700 → 0.9060 (+3.6pp)
  - OpenAI-1536 4-bit: 0.9550 → 0.9700 (+1.5pp)
  - OpenAI-3072 2-bit: 0.9120 → 0.9240 (+1.2pp)
  - OpenAI-3072 4-bit: 0.9670 → 0.9800 (+1.3pp)

  Same-session ARM and x86 speed benchmarks confirm no measurable
  search-latency change (deltas within FAISS noise floor on every
  cell). The correction adds one extra dot product per vector at
  encode time — a one-shot cost on the cold path, not visible to
  search.

#### Changed

- **On-disk format version bumped to 2** for both `.tv` and `.tvim`.
  `.tv` files now start with a 4-byte magic `"TVPI"` + 1-byte
  version. `.tvim` files use the existing magic with version byte
  bumped from 1 to 2.
- **Loading a turbovec ≤ 0.4.3 index raises with a clear error.**
  The per-vector scalar's meaning changed (`||v||` → `||v|| / <u_rot, x̂>`),
  so silently re-interpreting v1 files would produce wrong scores.
  The new loader detects v1 files by their format signature and
  raises `OSError` pointing the caller at rebuilding from source
  vectors.

#### Fixed

- **`turbovec.haystack.TurboQuantDocumentStore` clamps cosine scores
  to `[-1, 1]` before `scale_score` rescaling.** Cauchy–Schwarz
  bounds the true cosine in that range, but the LUT scoring kernel's
  float-precision noise can produce values slightly outside it —
  most visibly on a self-query, which is algebraically 1.0 but the
  kernel produces ~1.00016 after its per-sub-table calibration.
  Without the clamp, downstream consumers of `scale_score=True` saw
  scores `> 1.0` and the `[0, 1]` contract was violated. Dot-product
  path uses a sigmoid that is already bounded; no clamp needed there.

## turbovec 0.4.3 (Python package) — 2026-05-18

### turbovec — Python package (current: 0.4.2 → next: 0.4.3)

#### Added

- **Windows x64 wheel** (closes [#31](https://github.com/RyanCodrai/turbovec/issues/31)).
  Prior releases shipped only Linux x86_64/aarch64, macOS aarch64, and an
  sdist — Windows users running `pip install turbovec` fell through to
  the sdist and hit a `link.exe` build failure unless they had Rust + MSVC
  installed locally. The release workflow now also builds a
  `cp39-abi3-win_amd64` wheel and validates it by installing and running
  the core pytest suite (`test_index.py`, `test_id_map.py`,
  `test_filtering.py`) on the build runner before upload. Implementation
  in [#33](https://github.com/RyanCodrai/turbovec/pull/33).

  Intel Mac (macOS x86_64) was considered alongside Windows but blocked
  by GitHub's December 2025 deprecation of free-tier `macos-13` runners;
  tracked separately in [#34](https://github.com/RyanCodrai/turbovec/issues/34).

  No library changes in this release — same Python API, same on-disk
  format, same recall and throughput as 0.4.2. Pure platform-coverage
  patch.

## turbovec 0.4.2 (Python package) — 2026-05-17

### turbovec — Python package (current: 0.4.1 → next: 0.4.2)

#### Fixed

- **`numpy` is now a declared runtime dependency.** The Python package
  and every integration module imports `numpy` unconditionally, and the
  Rust extension's Python surface expects NumPy arrays as input. Prior
  releases relied on `numpy` being pulled in transitively via the
  framework extras (`langchain-core`, `llama-index-core`, `haystack-ai`).
  This broke `pip install turbovec[agno]` in clean environments because
  `agno` doesn't depend on `numpy`. `numpy>=1.20` is now declared in
  `[project].dependencies`, so it's installed regardless of which extra
  (or none) is selected.

## turbovec 0.4.1 (Python package) — 2026-05-17

### turbovec — Python package (current: 0.4.0 → next: 0.4.1)

#### Added

- **Agno integration** (`turbovec.agno`). New `TurboQuantVectorDb` class
  implementing Agno's `VectorDb` interface, structurally aligned with
  `agno.vectordb.lancedb.LanceDb` (the closest in-tree single-machine
  backend). Drop-in for callers that use `LanceDb` as their Agno
  knowledge backend.
  - Dim is sourced from `embedder.dimensions` (matches `LanceDb`); no
    baked-in default.
  - Filtered search uses the kernel-level `allowlist=` path: filters
    resolve to a handle allowlist before scoring, so selective filters
    return up to `limit` results from the filtered set instead of
    fewer-than-`limit` from a post-filter.
  - JSON side-car persistence (no pickle, no
    `allow_dangerous_deserialization` flag).
  - Constructor restricts `search_type=vector` and `distance=cosine`
    — turbovec doesn't ship a BM25/lexical index and stores
    unit-normalized vectors only. Non-vector / non-cosine constructions
    raise `ValueError` rather than silently misbehaving.
  - Honours `similarity_threshold` (cosine → relevance clamped to
    `[0, 1]` via `(s + 1) / 2`), `reranker` (optional rerank pass after
    vector retrieval), `content_id` / `content_hash` payload fields.
  - Full async surface: `async_*` variants for create/insert/upsert/
    search/drop/exists/name_exists, using the embedder's async batch
    paths when available.
  - Install: `pip install turbovec[agno]`.

## turbovec 0.3.0 (Rust crate) — 2026-05-17

## turbovec 0.3.0 (Rust crate) — 2026-05-17

### turbovec — Rust crate (current: 0.2.0 → next: 0.3.0)

#### Added

- **Search-time filtering.** New methods restrict the returned top-k to
  a caller-supplied subset of vectors. The kernel applies the filter at
  the heap-update site rather than via post-filtering, so selective
  filters return up to `k` results from the allowed set instead of
  fewer-than-`k` from an over-fetch pass. Output shape shrinks to
  `min(k, n_allowed)` — consistent with the existing `k > len(idx)`
  contract; no sentinel padding.
  ([#21](https://github.com/RyanCodrai/turbovec/issues/21))
  - `TurboQuantIndex::search_with_mask(queries, k, mask: Option<&[bool]>)`
    — slot bitmask, length equal to `len(idx)`.
  - `IdMapIndex::search_with_allowlist(queries, k, allowlist: Option<&[u64]>)`
    — external-id allowlist; translated to a slot bitmask internally
    via the existing `id_to_slot` map. Panics on empty allowlist or
    unknown ids.
  - Threaded through every scoring path: NEON (aarch64), AVX2
    (x86_64), AVX-512BW (x86_64), and the scalar fallback.

- **Lazy index construction.** The dim can now be deferred and inferred
  from the first batch of vectors, rather than committed at construction
  time. This is the same ergonomic improvement integration users were
  already getting through the framework wrappers, pulled down into the
  core so direct Rust users and any future integration get it for free.
  - `TurboQuantIndex::new_lazy(bit_width)` and
    `IdMapIndex::new_lazy(bit_width)` — construct an empty index with
    no committed dim.
  - `TurboQuantIndex::add_2d(vectors, dim)` and
    `IdMapIndex::add_with_ids_2d(vectors, dim, ids)` — add a flat
    vector batch with an explicit dim; locks the index dim on the
    first call, validates on subsequent ones. Existing `add(&[f32])` /
    `add_with_ids(&[f32], &[u64])` still work on a dim-known index and
    panic with a clear message on a lazy uncommitted one.
  - `TurboQuantIndex::dim_opt()` / `IdMapIndex::dim_opt()` return
    `Option<usize>` — `None` for the lazy uncommitted state. The
    existing `dim() -> usize` getters keep returning `usize`, with `0`
    as a non-breaking sentinel for the lazy state (the eager
    constructor asserts `dim >= 8`, so `0` doesn't collide).
  - File format: `.tv` and `.tvim` headers encode the lazy state via
    a `dim = 0` sentinel. Files written before this change always have
    `dim >= 8` and load cleanly into the eager state.

#### Changed

- `search`, `search_with_mask`, and `prepare` on `TurboQuantIndex`
  return empty results / are no-ops when called on a lazy
  uncommitted index, rather than panicking.

## turbovec 0.4.0 (Python package) — 2026-05-17

### turbovec — Python package (current: 0.3.0 → next: 0.4.0)

#### Added

- **Search-time filtering.** Same feature surfaced as keyword-only
  arguments on `search`:
  - `TurboQuantIndex.search(queries, k, *, mask=None)` — `mask` is a
    NumPy `bool` array of shape `(len(idx),)`.
  - `IdMapIndex.search(queries, k, *, allowlist=None)` — `allowlist`
    is a NumPy `uint64` array of external ids.
  - Pre-validates shape, dtype, emptiness and unknown ids and raises
    `ValueError` / `KeyError` rather than letting the Rust panic
    surface as `pyo3.PanicException`.
  ([#21](https://github.com/RyanCodrai/turbovec/issues/21))

- **Lazy construction.** `TurboQuantIndex(dim=None, bit_width=4)` and
  `IdMapIndex(dim=None, bit_width=4)` now accept an optional `dim`.
  When omitted, the dim is inferred from the first `.add(...)` /
  `.add_with_ids(...)` call using the input array's shape. The
  framework integrations all rely on this internally now.
- `.dim` property on both index types now returns `int | None` (was
  `int`); `None` means the index hasn't seen its first add yet.

#### Changed

- **Haystack integration** (`turbovec.haystack`):
  `TurboQuantDocumentStore` is now a structural drop-in for
  `haystack.document_stores.in_memory.InMemoryDocumentStore`. Audited
  against `haystack-ai 2.28.0` and brought up to parity. In addition
  to the earlier filter-resolution fix:
  - `dim` is now optional in the constructor; the index is built
    lazily on the first `write_documents`.
  - Constructor accepts `embedding_similarity_function`
    (`"cosine"` default, since turbovec stores unit-normalized
    vectors), `async_executor`, and `return_embedding` for parity
    with the reference. `scale_score=True` now uses the right
    per-similarity-function formula (`(s + 1) / 2` for cosine,
    `expit(s / 100)` for dot product), fixing a pre-existing bug.
  - 12 `*_async` variants added (`count_documents_async`,
    `filter_documents_async`, `write_documents_async`,
    `delete_documents_async`, `delete_all_documents_async`,
    `update_by_filter_async`, `count_documents_by_filter_async`,
    `count_unique_metadata_by_filter_async`,
    `get_metadata_fields_info_async`, `get_metadata_field_min_max_async`,
    `get_metadata_field_unique_values_async`, `embedding_retrieval_async`).
  - 8 utility methods added (`delete_all_documents`,
    `delete_by_filter`, `update_by_filter`, `count_documents_by_filter`,
    `count_unique_metadata_by_filter`, `get_metadata_fields_info`,
    `get_metadata_field_min_max`, `get_metadata_field_unique_values`),
    plus a `storage` property and `shutdown()`.
  - `write_documents` now validates its input and raises
    `ValueError("Please provide a list of Documents.")` on bad input
    instead of an opaque `AttributeError`.
  - Persistence methods renamed to match the reference:
    `save → save_to_disk`, `load → load_from_disk`. (No deprecation
    shims — pre-this-change persisted stores load fine, but the method
    names change.)

- **LangChain integration** (`turbovec.langchain`):
  `TurboQuantVectorStore` is now a structural drop-in for
  `langchain_core.vectorstores.in_memory.InMemoryVectorStore`. Audited
  against `langchain_core 0.3.63`. In addition to the earlier filter
  fixes:
  - `__init__` no longer requires a pre-built `IdMapIndex`. Lazy
    construction lets `TurboQuantVectorStore(embedding)` work
    directly — same no-arg ergonomics as the reference.
  - `_select_relevance_score_fn` override added — maps the raw cosine
    similarity into `[0, 1]` so `similarity_search_with_relevance_scores`
    and `as_retriever(search_type="similarity_score_threshold")` work.
    Result is clamped to `[0, 1]` to absorb the small overshoot caused
    by quantization noise.
  - `get_by_ids` / `aget_by_ids` implemented from the side-car
    docstore.
  - `add_documents` overrides the base-class default so partial
    `Document.id` is honoured per-document (some ids explicit, others
    UUID-generated) instead of being dropped wholesale.
  - True async overrides: `aadd_documents`, `aadd_texts` and
    `asimilarity_search_with_score` use `aembed_documents` /
    `aembed_query` for genuine async embedding generation;
    `asimilarity_search`, `asimilarity_search_by_vector`,
    `amax_marginal_relevance_search`, `afrom_texts`, `adelete` are
    explicit overrides too.
  - `delete` now returns `None` (was `bool`) and is a no-op when
    called with `ids=None` — matches the reference's contract.
  - `max_marginal_relevance_search` / `_by_vector` /
    `amax_marginal_relevance_search` raise `NotImplementedError` with
    a clear message rather than the base class's bare
    `NotImplementedError`. MMR isn't faithfully implementable on a
    quantized index because the algorithm requires full-precision
    candidate vectors that turbovec discards after encoding.
  - Persistence methods renamed: `save_local → dump`, `load_local →
    load`, matching the reference.

- **LlamaIndex integration** (`turbovec.llama_index`):
  `TurboQuantVectorStore` is now a structural drop-in for
  `llama_index.core.vector_stores.simple.SimpleVectorStore`. Audited
  against `llama_index.core 0.12.39`. In addition to the earlier
  filter fixes:
  - `__init__` no longer requires a pre-built `IdMapIndex`;
    `TurboQuantVectorStore()` works directly. `from_params(dim=None,
    bit_width=4)` is also lazy.
  - `get_nodes(node_ids, filters)` implemented (the reference raises
    NotImplementedError because it doesn't store nodes; we do).
    `clear()` resets state while preserving `bit_width`.
  - `to_dict` / `from_dict` for config round-trip.
  - `get(text_id)` raises `NotImplementedError` with an explanation —
    we can't return the original embedding (quantized away).
  - `delete_nodes(node_ids, filters)` now honours `filters` (previously
    raised). Both constraints intersect when supplied.
  - Async overrides for `async_add`, `adelete`, `adelete_nodes`,
    `aclear`, `aquery`, `aget_nodes`.
  - **StorageContext compatibility**: new
    `from_persist_dir(persist_dir, namespace, fs)` matching the
    reference's namespaced-filename convention, so
    `StorageContext.from_defaults(persist_dir=...)` works. The
    `persist` / `from_persist_path` on-disk layout is now stem-based:
    `persist_path` is a path *stem* and we write `{stem}.tvim` +
    `{stem}.nodes.json` next to each other. This fits StorageContext's
    file-shaped paths and lets multiple namespaced stores share a
    directory.

- **JSON side-cars across all three integrations.** Haystack, LangChain
  and LlamaIndex persistence now writes a plain-JSON side-car next to
  the binary `IdMapIndex` payload instead of a pickle. The
  `allow_dangerous_deserialization` flag is gone everywhere — loading
  is safe regardless of file provenance. Document / node metadata must
  be JSON-serializable, which matches the constraint the reference
  in-tree stores already impose. The side-car carries a
  `schema_version` field; loaders reject unknown versions instead of
  silently misinterpreting bytes.

[Unreleased]: https://github.com/RyanCodrai/turbovec/compare/py-v0.4.2...HEAD
[py-v0.4.2]: https://github.com/RyanCodrai/turbovec/compare/py-v0.4.1...py-v0.4.2
[py-v0.4.1]: https://github.com/RyanCodrai/turbovec/compare/py-v0.4.0...py-v0.4.1
