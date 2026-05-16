# Changelog

All notable changes to turbovec are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Rust crate (`turbovec` on crates.io) and the Python distribution
(`turbovec` on PyPI) version independently. Each release section below
is split by surface — a single feature can affect both, and its bullet
appears under each surface it touches.

## [Unreleased]

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

#### Changed

- **Haystack integration** (`turbovec.haystack`): `embedding_retrieval`
  now resolves `filters` to an allowlist before scoring, replacing the
  prior `top_k * 10` over-fetch + post-filter pass. Selective filters
  that previously could return fewer than `top_k` documents now return
  up to `top_k` matches from the filtered set. The "results may be
  fewer than `top_k` when filtering is restrictive" caveat in the
  docstring is gone.
- **LangChain integration** (`turbovec.langchain`):
  `similarity_search`, `similarity_search_with_score` and
  `similarity_search_by_vector` now accept a `filter` keyword — either
  a `dict[str, Any]` of metadata key/value pairs (AND of equality), or
  a `Callable[[dict], bool]` predicate over metadata. Previously these
  silently ignored any filter argument via `**_`.
- **LlamaIndex integration** (`turbovec.llama_index`): `query()` now
  honours `VectorStoreQuery.filters` and `VectorStoreQuery.doc_ids`.
  Supports `MetadataFilters` with the `EQ`, `NE`, `GT`, `LT`, `GTE`,
  `LTE`, `IN`, `NIN`, `TEXT_MATCH`, `CONTAINS` and `IS_EMPTY` operators,
  and `AND` / `OR` conditions (including nested `MetadataFilters`).
  Unsupported operators raise `NotImplementedError` rather than
  silently mismatching.

[Unreleased]: https://github.com/RyanCodrai/turbovec/compare/v0.2.0...HEAD
