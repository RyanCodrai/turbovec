# Changelog

All notable changes to turbovec are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Rust crate (`turbovec`) and the Python distribution (`turbovec` on PyPI)
version independently; entries note which surface a change applies to.

## [Unreleased]

### Added

- **Search-time filtering** on both index types. Pass an id allowlist (or a
  slot bitmask) to `search()` and the kernel honours it directly during the
  top-k heap update — no over-fetch dance, no recall hit on selective filters.
  Output shape shrinks to `min(k, n_allowed)` when the allowed set is smaller
  than `k`. ([#21](https://github.com/RyanCodrai/turbovec/issues/21))
  - Rust: `TurboQuantIndex::search_with_mask(queries, k, Option<&[bool]>)` and
    `IdMapIndex::search_with_allowlist(queries, k, Option<&[u64]>)`.
  - Python: keyword-only `mask=` on `TurboQuantIndex.search` and `allowlist=`
    on `IdMapIndex.search`.
  - Implemented across the NEON, AVX2, AVX-512BW and scalar scoring paths.

### Changed

- **Haystack integration** (`turbovec.haystack`): `embedding_retrieval` now
  resolves `filters` to an allowlist before scoring, replacing the prior
  over-fetch + post-filter pass. Selective filters that previously could
  return fewer than `top_k` documents now return up to `top_k` matches from
  the filtered set.
- **LangChain integration** (`turbovec.langchain`): `similarity_search`,
  `similarity_search_with_score` and `similarity_search_by_vector` now accept
  a `filter` keyword — either a `dict[str, Any]` of metadata key/value pairs
  (AND of equality), or a `Callable[[dict], bool]` predicate over metadata.
  Previously these silently ignored any filter argument.
- **LlamaIndex integration** (`turbovec.llama_index`): `query()` now honours
  `VectorStoreQuery.filters` and `VectorStoreQuery.doc_ids`. Supports
  `MetadataFilters` with the `EQ`, `NE`, `GT`, `LT`, `GTE`, `LTE`, `IN`,
  `NIN`, `TEXT_MATCH`, `CONTAINS` and `IS_EMPTY` operators, and `AND` / `OR`
  conditions (including nested `MetadataFilters`). Unsupported operators
  raise `NotImplementedError` rather than silently mismatching.

[Unreleased]: https://github.com/RyanCodrai/turbovec/compare/v0.2.0...HEAD
