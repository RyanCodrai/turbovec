# Changelog

All notable changes to turbovec are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Rust crate (`turbovec` on crates.io) and the Python distribution
(`turbovec` on PyPI) version independently. Each release section below
is split by surface — a single feature can affect both, and its bullet
appears under each surface it touches.

## [Unreleased]

### turbovec — Rust crate

#### Fixed

- **Index saves are atomic.** `io::write` / `io::write_id_map` (and
  `TurboQuantIndex::write` / `IdMapIndex::write` on top of them) now write
  to a sibling temp file, fsync, and rename over the destination, so a
  failed or interrupted save no longer destroys a previous good index at
  the same path; the TQ+ calibration-length assert also runs before any
  file is created. (#118)
- **Saving an index with ≥ 2³² vectors errors instead of silently
  wrapping.** `write_core` previously truncated `n_vectors` with `as u32`,
  producing a corrupt file that loaded clean with `n mod 2³²` vectors; it
  now returns `InvalidData`. (#119)
- **Float payloads are value-validated on load.** A `.tv` / `.tvim` with a
  non-finite or negative per-vector scale, a non-finite TQ+ shift, or a
  non-positive/non-finite TQ+ scale previously loaded clean and silently
  poisoned search results (NaN/Inf scores, vanishing or always-winning
  slots); such files are now rejected with `InvalidData`. (#122)
- **`search`/`prepare` on an empty index no longer build the dim×dim
  rotation matrix.** Searching an empty index is now O(1), which also stops
  a tiny file declaring a large `dim` with `n_vectors = 0` from driving a
  multi-gigabyte allocation on first search. (#123)
- **The published `.crate` now bundles the MIT LICENSE text.** The
  `LICENSE` file lived only at the repo root, outside the package
  directory, so cargo shipped the SPDX `license = "MIT"` metadata but not
  the notice itself (MIT requires the notice to accompany copies). A copy
  of the license is now committed inside `turbovec/`, which cargo
  auto-includes in the package. (#166)
- **Out-of-distribution vectors no longer explode their correction scale
  under frozen TQ+ calibration.** When a vector added after calibration was
  frozen reconstructed with a near-zero or negative inner product, the
  `inner.max(1e-10)` clamp turned the stored scale into up to ~1e10 (with
  a flipped sign for negative inners), letting that one vector falsely
  dominate every top-k. Reconstructions whose unit-space inner product
  falls at or below 0.1 are now treated as degenerate and store scale 0,
  so the vector scores ~0 and ranks last; any stored scale is thereby
  bounded by 10× the vector's norm. Healthy reconstructions sit well
  above the threshold (measured minima ≥ 0.56 even at 2-bit dim-8), so
  healthy vectors encode bit-identically to before on both the SIMD and
  scalar paths; the zero-vector behavior (scale 0) is unchanged. (#116)
- **`encode::encode` rejects dims that are not a multiple of 8 instead of
  writing out of bounds.** The packed layout allocates `dim / 8` bytes per
  bit-plane, so tail coordinates of a non-multiple-of-8 dim wrote past the
  end of each row — the top bit-plane panicked with an index-out-of-bounds
  and lower planes silently corrupted the next plane's bytes. The dead
  tail branch is removed and `encode` now panics up front with a clear
  message, matching the index-level validation. Unreachable through
  `TurboQuantIndex`, which already validated dim at construction. (#117)
- **A failed `add_2d` length check no longer wedges a lazy index.**
  `add_2d` committed the inferred dim before the `vectors.len() % dim`
  validation panicked, leaving the index with a locked dim and zero
  vectors — a follow-up add with a different dim saw a confusing
  `DimMismatch` instead of a fresh start. The length check now runs before
  the dim commit. (`IdMapIndex::add_with_ids_2d` already validated before
  committing and is unchanged.) (#129)
- **`IdMapIndex::search` rustdoc no longer promises a row stride of `k`.**
  The returned `(scores, ids)` are flattened with a stride of
  `effective_k = min(k, len)` — e.g. 5 vectors, 2 queries, `k = 100`
  returns 10 scores/ids per array, not 200. The doc now states the
  effective-k slicing formula and how callers recover the stride
  (`scores.len() / nq`); a regression test pins the behavior. Doc-only —
  the return shape itself is unchanged. (#120)
- Added missing rustdoc for `TurboQuantIndex` (struct summary) and for
  `SearchResults` — its `scores` / `indices` / `nq` / `k` fields
  (row-major `nq × k` layout, where `k` is the *effective* per-query
  result count) and the `scores_for_query` / `indices_for_query`
  accessors. (#162)

#### Changed

- **`TurboQuantIndex::from_parts` is now a public, validated constructor.**
  **Breaking (Rust crate).** It was `pub(crate)` and enforced its
  invariants with `assert!`; it is now `pub`, returns
  `Result<Self, FromPartsError>`, and checks every structural invariant at
  this single chokepoint — `bit_width ∈ {2,3,4}`, a committed `dim` a
  positive multiple of 8 and `≤ MAX_DIM`, `packed_codes` /`scales`/ TQ+
  array lengths (with the implied packed size computed via checked
  arithmetic, so huge `n_vectors` yields a named error rather than an
  overflow), the lazy-state constraints, and the same value-level checks
  as the file loader (finite non-negative per-vector scales, finite TQ+
  shifts, finite positive TQ+ scales — so an accepted index always
  survives its own `write` → `load` round-trip) — returning a named
  `FromPartsError` instead of panicking. This is the supported low-level
  construction path for embedders that hold an index payload in memory
  (e.g. a database page) and want to skip the `.tv`/`.tvim` file
  round-trip. The paired accessors `packed_codes()`, `scales()`,
  `tqplus_shift()` and `tqplus_scale()` are likewise promoted from
  `pub(crate)` to `pub` so an index round-trips through external storage.
  New public error type `FromPartsError`; `TurboQuantIndex` now derives
  `Debug`. (#141, #142; delivers the low-level API requested in #70)

#### Removed

- **The unchecked low-level kernels are no longer public.**
  **Breaking (Rust crate).** `codebook::codebook`, `encode::encode`,
  `pack::repack` and `search::search` are now `pub(crate)`. They trust
  their caller's invariants with no validation, so on the public surface
  they were a soundness and DoS hazard: `search::search` performed
  out-of-bounds reads / SIGBUS from inconsistent caller lengths — undefined
  behaviour reachable from safe code (#141); `encode`/`repack` panicked
  opaquely on malformed lengths or `bits == 0`, and `codebook` hung on an
  unbounded `2^bits` allocation for `bits` in ~32..63 and produced
  silently-wrong output for `bits ≥ 64` / degenerate `dim` (#142).
  *Migration:* construct through the validated `TurboQuantIndex::from_parts`
  or the high-level `TurboQuantIndex` / `IdMapIndex` types, which establish
  these invariants for you. The `dump_state` dev example, which existed
  only to dump the now-internal `codebook`, was removed with it. (#141, #142)
- Dead `avx2_block_epilogue` in `search.rs` (x86-only, ~190 lines, no
  callers). The live AVX2 epilogue helpers are `avx2_batch_flush_to_fa`
  and `avx2_post_flush_heap_update`; the dead copy's logic had drifted
  from them, so keeping it invited confusion in future kernel edits. No
  behavior change. (#134)
### turbovec — Python package

#### Added

- **`turbovec.__version__`.** The package now exposes the standard
  version attribute (resolved lazily via PEP 562 from the installed
  dist metadata, so `import turbovec` stays sub-millisecond —
  importing `importlib.metadata` eagerly would multiply the import
  time by ~25x). Falls back to `"0.0.0.dev0"` when no dist metadata
  is installed. (#153)

#### Changed

- **Haystack `write_documents` under `FAIL`/`NONE` now partial-writes like
  the reference instead of being atomic.** Previously the whole batch was
  validated up front and a `DuplicateDocumentError` left the store
  untouched. `InMemoryDocumentStore` instead commits each document as it
  iterates and raises on the *first* duplicate, persisting every
  preceding non-duplicate document. Per the maintainer ruling on #167,
  the store now matches that post-exception state exactly: documents are
  validated and committed one at a time in batch order, so after the
  raise everything before the first colliding id is persisted (an
  in-batch repeat keeps its already-committed first instance). Each
  individual document commit remains all-or-nothing, so a document
  failing turbovec's own embedding validation mid-batch persists the
  documents before it and leaves the index and id maps consistent.
  `OVERWRITE`/`SKIP` semantics and all success-path return counts are
  unchanged. (#167)

#### Fixed

- **Agno's `TurboQuantVectorDb` accepts `bit_width=3`.** The constructor
  guard rejected 3 even though the core `IdMapIndex` — and the langchain,
  haystack, and llama_index stores built on it — fully support it; the
  guard now accepts `{2, 3, 4}` and still raises `ValueError` for
  anything else. The integration docs and docstrings that described the
  contract as `{2, 4}` now state `{2, 3, 4}`, matching the core. (#138)
- **LlamaIndex `TurboQuantVectorStore.from_persist_dir(namespace=...)`
  rejects path-traversal namespaces.** `namespace` is composed into the
  side-car filename (`{persist_dir}/{namespace}__vector_store.json`), so a
  value containing a path separator or `..` (or an empty/`.` namespace)
  escaped `persist_dir` and read an arbitrary sibling/parent file. Such a
  namespace now raises `ValueError` naming the offending value, rather than
  silently loading a different store than the caller named. Any other
  namespace (alphanumerics, dash, underscore) is accepted verbatim. This
  deliberately diverges from `SimpleVectorStore`, which does not sanitize
  its namespace, in the safer direction. (#152)
- **Optional-dependency floors now match what the integrations actually
  need.** Three of the four extras declared minimums whose APIs the
  integration code relies on did not exist yet, so `pip install
  turbovec[haystack]`/`[agno]` could resolve to versions that crash on
  import or construction. Empirically verified floors (lowest version
  where the full per-integration test file passes):

  | extra | old floor | new floor |
  |---|---|---|
  | `langchain` | `langchain-core>=0.3` | unchanged (verified honest) |
  | `llama-index` | `llama-index-core>=0.11` | unchanged (see next bullet) |
  | `haystack` | `haystack-ai>=2.0` | `haystack-ai>=2.23.0` |
  | `agno` | `agno>=2.0` | `agno>=2.5.4` |

  haystack-ai below 2.1.0 was unimportable next to our integration, below
  2.16.0 lacked `ByteStream.to_dict`/`from_dict` (blob persistence), and
  below 2.23.0 could not serialize a pipeline containing the store;
  agno below 2.5.4 rejects the `similarity_threshold` kwarg the store
  passes to `VectorDb.__init__` (and below 2.2.0 all of the
  `id`/`name`/`description` kwargs too), so every construction failed.
  (#160)
- **LlamaIndex `ALL`/`ANY` metadata filters no longer crash on
  llama-index-core 0.11.x–0.12.5.** The operator dispatch referenced
  `FilterOperator.TEXT_MATCH_INSENSITIVE` (added in llama-index-core
  0.12.6) unconditionally before the `ALL`/`ANY` branches, so on older
  releases those queries — and the `FilterCondition.NOT` dispatch —
  died with `AttributeError`. The newer enum members are now resolved
  with `getattr` sentinels, keeping the honest `>=0.11` floor while
  still supporting `TEXT_MATCH_INSENSITIVE`/`NOT` where the installed
  version provides them. (#160)
- **Loading a missing index file raises `FileNotFoundError`.**
  `TurboQuantIndex.load` / `IdMapIndex.load` previously raised a bare
  `OSError` for a nonexistent path, so `except FileNotFoundError:`
  never matched — and the LlamaIndex `from_persist_path` was the one
  integration load that surfaced it (the other three open their JSON
  side-car with Python's `open()` first). The binding now maps
  `io::ErrorKind::NotFound` to `FileNotFoundError` and appends the
  offending path to every load error message; an existing-but-corrupt
  file still raises the plain `OSError` family. (#156)
- **An over-large `RAYON_NUM_THREADS` no longer makes the first
  `add`/`search` die with an uncatchable `PanicException`.** A value
  above the OS thread limit (`ulimit -u`) made rayon's lazy global
  thread-pool construction fail (EAGAIN) and panic. The module now
  builds the pool at import time when the variable is set, clamping
  the request to 4x the available parallelism with a `RuntimeWarning`
  naming the variable and the cap. When the variable is unset (or `0`,
  rayon's "auto"), nothing changes — rayon's lazy auto-sized pool is
  preserved exactly, and values at or under the cap keep producing
  byte-identical results. Note: an explicitly-set value above the cap
  that previously happened to work unclamped (e.g. `2000` under a
  permissive OS thread limit) is now clamped too and emits the warning;
  search/save results are byte-identical either way, only the thread
  count changes. (#158)
- **The bindings release the GIL around compute-bound core calls.**
  `TurboQuantIndex` / `IdMapIndex` `search`, `add` / `add_with_ids`,
  `prepare`, `write`, and `load` previously held the GIL for their full
  duration, stalling every other Python thread and serializing
  concurrent searches that the core index explicitly supports. The
  concurrency contract per index object is: reads (`search`, `prepare`,
  `write`, `contains`, `len`) may run in parallel with each other; a
  write (`add`, `add_with_ids`, `remove`, `swap_remove`) blocks until it
  has the index to itself and then succeeds — the same
  serialize-then-succeed outcome writes always had, now enforced by an
  internal reader-writer lock instead of the GIL. Inputs borrowed from
  numpy arrays (queries, vectors, ids, masks, allowlists) are
  snapshotted into owned buffers before the GIL is released, so a
  Python thread mutating an input array mid-call cannot corrupt or
  perturb the running operation. Long calls still cannot be interrupted
  with Ctrl-C mid-operation (signal polling is a separate concern).
  (#121)
- Added the `Operating System :: Microsoft :: Windows` classifier to the
  PyPI metadata — Windows x64 wheels have shipped since 0.4.3 but the OS
  classifiers listed only Linux and macOS. (#143)
- **LangChain: a `None` entry in an explicit `ids` list is replaced with a
  generated UUID** at add time (matching the reference
  `InMemoryVectorStore`), instead of being stored as `None` and silently
  rewritten to the string `"null"` by a `dump`/`load` round-trip. (#124)
- **LangChain: `add_texts` always returns a fresh `list[str]`** — passing
  `ids` as a tuple previously returned the tuple unchanged. (#126)
- **LangChain: a non-dict metadata entry (e.g. `None`) is rejected with a
  `TypeError` naming the bad entry, before any state is touched.**
  Previously the crash was an opaque `'NoneType' object is not iterable`
  raised *after* the vectors had been added to the index, leaving the
  docstore and index desynced in memory — and a subsequent `dump()`
  persisted the corruption. (#139)
- **LangChain: generator / one-shot-iterable inputs are materialized once
  at each entry point** (`add_texts` / `aadd_texts` `metadatas` and `ids`,
  `add_documents` / `aadd_documents`, `from_texts` / `afrom_texts`), and
  `from_texts` tests emptiness via `len()` so a numpy array of texts works.
  Previously such inputs were iterated more than once — drained on the
  first pass — producing misleading length-mismatch / `len()` /
  ambiguous-truth-value errors. `delete` / `adelete` get the same
  treatment: a multi-element numpy array of ids previously crashed on the
  `if not ids:` emptiness test. (#157)


- **LlamaIndex: `NE` / `NIN` metadata filters now match nodes missing the
  filtered key**, mirroring llama-index-core's `build_metadata_filter_fn`.
  Previously such nodes were silently dropped from filtered `query`,
  `get_nodes`, and `delete_nodes` results. (#132)
- **LlamaIndex: `get_nodes(node_ids=...)` returns nodes in requested-id
  order** (previously storage/insertion order), consistent with the
  LangChain integration's `get_by_ids`. The filters-only path keeps
  storage order. (#150)
- **LlamaIndex: `add()` accepts generators and other one-shot iterables.**
  The input was iterated twice, so a generator drained on the first pass
  and crashed with a misleading "expected 2D embedding batch, got 1D";
  it is now materialized once up front, as `async_add` already did.
  (#157, LlamaIndex case)
- **Haystack: `embedding_retrieval(filters={})` no longer raises
  `FilterError`.** An empty filter dict is now treated as "no filter",
  matching the reference `InMemoryDocumentStore` and the store's own
  `filter_documents`. (#131)
- **Haystack: `write_documents` no longer crashes on a `Document` whose
  `meta` is `None`** (off-contract, but `Document(..., meta=None)` keeps
  the `None` as-is); it is coerced to `{}`. (#139)
- **agno: `insert()` / `upsert()` accept numpy-array embeddings** instead
  of raising numpy's truth-value-ambiguous `ValueError`, matching
  LanceDb's tolerance. (#135)
- **agno: a `None`-valued metadata filter no longer matches documents
  missing the key.** `search(filters={"k": None})` returned — and
  `delete_by_metadata({"k": None})` deleted — the entire collection;
  both now require the key to be present and equal, matching LanceDb. (#144)
- **agno: concurrent `async_upsert` calls of the same `content_hash` no
  longer retain stale generations.** The previous generation's handles
  were captured before the awaited embed, so sibling tasks never removed
  each other's rows; `async_upsert` now awaits only the embedding and
  delegates to the sync `upsert` (last-writer-wins, same as sync). (#146)
- **agno: `drop()` on a store with a persistence `path` deletes the
  on-disk artifacts** (`index.tvim` / `docstore.json`), so a later
  `create()` starts empty instead of resurrecting the dropped data.
  Also, `delete_by_content_id(None)` / `update_metadata(None, ...)` are
  now no-ops instead of matching every document stored without a
  `content_id`. (#169)
- **agno: `search()` / `async_search()` deduplicate duplicate-content
  results.** `LanceDb.search` — the drop-in reference — unconditionally
  collapses the final result list by `md5(doc.content)`, keeping the
  first occurrence; turbovec returned every duplicate, handing callers
  (e.g. retrieved-chunks-to-LLM pipelines) duplicated context and a
  different result count. Per the maintainer ruling on the issue, dedup
  now runs as the final search step, after filtering and rerank —
  LanceDb's exact ordering — and, like LanceDb, without over-fetching,
  so a search returns fewer than `limit` documents when duplicate-content
  hits exist. Duplicate-content rows are still stored and individually
  deletable; only search results collapse. (#136)
- **Wrong dtype/ndim array arguments raise a clear `TypeError`** that names
  the argument and states expected vs got (e.g. "vectors must be a 2-D
  float32 array, got 2-D float64") instead of pyo3's opaque "'ndarray'
  object cannot be cast as 'ndarray'". Applies to `vectors`, `queries`,
  `ids`, `mask`, and `allowlist`; wrong dtypes are still rejected, never
  silently converted. (#127)
- **Negative and out-of-`uint64`-range integer arguments follow per-method
  semantics** instead of raising a bare, context-free `OverflowError`:
  `swap_remove` raises the `IndexError` its docstring documents, membership
  checks (`in` / `contains` / `remove`) return `False` for ids that can
  never be present, and `k` / `dim` / `bit_width` raise a `ValueError`
  naming the argument. (#128)
- **Integration load paths now validate the side-car's internal key-sets,
  not just the handle ↔ index bijection.** A desynced side-car (partial
  copy, stale backup, hand edit) previously loaded clean and failed later —
  an opaque `KeyError` mid-query, or silently missing results. All four
  integrations now raise a clean `ValueError` at load instead:
  - **agno**: `_load_from` skipped the index/side-car consistency check the
    other integrations run; a desynced `docstore.json` loaded clean and
    orphaned vectors were silently dropped from search results. It now
    calls the shared handle check. (#115)
  - **LangChain**: `load()` didn't require `docs` and `str_to_u64` to hold
    the same document ids; a `docs` entry missing for a mapped id raised
    `KeyError` inside `similarity_search`. (#125)
  - **LlamaIndex**: `from_persist_path` didn't require `nodes` and
    `node_id_to_u64` to hold the same node ids (nor the id map to be 1:1);
    a missing `nodes` entry raised `KeyError` inside `query()`. (#133)
  - **Haystack**: `load_from_disk` accepted a side-car with duplicate
    document ids, which collapsed the rebuilt id map and left a shadow
    document that was searchable but unreachable by id.
- **Integration saves no longer destroy a previously-good store on
  failure.** All four framework integrations (LangChain `dump`, LlamaIndex
  `persist`, Haystack `save_to_disk`, agno `save`) wrote the index and then
  truncate-and-wrote the JSON side-car in place, so a save that failed
  mid-serialization — e.g. a document whose metadata holds a `set` or
  ndarray — left the destination with a new index and a truncated side-car,
  unloadable and unrecoverable. Saves now serialize the side-car fully in
  memory first (bad metadata raises before any file is touched) and write
  both files via fsynced sibling temp files moved into place with
  `os.replace`, removing the temp files on failure. (#159)

### Benchmarks

- `benchmarks/download_data.py` downloads to a `.tmp` sibling and renames
  into place on success, so an interrupted download no longer leaves a
  partial file at the final path that the existence guard then treats as
  complete. (#140)
- **Misbehaving embedders now raise errors that name the embedder as the
  cause.** (#154) Three error-quality gaps — no data corruption or desync
  in any of them, just opaque errors:
  - LangChain `add_texts` / `aadd_texts` validate that the embedder
    returned one vector per text. A short batch previously surfaced as
    `expected N ids, got M` from the index (or an `IndexError` when the
    batch contained duplicate ids); it now raises
    `embedder returned X vectors for Y texts`. The query path likewise
    rejects `embed_query` returning `None` or a non-1D result with an
    error naming the embedder instead of an opaque PyO3 boundary `TypeError`.
  - Agno `insert` / `async_insert` no longer crash with
    `TypeError: object of type 'NoneType' has no len()` when a batch
    embedder returns `None` instead of the embeddings list — the
    documents are treated as un-embedded and the existing
    `failed to embed N document(s)` error is raised.
  - LangChain, Haystack, and LlamaIndex stores reject batches of empty
    (dim-0) embeddings — shape `(N, 0)` passed the 2D-batch guard and
    died in the index kernel with `vector buffer length 0 not a multiple
    of dim 0`; they now raise an error pointing at the embedder / embed
    model. (Agno already caught this mode via its missing-embedding
    check.)

### Docs

- Agno integration: the "Basic usage" example called
  `Knowledge.load_text(...)`, which no longer exists in current agno
  (2.7.x) and raised `AttributeError` on copy-paste. The example now uses
  `knowledge.add_content(text_content=...)`. (#164)
- `docs/api.md`: two points where the Rust API doesn't mirror the Python
  examples are now called out — a lazy index's first add on the Rust API
  must use `add_2d` / `add_with_ids_2d` (the flat forms require an
  already-committed dim and panic otherwise); and the allowlist result
  width is `min(k, unique ids in allowlist)` — the allowlist is
  deduplicated, so repeated ids don't widen the result. (#168)
- The `[Unreleased]` compare link pointed at the stale `v0.8.1` tag,
  folding the entire 0.9.0 release into "Unreleased"; it now compares
  against `v0.9.0`. (#143)

## turbovec 0.8.0 (Python package) + turbovec 0.9.0 (Rust crate) — 2026-06-10

Security-audit release. Two adversarial audit passes over the core crate,
the Python binding, and the framework integrations, hardening the
untrusted-file load path and the Python API surface and fixing several
data-integrity bugs in the integration wrappers. Resolves #104, #105, and
#106. No on-disk format change (still `.tv` / `.tvim` v3).

A few fixes change observable behavior — see **Changed** under each surface.
They turn previously-undefined or silently-wrong situations into clean,
typed errors, so the bump is minor rather than patch.

### turbovec — Rust crate (current: 0.8.1 → next: 0.9.0)

#### Fixed

- **Untrusted index files are validated before allocation on load.** A
  crafted `.tv` / `.tvim` could previously trigger an integer-overflow in
  the packed-size computation, drive a multi-gigabyte allocation from a
  tiny file, divide-by-zero in the repack step, or load a structurally
  invalid index that returned silently-wrong scores (a `bit_width` of 5–8
  passed the old length check). The loader now range-checks `bit_width` and
  `dim`, computes every size with checked arithmetic, and reads each payload
  through a length-capped reader. (#105)
- **x86 scalar fallback returned wrong results.** On pre-AVX2 x86 (or VMs
  without AVX2), `score_query_into_heap` read the perm0-interleaved code
  layout as if sequential, producing an incorrect top-k. It now
  de-interleaves correctly; verified end-to-end against the SIMD kernels on
  AVX-512 hardware. (#106)

#### Changed

- **`AddError` and `ConstructError` are now `#[non_exhaustive]`.** Downstream
  `match` on these enums must carry a wildcard arm; in exchange, future error
  variants are no longer breaking changes. (The new `DimTooLarge` variant is
  why this release is the moment to make the switch.)
- **`dim` is capped at `MAX_DIM` (65536)** at construction, first add, and
  load. `search` lazily builds a `dim`×`dim` rotation matrix whose size is
  not bounded by any file, so an unbounded `dim` was a load-time
  resource-exhaustion vector. Larger dims now return a typed error.
- **A zero-`dim` lazy add is rejected** with `AddError` instead of panicking
  with a divide-by-zero and wedging the index.

#### Removed

- Dead, untested `pack::repack_3bit` (no callers; 3-bit goes through
  `repack`).

#### Other

- The crate now fails to compile on non-64-bit targets (a `compile_error!`
  gated on `target_pointer_width`). The size/offset arithmetic in
  `encode`/`pack`/`search` assumes 64-bit `usize`; refusing to build on
  32-bit/wasm avoids shipping a silently-unsafe (potential out-of-bounds)
  build there.

### turbovec — Python package (current: 0.7.1 → next: 0.8.0)

#### Fixed

- **`search()` no longer panics on NaN / Inf / oversized query
  coordinates.** These previously raised an uncatchable `PanicException`
  (a `BaseException`); they now raise `ValueError`, matching `add`. (#105)
- **Loading a malformed `.tv` / `.tvim` raises a clean error** instead of
  panicking or driving a huge allocation (the Rust load-path hardening
  above, surfaced through the binding).
- **agno: duplicate derived `doc_id` no longer orphans vectors.** Two
  documents that derive the same id (a repeated `doc.id`, or identical
  content) are both kept and both deletable, matching agno's reference
  store (LanceDb appends). Previously the earlier vector was counted and
  searchable but unreachable by id, and leaked on every upsert. (#104)
- **agno: `delete_by_name` / `delete_by_content_id` / `delete_by_metadata`
  no longer over-delete.** When distinct documents collided on a
  content-derived `doc_id`, deleting by one attribute also deleted the
  id-twin; deletion now targets only the documents matching the predicate.
- **LangChain / Haystack / LlamaIndex: a persisted JSON side-car that is out
  of sync with its `.tvim` index now raises a `ValueError` at load** instead
  of an opaque `KeyError` deep inside a later query.
- **Internal binding result-shape errors map to `RuntimeError`** rather than
  an uncatchable panic.

#### Changed

- `search()` and the index constructors now raise `ValueError` for
  non-finite query values and for `dim` outside the supported range, where
  some of these inputs previously panicked or were silently accepted.

### Docs

- Corrected stale benchmark figures in the README (recall deltas, ARM/x86
  speed ranges) to match the current `benchmarks/results/`; several had
  drifted from before the TQ+ calibration step landed.

## turbovec 0.7.1 (Python package) + turbovec 0.8.1 (Rust crate) — 2026-06-09

Bug-fix release. Two data-safety fixes in the Python integration wrappers'
add/upsert paths, plus a source-build fix for the Python extension on macOS.
The Rust crate is functionally unchanged — only non-behavioral cleanups —
but is re-released to keep crates.io in sync with the source tree. No
on-disk format change (still `.tv` / `.tvim` v3).

### turbovec — Rust crate (current: 0.8.0 → next: 0.8.1)

#### Changed

- Internal cleanup only, **no behavior change** — the published crate
  behaves identically to 0.8.0. Cleared three build warnings (two unused
  bindings in the NEON scoring kernels; the scalar `score_query_into_heap`
  fallback is now `cfg`-gated out of `aarch64` builds, where the NEON
  kernel is always used and it was dead code) and corrected stale SIMD
  module/kernel doc comments.

### turbovec — Python package (current: 0.7.0 → next: 0.7.1)

#### Fixed

- **Intra-batch duplicate ids no longer orphan vectors** in the LangChain
  and Haystack integrations. A repeated id within a single `add_texts` /
  `add_documents` / `write_documents` call previously added one vector per
  row while the id→handle map kept only the last, leaving the earlier
  vectors live in search but mapped to the wrong document and unreachable
  for delete. Both now resolve duplicates the way their reference stores
  do — LangChain (`InMemoryVectorStore`) keeps the last occurrence;
  Haystack (`InMemoryDocumentStore`) applies the `DuplicatePolicy` against
  the batch-so-far. Fixes #90.
- **Upsert no longer destroys existing data when the new batch fails
  validation**, across all four integrations (LangChain, LlamaIndex,
  Haystack, Agno). The old vectors for matching ids were deleted *before*
  the incoming batch was validated/encoded, so a dimension change or a
  non-finite embedding left the store with the originals already gone. The
  delete is now deferred until after the add succeeds (Agno captures the
  previous generation's handles and removes them after `insert`). Fixes
  #89.
- **Plain `cargo build` of the extension now links on macOS.** Building
  `turbovec-python` from source failed with "symbol(s) not found for
  architecture arm64" because nothing emitted the Python extension-module
  linker args (maturin injects them; a bare `cargo build` did not). Added a
  `build.rs` calling `pyo3_build_config::add_extension_module_link_args()`.
  Prebuilt wheels were unaffected. Fixes #92.

## turbovec 0.7.0 (Python package) + turbovec 0.8.0 (Rust crate) — 2026-05-30

Audit-driven correctness pass on every layer (Rust core, Python bindings,
four integration wrappers). Headline: 14 active bugs found and fixed,
hundreds of regression tests added, doc drift cleaned up across the
public API. No on-disk format change (still `.tv` / `.tvim` v3).

### turbovec — Rust crate (current: 0.7.0 → next: 0.8.0)

#### Added

- **`AddError::InvalidInputValue { vector_index, coord_index, value }`** —
  new error variant returned by `TurboQuantIndex::add_2d` and
  `IdMapIndex::add_with_ids_2d` when an input coordinate is non-finite
  (NaN, +Inf, -Inf) or has magnitude `>= 1e16`. Without this validation
  the encode pipeline silently corrupted the index: NaN/Inf propagated
  through `simd_norm` and poisoned `vec_scales[slot] = NaN`, making the
  slot exist in `len()` but unreachable through `search`; huge magnitudes
  overflowed the f32 norm to `+Inf`, making the slot win every query.
- **Scalar fallback in the x86_64 search dispatch.** Previously, `search`
  on an x86_64 CPU without AVX-512 BW or AVX2 silently returned empty
  top-k results for every query (the SIMD `unsafe { if/else if }` block
  had no `else`). Rare in practice on modern hardware but the failure
  mode was the worst kind.

#### Changed

- **Breaking**: `AddError` no longer derives `Eq` (the new
  `InvalidInputValue` variant carries an `f32`, which isn't `Eq` because
  `NaN != NaN`). `PartialEq` is still derived. Downstream code that
  pattern-matches `AddError` exhaustively will need to add the new
  variant.
- `TurboQuantIndex::add` / `add_2d` / `search` / `search_with_mask` now
  reject non-finite / huge-magnitude inputs at entry. `add` and `search`
  panic with a clear message (matching their existing precondition-
  panic style); `add_2d` and `add_with_ids_2d` return
  `Err(InvalidInputValue)` for callers handling untrusted input.
- `TurboQuantIndex::from_parts` asserts structural invariants
  (packed_codes / scales / TQ+ length relationships) at entry, catching
  any future caller that bypasses the read-layer validation.
- Rustdoc on `add`, `add_2d`, `search`, `search_with_mask`, and
  `IdMapIndex::add_with_ids` now documents every panic condition
  introduced by the input validation.

#### Fixed

- **`IdMapIndex::add_with_ids_2d` partial-mutation on inner failure.**
  ID tables (`id_to_slot` / `slot_to_id`) were mutated before delegating
  to the inner `add_2d`. If the inner call returned `Err` (e.g.
  `DimMismatch` on a committed-dim index), the ID tables retained `n`
  ghost entries pointing at slots that didn't exist in the inner index —
  corrupting later `search_with_allowlist` / `remove`. Fixed by capturing
  `base_slot` before, running inner add first, mutating ID tables only
  on success.
- **v2-loaded index + `add` silently mis-encoded new vectors.** Loading a
  pre-TQ+ (v2) file left `tqplus_shift` empty; the next `add` saw
  `existing = None`, fit fresh calibration, encoded the new batch with
  that calibration — but then silently dropped the fitted shift/scale
  because the `n_vectors != 0` else branch only extended `packed_codes`
  / `scales`. The new vectors then got searched against identity
  calibration, producing silently wrong scores. Fixed by populating
  explicit identity TQ+ in `from_parts` when loading a v2-shaped state.
- **Empty first add froze identity calibration forever.** `add(&[])`
  hit the `n < TQPLUS_MIN_SAMPLES` branch in `encode`, returned
  identity, and the `n_vectors == 0` branch wrote it to
  `self.tqplus_shift`. Every subsequent add — even a million-vector
  batch with rich distribution — then saw `existing = Some(identity)`
  and silently skipped fresh fitting. Fixed by short-circuiting `add`
  to a true no-op when `n == 0`.

### turbovec — Python package (current: 0.6.0 → next: 0.7.0)

#### Changed

- **Breaking** (typed-exception hygiene): `TurboQuantIndex.add` /
  `search` and `IdMapIndex.add_with_ids` / `search` now raise
  `ValueError` for non-finite or huge-magnitude coordinates, non-
  contiguous numpy arrays, and wrong-dim queries. Previously these
  surfaced as Rust panics → `PanicException` in Python.
- **Breaking**: `TurboQuantIndex.swap_remove` now raises `IndexError`
  for out-of-range indices (was a Rust panic).
- `IdMapIndex.search` and `TurboQuantIndex.search` now return consistent
  shapes for empty queries — `(0, min(k, n_vectors, n_allowed))` on
  both. Previously `IdMapIndex` returned `(0, k)` (raw `k`), diverging
  from `TurboQuantIndex`'s `(0, min(k, n))`. For `IdMapIndex`, the
  `effective_k` computation also now dedups the allowlist for the
  `nq == 0` path, matching the kernel's mask-based dedup for `nq > 0`.

#### Fixed

- **`turbovec.langchain.TurboQuantVectorStore`**: `similarity_search`,
  `similarity_search_with_score`, and `similarity_search_by_vector` now
  populate `Document.id` on returned hits (was silently `None`). The
  `Document` passed to user-supplied filter callables also carries
  `.id` so predicates can filter on it. Fixes #81.
- **`turbovec.haystack.TurboQuantDocumentStore`**: `Document.blob` and
  `Document.sparse_embedding` now survive write → retrieval round-trip
  (were silently dropped). Docstore schema bumps `v1 → v2` with
  backward-compat load. Filter shape validation tightened to match
  `InMemoryDocumentStore` (bare `{"field": "x"}` shapes are rejected).
  Docstring scoped back from "matches the public surface of
  `InMemoryDocumentStore`" since `bm25_retrieval` is not implemented.
- **`turbovec.llama_index.TurboQuantVectorStore`**: full `BaseNode`
  fidelity through `query` / `get_nodes` / persist+load. PREVIOUS /
  NEXT / PARENT / CHILD relationships, `excluded_embed_metadata_keys` /
  `excluded_llm_metadata_keys`, template fields (`text_template`,
  `metadata_template`, `metadata_separator`), `start_char_idx` /
  `end_char_idx`, and `mimetype` were silently dropped — now preserved
  via `node_to_metadata_dict` / `metadata_dict_to_node`. Nodes schema
  bumps `v1 → v2` with backward-compat load. Plus:
  - `FilterCondition.NOT` now supported (was `NotImplementedError`).
  - `FilterOperator.TEXT_MATCH` is now case-sensitive (matches the
    reference; previously silently lowercased both sides).
  - `FilterOperator.TEXT_MATCH_INSENSITIVE`, `ALL`, `ANY` added.
  - `query.mode != VectorStoreQueryMode.DEFAULT` raises
    `NotImplementedError` instead of silently behaving as DEFAULT.
  - `add()` rejects intra-batch duplicate `node_id`s with a clear
    `ValueError` (previously, the index ended up with both vectors but
    only the last node_id mapped back to one, orphaning the first handle
    and silently returning the second node's payload attached to the
    first node's vector).
- **`turbovec.agno.TurboQuantVectorDb`**: `embedder` is now threaded
  through returned `Document` objects so `doc.embed()` / `doc.async_embed()`
  work on retrieved hits (previously raised "No embedder provided").
  Empty query strings short-circuit to `[]` (matching LanceDb).

## turbovec 0.6.0 (Python package) + turbovec 0.7.0 (Rust crate) — 2026-05-27

### turbovec — Rust crate (current: 0.6.0 → next: 0.7.0)

#### Added

- **TQ+ per-coordinate calibration.** Before the data-oblivious rotation,
  every coordinate is shifted by its empirical 5th percentile and scaled
  so that the 5–95% range maps to `[0, 1]`. The shift/scale pair is
  fit incrementally from the cold-path `add` data, so the index stays
  online — no separate train pass, no rebuilds as the corpus grows.
  At search time, the same affine is applied to the query before the
  rotation. Recall@1 lifts across published cells:
  - GloVe-200 4-bit:   0.8440 → 0.8498 (+0.6pp)
  - OpenAI-1536 2-bit: 0.876  → 0.891  (+1.5pp)
  - OpenAI-1536 4-bit: 0.966  → 0.974  (+0.8pp)
  - OpenAI-3072 2-bit: 0.911  → 0.929  (+1.8pp)
  - OpenAI-3072 4-bit: 0.971  → 0.974  (+0.3pp)

  No public API change — TQ+ is always-on. The cost is one extra pass
  per `add` batch to update the running quantile estimates, paid once
  on the cold path; search latency is essentially unchanged.

- **Cross-arch top-K parity.** The AVX2 and AVX-512 BW kernels now
  produce byte-identical top-K result sets to the NEON kernel for any
  deterministic input. Per-vector f32 scores still differ by ~1e-5
  relative across arches (different SIMD reduction orders), but those
  rank swaps are confined to within-tie vectors and never change set
  membership. Verified via the new `examples/kernel_xtest.rs` smoke
  test (sha256 of sorted-per-query top-K indices matches across all
  three SIMD paths).

#### Changed

- **On-disk format version bumped to 3** for both `.tv` and `.tvim`.
  v3 appends a TQ+ trailer (per-coord shift + scale arrays) after the
  existing scales section. The v3 reader is **backward-compatible**:
  v2 files load with empty TQ+ vectors (identity calibration). Files
  written by 0.7.x cannot be loaded by 0.6.x or older; there's no
  forward-compat shim. Reindexing from source vectors picks up the
  TQ+ recall lift; loading an old v2 file gives you the pre-TQ+
  numbers.

- **x86 LUT-build is no longer data-dependent.** The AVX2 and AVX-512
  BW kernels previously capped `max_lut` at `min(127, 65535 / n_byte_groups)`
  to keep their no-flush u8→i16 accumulators in range — which at
  d=1536/4-bit clamped to 42, and at d=3072/4-bit to 21, opening a
  visible recall gap vs ARM (−1.6pp and −5.5pp respectively). Both
  kernels now batch the inner loop by `FLUSH_EVERY=256` byte-groups
  and run a mini-epilogue (SUB-trick + i16→f32 + fmadd into per-query
  f32 accumulators) at the end of each batch — the same structure
  NEON has used since 0.5.x. `max_lut` is now unconditionally 127 on
  every arch. x86 speed is essentially flat vs the previous release
  (the per-batch flush eliminates the same work from the single final
  epilogue).

### turbovec — Python package (current: 0.5.3 → next: 0.6.0)

#### Added

- **TQ+ per-coordinate calibration.** Same kernel-level change as the
  Rust crate; Python users see no API change. `TurboQuantIndex.add()`
  carries a small extra pass per batch to update the running quantile
  estimates (one-shot cold-path cost; search latency unchanged), and
  `.search()` returns higher recall on the cells listed above. The
  README's "How it works" section documents the calibration step.

#### Changed

- **On-disk format version bumped to 3** for both `.tv` and `.tvim`.
  Same forward-compat policy as the Rust crate: old v2 files load
  fine into 0.6.0+ (with identity calibration), but indexes written
  by 0.6.0+ cannot be loaded by ≤ 0.5.3. Reindex from source vectors
  to pick up the recall lift.

#### Fixed

- **x86/ARM recall parity at d=1536 and d=3072, 4-bit.** Previous
  releases silently produced lower recall on x86 than ARM at high
  dim — most visibly at d=3072/4-bit where x86 measured 0.919 @1 vs
  ARM's 0.974 (−5.5pp). Same fix as the Rust crate (porting the
  ARM-style periodic accumulator flush to AVX2 and AVX-512 BW). x86
  search latency is essentially unchanged.

## turbovec 0.5.3 (Python package) + turbovec 0.6.0 (Rust crate) — 2026-05-25

### turbovec — Rust crate (current: 0.5.0 → next: 0.6.0)

#### Changed

- **BREAKING:** `TurboQuantIndex::new`, `TurboQuantIndex::new_lazy`,
  `IdMapIndex::new`, and `IdMapIndex::new_lazy` now return
  `Result<Self, ConstructError>` instead of panicking on invalid
  input. The new `turbovec::ConstructError` enum covers `bit_width`
  out of `{2, 3, 4}` and `dim` not a positive multiple of 8 (which
  also closes a latent hole where `dim = 0` was silently accepted —
  the previous `dim % 8 == 0` assertion vacuously passed for zero,
  then divided-by-zero on the first `add`).

  Migration: append `?` (or `.unwrap()` in tests/binaries) to
  existing constructor calls. Mirrors the [`AddError`](src/error.rs)
  pattern from the previous release.

- **Encode is 2–3× faster on aarch64.** SIMD-ifies the quantize +
  scale + bit-pack inner loop via NEON (compare against boundaries
  in 8 lanes at a time, weighted horizontal-add for the bit-pack)
  and fuses the three passes so there's no intermediate
  `codes: Vec<u8>` allocation. Rayon parallelises across rows on
  both aarch64 and x86_64; x86_64 keeps the existing scalar inner
  loop. Recall is bit-identical to the previous release at every
  published cell (verified against `benchmarks/suite/recall_*.py`
  on M3 Max). Measured throughput on M2 Pro, single-threaded:
  - d=768, 4-bit: 22.5K → 66.3K vec/sec (2.9×)
  - d=1536, 4-bit: 9.5K → 21.9K vec/sec (2.3×)
  - d=1536, 2-bit: 16.6K → 25.7K vec/sec (1.5×)

- **Codebook is now cached across incremental `add` calls.** The
  Lloyd-Max boundaries and centroids are a deterministic function
  of `(bit_width, dim)`, so recomputing them on every `add` was
  wasted work. They're now stored in `OnceLock` cells (the same
  pattern already used for the rotation matrix) and reused across
  calls. No behaviour change; faster incremental indexing.

### turbovec — Python package (current: 0.5.2 → next: 0.5.3)

#### Fixed

- **Linux wheels now actually import.** Every Linux wheel since
  Linux build support was added had a missing `DT_NEEDED` entry for
  `libopenblas`, so `import turbovec` failed at the dynamic linker
  step with `undefined symbol: cblas_sgemm` — even on systems that
  had OpenBLAS installed. The wheel now declares the dependency
  explicitly, and `auditwheel` bundles a self-contained copy of
  `libopenblas` (plus its `libgfortran` / `libquadmath` runtime
  deps) into `turbovec.libs/`. Linux wheel size grows from ~1.8 MB
  to ~11 MB (aarch64) / ~42 MB (x86_64) as a consequence — the
  bundled OpenBLAS contains kernel variants for many micro-archs
  and dispatches at runtime. The Linux release CI now also runs
  `pytest` against the freshly-built wheel on native runners so
  this class of bug can't ship silently again.

#### Changed

- **`TurboQuantIndex` and `IdMapIndex` constructors raise
  `ValueError` on bad input** (`bit_width` outside `{2, 3, 4}`,
  `dim` not a positive multiple of 8, including the previously
  silently-accepted `dim = 0` case). Previously these surfaced as
  `pyo3_runtime.PanicException`, which subclasses `BaseException`
  and so wasn't caught by `except Exception:` — user code can now
  recover from a configuration error as a normal usage error.

- **Encode (build-time, not query-time) is faster on aarch64.**
  Same kernel-level change as the Rust crate; Python users see no
  API change and bit-identical recall at every published cell.
  Building an index with `TurboQuantIndex.add()` is ~2–3× faster on
  M-series macOS and Linux aarch64. x86_64 sees the Rayon
  parallelism but not the SIMD kernel.

## turbovec 0.5.2 (Python package) + turbovec 0.5.0 (Rust crate) — 2026-05-21

### turbovec — Rust crate (current: 0.4.1 → next: 0.5.0)

#### Changed

- **BREAKING:** `TurboQuantIndex::add_2d`, `IdMapIndex::add_with_ids_2d`,
  and `IdMapIndex::add_with_ids` now return `Result<(), AddError>`
  instead of panicking on invalid input. The new `turbovec::AddError`
  enum covers dim mismatch, `dim % 8 != 0` on lazy-commit, vector
  buffer length not a multiple of `dim`, ids/vectors count mismatch,
  and duplicate ids. The low-level `TurboQuantIndex::add(&[f32])` and
  constructor asserts are unchanged — they still panic, since those
  signal contract violations rather than user-input errors.

  Migration: append `?` (or `.unwrap()` in tests/binaries) to existing
  calls. Match on `AddError` if you need to recover from specific
  failure modes.

### turbovec — Python package (current: 0.5.1 → next: 0.5.2)

#### Changed

- **Dim mismatch on `add` / `add_with_ids` now raises `ValueError`**
  instead of surfacing a `pyo3_runtime.PanicException` with a Rust
  backtrace. The previous `PanicException` subclassed `BaseException`
  and so was not caught by `except Exception:` — user code can now
  recover from a wrong-shape batch as a normal usage error. The same
  applies to duplicate ids and length mismatches on
  `IdMapIndex.add_with_ids`.

## turbovec 0.5.1 (Python package) + turbovec 0.4.1 (Rust crate) — 2026-05-18

### turbovec — Rust crate (current: 0.4.0 → next: 0.4.1)

#### Added

- **Block-level early exit for selective mask searches** (closes
  [#30](https://github.com/RyanCodrai/turbovec/issues/30)). When a
  search is issued with `Some(mask)` the SIMD kernels now check
  whether each 32-vector block contains any allowed slots before
  doing the LUT lookup + popcount + score-decode work for that
  block. If not, the entire block is short-circuited at one
  integer-load + branch per block. The AVX-512BW path additionally
  short-circuits 64-vector pairs at once where possible.

  Measured speedup at 1% selectivity, 100K vectors, d=1536 (mask
  allowing the last 1K slots): **6.4× on ARM (M3 Max), 12.7× on x86
  (Sapphire Rapids c3-standard-8)**. Unmasked search latency is
  unchanged (the guard only fires when a mask is passed).

  Public API: no change to existing surfaces.

- **`turbovec::search::BLOCKS_SKIPPED_BY_MASK`** — atomic counter
  incremented each time a block is short-circuited. Accessors
  `blocks_skipped_by_mask()` and `reset_blocks_skipped_by_mask()`
  are exposed for hybrid-retrieval telemetry. AVX-512BW pair-level
  skips count as 2.

### turbovec — Python package (current: 0.5.0 → next: 0.5.1)

#### Added

- **Block-level early exit for selective `search_with_mask` calls.**
  Same kernel-level change as the Rust crate; Python users see
  identical API and unchanged unmasked latency. Selective masks now
  run substantially faster (≈6–13× at 1% selectivity, scaling with
  index size — larger indices amortize fixed per-query cost more
  and see larger speedups). Closes
  [#30](https://github.com/RyanCodrai/turbovec/issues/30).

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

[Unreleased]: https://github.com/RyanCodrai/turbovec/compare/v0.9.0...HEAD
[py-v0.4.2]: https://github.com/RyanCodrai/turbovec/compare/py-v0.4.1...py-v0.4.2
[py-v0.4.1]: https://github.com/RyanCodrai/turbovec/compare/py-v0.4.0...py-v0.4.1
