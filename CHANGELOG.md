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

#### Added

- **`TurboQuantIndex::serialized_len()` (#409).** The exact number of
  bytes `to_bytes()` returns and `write` puts in the file, from the
  index's geometry alone — no serialization, no allocation. Exact, not an
  upper bound, for sizing a buffer, a database column or a quota check
  before paying for the bytes. `to_bytes` uses it to allocate its buffer
  once.

- **`search::blocks_skipped_by_mask()` now returns `Option<u64>` (#368).**
  Counting mask-skipped blocks costs an atomic RMW per skipped block on a
  shared cache line, so it is compiled out unless the new off-by-default
  `mask-skip-counter` feature is enabled (#294). Previously the accessor
  returned a plain `0` in that case, which a telemetry consumer cannot
  distinguish from "no blocks were skipped" — two different facts sharing
  one representation. `None` now means "this build does not count".
  `BLOCKS_SKIPPED_BY_MASK` itself is no longer public for the same reason:
  reading the static directly reproduces the ambiguity the `Option` exists
  to remove. Migration: match on the `Option`; enable `mask-skip-counter`
  if you want the numbers.
- **New off-by-default cargo feature `mask-skip-counter`** — see above.
- **`TurboQuantIndex::try_search` and `TurboQuantIndex::try_search_with_mask`
  return `Result<SearchResults, SearchError>` (#351).** The search path had
  no non-panicking form: a query buffer whose length is not a multiple of
  `dim`, a non-finite or `>= 1e16` coordinate, or a mask sized for a
  different index each aborted the calling thread. All three arrive from
  outside the process in a real service, and the Python binding already
  pre-validated exactly these three and raised `ValueError`, so Rust
  callers were the only ones without a recoverable error. `search` /
  `search_with_mask` are not deprecated, and their signatures, results
  and validation order are unchanged — they now delegate to the checked
  forms and panic with the error's `Display`. **Their panic text did
  change at three of the four sites** (four sites, three conditions —
  the mask-length check has one site for an empty index and one for a
  populated one). Those three were raised by `assert_eq!`, so the
  payload carried an ``assertion `left == right`` prefix plus `left:` /
  `right:` lines; it is now the error message alone. The fourth, the
  non-finite-coordinate panic, is byte-identical — it was always a
  `panic!`. At the two mask sites the message text was already inside
  the old payload, so a `should_panic(expected = "mask length")` still
  matches; the ragged-buffer assert carried no message at all, so its
  old payload and its new one (`query buffer length 65 not a multiple of
  dim 64`) share nothing but the two numbers, and any `expected =` string
  that matched the old one will not match the new.
  Reach for `try_search` when the query vectors are untrusted; keep
  `search` when a malformed query would be a bug in your own code.
- **`turbovec::expected_codebook` and `turbovec::MIN_INPUT_NORM` are public.**
  `expected_codebook` gives callers of the raw `io::*` writers the codebook
  arrays a v6 file must embed; `MIN_INPUT_NORM` documents the norm at or
  below which a vector has no representable direction and is stored with
  scale 0 (#286).
- **`turbovec::set_warning_hook` (and `turbovec::WarningHook`) route the
  library's non-fatal diagnostics (#365, #390).** `set_warning_hook(Some(f))`
  sends them to `f` — forward them into `log`, `tracing`, or whatever the
  embedder actually uses — and `set_warning_hook(Some(|_| {}))` silences
  them. `None` restores the stderr default. There is one such diagnostic
  today: the post-commit durability shortfall from #365.
- **v6 loads reject a file whose embedded codebook is not a valid Lloyd-Max
  codebook for its `(bit_width, dim)` (#320).** A degenerate codebook —
  collapsed or reversed centroids — previously loaded clean and silently
  mis-scored every query. New rejection class for anyone hand-writing files
  through the raw `io::*` writers.
- **Optional fast-durability writes (#274).** `write` stays fully durable
  by default (temp file, fsync, atomic rename, and now a parent-directory
  fsync so the rename itself is on stable storage — closing a gap between
  the documented power-loss guarantee and the implementation). New
  `TurboQuantIndex::write_with_durability` / `IdMapIndex::write_with_durability`
  take an `io::Durability`: `Fast` keeps the temp-file + atomic-rename
  protocol — the destination can never hold a torn index and the previous
  file survives a process crash — but skips fsync (not power-loss-safe;
  documented). Byte-identical output either way. Measured on the 200k ×
  768 4-bit reference workload: x86 386 → 286 ms, ARM 191 → 119 ms.
- **In-memory serialization: `to_bytes` / `from_bytes` on both index
  types, and generic `Read`/`Write` I/O entry points.**
  `TurboQuantIndex::to_bytes` / `IdMapIndex::to_bytes` serialize an
  index to its `.tv` / `.tvim` wire format in memory — byte-identical
  to the file `write(path)` produces — and `from_bytes` mirrors `load`
  with exactly the same validation (version handling, structural and
  value-level checks, the `.tvim` duplicate-id check), so bytes and the
  file they came from load, or fail, identically. `write_to_writer<W:
  Write>` / `load_from_reader<R: Read>` are the generic-sink forms; the
  `io` module gains the matching raw entry points `io::write_to`,
  `io::load_from`, `io::write_id_map_to` and `io::load_id_map_from`.
  `IdMapIndex` now derives `Debug`.
  This delivers the in-memory I/O half of #70 (the `from_parts` half
  landed in #204) and is the substrate for the Python stores' pickle
  support. (#148, #149, #70)
- **Public items added since 0.9.0 that no entry above announces (#344).**
  Each is `pub` and reachable from a downstream crate, so listing them is
  the difference between a documented surface and one a reader has to
  diff for:
  - `io::CodePayload` — the tagged code-bytes type the `io::load*`
    readers now return in place of `Vec<u8>`; see the reader signature
    change under Changed.
  - `TurboQuantIndex::packed_ready` and `IdMapIndex::packed_ready` —
    whether the packed bit-plane rows are materialized. After a v6 load
    they are not, and no mutation materializes them, so this is how a
    caller tells a load-seeded index from a freshly-built one.
  - `search::single_query_parallelizes` and
    `search::SINGLE_QUERY_PARALLEL_MIN_BLOCKS` — the size half of the
    single-query parallel gate. The threshold entry under Changed
    describes moving the constant but never says it became public.
  - `TurboQuantIndex::add_parallelizes` and
    `turbovec::validation_parallelizes` — whether an `add` of `n_rows`,
    or input validation over `len` values, injects rayon work that is
    not proportional to the row count. Bindings that must control which
    pool that work lands in gate on these (#288, #364).
  - `rotation::Rotation` (with `new`, `dim`, `apply`,
    `apply_with_scratch`, `apply_scaled_into`) and `rotation::K` — the
    block-Hadamard rotation itself, which replaced the removed
    `make_rotation_matrix` below. `apply_scaled_into` appears above only
    in a test-hardening note, never as new API.

#### Changed

- **The #383 below-the-table add gate is pinned structurally, not by wall
  clock (#409, #420).** `deferred_adds_below_the_table_do_not_scale_with_n`
  now asserts that the load-time sorted table is byte-identical after the
  adds and that the deferred set grew by exactly the rows added — the
  mechanical statement of "a below-the-table add does not rewrite the
  table". The old form divided per-add time at 200k vectors by per-add
  time at 25k and required the ratio under 3. That passed on main for
  arithmetic rather than for the property: the n-dependent part of a
  deferred add is only ~2 ps per vector (~400 ns at n = 200k), and it was
  being divided by a ~3000 ns constant, so it read as 1.1x. Removing that
  constant (above) left the same slope on a ~350 ns base and the gate
  failed at 4.5x on CI while the path had become several times faster at
  every size measured. A ratio cannot outlive its own denominator; the
  replacement is machine-independent and fails in microseconds. It pins
  both halves of the property: the write side (the table is not
  rewritten) and the read side (the presence check stays a binary
  search, asserted by counting comparisons — a linear scan there is O(n)
  per add while leaving every structural assertion intact).

- **`to_bytes` sizes its buffer up front (#409).** It allocates
  `serialized_len()` bytes once instead of growing from empty, so peak
  live memory while serializing is the payload rather than roughly three
  times it, and the returned `Vec` has no spare capacity. On every
  architecture except x86-64, a warm search cache is written straight
  through: its bytes are already the sequential layout the format
  persists, so no intermediate copy is made. x86-64 still materializes
  one — the native cache is nibble-interleaved there and the
  de-interleave needs a positioned sink to stream, which a bare
  `io::Write` is not; the file writer, which has one, already streams it
  chunk-wise.

- **Building the SIMD-blocked layout allocates a fixed number of buffers,
  independent of index length (#409).** The packed→blocked extraction step
  materialises one flat `n_vectors * n_byte_groups` buffer with a row
  stride instead of a `Vec<Vec<u8>>` (one heap allocation per vector plus
  the outer pointer vector), and the 4 KB per-bit-width extraction table is
  built once per process rather than on every call. Warming a 4096-vector
  index makes 11 allocations where it previously made 4107; the saving is
  proportional to length, and it is paid in full by the single-row `add`
  on a lazily-loaded index, which extracts one row per call. Byte output
  is unchanged on every architecture.

- **A single query enters the fork-safe rayon pool only from 32768 vectors,
  not 8192 (#336).** `search::SINGLE_QUERY_PARALLEL_MIN_BLOCKS` went from
  256 to 1024 blocks — one full `MIN_TILE_BLOCKS` tile, which is the
  granularity at which the batch dispatch itself splits the block axis. At
  256 the gate fired four tile-widths early: the pool `install` handoff was
  larger than the entire scan it was paying for, producing an undocumented
  latency cliff at exactly n = 8192 where a 0.4% larger index made an nq=1
  search several times slower. Measured A/B interleaved (14-core arm64,
  dim=128, k=10, nq=1, inline vs pooled): 0.64x at n=8192, 0.77x at 16384,
  0.98x at 32768, 1.34x at 65536 — inline wins up to the new threshold and
  loses above it. At `RAYON_NUM_THREADS=1` inline is never slower at any
  size. Results are unchanged: both dispatch paths merge in the same
  (score desc, index asc) order, which the existing cross-path equality
  tests pin. Callers who read the constant to size a benchmark or a test
  index will need to re-derive from it rather than hard-code 8192; the
  in-tree tests now do exactly that.
- **The block-axis tile count is one shared function across both
  architectures.** `MIN_TILE_BLOCKS` is hoisted out of the two dispatch
  bodies and the range count comes from a single `n_block_ranges`, which
  clamps an `nq == 1` search that `single_query_parallelizes` reports as
  serial to exactly one range. That clamp is what makes the threshold safe
  to move at all: without it, raising the gate past the tile granularity
  would split the block axis on a call the Python bindings had already
  decided to run outside the fork-safe pool (the #147 invariant).

- **Encoded bytes now have an absolute golden anchor, not just cross-platform
  agreement (#352, #346).** Determinism was previously checked only by the
  `Encode fingerprint agrees across OSes` CI leg, which compares three
  operating systems inside a single locked build — structurally blind to any
  change that moves every platform together. `tests/encode_fingerprint.rs`
  freezes all six fingerprint columns (boundaries, centroids, calibration,
  codes, scales, file) for the six `(dim, bit_width)` cells, so a `statrs`
  bump, a libm change or a retuned reduction order fails loudly instead of
  silently re-encoding every future index. The fixture and hashing moved to
  `tests/common/fingerprint.rs`, shared with `examples/encode_hash`, so the
  anchor and the cross-OS leg cannot drift apart. The two batch-size
  thresholds that decide encoded bytes are pinned alongside it:
  `RECON_TABLE_MIN_ROWS` must *not* change them and `TQPLUS_MIN_SAMPLES`
  must change them at exactly 1000 rows. Only an affirmative
  `TURBOVEC_REFREEZE` value re-freezes — empty, `0`, `false`, `no` and `off`
  compare as usual, so a stray environment variable cannot turn the anchor
  into a silent no-op. No behaviour change.
- **The quantize kernels' f64 reconstruction table is built by a named
  `build_recon_table` instead of an inline closure (#369).** Purely so the
  kernel identity test can call the *production* builder; a test that
  rebuilt the table itself could not see a divergence between the builder
  and the kernels' inline expression. Its entries are held to the kernels'
  inline expression at f64 precision, bit for bit, so a reassociation that
  the f32 packed bytes would round away is still caught.
  `RECON_TABLE_MIN_ROWS` is now a named constant next to
  `KERNEL_USES_RECON_TABLE`, pinned so raising it fails the build rather
  than quietly narrowing the threshold test. Same table, same bytes.
- **`Rotation::apply_scaled_into` — the entry point that produces every
  encoded byte — has direct tests (#372), and the recon-table/inline paths
  are compared against each other rather than only each against the scalar
  reference (#369).** Both were previously asserted only in doc comments.
  Test-only; no behaviour change.
- **`IdMapIndex::remove` updates its tables only after the inner removal
  returns (#380).** Ordering hardening rather than a fix for reachable
  misbehaviour: no unwind is reachable from `remove`, whose slot comes
  from the id table and so is in bounds by construction — the documented
  `idx >= n_vectors` panic in `TurboQuantIndex::swap_remove` cannot fire
  for it. Past that assert, `swap_remove` calls `packed_mut()` only when
  the packed rows are already materialized, so the lazy O(n·dim) rebuild
  never fires from a remove, and the rest is in-bounds indexing and
  allocation-free lane ops. Taking the id out of `id_to_slot` before that
  call was nonetheless the wrong order: were the inner removal ever to
  become fallible, a caught panic would leave the id gone from the map,
  still present in `slot_to_id`, and `slot_to_id` one entry longer than
  the inner index — the vector searchable but unresolvable, with every
  later `remove` computing the swap target off the wrong length. The
  removal now runs first, matching the "index first, then the maps" order
  the Python stores' delete paths use. No behaviour change.

- **x86 search dispatch now tests every CPU feature the kernels declare
  (#291).** The AVX2 gates additionally require FMA and the AVX-512 gates
  additionally require AVX2+FMA, matching what those kernels execute. On a
  CPU advertising AVX2 without FMA (reachable via hypervisor CPU models)
  the previous gates selected a kernel that would SIGILL on first search;
  such hosts now take the next supported path instead.
- **Masked single-query search is block-parallel (#295).** Filtered search
  previously ran serial over blocks regardless of core count. Measured at
  n=400k, d=128, 4-bit: an all-true mask went 2.04 → 0.35 ms multi-threaded
  and 2.04 → 1.11 ms single-threaded. This changes the performance profile
  of the filtered-search path specifically.
- **`IdHasher` mixes the low bits (#311).** Ids that are multiples of 2^32
  — the common `shard << 32 | seq` layout — previously collided into one
  bucket region, making add/lookup/remove quadratic. Measured over 100k
  such ids: add 1017 → 60 ms, lookup 456 → 0.2 ms, remove 448 → 0.5 ms.
  Sequential-id removes cost ~5 → ~11 ns each, the price of mixing.
- **The GIL is released at more binding sites (#288, #289, #319, #321).**
  `remove` / `swap_remove` probes, the deferred id-slot map build, and
  query validation now run detached, so they no longer stall other Python
  threads while a bulk write holds the lock.
- **AVX-512BW paired-block scoring matches NEON in two more geometries
  (#314).** With `n_byte_groups == 1` the kernel previously scored the bias
  alone, and an odd trailing group could be folded into an already-full
  flush batch, diverging from NEON's rounding. Both were unreachable with
  current legal dims.

- **`SearchResults` derives `Debug`, `Clone` and `PartialEq` (#351).** It
  previously implemented nothing at all, on the type every search
  returns. A downstream struct holding one could not `#[derive(Debug)]`,
  `dbg!(results)` did not compile, results could not be cached or cloned,
  and `assert_eq!` in a user's test was unavailable. `Eq`/`Hash` are
  deliberately absent: `scores` holds `f32`.
- **`SearchError` gains three variants and no longer derives `Eq`
  (#351).** `QueryBufferNotMultipleOfDim`, `InvalidQueryValue` and
  `MaskLengthMismatch` are what the new `try_search` returns; the enum is
  `#[non_exhaustive]`, so adding them is not breaking. `Eq` goes because
  `InvalidQueryValue` carries an `f32` — the same reason `AddError` and
  `FromPartsError` do not derive it. `PartialEq` is unchanged and covers
  every comparison the existing variants supported. `SearchError` itself
  is unreleased (it landed in this same section under #318), so no
  published version is affected.
- **Breaking: `IdMapIndex::search_with_allowlist` returns
  `Result<(Vec<f32>, Vec<u64>), SearchError>` (#318).** It previously
  panicked on an empty allowlist and on an allowlist id missing from the
  index. Both are input conditions — allowlists are built from the
  caller's own metadata store, which drifts out of step with the index —
  so in a service they killed the worker instead of returning an empty
  page. They are now the new `SearchError::AllowlistEmpty` and
  `SearchError::UnknownId(u64)` (`#[non_exhaustive]`, like the crate's
  other error enums). The allowlist-free `IdMapIndex::search` is
  unchanged and still returns the tuple directly. Migration: add `?` or
  `.unwrap()` at `search_with_allowlist` call sites. The Python binding
  already raised `ValueError` / `KeyError` for both and is unaffected.
- **`TurboQuantIndex::dim()` / `IdMapIndex::dim()` are deprecated in favour
  of `dim_opt()` (#318).** They still return `usize`, still return the `0`
  sentinel for a lazy index, and still behave exactly as before on a
  committed index — nothing breaks. The deprecation is the signal: `0` is
  only safe for comparisons, but callers do arithmetic with a dim, so
  `buf.len() / idx.dim()` divided by zero and `vec![0.0f32; idx.dim()]`
  silently built a zero-length buffer. `dim_opt() -> Option<usize>` makes
  the uncommitted case impossible to ignore.
- **Stored per-vector scales may differ by ~1 ULP from earlier v5
  builds** for newly encoded vectors: the scale's f64 reconstruction
  inner product now accumulates through four fixed chains instead of
  one serial chain (deterministic, identical across platforms and
  thread counts; packed codes are unchanged and previously written
  files load byte-identical). Recall is unaffected.
- **Codebook boundaries are now the f32 midpoints of the f32 centroids**,
  rather than the f64 midpoints cast once to f32 — which makes the whole
  Lloyd-Max codebook reproducible across platforms, closing the second
  and last open input in the v5 determinism scope (#259 finding 2).
  The cross-OS fingerprint CI leg caught this on its first run: Linux,
  macOS and Windows each produced a *different* codebook, while
  calibration, codes and scales were byte-identical on all three. The
  f64 iteration is not bit-reproducible (`statrs`'s Beta cdf/pdf bottom
  out in `ln`/`exp`, which differ by ~1 ulp between libms, and the
  adaptive-Simpson recursion can branch differently; at 4 bits the loop
  also exhausts `max_iter` without reaching `tol`, so the f64 centroids
  settle only to ~1e-8). Casting a centroid to f32 absorbs all of that —
  measured invariant under pdf perturbations up to 1e-10 relative — but
  the *midpoint* computed in f64 sat a fraction of an f32 ulp from a
  rounding boundary and flipped under a 1e-15 perturbation at every
  (bits, dim) cell tested. Averaging the already-rounded f32 centroids
  removes the knife-edge by construction: f32 add is correctly rounded
  and `* 0.5` is exact. Boundaries move by at most 1 ULP versus earlier
  unreleased builds, so a coordinate sitting exactly on one can change
  code; both formats are unreleased, so no published index is affected.
- **The per-vector norm has one frozen reduction order on every
  architecture** — `c[j % 8] += x*x`, combined
  `((c0+c1)+(c2+c3)) + ((c4+c5)+(c6+c7))`, with separate multiply and
  add rather than an FMA. It was previously two different reductions:
  aarch64 accumulated four chains through `vfmaq_f32` (one rounding
  where the scalar path has two) while everything else summed serially,
  and those disagree in the last ulp. Since `1/||v||` rides the first
  rotation gather, that reached every encoded byte — the remaining
  cross-platform encode input the v5 determinism scope flagged
  (#259 finding 1), now closed by construction rather than by
  observation. **Newly encoded vectors can differ from earlier
  unreleased builds by ~1 ULP in the stored scale**, and at an exact
  boundary tie by one code. Measured recall is unchanged (R@1 and R@4
  identical at d1536 2/4-bit and d3072 4-bit; R@16/R@64 move by
  <3e-4, i.e. a handful of near-ties reordering). Both v5 and v6 are
  unreleased, so no published index is affected.
- `add` on a populated index no longer holds allocation-sized
  intermediates: encode appends in place and reuses a per-index scratch
  buffer. The buffer is retained at the previous call's demand plus half
  again, and only shrunk when its capacity exceeds twice that — so
  repeated, growing and jittering batch sizes keep their warm allocation,
  while a one-shot bulk load has no previous demand and releases outright.
- **`MAX_DIM` lowered from 65536 to 16384.** A loaded `.tv`/`.tvim`
  header declaring a huge `dim` drives allocations (codebook, blocked
  layout, per-query rotate scratch) not bounded by the file's own size,
  so the old cap — documented as the bound that "rejects the
  catastrophic cases" — still permitted a ~16 KB internally-consistent
  file to demand multi-gigabyte buffers at load or first search. 16384
  leaves >4× headroom over the largest embedding dimensions in common
  use (~4096; rare research models reach 8k–12k). The cap is enforced
  identically at construction, first add, and load — any index this
  build can create it can also load back. (#123)

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
- **Save-path performance (#274).** Mutations now maintain the SIMD-blocked
  cache incrementally (only touched blocks recompute), so a mutate-then-save
  no longer pays the full O(n·dim) repack: post-mutation saves dropped from
  1037 → 391 ms (x86) and 495 → 131 ms (ARM), equal to warm saves — also
  resolving the mutate-then-save item tracked in #273. On x86, path writes
  use parallel positioned writes for the codes section (small additional
  win; ARM keeps the streamed writer, where the same technique regresses).
  Temp files now carry a per-process sequence number so concurrent saves to
  one path cannot interleave.
- **File format v6 for `.tv` / `.tvim`: the file *is* the search-ready
  index — loads skip the first-search rebuild entirely (#68).** The code
  payload is now stored in the arch-neutral *sequential blocked* layout
  (32-vector blocks, one code byte per lane) instead of per-vector
  bit-plane rows, and the file embeds the Lloyd-Max codebook (~124 bytes).
  A load seeds the search caches directly: non-x86 consumes the stored
  layout as-is; x86 applies one cheap in-block nibble interleave (a
  threaded SSSE3 kernel with streaming stores and software prefetch —
  ~2 ms for a 77 MB payload vs ~400 ms for the bit-plane repack it
  replaces). Measured cold start (load → first search, 200k × dim 768,
  Apple M-series): 447 ms → 12 ms. At 2- and 4-bit the code payload is a
  permutation of the same bytes (file size unchanged apart from padding
  to whole 32-vector blocks and the ~124-byte codebook); at 3-bit the
  blocked layout stores one code per nibble, growing the code payload by
  ~33% versus the packed rows.
  - **One file.** The derived state lives inside the index — no sidecar
    files, nothing extra to ship, copy, or clean up.
  - **The format adds no platform dependence.** The stored layout and
    embedded codebook are pure functions of the index content: a v6 file
    loaded and re-saved on a different architecture is byte-identical
    (verified ARM → x86 through the SIMD interleave kernels), and readers
    use the writer's codebook instead of recomputing it — removing the
    cross-libm codebook variance from the search path. (Encoding the
    same *raw vectors* on different platforms can still differ per the
    v5 determinism scope below; v6 neither adds to nor removes that.)
  - **v5 files load unchanged.** v5 stored the same codes in a different
    layout, so the v6 loader accepts v5 and converts on load (identical
    search results); re-saving emits v6. Versions ≤ 4 remain refused with
    the rebuild hint. The writer emits v6 only.
  - **Mutations never pull the packed rows back.** In the window after a
    v6 load the blocked cache is authoritative and the packed bit-plane
    rows stay unbuilt: `add` lazy-appends to the blocked cache and
    `swap_remove` patches it with O(dim) lane ops, so
    `TurboQuantIndex::packed_ready()` stays `false` for the index's whole
    lifetime unless something explicitly asks for the packed rows. A
    write serializes straight out of the blocked cache. Measured on a
    dim-64 index: `packed_ready=false` after the load (len 100), still
    `false` after an `add` (len 110) and after a `swap_remove` (len 109),
    and the file written from that mutated index is byte-identical to
    one built from the same content from scratch, with identical search
    results.
  - `TurboQuantIndex::codes_blocked_seq` / `codebook_for_write` expose the
    v6 payload parts for embedders serializing through the raw `io::*`
    writers (whose code-payload parameter is now the blocked layout).
- **x86 insertion is 1.4-3.5x faster again** on top of the pass below
  (#273). The x86 encode path had a NEON-shaped hole in it: the
  Walsh-Hadamard butterfly ran as a radix-2 ladder (9 memory passes over
  the block at dim 1536, where the NEON path had already moved to
  radix-8), the permutation gather was scalar, the bit-packer OR-ed one
  bit at a time into a pre-zeroed row, and the reconstruction operand
  came from a hoisted table that every row streamed in full. All four
  are now closed, plus an AVX-512 butterfly and the L1-sized calibration
  transpose below. Every change is bit-identical — packed codes and
  stored scales are unmoved, enforced against the scalar reference on
  every SIMD path the host can run. Measured on x86 (Cascade Lake,
  interleaved A/B, synthetic 1536/3072-dim corpora; the official cells
  are pending a run on the GCP Sapphire Rapids instance):

  | cell | cold bulk | warm append | single add |
  |---|---|---|---|
  | d1536 2-bit ST | +72% | +2.2x | -66% |
  | d1536 2-bit MT | +53% | +45% | -66% |
  | d1536 4-bit ST | +91% | +3.2x | -74% |
  | d1536 4-bit MT | +67% | +2.6x | -74% |
  | d3072 2-bit ST | +61% | +2.8x | -64% |
  | d3072 2-bit MT | +42% | +2.6x | -64% |
  | d3072 4-bit ST | +75% | +3.5x | -73% |
  | d3072 4-bit MT | +63% | +3.4x | -71% |

  Removal is unchanged: `swap_remove` was already an O(1) swap-and-pop
  and none of this touches it.
- **Insertion and removal are substantially faster** across a ~35-commit
  optimization pass (arm d=1536 2-bit: cold bulk add ~4.7x, warm append
  ~4-6x, single add ~3x, removals ~25% faster; x86 gains larger from a
  lower base). Encode kernels are now SIMD on both aarch64 (NEON) and
  x86_64 (AVX2, runtime-detected with a scalar fallback); packed codes
  are bit-identical across the scalar, NEON, and AVX2 paths, enforced
  by cross-path identity tests.
- **File format v5 for `.tv` / `.tvim`: a deterministic block-Hadamard
  rotation, replacing the dense QR rotation (hard break).** The
  coordinate rotation that every quantized code is encoded through is now
  a globally-permuted block-Hadamard transform at k=2 rounds (ChaCha8-
  seeded ±1 sign flips → per-block normalized Walsh-Hadamard butterfly →
  a global Fisher-Yates permutation applied *before* every Hadamard,
  twice), applied in place with no matrix and no GEMM. Each round is
  permute → sign-flip → block-Hadamard; the leading permutation makes the
  transform **order-invariant**, so importance-ordered embeddings
  (matryoshka/MRL, PCA) are handled the same as any other coordinate
  ordering. The rotation is **bit-for-bit deterministic across platforms,
  CPU architectures, and thread counts** (only integer permutations and
  basic f32 add/sub/scale — no FMA, no reductions, no transcendentals;
  golden-bytes-pinned) — the property the QR rotation lacked (#206): the
  old rotation read the global rayon parallelism and used `faer`'s
  order-dependent parallel Householder reduction plus a transcendental
  sampler, so its output changed with `RAYON_NUM_THREADS` (dim ≥ 1536) and
  between libm implementations (dim ≥ 3072), and the rotate GEMM
  dispatched to a per-OS BLAS backend so the *encoded bytes* differed by
  platform. The new transform removes all three causes; recall is neutral
  versus the QR rotation (measured at dim 768/1000/1536/3072, 2/4-bit,
  including importance-ordered profiles).

  *Determinism scope:* the whole encode pipeline is bit-identical across
  thread counts on a given machine (verified). Full cross-platform byte
  identity is not yet claimed: the per-vector norm uses an FMA on aarch64,
  and the Lloyd-Max codebook is computed at runtime from `statrs` Beta
  cdf/pdf (transcendentals, the same cross-libm class as #206's finding 2)
  and is not golden-pinned. f64→f32 rounding very likely absorbs both, but
  a cross-OS byte-hash CI leg (see the recommended follow-up) is what would
  prove it.
  - **Hard break.** The rotation change rewrites every encoded byte, so
    v5 is not backward compatible. The writer emits version 5 only; the
    loader accepts version 5 only and refuses any version 1–4 index with
    a clean, actionable `InvalidData` error — *"format version N …
    incompatible with the … v5 rotation … rebuild the index"* — never a
    silent mis-decode and never a panic. There is no in-place migration;
    rebuild from the source vectors. (Format v4 — a rotation-drift
    fingerprint — was never released; it is superseded by v5. The v5
    rotation is deterministic, so no drift fingerprint is needed and the
    v4 header field is dropped.)
  - **64-bit `n_vectors`.** The count field is a u64, so indexes with
    ≥ 2³² vectors serialize exactly instead of erroring at the v3 u32
    ceiling. The in-memory top-k heap index slots widen in lockstep
    (u32 → u64) so results above slot 2³² − 1 cannot truncate. (#119)

  The ChaCha8 seed is frozen and pinned (`rand_chacha` is depended on at
  an exact version) with a golden-bytes test guarding the stream, so a
  future dependency release cannot silently change the wire format.
  Version-5 files are **not readable by earlier turbovec releases**:
  their loaders reject the version byte with a clean "unsupported format
  version" error (no silent misparse). (#206)
- **Breaking (Rust crate): the raw `io::*` readers return an
  `io::CodePayload` where they returned `Vec<u8>` (#344).** The v6 entry
  above records this for the *writers* — "whose code-payload parameter is
  now the blocked layout" — but the readers changed too and were never
  mentioned. `io::load` and `io::load_id_map` now yield
  `(.., CodePayload, ..)`; at 0.9.0 the same slot was `Vec<u8>`. Any
  embedder deserializing through those two entry points fails to
  compile. The new `io::load_from` / `io::load_id_map_from` readers
  added in this release (see above) yield `CodePayload` too, but have no
  0.9.0 form to break.
  *Migration:* match the payload instead of using it directly —
  `CodePayload::Packed(codes)` is the old `Vec<u8>` of per-vector
  bit-plane rows (v5 files), `CodePayload::BlockedSeq { codes,
  boundaries, centroids }` is the v6 sequential blocked layout exactly as
  stored plus the file's embedded Lloyd-Max codebook, and
  `CodePayload::BlockedNative { .. }` is the same codes already
  transformed into this platform's kernel layout. Callers with no reason
  to touch the payload should use `TurboQuantIndex::load` /
  `from_bytes` (and the `IdMapIndex` pair), which take no payload
  argument and are unaffected.

#### Removed


- **The OpenBLAS / Accelerate dependency (and `faer`, `ndarray`,
  `rand_distr`).** The only use of a BLAS backend was the rotation GEMM;
  the v5 block-Hadamard rotation is applied in place with no matrix
  multiply, so the native BLAS link, the `build.rs` link-directive
  shim, and the `faer` / `ndarray` (blas feature) / `rand_distr`
  dependencies are all gone. The crate now builds with a plain `cargo
  build` and no native toolchain, which removes most of what took the
  Linux x86_64 wheel from ~1.8 MB to ~42 MB. (#206)

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
- **`rotation::make_rotation_matrix` (#344).** **Breaking (Rust crate).**
  It was `pub` in the `pub mod rotation` at 0.9.0 and returned the dense
  `dim`×`dim` rotation as a `Vec<f32>`. The v5 block-Hadamard rotation
  (see Changed) applies its transform in place and never materializes a
  matrix, so there is nothing left for the function to return.
  *Migration:* there is no drop-in replacement, and the substitute is not
  the same rotation — code that reproduced turbovec's encoding externally
  must switch transforms rather than translate. Use the public
  `rotation::Rotation`: `Rotation::new(dim)` then `apply` /
  `apply_with_scratch` / `apply_scaled_into`, which is the transform the
  encoder itself uses.

#### Fixed

- **An empty query batch no longer panics with a divide-by-zero (#349).**
  The batch dispatch splits the block axis into
  `(n_threads * 4).div_ceil(n_quads)` ranges, where
  `n_quads = nq.div_ceil(QBS)` — zero when `nq == 0`, so `search(&[], k)`
  aborted the calling thread with `attempt to divide by zero`. It hit at
  every index size and on both index types, `search`, `search_with_mask`,
  `IdMapIndex::search` and `search_with_allowlist` alike, whenever the
  search ran on a rayon pool with more than one thread; a single-threaded
  pool returns before the division. `n_quads` is now clamped to 1 at both
  batch dispatches. The tile loop is empty at `nq == 0` either way, so the
  merge yields the same empty result. An empty batch stays a legal no-op
  returning an empty `SearchResults` rather than becoming a
  `SearchError`: it is a routine input — a filter that matched nothing,
  an empty request page — and it already returned empty results wherever
  it did not panic.
- **The public Rust surface is fully documented, and two more panics have
  a `# Panics` heading (#324).** `RUSTFLAGS="-W missing_docs" cargo build
  -p turbovec` reported 55 warnings and now reports 0: the `AddError` and
  `ConstructError` enums themselves, every named field of every
  struct-variant in `AddError` / `SearchError` / `FromPartsError`, the
  `io::Durability` variants, the `io::CodePayload` payload fields, and
  `len` / `is_empty` / `bit_width` on both index types.
  `TurboQuantIndex::swap_remove` (panics when `idx >= len()`) and
  `IdMapIndex::add_with_ids` (panics on a lazy index, where there is no
  dim to split the buffer by) stated their panic in trailing prose, so
  rustdoc rendered no Panics section for either.
- **`search::single_query_parallelizes` no longer claims to be "the
  single source of truth for the gate" (#324).** It is the size half, and
  the whole gate only on aarch64; the x86 dispatch additionally requires
  runtime AVX2+FMA (or AVX-512) and, without it, runs an nq=1 scan
  serially at a size the predicate calls parallel. What the predicate
  really guarantees is one-directional — `false` means the core never
  splits the block axis, on every target — and that is the direction the
  Python bindings' pool routing depends on. The doc now says so. It also
  says how the predicate is actually reached: neither dispatch calls it
  directly — each re-tests the constant inline, and nothing makes those
  inline conditions agree with it — but a single query sent down the
  batch path meets it again inside `n_block_ranges`, whose `nq == 1`
  clamp pins the block-range count at 1. That clamp is a drift guard,
  inert while `SINGLE_QUERY_PARALLEL_MIN_BLOCKS` and `MIN_TILE_BLOCKS`
  are equal (both 1024), since the tile-granularity term already pins
  the count at 1 on its own.
- **Two `no_run` doctests now execute (#324).** The `id_map` module
  header and the `TurboQuantIndex::from_parts` example touch no
  filesystem, so `no_run` bought nothing and their `assert_eq!`s never
  ran. `cargo test -p turbovec --doc` still runs 4 tests, but only 1 is
  now compile-only instead of 3 — 3 execute where 1 did, in ~1.1 s. The
  crate-header example keeps `no_run`: its point is
  `write("index.tv")` / `load("index.tv")`, which would drop a file in
  the test's working directory.
- **`IdMapIndex` id lookups stay flat for composite ids at every shift
  width, not just up to 32 (#385).** The id hasher's finalizer was a
  single `z ^ (z >> 32)`. `id = i << s` zeroes the low `s` bits of the
  Fibonacci product, and for `s > 32` bits `32..s` are zero too, so the
  single fold laid zeroes over the low `s - 32` bits — exactly the bits
  hashbrown uses as the bucket index. `shard << 48 | seq` ids therefore
  landed in one bucket at every table size and the map degraded to a
  linear scan, the same failure mode #311 repaired for `s <= 32`. The
  finalizer now runs two splitmix-style rounds, so the second multiply
  re-spreads the folded-in entropy before the final fold. Measured on
  60k `i << 48` ids: `remove` went from ~5.2 µs to ~18 ns each. Hash
  values change, so iteration order over `IdMapIndex`'s internal maps
  changes — it was never ordered, and no API exposes it. Encoded bytes,
  search results and file formats are unaffected.
- **`IdMapIndex::search_with_allowlist` reports every condition its error
  type declares (#412).** The method returns
  `Result<_, SearchError>`, but the two query-shape conditions —
  `QueryBufferNotMultipleOfDim` and `InvalidQueryValue` — escaped as
  panics from the inner index instead of being returned, even though
  `SearchError` carries a variant for each. A service that matched on the
  error and mapped it to a 400 still lost the request thread to a ragged
  body. Both now arrive as `Err`, with and without an allowlist. The
  panicking sibling `IdMapIndex::search` is unchanged in behaviour: it
  re-panics with the error's `Display`, which is the same message it
  raised before, and it now carries a `# Panics` section naming both
  conditions. The `SearchError` variant table records `yes` for the pair
  where it previously read `no (panics)`.
- **`io::write_to` and the other raw `write*` entry points reject a
  `bit_width` too large to describe a codebook, identically in every
  build profile (#411).** `assert_codebook_lengths` computed
  `1usize << bit_width` unguarded, so at 64 and above debug panicked
  `attempt to shift left with overflow` — naming neither the argument nor
  the function — while release masked the shift to `<< 0` and carried on
  with one level, which is satisfiable: a caller passing one centroid and
  no boundaries got `Ok(())` and a 26-byte file whose header no reader
  accepts. One actionable message now covers both profiles. The bound is
  the shift, not the format's 2..=4: widths below 64 but outside that
  range still write and are still refused by the load-side header check,
  unchanged. The `# Panics` sections on the six `write*` entry points now
  state the `bit_width` bound alongside the slice-length invariants, so
  they remain an exhaustive list.
- **The reconstruction arithmetic the quantize kernels share with the
  hoisted table is defined once (#410).** `build_recon_table` and the
  scalar and aarch64 kernels each spelled out `centroid * inv_scale -
  shift` in f64 separately, pinned only by a test that compared the
  builder against a hand-copied transcription. That pinned the builder,
  not the kernels: reassociating a kernel's inline branch left the whole
  suite green while roughly a third of the reconstructions diverged from
  the table, because the only cross-path test compares f32 outputs and
  absorbs a sub-f32 difference. All three now call one `recon_entry`
  helper, making the bit-identity structural; only the AVX2 packed form
  stays hand-mirrored, backstopped by the existing avx2-vs-scalar
  assertion. No encoded byte changes — the encode fingerprint is
  unmoved, aarch64 machine code is instruction-for-instruction identical,
  and the x86_64 instruction multiset is unchanged but for two fewer
  `xorps`.
- **`RECON_TABLE_MIN_ROWS` is exported as `#[doc(hidden)]` so its test
  derives the threshold instead of copying it (#410).** The end-to-end
  test that drives the table/inline switch kept its own `THRESHOLD = 16`,
  guarded from the crate side by a compile-time assertion. That guard ran
  one way only — it caught the constant moving out from under the copy,
  but lowering the copy compiled clean and quietly put both batch depths
  below the real threshold, leaving the test comparing the inline path
  against itself. The copy and the guard are both gone.
- **An `add` that crosses the 1000-vector threshold fits a real TQ+
  calibration even when every earlier row has been removed (#360, #366).**
  A sub-threshold `add` commits an explicit *identity* calibration for the
  rows it stores, and `swap_remove`-ing all of them leaves that identity
  committed beside an empty warm-up buffer. The crossing add then had no
  buffered rows to re-encode, so it took the plain bulk-add path, where
  `encode` saw a committed calibration and reused it — the index was
  frozen to identity for the rest of its life, at reduced recall, while
  `calibration_state()` still reported the recoverable `WarmingUp`. An
  empty buffer means no stored rows, so the committed identity describes
  nothing and is now discarded before the batch is encoded. Draining a
  **fitted** index to zero still keeps its calibration (#284) —
  unchanged.
- **`TurboQuantIndex::write` / `to_bytes` and the `IdMapIndex` pair
  document the warm-up forfeit (#361, #366).** The format carries no
  warm-up buffer, so serializing an index that is still `WarmingUp`
  commits the *reloaded* copy to `Identity` calibration for good; the
  original is unaffected. Only the `CalibrationState::Identity` enum doc
  said so, and `to_bytes` is what a clone-by-round-trip goes through.
- **`IdMapIndex::prepare()` now warms the lazy id → slot map (#348).** It
  only forwarded to `inner.prepare()`, so `id_to_slot` stayed unbuilt and
  the first `search_with_allowlist`, `contains` or `remove` after a load
  still paid the O(n) build the method exists to absorb — measured 2.58 ms
  for the first allowlist search vs 0.73 ms warm on a 500k index, while
  `prepare()` itself returned in 0.01 ms. Materializing the map also
  releases the load-time `sorted_ids`/`deferred_added` side-tables, i.e.
  `prepare()` now reaches exactly the steady state a first allowlist
  search would have reached. Still idempotent and O(1) once warm.
- **Every raw `io::write*` entry point rejects a code or scale buffer that
  disagrees with the header it is written under (#407).** `scales.len()`
  must equal `n_vectors`, and `codes_blocked_seq.len()` must be the
  blocked-layout size `(bit_width, dim, n_vectors)` implies — `n_vectors`
  rounded up to whole 32-vector blocks times `dim / (8 / bit_width)`
  bytes. These are the two conditions `TurboQuantIndex::from_parts`
  already returns `PackedCodesLengthMismatch` / `ScalesLengthMismatch`
  for, so both entry points to the format now agree on what a valid index
  is. A violation panics, alongside the existing TQ+-calibration,
  codebook-length and `slot_to_id`-length invariants and for the same
  reason — the `io::Result` reports what happened to the sink, not a
  caller-assembled shape — and it panics before anything is written, so
  an existing index at the destination is never truncated or replaced.
  Both sections are sized from the header on load, never from a length
  prefix, so an inconsistent buffer previously produced not a rejected
  file but an undefined one: a 16-byte-short codes buffer on a 16×64
  4-bit index wrote a file that loaded clean and returned a top score of
  1.0073 against unit-norm rows, above the cosine ceiling, and a
  compensating pair that preserves the total byte count shifts nothing
  downstream, so no header-derived check can fire on it at all. Writers
  that take these buffers from `codes_blocked_seq()` / `scales()` on a
  real index, and every path through `TurboQuantIndex::write` /
  `to_bytes`, are unaffected. Widths outside 2..=4 skip the codes check
  and are still refused by header validation on load (#411).
- **The codebook and accepted-codebook memos no longer take a blocking
  lock on the load path (#390).** Both memos added with the load-time
  codebook validation were `Mutex`-guarded and taken with `lock()`. `fork`
  clones only the calling thread, so a child inherits every mutex in the
  state it had at the fork — and one held by a thread the child does not
  have is never unlocked. Both memos sit on the **load** path, which is
  the first thing a forked worker touches, so a fork landing in that
  window left the child hanging on its first `load` with no error: the
  #147/#288/#321/#364 failure mode in a new place. Both are now taken with
  `try_lock`, which cannot block; a lock that cannot be taken is just a
  memo miss, and a miss is only ever slower, never wrong, because the
  memoised values are pure functions of `(bit_width, dim)`. The
  stale-temp sweep's `SWEPT` set, the same shape on the save path, is
  `try_lock` for the same reason. Memoisation is unaffected: a repeated
  load stays at ~90 µs against a ~68 ms Lloyd-Max solve, at one thread
  and at the default thread count.
- **A durability shortfall is no longer written unconditionally to
  stderr (#365, #390).** When a save's post-rename parent-directory fsync
  fails, the save has already committed and must not be reported as an
  error — but the shortfall has to stay visible. It was reported with
  `eprintln!`, which a service that captures its logs structurally never
  sees and no caller can turn off. It now goes through a process-global
  warning hook (`turbovec::set_warning_hook`); with no hook installed the
  default sink is still stderr, so nothing is silently dropped.
- **Rust docs: the crate header no longer describes a cache strategy
  `add` abandoned (#324).** The docs.rs landing text — the first thing a
  reader sees — said `add` "extends the packed codes and invalidates the
  blocked layout cache by replacing its `OnceLock`". Neither half is
  true: `add` maintains the blocked cache in place through `get_mut`,
  and after a v6 load it appends into that cache and leaves the packed
  rows unmaterialized entirely. The replacement states the invariant a
  reader can actually rely on (every populated cache describes exactly
  the rows the index holds, whenever the index is reachable through
  `&self`) and why it holds by construction, instead of narrating which
  buffer a particular mutator touches — the detail that went stale.
- **Undocumented panics on the public `rotation::Rotation::new` and the
  six `io::write*` entry points now have `# Panics` sections (#324).**
  `Rotation` is reachable without `TurboQuantIndex`, and its `MAX_DIM`
  ceiling was visible only in an implementation comment. The raw writers
  abort on a length inconsistency among four of their six slice
  arguments (five of seven for the `write_id_map*` trio), which their
  `io::Result<()>` signature does not suggest; the docs also now say
  which arguments are *not* checked — `codes_blocked_seq` and `scales`
  are written through as given by the writer, and the loader's own
  length checks do not reliably catch an inconsistent one. A wrong
  length shifts every later section of the file, so the load may error
  *or* may succeed and silently mis-score, depending on what the shifted
  bytes land on, and which dominates varies sharply with index geometry.
  A compensating pair that keeps the total byte count unchanged shifts
  nothing and has loaded clean in every configuration tested. The docs
  say that rather than promising a failure mode that does not hold; the
  underlying gap, with the measured sweep, is tracked as #407. `TurboQuantIndex::write`
  and `TurboQuantIndex::load` had no documentation at all despite
  `from_bytes` pointing readers at `load`; `SearchResults::scores_for_query`
  / `indices_for_query` documented their panics in prose without the
  heading that puts them on docs.rs. No behaviour changed.
- **A one-shot bulk `add()` no longer pins its rotated-batch scratch for
  the index's lifetime (#333).** The encode scratch only shrank when
  `capacity > 4 x this call's length` — a test the call that *grew* the
  buffer can never pass, since growing leaves capacity and length equal.
  So the batch that allocated the buffer was exactly the one that could
  not release it, and a copy-paste `index.add(embeddings)` kept a full
  rotated copy of the batch until the index was dropped. (A later,
  smaller add *did* release it; retention was permanent only for the
  common shape where no smaller add follows.)
  Retention is now sized from the previous call's demand plus half again,
  and only applied when capacity exceeds twice that. The slack preserves
  the amortized growth headroom a growing or jittering batch size relies
  on, and the hysteresis keeps ordinary shapes from shrinking at all;
  a one-shot bulk add has no previous demand and so releases outright.
  There is no retention floor — `Vec::reserve` from zero capacity
  allocates once, so a floor has no allocation cascade to prevent.
  Measured with a counting global allocator, dim 768 at 2-bit, single
  thread: a 200k one-shot add retains **623.3 MB before, 37.4 MB after**
  against a 36.6 MB index, and the total allocation count over a run is
  unchanged to within one — 520 -> 521 for twelve equal 50k adds,
  743 -> 744 for twenty adds growing 5% each, 740 -> 741 for twenty
  jittering between 45k and 55k. Add throughput is unchanged at default
  threads and at `RAYON_NUM_THREADS=1`.
  Note the numbers above are live heap. On macOS this does not show up in
  RSS at all: `ps` reports the same resident size with and without the
  fix, for reasons not fully established — the freed spans stay resident
  even in a sequential build-and-drop loop where they ought to be reused.
  The allocator-level win is solid; the resident-size win is unverified
  on any platform.
- **Deferred-window adds no longer cost O(n) when the new ids sort below
  the retained id table (#383).** After a load, `IdMapIndex` keeps the
  load-time sorted id table alive so post-load adds can validate new ids
  by binary search instead of forcing the O(n) `id → slot` map build. The
  merge that kept it current broke out early only when the new ids all
  sorted *above* the table's tail, so an id sorting below rewrote all n
  entries — per add, under the write lock, quadratic over a chatty
  post-load pattern. Ids added inside the deferred window now go into a
  side hash set and the load-time table is never rewritten; a presence
  check is one binary search plus one hash lookup, and an add costs
  O(rows added) wherever the new ids sort. Measured at dim=32, 4-bit,
  interleaved A/B, µs per single-row add with ids below the table:
  52.5/102.2/201.2/405.3 → 3.5/3.1/3.2/3.1 at n = 50k/100k/200k/400k
  (and 51.8/100.2/198.0/393.9 → 3.5/3.7/3.1/3.0 at
  `RAYON_NUM_THREADS=1`). Ids sorting above the table were already flat
  and are unchanged. Note the side set is retained, like the sorted
  table, until something materializes the map.
- **A panicking first add no longer wedges a lazy index at a committed
  dim (#380).** `add_2d` locked the inferred dim before the encode, so a
  caught encode panic left an index with a dim and no vectors, and the
  follow-up `add_2d` at a different dim got `DimMismatch` instead of the
  fresh start #129 established. The dim — and the rotation, boundary and
  centroid caches derived from it — are now rolled back if the add
  unwinds; rolling back the dim alone would leave the next add at a
  different dim panicking inside `rotation` instead of starting fresh.

- **A caught panic in the eager add's cache repack no longer leaves the
  stored codes ahead of the row count (#388).** The blocked-cache patch is
  fallible, and `packed_codes` and `scales` were published before it while
  `n_vectors` was published after — so a caught `PanicException` left both
  buffers holding the failed batch's rows against the old count, and the
  next add addressed past the orphans. That is silent slot corruption
  rather than a detectable inconsistency. Reordering alone is not a fix:
  both buffers are taken out of the index before encoding, so deferring
  their publication makes a panic drop them entirely and leave the index
  with *empty* buffers against a non-zero count. The repack now runs under
  a guard that truncates both back to their pre-call lengths and
  republishes them, matching the contract `encode`'s own guard keeps.

- **The v6 codebook check no longer puts a Lloyd-Max solve on the load
  path (#357).** Validating the embedded codebook by recomputing it and
  comparing cost 25–100 ms — two orders of magnitude more than the load
  it guarded. It is replaced by the two properties that *define* the
  codebook and cost microseconds: each centroid must equal the Beta
  conditional mean of its own cell (the Lloyd-Max fixed point, evaluated
  in closed form), and each boundary must be the exact f32 midpoint of
  its neighbouring centroids. Rejection strength is unchanged or better —
  the boundary identity is now bit-exact rather than compared at 1e-4 —
  and `codebook(bit_width, dim)` is memoised process-globally, so repeat
  builds and saves of the same shape no longer re-solve either. Measured
  cold load (file → first search, 20,000 × 768 4-bit, interleaved A/B):
  66.5 → 1.00 ms at default threads, 66.6 → 0.99 ms at
  `RAYON_NUM_THREADS=1`.
- **A save that committed is no longer reported as a failure (#365).**
  The rename is the commit point; a parent-directory fsync failing after
  it left the *new* file in place while `write` returned `Err`,
  contradicting the documented "the previous file at `path` is left
  untouched" guarantee and sending callers with a rollback policy down a
  destructive path (the error cleanup also silently no-opped, since the
  temp name no longer existed). Such a failure is now a durability
  shortfall, warned about on stderr, and the save reports success.

- **Add-path and loader error messages now name the condition that
  actually occurred (#329).** An id repeated inside a single
  `add_with_ids` batch reported "id N already present in index" — false
  on an empty index, and it sent callers hunting for a phantom prior
  insert; it is now the new `AddError::DuplicateIdInBatch`, "duplicate
  id N appears more than once in this batch". A zero-width batch
  (`dim == 0`, usually an embedder returning empty embeddings) was folded
  into "vector buffer length 0 not a multiple of dim 0" — mathematically
  nonsense and the wrong cause — and is now the new `AddError::ZeroDim`
  on both `TurboQuantIndex::add_2d` and `IdMapIndex::add_with_ids_2d`.
  The pre-v5 rejection hint no longer names a version number: it pointed
  at a release that does not exist and at a remedy the reader was already
  running. The `.tvim` wrong-magic error now reads "not a turbovec .tvim
  file", matching its `.tv` counterpart.
- **A zero-row `add_2d` no longer commits a lazy index's dim (#308).**
  `add_2d` set `self.dim` before delegating to `add`, whose zero-row
  no-op guard then returned — so an empty batch permanently locked the
  dim of a lazy index, changed its serialized bytes (the `dim=0` sentinel
  became the batch's dim) and survived save/load, making a later add of
  the real dimensionality fail with `DimMismatch`.
  `IdMapIndex::add_with_ids_2d` inherited it. A zero-row batch is now a
  true no-op: `dim` is still validated (a mismatch against an
  already-committed dim, or a malformed lazy first dim, reports the same
  error as before), but nothing is committed and the serialized bytes are
  byte-identical to a pristine lazy index. Realistic trigger: a lazily
  constructed framework store where `add_texts([])` or a filtered-to-empty
  batch preceded the first real batch.
- **TQ+ calibration is now a warm-up lifecycle instead of hidden
  first-add state (#107, #284, #285, #303, #317).** An index buffers its
  raw rows until it has seen 1000 vectors, then fits the calibration and
  re-encodes those rows with it, in slot order — so a first `add` of
  1–999 vectors (or a stream of 500-vector batches, the default shape of
  every framework integration) no longer locks identity calibration and
  silently forfeits the TQ+ recall gain for the index's whole life. The
  buffer is bounded by 1000 rows and mirrors `swap_remove`. Three further
  entrances to a mis-declared calibration are closed with it: the commit
  site now writes only a calibration `encode` actually fitted, so
  draining an index to empty and re-adding no longer overwrites the
  fitted calibration with identity (#284); both v6 load arms of
  `from_loaded` route through the same identity-population `from_parts`
  performs, so a v6 file with an empty TQ+ trailer plus a later `add` no
  longer produces vectors that `len` counts but search can never return
  (#303); and the new `TurboQuantIndex::calibration_state` /
  `IdMapIndex::calibration_state` accessors make the state queryable
  instead of invisible (#317). No file-format change: a stored index
  always declares exactly the calibration its codes were encoded with,
  and files written by earlier versions load unchanged.
- **Declared MSRV corrected from 1.83 to 1.89 — the crate did not build
  on the version it advertised.** The AVX-512 search kernel added in the
  v6 cycle uses `_mm512_*` intrinsics and the `avx512f`/`avx512bw`
  `target_feature` gates, all of which stabilized in Rust 1.89; on 1.83
  `cargo check -p turbovec` fails outright with `use of unstable library
  feature 'stdarch_x86_avx512'` (67 errors), so a downstream consumer
  pinned to the declared MSRV got a hard compile error rather than a
  scalar fallback. Both packages now declare `rust-version = "1.89"`,
  verified by a clean `cargo +1.89 check` of each plus the full test
  suite (19 suites) on 1.89. Found by `clippy::incompatible_msrv` while
  adding the AVX-512 butterfly, which raised the same lint against the
  pre-existing search kernel.
- **Declared MSRV corrected from 1.70 to 1.83.** The
  `rust-version = "1.70"` declared in both `Cargo.toml`s was never
  accurate: when it was introduced (2026-04-13, chosen for the crate's
  own `OnceLock` use), the dependency tree already required newer
  toolchains — `pest` 2.8.6 (transitive via
  `faer` → `npyz` → `py_literal`) requires rustc 1.83, `faer` 0.20.2
  requires 1.81, and the v4 `Cargo.lock` needs cargo ≥ 1.78 to parse.
  Both packages now declare `rust-version = "1.83"`, verified by a clean
  `cargo +1.83 check` of each. (#182)
- **Pre-AVX2 x86-64 CPUs no longer SIGILL before the scalar fallback can
  run.** The repo-level `.cargo/config.toml` set a global
  `target-cpu=x86-64-v3` (AVX2/FMA/BMI2) baseline, so every *plain* (non-
  `#[target_feature]`) function — including the runtime-dispatch prologue
  and the pre-AVX2 scalar fallback itself — was compiled with AVX/VEX
  instructions and faulted on pre-Haswell CPUs before
  `is_x86_feature_detected!` could ever select the fallback. The baseline
  is now `x86-64-v2`; the AVX2 and AVX-512 kernels stay runtime-dispatched
  and are `#[target_feature]`-gated, so they are compiled with their full
  feature sets regardless of the baseline (re-tuned by the baseline's
  tuning model). Applies only to builds made from a repo checkout — CI,
  benchmarks, and the wheel pipeline; the published crates.io `.crate`
  does not contain `.cargo/config.toml`, so `cargo add turbovec` users
  were never affected. (#137)
- **Index saves are atomic.** `io::write` / `io::write_id_map` (and
  `TurboQuantIndex::write` / `IdMapIndex::write` on top of them) now write
  to a sibling temp file, fsync, and rename over the destination, so a
  failed or interrupted save no longer destroys a previous good index at
  the same path; the TQ+ calibration-length assert also runs before any
  file is created. (#118)
- **Saving an index with ≥ 2³² vectors no longer silently wraps.**
  `write_core` previously truncated `n_vectors` with `as u32`, producing
  a corrupt file that loaded clean with `n mod 2³²` vectors. Resolved by
  the format-v4 64-bit count field (see *Added* above), which stores the
  exact count. (#119)
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

### turbovec — Python package

#### Added

- **Interruptible long search/add (#216).** A large batch `search` / `add`
  / `add_with_ids` is now processed one row-slice at a time (default
  `turbovec.BATCH_CHUNK_SIZE = 1000`, overridable per call with
  `chunk_size=`), so control returns to Python between slices and a queued
  Ctrl-C is serviced there instead of at the end of the call. The GIL was
  already released (#186), but Python delivers signals on the main thread —
  the one parked inside the Rust kernel — so a Ctrl-C used to be queued
  until the whole call returned. Measured on a ~7.2 s batch search: the
  Ctrl-C delay dropped from ~5.4 s (queued to the end) to ~10 ms (within
  one slice). Pure-Python wrappers over the native kernels — no core
  change. Chunked results are identical to a single call (each `search`
  slice reads one coherent snapshot of the query array, preserving the
  mid-search-mutation guarantee; each `add` slice is committed atomically).
  Throughput cost is asymmetric: `search` is unaffected (~0 %), but a
  chunked `add` / `add_with_ids` pays a snapshot, per-slice validation and
  dispatch, and (`add_with_ids`) an O(n) pre-existing-id check — measured
  at roughly 2–7× the unchunked wall time at the default `chunk_size=1000`
  (the base add is fast, so fixed per-slice overhead dominates the ratio;
  it varies with dim/batch/machine). The absolute overhead is small, on the
  order of ~1–10 µs/vector. For a throughput-critical one-shot bulk load,
  pass `chunk_size=0` to run the add whole at full speed. A cancelled `add` commits the completed slices and raises — the
  index stays consistent and queryable at that count. Two calls stay
  indivisible and deaf to Ctrl-C by design: a single huge query
  (`nq == 1`) and the *first* add into an empty index (it fits and locks
  the TQ+ calibration from its batch, so slicing it would change the whole
  index's quantization). Making those interruptible needs a core
  cancellation poll (`PyErr::CheckSignals` in the hot loops) — the deferred
  follow-up.
- `write(path, durable=False)` on `TurboQuantIndex` and `IdMapIndex`:
  keeps atomic-replace semantics but skips fsync (not power-loss-safe) —
  see the Rust-surface entry for details and measurements (#274).

- **Per-store similarity modes for all four integration stores**
  (LangChain, Haystack, LlamaIndex, Agno), fixed at construction and
  recorded in the persisted side-car. Two modes:
  - **`cosine` (the default).** Document vectors are L2-normalized
    before they reach the quantized index and query vectors before
    search, so raw scores are true cosine similarity in `[-1, 1]` for
    embeddings of any magnitude, and ranking matches each framework's
    in-tree reference store (`InMemoryVectorStore`,
    `InMemoryDocumentStore`'s cosine branch, `SimpleVectorStore`).
    Zero vectors cannot be normalized and are kept as-is — they score
    `0` against everything, matching the references.
  - **`dot_product` (explicit opt-in).** Raw vectors, raw
    inner-product scores, magnitude-aware ranking — exactly the
    previous behavior. Absolute score thresholds are dataset-relative
    in this mode and need calibrating per embedder.

  Parameter surface per store: Haystack's existing native
  `embedding_similarity_function` ("cosine"/"dot_product") now selects
  the mode (and, as before, the `scale_score` formula); Agno's existing
  `distance` parameter accepts `Distance.cosine` (default) and now also
  `Distance.max_inner_product` (`Distance.l2` still raises); LangChain
  and LlamaIndex gain a `similarity: str = "cosine"` keyword (a
  documented turbovec extension — their references compute cosine
  unconditionally). Unknown mode values raise `ValueError`. (#114)

- **`turbovec.__version__`.** The package now exposes the standard
  version attribute (resolved lazily via PEP 562 from the installed
  dist metadata, so `import turbovec` stays sub-millisecond —
  importing `importlib.metadata` eagerly would multiply the import
  time by ~25x). Falls back to `"0.0.0.dev0"` when no dist metadata
  is installed. (#153)
- **`to_bytes()` / `from_bytes(data)` on `TurboQuantIndex` and
  `IdMapIndex`.** Serialize an index to `bytes` in its `.tv` / `.tvim`
  wire format — byte-identical to the file `write(path)` produces —
  and reconstruct it from those bytes with exactly the same validation
  as `load` (corrupt or drifted payloads raise `ValueError`). Both run
  with the GIL released; `from_bytes` accepts `bytes` or `bytearray`.
  This is the in-memory persistence path (caches, databases, pickling)
  that previously required a filesystem round-trip. (#148, #70)
- **All four integration stores are picklable and copyable.** The
  LangChain, Haystack, LlamaIndex, and Agno stores implement
  `__getstate__` / `__setstate__` (the Rust index rides along as its
  `.tvim` bytes; the per-store lock — and Haystack's async executor —
  are excluded and recreated on restore), `__deepcopy__`, and
  `__copy__`. The state is snapshotted under the store's writer lock,
  so a pickle overlapping a write captures a consistent index/side-car
  pair (per-doc payload dicts are copied deeply enough that an
  in-place metadata update landing mid-pickle cannot tear the
  snapshot). `pickle.loads(pickle.dumps(store))` round-trips
  documents, metadata, search results, the similarity mode, and the
  handle counter, including into `multiprocessing` spawn workers. `copy.copy` deliberately equals
  `copy.deepcopy`: there is no meaningful shallow copy of a store —
  sharing the mutable Rust index means mutations bleed between the
  copies (see *Fixed*). (#148, #149)
- **`TurboQuantIndex` and `IdMapIndex` support `pickle`, `copy.copy` and
  `copy.deepcopy` (#340).** Both classes implement `__reduce__`, reducing
  to `from_bytes(to_bytes())` — so a bare index can cross a
  `multiprocessing` `spawn` boundary (the default start method on macOS
  and Windows) and any user container holding one can be deep-copied. A
  reconstructed index is fully independent of the original. Pickle
  inherits the `to_bytes` persistence contract unchanged, including the
  `RuntimeWarning` that an index below the 1000-vector TQ+ sample
  threshold reloads committed to identity calibration — check
  `index.calibration_state` before serializing. Previously only the four
  integration stores implemented the protocol; the bare index raised
  `TypeError: cannot pickle`.
- **Both index classes are weakly referenceable (#340).** An index can be
  held in a `weakref.WeakValueDictionary` — the standard way to key a
  per-tenant cache without pinning its memory. `weakref.ref(index)`
  previously raised `TypeError`.

#### Changed

- **LangChain / LlamaIndex / Agno async methods no longer block the event
  loop, and `asyncio.wait_for` now works on them (#342).** The `a*` /
  `async_*` methods ran their index work inline on the loop thread, so a
  large `aadd_texts` blocked the loop for the operation's full duration
  and a deadline could never be delivered at all — `await
  asyncio.wait_for(..., timeout=0.05)` ran to completion with no
  `TimeoutError` raised and every document committed. Each method now
  runs its sync body on a worker thread via
  `asyncio.to_thread`, matching `VectorStore`'s own `run_in_executor`
  defaults and the `asyncio.to_thread` shape Agno's in-tree sync-backed
  vector DBs use. One offload per method, never one per chunk, so the
  locked bodies stay atomic and the issue-#146 / #89 orderings are
  unchanged. Agno's `async_exists` / `async_name_exists` /
  `async_get_count` still answer inline — O(1) reads where a thread hop
  costs more than it saves. **Cancellation is partial by design:** the
  awaiting caller is released promptly, but cancelling does not decide
  what happened to the write. A worker that already started runs the call
  to completion (work inside the Rust core is not interruptible) and the
  write commits in full; a call still queued behind a saturated executor
  is cancelled before it ever runs and nothing is written. A cancelled
  write is therefore "outcome unknown" — it may have fully committed, or
  may never have begun — so make retries idempotent. The one guarantee is
  that the outcome is all-or-nothing: the store is never left torn.
  Documented per integration.

- **Index file format break (v5): saved indexes from older versions no
  longer load.** The Python package inherits the Rust crate's format v5
  rotation break (see the Rust section): any `.tv` / `.tvim` file, pickle,
  or `to_bytes` payload written by an earlier turbovec — including the
  framework stores' on-disk state — is refused on load with an actionable
  "rebuild the index" error rather than silently mis-decoding. Rebuild
  affected indexes from the source vectors. In exchange, encoded output is
  now deterministic across platforms and thread counts, and the wheel no
  longer bundles OpenBLAS. (#206)

- **LangChain: non-`str` ids are now rejected with `TypeError` at the
  add boundary** (`add_texts` / `aadd_texts` / `add_documents` /
  `from_texts` and async variants), naming the offending id, its type,
  and its position, before any embedding-store mutation. Previously an
  off-contract id (the declared type is `list[str]`) was accepted
  in-memory and then corrupted by JSON persistence: an `int` id `2`
  round-tripped as the string `"2"`, and an int id coexisting with its
  equal-looking str id (`2` + `"2"`) produced a duplicate JSON key on
  `dump` that `load` collapsed — one document silently destroyed and
  the side-car left unloadably out of sync with the index. This is a
  deliberate safer-than-reference deviation: `InMemoryVectorStore`
  accepts non-str ids and exhibits the same dump/load corruption.
  `None` entries in an explicit ids list are still replaced with
  generated UUIDs; `bool` (a subclass of `int`) is rejected like any
  other non-str type. (#124)
- **Default scoring of the four integration stores changes for
  non-unit-normalized embeddings.** Under the new `cosine` default the
  stores normalize documents at add time and queries at search time, so
  scores and ranking are cosine — previously they were raw inner
  products (magnitude-aware). For unit-normalized embedders (OpenAI,
  Cohere, sentence-transformers with `normalize_embeddings=True`) the
  scores are identical up to quantization noise and nothing changes.
  For non-unit embedders, freshly built stores now rank by angle
  rather than by `‖v‖·‖q‖·cosθ`, and score magnitudes move from
  unbounded to `[-1, 1]` — this is the fix for #114; opt into
  `dot_product` mode to keep the old ranking. **Persisted stores are
  unaffected:** a side-car written before the mode field existed holds
  raw vectors and loads in `dot_product` mode, keeping scoring
  byte-identical with zero migration (see the schema notes below).
  (#114)
- **Integration side-car schemas record the similarity mode** (each
  bumped per its own versioning convention; every loader still accepts
  all older versions):
  - LangChain `docstore.json` v1 → v2 (`similarity` field; v1 loads as
    `dot_product`).
  - LlamaIndex `nodes.json` v2 → v3 (`similarity` field; v1/v2 load as
    `dot_product`).
  - Agno `docstore.json` v1 → v2 (`distance` field; v1 loads as
    `max_inner_product`, updating `self.distance` to match; a v2 file
    whose recorded mode conflicts with the constructor's `distance`
    raises `ValueError` at `create()` because Agno's
    construct-then-load shape means both sides hold a mode).
  - Haystack `docstore.json` v2 → v3 (`vectors_normalized` field;
    v1/v2 vectors were always written raw whatever their recorded
    `embedding_similarity_function`, so they load with normalization
    off and keep the recorded function for the `scale_score` formula —
    byte-identical behavior, including the saturating cosine-branch
    `scale_score` those stores had). New writes into a legacy-loaded
    store stay raw (mixing normalized and raw rows would corrupt
    ranking). (#114)
- **All four integration stores are now safe for concurrent
  multi-threaded use; writes serialize on a per-store lock.** The
  LangChain, Haystack, LlamaIndex, and Agno stores adopt a layered
  design measured in the #161 research: every mutating method
  (add/write/insert, delete, upsert, update, clear/drop — sync and
  async) and every save path (`dump` / `save_to_disk` / `persist` /
  `save`) serializes on a per-store `threading.RLock`; reads take no
  lock, so concurrent searches keep overlapping and scaling across
  threads (#186). Adds now populate the side-car maps *before* the
  index insert (with a failure-unwind preserving the #89
  "a failed add never destroys existing data" guarantee) and deletes
  remove from the index *first*, so a search can never surface a
  handle that doesn't resolve; result translation skips handles whose
  entries a concurrent delete removed mid-search, and filtered
  searches retry a stale allowlist and fall back to a non-raising
  post-filtered search under sustained churn. The resulting contract:
  a read overlapping a write sees pre- or post-write state, and under
  heavy concurrent churn a search may transiently return fewer than
  `k` results. There is still no cross-call atomicity, and
  multi-process access remains unsupported. (#161)
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

#### Removed

- **Wheels no longer ship a `turbovec.mlx` namespace package (#305).**
  Locally-built wheels picked up stale `__pycache__` for an `mlx`
  subpackage whose sources no longer exist, so `import turbovec.mlx`
  succeeded and yielded an empty module. It now raises `ImportError`.

#### Fixed

- **`search()` on an empty query batch no longer raises `PanicException`
  (#349).** `ix.search(np.zeros((0, dim), np.float32), k)` reached a
  divide-by-zero in the core's block-range tiling (see the Rust entry),
  which crossed the PyO3 boundary as `pyo3_runtime.PanicException` rather
  than any exception a caller would think to catch. It needed the batch to
  run in the fork-safe pool, and for `nq == 0` that happens only when
  `single_query_parallelizes(len(index))` is true — `len` rounded up to
  32-vector blocks reaching `SINGLE_QUERY_PARALLEL_MIN_BLOCKS`, then 256,
  so 8161 vectors and up, matching the bisect in the issue (8160 fine,
  8161 panicking). Below that the extension's global rayon pool is pinned
  to a single sentinel thread and the one-thread path returns before the
  division. `TurboQuantIndex.search` and `IdMapIndex.search` now return
  their documented `(0, effective_k)`-shaped arrays at any index size,
  with or without `mask=` / `allowlist=`.
- **A completed `save()` is durable: the integrations fsync the directory
  the index and side-car were renamed into (#350).** `os.replace`
  publishes a name by updating the *directory*, so fsyncing the two temp
  files made their contents durable but not the renames that named them.
  A power loss after a save returned success could leave the directory
  entry unwritten and the store back at its previous contents. All four
  stores (LangChain, LlamaIndex, Haystack, agno) share the write path and
  get the fix together, matching the Rust writer's parent-dir fsync. When
  the index and side-car live in different directories, both are synced.
  The fsync is skipped on Windows, which has no directory-fsync
  equivalent, and a filesystem that refuses `fsync` on a directory fd is
  tolerated rather than turning a completed save into an error.
- **A side-car `schema_version` must be an integer, not merely equal to
  one (#350).** The version gate was `version not in compat`, and `==`
  crosses numeric types in Python, so `2.0` and `true` were accepted as
  versions 2 and 1 — a side-car from a non-Python writer (JSON has a
  single number type) passed a gate it does not actually match. A version
  is an identifier rather than a quantity, so the type must match too.
  All four stores share one `check_schema_version` helper; their error
  messages and the versions they accept are unchanged, and `"2"`, `None`
  and unknown integers are rejected exactly as before.
- **A `mask=` whose bytes are not 0 or 1 now filters by numpy's own
  truthiness (#349).** numpy stores `bool_` in one byte and does not
  constrain the value, so `np.array([2], np.uint8).view(bool)` hands
  Python a `bool` array numpy reports as truthy. Those bytes were read
  straight into Rust `bool`, which may only hold 0 or 1 — undefined
  behavior, and it mis-filtered concretely: a mask selecting slots
  `{3, 9, 40}` returned five results drawn from slots the mask never
  selected while dropping ones it did. The mask buffer is now read as
  bytes and compared `!= 0`, matching numpy. A clean `bool` mask is
  unaffected, and the dtype and C-contiguity errors are unchanged.
- **`type(index).__module__` reports `turbovec._turbovec`, not `builtins`
  (#340).** Anything recording `f"{cls.__module__}.{cls.__name__}"` — a
  framework `to_dict`, a `spawn` payload, a Sphinx cross-reference —
  stored `builtins.TurboQuantIndex`, which resolves nowhere, and
  `pickle.dumps(TurboQuantIndex)` (the *class*, not an instance) raised
  `PicklingError`. This also changes the class name in `TypeError`
  messages and in `repr(type(index))`.
- **`inspect.signature()` reports the documented `chunk_size` kwarg on
  `search` / `add` / `add_with_ids` (#340).** The chunking wrappers set
  `__wrapped__`, which `inspect.signature` follows by default, so it
  reported the *native* signature: `chunk_size` was invisible to `help()`
  and IDE completion, and `Signature.bind` rejected it — so every
  signature-driven caller (`pydantic.validate_call`, framework tool-arg
  introspection, CLI adapters) refused a parameter that works at
  runtime. The wrappers now carry an explicit `__signature__` (the native
  parameters plus keyword-only `chunk_size=None`) and report the public
  method's `__qualname__` / `__module__` rather than the internal
  `_make_search.<locals>.search` closure. `__wrapped__` is still set.
- **A float `turbovec.BATCH_CHUNK_SIZE` is coerced with `int()` like an
  explicit `chunk_size=` argument (#345).** The coercion documented for
  the slice size applied only to the per-call argument, so assigning a
  float to the public constant surfaced as `TypeError: 'float' object
  cannot be interpreted as an integer` from inside the wrapper, on an
  `add` with nothing wrong with it.
- **The warm-up serialization `RuntimeWarning` is one-shot per index, not
  per process (#360, #366).** Every index warns the first time it is
  serialized while `calibration_state` is `"warming_up"` **and it holds at
  least one vector**, rather than one index per process doing so. What
  reaches the user is then up to the filter chain: under the default
  configuration CPython dedupes per `(text, category, module, lineno)`, so
  tenants holding the *same* number of vectors and saving from one shared
  call site still collapse to a single warning — measured, 3 tenants of 10
  vectors deliver 1 under default filters, 3 under `always`, and 3 under
  default filters once their counts differ (10/11/12). So the per-tenant
  gain is real but conditional; the unconditional part is that the library
  no longer suppresses anything after the first index. A warming-up index
  drained to **zero** vectors is serialized silently and its reload is
  permanently identity — that is #418, and out of scope here. The latch is
  consumed only once
  `warnings.warn` has returned, so a `simplefilter("error")` save — which
  raises out of the warn and is routed to `sys.unraisablehook` — leaves
  the index able to warn again, and `pytest.warns` around a warming-up
  save no longer depends on what an earlier test in the session saved. A
  serialization under an `ignore` filter still consumes that index's
  latch: `warn` reports the same thing whether the chain delivered the
  warning or dropped it, so there is nothing to branch on (#360).
- **The warm-up serialization warning is attributed to the caller's own
  file (#366).** A Rust frame is not a `warnings` stack level, so the
  warning was credited to the nearest Python frame — for every
  integration store's save path that is `turbovec/_persist.py`, a
  turbovec internal the user never wrote, which also keyed
  `__warningregistry__` there. It now names the first frame outside the
  `turbovec` package, i.e. the `write()` / `dump()` / `persist()` /
  `copy.copy()` call the user made. The core crate's durability warning
  (#365) now shares that emitter, but its attribution is **unchanged**: it
  is raised with no Python frame on the stack, so the walk finds none and
  falls back to CPython's `sys:1`, exactly as before.
- **The warm-up warning says "serializing", and mentions copying.** It
  also fires from `to_bytes`, which is the path `pickle`, `copy.copy` and
  `copy.deepcopy` take on all four integration stores — so "saving an
  index" pointed the reader at a save call that does not exist. `write`,
  `to_bytes` and the stores' copy/pickle sections now state that a copy of
  a store below 1000 vectors is permanently committed to `"identity"`
  while the original keeps its warm-up buffer (#366).
- **`IdMapIndex.prepare()` warms the id map and has a docstring (#348).**
  It inherits the Rust-side fix above, so the first `search(...,
  allowlist=)`, `contains()` or `remove()` after a load no longer pays an
  O(n) build that `prepare()` promised to absorb. `inspect.getdoc()`
  previously returned `None` for it while `docs/api.md` advertised it as
  "same as `TurboQuantIndex`".
- **nq=1 searches on indexes of 8192–32767 vectors no longer take the
  process-wide rayon pool (#336).** They ran on the shared pool for work
  too small to split, which cost the `install` handoff *and* serialized
  every concurrent caller behind one queue. Measured at
  `RAYON_NUM_THREADS=1`, n=16384, 14 Python threads: 19,078 → 70,865
  queries/s (3.71x), with thread scaling going from 1.24x to 4.32x. At
  default threads and n=32768 the same comparison is 20,001 → 49,007 q/s
  (2.45x). This does **not** remove the ceiling reported in #336 for
  larger indexes: work that genuinely splits still goes through the one
  process-local pool and is still capped by `RAYON_NUM_THREADS`, which is
  inherent to the fork-safe single-pool design (#147/#288/#321/#364).
- **A save whose parent-directory fsync fails now raises a
  `RuntimeWarning` instead of printing to stderr (#365, #390).** The save has
  committed and still succeeds, but the durability shortfall was written
  straight to stderr by the core crate — unfilterable, and invisible to
  `logging.captureWarnings(True)`. The extension now points the core's
  warning hook at Python's `warnings`, so it behaves like the warm-up
  save warning: filterable, capturable, assertable with `pytest.warns`.
- **A one-shot bulk `add()` / `add_with_ids()` no longer pins its
  GIL-safety snapshot for the index's lifetime (#333).** The snapshot
  buffer carried the same unsatisfiable shrink condition as the core's
  encode scratch and now follows the same policy — retain the previous
  call's length plus half again, and only shrink when capacity exceeds
  twice that. Together with the core fix this drops both copies of a bulk
  batch that an index used to hold after `add()` returned.
  As with the core entry, the measured win is in live heap and does not
  appear in macOS RSS, for reasons not fully established; treat the
  resident-size effect as unverified.

- **The JSON side-car no longer writes data it cannot read back
  (#350).** ⚠️ **Breaking for stores holding non-finite floats
  anywhere in the side-car — see the migration note below.** Two
  payloads passed `json.dumps` but did not survive the file, silently,
  across all four integrations' save paths.
  *Non-string metadata keys* were stringified, so `{1: "int-one", "1":
  "str-one"}` landed on disk as a single `{"1": "str-one"}` — one entry
  gone, with `save()` returning success (`True`/`1` and `2020`/`"2020"`
  collided the same way). *NaN and Infinity* were emitted as bare tokens
  RFC 8259 forbids: `jq .` rewrites `NaN` to `null` and `serde_json` /
  `JSON.parse` reject the file outright — in a side-car documented as
  plain, inspectable JSON. Both now raise before any file is touched:
  `TypeError` for a non-str key, `ValueError` for a non-finite float,
  each naming the exact path to the offending entry.

  **Migration.** The two halves differ in impact and it is worth being
  precise about which affects you:

  - *Non-str keys* — no working code is affected. Those saves were
    already lossy on reload (the keys came back as strings, and colliding
    entries were simply gone), so the previous behaviour reported success
    for a save that had not preserved the data. If you relied on it,
    stringify the keys at the call site: `{str(k): v for k, v in ...}`.
  - *NaN / Infinity* — **this is a genuine break.** Python's `json` both
    writes and reads the non-standard tokens, so such metadata *did*
    round-trip correctly through turbovec's own `save`/`load`, and that
    now raises `ValueError`. The change is still deliberate: the file
    those saves produced was not JSON, and every non-Python consumer
    either rejects it or (jq) quietly rewrites the value to `null`. If
    you legitimately carry non-finite numbers, sanitize before saving —
    `None` for "no value" (it round-trips as `null` and is valid JSON),
    or a finite sentinel your pipeline agrees on.

  Validation walks the whole payload, so serializing the side-car costs
  roughly twice what it did: on a 200k-document payload with 4-field
  metadata, 0.31 s → 0.65 s. That is the side-car step only; a full
  `save()` also writes and fsyncs the index.

- **LangChain: a dict filter with a `None` value no longer matches
  documents that lack the key (#381).** `filter={"g": None}` was compiled
  to `doc.metadata.get("g") is None`, and `dict.get` cannot tell "absent"
  from "present and None" — so every document with no `g` key at all came
  back. A dict entry now requires the key to be present, matching the
  predicate a user would write by hand and agreeing with the agno store's
  dict filter (#144). Absence is still expressible through the callable
  filter form (`lambda doc: "g" not in doc.metadata`), which is the only
  form langchain_core's own `InMemoryVectorStore` accepts.
- **Adds and removes on a loaded index are no longer permanently routed
  through the rayon pool (#392).** The bindings chose between an
  uncontended fast path and a `py.detach` + pool handoff by probing
  `packed_ready()`, which was documented as "false only until the first
  mutation after a load". That stopped being true: no mutation on a
  v6-loaded index materializes the packed bit-plane rows any more — `add`
  lazy-appends to the blocked cache and `swap_remove` patches it with
  O(dim) lane ops — so the probe stayed false for the index's whole
  lifetime and *every* add and remove paid a pool handoff costing far
  more than the operation. `swap_remove` and single-row `add` drop the
  probe entirely; `IdMapIndex.remove` keeps one, but on `slots_ready()`,
  the structure whose first build genuinely is O(n) with the GIL held
  (#319) and which does flip to true after one remove. The penalty was a
  fixed per-call pool handoff, so its size depends on how contended the
  pool is and is not a stable figure — measured between 20x and 280x the
  cost of the same operation on a fresh index across ops, thread counts
  and machine load, and in the worst samples far higher. After the fix a
  loaded index costs 1.4x a fresh one for `IdMapIndex.remove`, 2.2x for
  `swap_remove` and ~3x for a single-row `add` (100k × 128, 4-bit), at
  both default threads and `RAYON_NUM_THREADS=1`; those figures are
  stable and reproduce. Fresh-index cost is unchanged. The residual is
  real work a loaded index does and a fresh one does not — blocked-cache
  lane ops, and in the add case a per-call extract-LUT rebuild — not
  overhead.
- **agno: a failed load no longer leaves a half-loaded store (#380).**
  `_load_from` replaced `_index`, `_u64_to_doc`, `_next_u64` and all three
  reverse indexes *before* the side-car/index consistency check that can
  raise, so a store whose load failed still reported `exists() is True`
  and a retried `create()` returned silently as "already created",
  handing back the half-load. The new state is now built into locals and
  committed in one block after every check has passed — a store whose
  load raised is one the method never touched. agno is the only
  integration that loads in place; the other three return a fresh object
  and were already safe.

- **agno: a half-present save loaded silently empty and was then
  overwritten (#328).** `create()` caught the `FileNotFoundError` that
  `_load_from` raises for a folder holding only one of
  `index.tvim` / `docstore.json` and built a fresh empty index instead,
  so the next `save()` overwrote the surviving file and the data was
  gone. Only a folder with *neither* artifact is treated as a fresh
  path now; a partial store propagates the "missing one of ..." error.
  The other three stores load through explicit classmethods that already
  propagate, and are unchanged.

- **`write()` errors now name the file and use `FileNotFoundError`
  (#329).** `TurboQuantIndex.write` / `IdMapIndex.write` raised a bare
  `OSError: No such file or directory (os error 2)` identifying no path,
  so a batch job writing several paths could not tell which failed and
  `except FileNotFoundError:` around a write never matched. Both now go
  through the same path-appending helper `load` uses. Duplicate-id and
  zero-width-batch messages from the add path are corrected with the
  Rust-side change above, and a persisted-store corruption message no
  longer leads with the internal "handle" vocabulary.
- **A zero-row `add` / `add_with_ids` no longer commits a lazy index's
  dim (#308).** `TurboQuantIndex(bit_width=4)` followed by
  `idx.add(np.zeros((0, 768), np.float32))` left `idx.dim == 768`, so the
  next real batch of a different dimensionality raised
  `ValueError: dim mismatch`, and the wedged dim survived
  `write` / `load` and `to_bytes` / `from_bytes`. An empty batch is now
  the documented no-op: `idx.dim` stays `None` and `to_bytes()` is
  byte-identical to a pristine lazy index.
- **Framework integrations: four parallel implementations of the same
  semantics, brought back into line (#321, #302, #322, #301).** The
  langchain / llama_index / haystack / agno stores each re-implement the
  same store contract, so a fix landed on one has repeatedly been missed
  on its siblings. This round closes four such gaps:
  - **agno: a failed `insert` could destroy a pre-existing document.**
    The other three stores capture the previous state before the
    maps-first write and restore it when the index add raises; agno's
    unwind popped the handle unconditionally, so when a corrupt
    `next_u64` watermark reissued a live handle the unwind deleted the
    *victim's* payload and unlinked the *new* document's id and name.
    agno now captures and restores like its siblings, restoring the
    "a failed add never destroys existing data" guarantee.
  - **Persisted-store validation now checks the handle watermark.**
    `check_persisted_handles` verified duplicate handles, count parity
    and index membership, but never that `next_u64` sits at or above the
    largest handle in use — so a stale, hand-edited or partially-written
    side-car loaded cleanly and then failed every subsequent write with
    a leaked internal handle id. All four stores inherit the check.
  - **llama_index: `delete(None)` wiped every parentless node.** Nodes
    with no SOURCE relationship stored `ref_doc_id = None`, so
    `delete(node.ref_doc_id)` on a parentless node deleted all of them.
    A parentless node is now filed under the literal `"None"`, matching
    `SimpleVectorStore`: `delete(None)` is a no-op, `delete("None")`
    targets them.
  - **llama_index metadata filters: two divergences from
    `SimpleVectorStore`.** `TEXT_MATCH` is case-**insensitive** again
    (the reference lowercases both sides), and a **missing** metadata key
    now fails `NE`/`NIN` as it already failed every other operator — the
    reference returns `False` for all operators once the value is absent.
    The previous behaviour was justified against
    `vector_stores.utils.build_metadata_filter_fn`, which does not exist
    in the supported llama-index-core range; `simple.py` holds the only
    in-tree evaluator and is now the reference of record.
  - **langchain: `dot_product` mode no longer fakes a `[0, 1]`
    relevance.** The relevance mapping clamped, so every raw inner
    product `>= 1.0` became exactly `1.0` — a `similarity_score_threshold`
    retriever admitted unrelated documents, and the clamp also suppressed
    the out-of-range warning `VectorStore` emits. In `dot_product` mode
    the mapping is now unclamped and selecting a relevance fn emits a
    `UserWarning`; cosine (the default) is unchanged and still clamped.
  - **Reference-parity API gaps.** langchain gains
    `similarity_search_with_score_by_vector` (and its `a`-prefixed
    variant) — the only non-deprecated public method the
    `InMemoryVectorStore` reference exposes and we lacked, so user code
    got `AttributeError` rather than `NotImplementedError`. haystack's
    `embedding_retrieval` now performs the reference's up-front
    `ValueError("query_embedding should be a non-empty list of floats.")`
    instead of returning `[]` on an empty store or reporting a dim
    mismatch. `top_k=-1` still raises here where the reference returns
    `n - 1` documents: a negative count is a caller bug, not a request.
- **TQ+ calibration warm-up (#107, #284, #285, #303, #317).** See the
  Rust-crate entry for the lifecycle change. On the Python side: both
  index types gain a read-only `calibration_state` property
  (`"warming_up"` / `"fitted"` / `"identity"`); saving an index that is
  still warming up emits a one-shot `RuntimeWarning`, because a file
  carries no warm-up buffer and the reloaded copy is committed to
  identity calibration for good; and the interruptibility wrapper no
  longer chunks an add into a warming-up index, since the calibrating
  add must see its whole batch to stay bit-identical to an unchunked
  one (it already made the same exception for the first add).
- **Fork safety: turbovec no longer deadlocks in a `fork()`ed child.**
  rayon's thread pool does not survive `fork()` — its worker threads live
  only in the parent, so the first parallel op a forked child ran (a batch
  search, or any `add`) injected work into a worker-less registry and hung
  forever. This wedged the default configurations of `multiprocessing`
  (fork start method — the Linux default through 3.13), gunicorn
  `--preload`, Celery prefork, and PyTorch `DataLoader(num_workers>0)`;
  single-query searches "worked" only by accident (a length-1 parallel
  iterator folds inline and never enters the pool), so smoke tests passed
  while the first batch or write silently wedged. The extension now routes
  every rayon-using kernel through a process-local pool: a forked child
  detects the fork (via `os.register_at_fork`, with a `pthread_atfork`
  backstop) and transparently rebuilds the pool on its first call, so its
  parallel ops run on live workers. Inherited-index searches return
  identical results in parent and child; the single-query hot path is
  unchanged (within measurement noise). Remaining unsafe cases (documented,
  no library fix): forking *while another thread is mid-`add`/`search`*
  (POSIX async-signal-safety limit) and co-loaded OpenMP/MKL, which stays
  independently fork-unsafe. (#147)
- **LlamaIndex: dotted namespaces no longer silently collide in a shared
  `persist_dir`.** The persistence stem handling used `with_suffix`,
  which re-split the stem at its last dot, so namespaces `v1.2` and
  `v1.3` both persisted to `v1.tvim` / `v1.nodes.json` — the second
  persist silently overwrote the first (data loss), and
  `from_persist_dir(namespace="v1.2")` returned the other namespace's
  data. Extensions are now appended to the full namespace-derived stem
  (`v1.2__vector_store.tvim` / `v1.2__vector_store.nodes.json`), so
  dotted namespaces coexist. Non-dotted namespaces keep byte-identical
  file paths — no migration. A store persisted by an earlier release
  under a dotted namespace still loads: when the correct filename is
  absent but the old mangled one exists, `from_persist_path` falls back
  to it (safe — the mangling meant at most one store could survive per
  mangled prefix), and the next `persist` writes the correct names.
  `_validate_namespace` additionally rejects `:` (a Windows
  drive-relative name like `C:foo` escapes `persist_dir` with no
  separator), extending the #152/#197 guard. (#200)
- **Threshold and relevance-score paths are no longer broken for
  non-unit-normalized embeddings** (the three wave-8 findings on #114
  — all symptoms of the mappings assuming cosine input while the
  engine returned raw inner products; fixed by the `cosine` default
  above, which makes the existing `(sim + 1) / 2` mappings correct):
  - **LangChain:** `as_retriever(search_type="similarity_score_threshold")`
    returned `[]` for any threshold when all raw inner products fell
    below `-1` (every relevance clamped to `0.0`), and distinct
    high-scoring documents all clamped to relevance `1.0`
    (indistinguishable — no threshold could separate them). Relevance
    scores are now distinct, ordered, and threshold-usable.
  - **Agno:** `similarity_threshold` was effectively inert — small
    thresholds discarded everything on all-negative raw scores and
    large thresholds admitted everything on large-positive ones. It now
    behaves as a true `[0, 1]` relevance cutoff under the default mode.
  - **Haystack:** `embedding_retrieval(..., scale_score=True)` on the
    default cosine function collapsed distinct scores to `1.0`,
    violating the "same ranking, mapped into [0, 1]" contract; scaled
    scores are now distinct and order-preserving. (#114)
- **Declared MSRV corrected from 1.70 to 1.83.** Building the PyPI
  package from source (sdist, or a platform with no prebuilt wheel) now
  correctly requires rustc 1.83 — the toolchain the dependency tree has
  in fact required all along; the 1.70 declared in
  `turbovec-python/Cargo.toml` was never sufficient. Prebuilt-wheel
  users are unaffected. (#182)
- **Pickling a LlamaIndex store no longer silently returns an EMPTY
  store (total data loss).** `pickle.dumps` / `pickle.loads` of a
  populated LlamaIndex `TurboQuantVectorStore` *appeared* to succeed
  but dropped the Rust index on the floor: it lives in a pydantic
  `PrivateAttr`, and the inherited `BaseComponent.__getstate__`
  removes any private attribute that fails `pickle.dumps` with only a
  log warning — so the store deserialized valid-looking and every
  query returned `[]`. This hit exactly the scenarios users pickle
  for (caching, `multiprocessing`, Ray, Celery): ship a populated
  store, workers silently search an empty one. The store now pickles
  faithfully via the new `to_bytes` / `from_bytes` core API; the
  LangChain, Haystack, and Agno stores — which previously raised
  `TypeError: cannot pickle 'builtins.IdMapIndex' object` — now
  round-trip too. (#148)
- **`copy.copy` of a store no longer aliases the Rust index, and
  `copy.deepcopy` works.** A shallow copy of any of the four stores
  shared the underlying mutable `IdMapIndex` and side-car maps, so
  mutating the copy silently mutated the original (and vice versa) —
  and `copy.deepcopy`, the only safe alternative, raised `TypeError`
  because deepcopy rides the pickle path. Both now return a fully
  independent store (`__copy__` is deliberately identical to
  `__deepcopy__`; a shared-index "shallow" copy is precisely the bug).
  (#149)
- **LlamaIndex `add()` no longer loses data under concurrent calls.**
  `_next_u64 += 1` on a pydantic `PrivateAttr` is not atomic under
  the GIL, so two concurrent `add()` calls could issue the same
  handle — the second batch was rejected with an opaque
  `ValueError: id already present` and its documents silently never
  stored (measured: 0.78% of adds under 2-thread ingest). Handle
  issuance now happens under the store's writer lock. (#161)
- **Concurrent search / retrieval no longer raises transient
  `KeyError` / `RuntimeError: dictionary changed size during
  iteration` while another thread writes.** This closes the measured
  crash classes in all four stores, including Agno's
  delete-vs-delete cleanup-scan crash and the misleading
  `KeyError: allowlist contains id(s) not present in index` from a
  delete racing a filtered search. Load-time validation is
  unchanged: a corrupt persisted store still fails loudly at load.
  (#161)
- **The x86_64 Linux and Windows wheels run on pre-AVX2 CPUs.** The
  wheels are built inside the repo checkout, so they inherited the repo
  `.cargo/config.toml`'s global `target-cpu=x86-64-v3` baseline: every
  plain (non-`#[target_feature]`) function — including the runtime
  dispatch and the scalar fallback itself — contained AVX/VEX
  instructions, and importing-and-searching on a pre-Haswell x86-64 CPU
  faulted (SIGILL) before the fallback could be selected. The wheels now
  build at the `x86-64-v2` baseline; AVX2/AVX-512 hardware still gets the
  `#[target_feature]`-gated SIMD kernels via runtime dispatch, compiled
  with their full feature sets regardless of the baseline (re-tuned by
  the baseline's tuning model). (#137)
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

- **Official persistence cells, x86 insert re-measure, and ARM
  re-baseline (#279, #280).** The published ARM benchmark environment
  moved from an Apple M3 Max laptop to a **GCP c4a-standard-8 (Google
  Axion, 8 vCPU)** instance — release build, idle box — and every ARM
  cell (search, insert, remove, persist) was re-measured there. The x86
  cells stay on the same GCP c3-standard-8 (Sapphire Rapids) box; the
  x86 insert and persist cells were re-measured on a clean release build
  at the PR base commit (fresh `target/` + `maturin develop --release`,
  provenance verified after an earlier run reused a pre-#277 build). The
  fresh clean-build run agreed with the committed x86 insert numbers
  within measurement noise across all 8 cells, so the committed bytes
  were retained: #277's encode speedup was measured on Cascade Lake and
  does not move Sapphire-Rapids bulk insert. (The agreement is what the
  ST≈MT single-add invariant confirms — single `add()` is serial, so a
  cell's ST and MT single-add timings must match, and across the grid
  they do.) All 16 `speed_persist_*` cells
  (arm + x86, both threadings) are now recorded in `benchmarks/results/`
  and `create_diagrams.py` renders matching
  `docs/{arm,x86}_persist_{st,mt}.svg` save/load figures (save-warm and
  load→first-search as precision-matched TurboQuant-vs-FAISS pairs; the
  mutate→save→load→search round-trip, which FAISS has no measured
  equivalent for, shown TurboQuant-only). README search prose (ARM now
  16–24%) and the ARM figure labels were updated to the new environment.
- **Persistence benchmarks join the suite** (#275): `speed_persist_*`
  for every (dim, bit width, arch, threading) cell, covering write in
  both states (warm blocked cache vs invalidated by a mutation — ~5x
  apart, so a single "save time" would hide the interesting half),
  `load` and `load → first search` separately (the gap v6 removed),
  and `mutate → save → load → first search` as one checkpoint/resume
  pipeline, each against FAISS `write_index` / `read_index`. Page-cache
  state is warm and stated; the fsync + atomic rename turbovec does and
  FAISS does not is called out in the scripts as a deliberate
  durability difference rather than a gap to close.
- **`examples/insert_bench`**: a Rust harness reproducing the suite's
  four mutation metrics on deterministic synthetic vectors, so an
  optimization hypothesis can be measured in seconds without the OpenAI
  corpus or FAISS. Official numbers still come from
  `benchmarks/suite/`.
- Insertion and removal speed benchmarks join the suite. (#65) For every
  published search-speed cell (d=1536/3072 × 2-bit/4-bit × ARM/x86 ×
  ST/MT), `benchmarks/suite/` gains `speed_insert_*` — bulk `add()` into
  an empty index (rotation/codebook init + TQ+ calibration fit included),
  a warm append with calibration frozen (the steady-state encode path),
  single-vector `add()` latency, and a FAISS `IndexPQFastScan` bulk-add
  baseline — and `speed_remove_*` — `IdMapIndex.remove(id)` per-op
  latency and throughput against a raw `TurboQuantIndex.swap_remove`
  baseline, isolating the id-map layer's bookkeeping. Fresh index per
  timed run (add is cumulative, remove shrinks the index), fixed seeds,
  median of 5, results in `benchmarks/results/` like the rest of the
  suite. ARM (Apple M3 Max) and x86 (Intel Sapphire Rapids, c3-standard-8)
  results are both recorded. `create_diagrams.py` gains matching figures
  (insertion throughput ST/MT, removal latency) for both architectures.
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

### CI

- **MSRV leg**: reads `rust-version` out of both manifests, checks they
  agree, and builds with exactly that toolchain. The declared MSRV has
  been wrong twice (1.70 → 1.83 → 1.89) and both times it took a human
  to notice; now it cannot drift from reality silently.
- **SIMD coverage gate, wired into the Rust legs.**
  `TURBOVEC_REQUIRE_SIMD=avx2,avx512f` makes the kernel identity tests
  *fail* when a listed feature is missing rather than silently skipping
  the paths gated on it — without it a runner lacking a feature
  exercises nothing and still reports green, so the absence of coverage
  is invisible. CI sets `avx2`, which every GitHub-hosted x86 runner
  has. AVX-512 is deliberately not required there (hosted runners do not
  guarantee it), so those kernels remain single-machine-verified until a
  designated runner or an Intel SDE leg covers them.
- **Cross-OS encode fingerprint leg** (#259). `examples/encode_hash`
  encodes a fixed LCG fixture across six (dim, bit width) cells and
  prints a hash per pipeline stage — codebook, calibration, codes,
  scales, whole file. Each OS in the matrix runs it; a `needs:` job
  fails unless all three agree. Hashing per stage means a divergence
  names the stage that drifted — which it did on the very first run:
  the codebook differed on all three OSes while calibration, codes and
  scales matched, localizing the cause to the boundary midpoints (fixed
  above) rather than to "the encode". Boundaries and centroids are
  hashed as separate columns for exactly that reason.

### Docs

- `docs/api.md`: the two FAISS analogues used as shorthand are replaced
  with direct descriptions — `swap_remove` is "not a shift" because the
  slots after `i` do not move down by one, and `IdMapIndex` is described
  as a hash-table-backed `u64 id ↔ slot` mapping rather than by
  comparison. (#344)
- README gains an "Insertion & Removal Speed" section after Search Speed:
  ARM insertion-throughput (ST/MT) and removal-latency figures generated
  from the new `speed_insert_*` / `speed_remove_*` results, with the
  measurement setup stated and results linked. x86 figures follow once
  the x86 cells are run. (#65)
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
