//! Runtime-cache sidecar for `.tv` / `.tvim` index files (issue #68).
//!
//! Loading an index is cheap, but the first search then rebuilds two
//! pieces of derived state: the Lloyd-Max codebook (Beta-distribution
//! numerical integration) and — dominating on real indexes — the
//! SIMD-blocked code layout, an O(n·dim) repack of every stored vector
//! ([`pack::repack`]). For short-lived processes that cost is paid on
//! every start. This module persists that derived state in a sidecar
//! file next to the index so a later load can seed the lazy caches and
//! skip the rebuild entirely.
//!
//! # Design contract
//!
//! * The `.tv` / `.tvim` file stays the only authoritative, portable
//!   artifact. The sidecar is a disposable, backend-specific
//!   accelerator: deleting it never loses data and never forces an
//!   index rebuild.
//! * **Fail closed.** A missing, truncated, corrupt, stale, or
//!   wrong-backend sidecar is silently ignored and the caches fill
//!   lazily on first search, exactly as without this module. Seeding
//!   can therefore never make a load fail, and a sidecar is never
//!   trusted unless every validation below passes.
//! * **Load never writes.** Only the path-based `write` methods create
//!   or refresh sidecars, so read-only deployments and the
//!   byte-oriented APIs (`to_bytes` / `from_bytes` / reader / writer)
//!   are completely unaffected.
//! * **Content binding.** The sidecar stores a 64-bit hash of the
//!   authoritative `packed_codes` (the exact input of the repack) plus
//!   the header fields the derived data depends on. Size/mtime are not
//!   used: if the index bytes change under the sidecar — even with the
//!   same length — the hash mismatch rejects it. A whole-file trailer
//!   hash additionally rejects truncation and bit rot inside the
//!   sidecar itself. (The hash is a fast non-cryptographic mix: the
//!   sidecar sits in the same directory as the index, so anyone who
//!   can forge it can rewrite the index itself — staleness detection,
//!   not tamper-proofing, is the goal.)
//! * The blocked layout differs per architecture ([`pack::repack`]'s
//!   x86 FAISS-style interleave vs the sequential layout elsewhere), so
//!   the backend id is part of both the filename and the header:
//!   `index.tvim` gets `index.tvim.x86_64-faiss-v1.cache` on x86-64 and
//!   `index.tvim.sequential-v1.cache` elsewhere. Machines of different
//!   architectures sharing a directory each keep their own sidecar.
//!
//! # File format (TVRC, version 1, little-endian)
//!
//! ```text
//! magic            4  b"TVRC"
//! version          1  u8 = 1
//! backend_len      1  u8
//! backend          k  ASCII backend id
//! bit_width        1  u8
//! dim              4  u32
//! n_vectors        8  u64
//! packed_hash      8  u64  hash of the index's packed_codes
//! boundaries   4(L-1)  f32 Lloyd-Max boundaries, L = 2^bit_width
//! centroids      4·L   f32 Lloyd-Max centroids
//! blocked          …   SIMD-blocked codes (length derived from header)
//! trailer_hash     8  u64  hash of every preceding byte
//! ```
//!
//! Every section length is a deterministic function of
//! `(backend, bit_width, dim, n_vectors)`, so the loader computes the
//! exact expected file size from the already-validated index before
//! reading a byte — a sidecar of any other size is rejected without
//! allocating, which also bounds the read by data the index has
//! already accounted for.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::{codebook, io, pack};
use crate::{BlockedCache, TurboQuantIndex, BLOCK};

const MAGIC: &[u8; 4] = b"TVRC";
const VERSION: u8 = 1;

/// Identity of the SIMD-blocked layout this build produces — must be
/// bumped if [`pack::repack`]'s output layout ever changes shape.
#[cfg(target_arch = "x86_64")]
const BACKEND_ID: &str = "x86_64-faiss-v1";
#[cfg(not(target_arch = "x86_64"))]
const BACKEND_ID: &str = "sequential-v1";

/// Fixed byte overhead: magic + version + backend_len + bit_width +
/// dim + n_vectors + packed_hash + trailer_hash.
const FIXED_OVERHEAD: usize = 4 + 1 + 1 + 1 + 4 + 8 + 8 + 8;

/// Sidecar path for an index file: the full file name plus
/// `.<backend>.cache`, so `.tv` and `.tvim` siblings never collide and
/// each architecture keeps its own sidecar.
fn cache_path(index_path: &Path) -> PathBuf {
    let mut name = index_path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    name.push(format!(".{BACKEND_ID}.cache"));
    index_path.with_file_name(name)
}

/// Fast 64-bit content hash, 8 bytes per round with a multiply-xor mix
/// and the length folded into the seed (so a zero-padded tail cannot
/// collide with real trailing zeros). Non-cryptographic by design —
/// see the module docs for the threat model.
fn hash64(bytes: &[u8]) -> u64 {
    const MUL: u64 = 0x2545_F491_4F6C_DD1D;
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15 ^ (bytes.len() as u64);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().expect("chunk is 8 bytes"));
        h = (h ^ w).wrapping_mul(MUL);
        h ^= h >> 29;
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rem.len()].copy_from_slice(rem);
        h = (h ^ u64::from_le_bytes(tail)).wrapping_mul(MUL);
        h ^= h >> 29;
    }
    h ^= h >> 32;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^ (h >> 32)
}

/// Section sizes implied by `(bit_width, dim, n_vectors)`. Mirrors
/// [`pack::repack`]'s layout math exactly; `None` on arithmetic
/// overflow (checked because `n_vectors` originates from a file
/// header, even though the index-level validation already bounds it).
struct Expected {
    n_levels: usize,
    n_blocks: usize,
    blocked_len: usize,
    total_len: usize,
}

fn expected(bit_width: usize, dim: usize, n_vectors: usize) -> Option<Expected> {
    let n_levels = 1usize << bit_width;
    let codes_per_byte = 8 / bit_width;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = n_vectors.checked_add(BLOCK - 1)? / BLOCK;
    let blocked_len = n_blocks.checked_mul(n_byte_groups)?.checked_mul(BLOCK)?;
    let codebook_bytes = 4 * (2 * n_levels - 1);
    let total_len = FIXED_OVERHEAD
        .checked_add(BACKEND_ID.len())?
        .checked_add(codebook_bytes)?
        .checked_add(blocked_len)?;
    Some(Expected {
        n_levels,
        n_blocks,
        blocked_len,
        total_len,
    })
}

/// Codebook arrays must look like a codebook: finite, inside the
/// quantizer's `[-1, 1]` support, and non-decreasing. The trailer hash
/// already rejects accidental corruption; this is defense in depth so
/// even a hash-consistent but nonsensical payload cannot poison a
/// later `add`'s encode path.
fn valid_codebook_values(vals: &[f32]) -> bool {
    vals.iter().all(|v| v.is_finite() && v.abs() <= 1.0)
        && vals.windows(2).all(|w| w[0] <= w[1])
}

struct Cursor<'a> {
    buf: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.buf[self.off..self.off + n];
        self.off += n;
        s
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().expect("4 bytes"))
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().expect("8 bytes"))
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        self.take(4 * n)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
            .collect()
    }
}

/// Derived state recovered from a validated sidecar.
struct Payload {
    boundaries: Vec<f32>,
    centroids: Vec<f32>,
    blocked: Vec<u8>,
    n_blocks: usize,
}

/// Read the sidecar next to `index_path` and validate it against the
/// already-loaded `index`. Returns the derived payload only when every
/// check passes: exact file size, trailer hash, magic/version/backend,
/// header-field equality, packed-codes content hash, and codebook value
/// sanity. Any failure — including plain absence — returns `None`.
fn read_validated(index: &TurboQuantIndex, cache: &Path) -> Option<Payload> {
    let dim = index.dim?;
    if index.n_vectors == 0 {
        return None;
    }
    let exp = expected(index.bit_width, dim, index.n_vectors)?;

    let f = File::open(cache).ok()?;
    if f.metadata().ok()?.len() != exp.total_len as u64 {
        return None;
    }
    let mut buf = Vec::with_capacity(exp.total_len);
    // take() re-bounds the read in case the file grew between the
    // stat and here; the equality check below then rejects it.
    f.take(exp.total_len as u64 + 1).read_to_end(&mut buf).ok()?;
    if buf.len() != exp.total_len {
        return None;
    }

    let (body, trailer) = buf.split_at(exp.total_len - 8);
    if hash64(body) != u64::from_le_bytes(trailer.try_into().ok()?) {
        return None;
    }

    let mut c = Cursor { buf: body, off: 0 };
    if c.take(4) != MAGIC
        || c.u8() != VERSION
        || c.u8() as usize != BACKEND_ID.len()
        || c.take(BACKEND_ID.len()) != BACKEND_ID.as_bytes()
        || c.u8() as usize != index.bit_width
        || c.u32() as usize != dim
        || c.u64() != index.n_vectors as u64
        || c.u64() != hash64(&index.packed_codes)
    {
        return None;
    }
    let boundaries = c.f32s(exp.n_levels - 1);
    let centroids = c.f32s(exp.n_levels);
    if !valid_codebook_values(&boundaries) || !valid_codebook_values(&centroids) {
        return None;
    }
    let blocked = c.take(exp.blocked_len).to_vec();
    Some(Payload {
        boundaries,
        centroids,
        blocked,
        n_blocks: exp.n_blocks,
    })
}

/// Best-effort: seed `index`'s lazy caches from a valid sidecar next to
/// `index_path`. On any validation failure this is a no-op — the caches
/// fill lazily on first search. Never touches the filesystem beyond one
/// read, and never fails the enclosing load.
pub(crate) fn seed(index: &TurboQuantIndex, index_path: &Path) {
    let Some(payload) = read_validated(index, &cache_path(index_path)) else {
        return;
    };
    let _ = index.boundaries.set(payload.boundaries);
    let _ = index.centroids.set(payload.centroids);
    let _ = index.blocked.set(BlockedCache {
        data: payload.blocked,
        n_blocks: payload.n_blocks,
    });
}

/// Best-effort: bring the sidecar next to `index_path` in sync with
/// `index`, which the caller has just written to `index_path`.
///
/// * Empty or lazy index → remove any existing sidecar.
/// * An existing sidecar that already validates against the current
///   content is left untouched (an unchanged rewrite costs one
///   sequential read, not a repack).
/// * Otherwise the derived state is materialised (reusing the lazy
///   caches when already populated) and written whole via the same
///   atomic temp-then-rename used for the index itself.
///
/// Errors are deliberately swallowed: the sidecar is an optimization,
/// and the authoritative write this call follows has already succeeded.
pub(crate) fn persist(index: &TurboQuantIndex, index_path: &Path) {
    let cache = cache_path(index_path);
    let Some(dim) = index.dim else {
        let _ = std::fs::remove_file(&cache);
        return;
    };
    if index.n_vectors == 0 {
        let _ = std::fs::remove_file(&cache);
        return;
    }
    if read_validated(index, &cache).is_some() {
        return;
    }
    let Some(exp) = expected(index.bit_width, dim, index.n_vectors) else {
        return;
    };

    if index.boundaries.get().is_none() || index.centroids.get().is_none() {
        let (b, c) = codebook::codebook(index.bit_width, dim);
        let _ = index.boundaries.set(b);
        let _ = index.centroids.set(c);
    }
    let boundaries = index.boundaries.get().expect("boundaries just initialised");
    let centroids = index.centroids.get().expect("centroids just initialised");
    let blocked = index.blocked.get_or_init(|| {
        let (data, n_blocks) =
            pack::repack(&index.packed_codes, index.n_vectors, index.bit_width, dim);
        BlockedCache { data, n_blocks }
    });
    // If the in-memory shapes ever disagree with the derived layout,
    // skip rather than persist a sidecar the loader would reject.
    if boundaries.len() != exp.n_levels - 1
        || centroids.len() != exp.n_levels
        || blocked.data.len() != exp.blocked_len
        || blocked.n_blocks != exp.n_blocks
    {
        return;
    }

    let mut buf = Vec::with_capacity(exp.total_len);
    buf.extend_from_slice(MAGIC);
    buf.push(VERSION);
    buf.push(BACKEND_ID.len() as u8);
    buf.extend_from_slice(BACKEND_ID.as_bytes());
    buf.push(index.bit_width as u8);
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    buf.extend_from_slice(&(index.n_vectors as u64).to_le_bytes());
    buf.extend_from_slice(&hash64(&index.packed_codes).to_le_bytes());
    for &v in boundaries {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &v in centroids {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&blocked.data);
    let trailer = hash64(&buf);
    buf.extend_from_slice(&trailer.to_le_bytes());

    let _ = io::write_atomic(&cache, |w| w.write_all(&buf));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IdMapIndex, TurboQuantIndex};
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("turbovec-rtc-{}-{}", nonce, name));
        std::fs::create_dir(&p).unwrap();
        p
    }

    fn lcg_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n * dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 32) as u32 as f64 / 2_147_483_648.0 - 1.0) as f32
            })
            .collect()
    }

    const DIM: usize = 16;
    const N: usize = 40; // spans two 32-vector blocks

    fn build(bits: usize) -> TurboQuantIndex {
        let mut idx = TurboQuantIndex::new(DIM, bits).unwrap();
        idx.add_2d(&lcg_vectors(N, DIM, 0xDECAF), DIM).unwrap();
        idx
    }

    fn seeded(idx: &TurboQuantIndex) -> bool {
        idx.blocked.get().is_some()
            && idx.boundaries.get().is_some()
            && idx.centroids.get().is_some()
    }

    /// Baseline: the same file loaded through the byte API, which never
    /// sees a sidecar.
    fn baseline_search(path: &Path, queries: &[f32], k: usize) -> (Vec<f32>, Vec<i64>) {
        let idx = TurboQuantIndex::from_bytes(&std::fs::read(path).unwrap()).unwrap();
        assert!(!seeded(&idx));
        let res = idx.search(queries, k);
        (res.scores, res.indices)
    }

    fn assert_search_parity(idx: &TurboQuantIndex, path: &Path) {
        let queries = lcg_vectors(3, DIM, 0xC0FFEE);
        let (base_scores, base_indices) = baseline_search(path, &queries, 5);
        let res = idx.search(&queries, 5);
        assert_eq!(res.scores, base_scores, "seeded/rebuilt score mismatch");
        assert_eq!(res.indices, base_indices, "seeded/rebuilt index mismatch");
    }

    /// Patch `bytes` at `offset` and rewrite the trailer hash so only
    /// the field-level checks can reject it.
    fn patch_with_valid_trailer(bytes: &mut [u8], offset: usize, value: u8) {
        bytes[offset] = value;
        let len = bytes.len();
        let h = hash64(&bytes[..len - 8]);
        bytes[len - 8..].copy_from_slice(&h.to_le_bytes());
    }

    #[test]
    fn write_creates_sidecar_and_load_seeds_all_caches() {
        for bits in [2usize, 3, 4] {
            let dir = temp_dir(&format!("roundtrip-{bits}"));
            let path = dir.join("index.tv");
            let idx = build(bits);
            idx.write(&path).unwrap();

            let cache = cache_path(&path);
            assert!(cache.exists(), "write must create the sidecar");

            let loaded = TurboQuantIndex::load(&path).unwrap();
            assert!(seeded(&loaded), "bits={bits}: load must seed from the sidecar");

            // Seeded blocked data must be byte-identical to a fresh repack.
            let (fresh, n_blocks) =
                pack::repack(&loaded.packed_codes, loaded.n_vectors, bits, DIM);
            let cached = loaded.blocked.get().unwrap();
            assert_eq!(cached.data, fresh);
            assert_eq!(cached.n_blocks, n_blocks);

            let queries = lcg_vectors(3, DIM, 0xC0FFEE);
            let (base_scores, base_indices) = baseline_search(&path, &queries, 5);
            let res = loaded.search(&queries, 5);
            assert_eq!(res.scores, base_scores);
            assert_eq!(res.indices, base_indices);
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn load_without_sidecar_stays_lazy_and_never_writes() {
        let dir = temp_dir("no-sidecar");
        let path = dir.join("index.tv");
        build(4).write(&path).unwrap();
        std::fs::remove_file(cache_path(&path)).unwrap();

        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert!(!seeded(&loaded), "no sidecar → caches stay lazy");
        assert_search_parity(&loaded, &path);

        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["index.tv"], "load must never create files");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn every_single_byte_corruption_is_rejected() {
        let dir = temp_dir("byte-corruption");
        let path = dir.join("index.tv");
        build(2).write(&path).unwrap();
        let cache = cache_path(&path);
        let good = std::fs::read(&cache).unwrap();

        for i in 0..good.len() {
            let mut bad = good.clone();
            bad[i] ^= 0xFF;
            std::fs::write(&cache, &bad).unwrap();
            let loaded = TurboQuantIndex::load(&path).unwrap();
            assert!(
                !seeded(&loaded),
                "corrupt byte {i} must be rejected, not seeded",
            );
            // Search parity after rejection is spot-checked (a full
            // per-byte sweep would rebuild the codebook hundreds of
            // times); rejection itself is asserted for every byte.
            if i == 0 || i == good.len() / 2 || i == good.len() - 1 {
                assert_search_parity(&loaded, &path);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truncation_and_trailing_bytes_are_rejected() {
        let dir = temp_dir("truncation");
        let path = dir.join("index.tv");
        build(2).write(&path).unwrap();
        let cache = cache_path(&path);
        let good = std::fs::read(&cache).unwrap();

        for len in [0, 1, 4, good.len() / 2, good.len() - 9, good.len() - 1] {
            std::fs::write(&cache, &good[..len]).unwrap();
            let loaded = TurboQuantIndex::load(&path).unwrap();
            assert!(!seeded(&loaded), "truncated to {len} must be rejected");
        }

        let mut longer = good.clone();
        longer.push(0);
        std::fs::write(&cache, &longer).unwrap();
        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert!(!seeded(&loaded), "trailing byte must be rejected");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hash_valid_but_mismatching_fields_are_rejected() {
        let dir = temp_dir("field-mismatch");
        let path = dir.join("index.tv");
        build(2).write(&path).unwrap();
        let cache = cache_path(&path);
        let good = std::fs::read(&cache).unwrap();

        // (offset, value, label): version byte, a backend-id byte, and
        // the low byte of n_vectors — each patched with a recomputed
        // trailer hash so only the field equality checks can reject.
        let n_off = 4 + 1 + 1 + BACKEND_ID.len() + 1 + 4;
        let cases = [
            (4usize, 2u8, "version"),
            (6, good[6] ^ 0x20, "backend id"),
            (n_off, good[n_off].wrapping_add(1), "n_vectors"),
        ];
        for (off, val, label) in cases {
            let mut bad = good.clone();
            patch_with_valid_trailer(&mut bad, off, val);
            std::fs::write(&cache, &bad).unwrap();
            let loaded = TurboQuantIndex::load(&path).unwrap();
            assert!(!seeded(&loaded), "{label} mismatch must be rejected");
            assert_search_parity(&loaded, &path);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stale_sidecar_is_rejected_when_index_content_changes() {
        let dir = temp_dir("stale");
        let path = dir.join("index.tv");
        build(4).write(&path).unwrap(); // index A + sidecar for A

        // Replace the authoritative file with different content of the
        // SAME shape through the raw io layer (which manages no
        // sidecars), leaving A's sidecar in place — same file size,
        // same header fields, different packed bytes.
        let mut b = TurboQuantIndex::new(DIM, 4).unwrap();
        b.add_2d(&lcg_vectors(N, DIM, 0xBEEF), DIM).unwrap();
        io::write(
            &path,
            b.bit_width(),
            DIM,
            b.len(),
            b.packed_codes(),
            b.scales(),
            b.tqplus_shift(),
            b.tqplus_scale(),
        )
        .unwrap();

        let loaded = TurboQuantIndex::load(&path).unwrap();
        assert!(!seeded(&loaded), "stale sidecar must fail the content hash");
        assert_search_parity(&loaded, &path);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rewrite_repairs_corrupt_sidecar_and_empty_write_removes_it() {
        let dir = temp_dir("repair-remove");
        let path = dir.join("index.tv");
        let idx = build(4);
        idx.write(&path).unwrap();
        let cache = cache_path(&path);

        // Corrupt, then write the same index again: persist must
        // detect the invalid sidecar and rewrite it.
        let mut bytes = std::fs::read(&cache).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&cache, &bytes).unwrap();
        idx.write(&path).unwrap();
        assert!(seeded(&TurboQuantIndex::load(&path).unwrap()));

        // Writing an empty index over the same path removes the sidecar.
        TurboQuantIndex::new(DIM, 4).unwrap().write(&path).unwrap();
        assert!(!cache.exists(), "empty write must remove the sidecar");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn add_after_seeded_load_then_write_refreshes_sidecar() {
        let dir = temp_dir("add-after-load");
        let path = dir.join("index.tv");
        build(4).write(&path).unwrap();

        let mut loaded = TurboQuantIndex::load(&path).unwrap();
        assert!(seeded(&loaded));
        loaded.add_2d(&lcg_vectors(7, DIM, 0xF00D), DIM).unwrap();
        loaded.write(&path).unwrap();

        let reloaded = TurboQuantIndex::load(&path).unwrap();
        assert!(seeded(&reloaded), "refreshed sidecar must seed again");
        assert_eq!(reloaded.len(), N + 7);
        assert_search_parity(&reloaded, &path);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn idmap_write_and_load_seed_inner_caches() {
        let dir = temp_dir("idmap");
        let path = dir.join("index.tvim");
        let mut idx = IdMapIndex::new(DIM, 4).unwrap();
        let ids: Vec<u64> = (100..100 + N as u64).collect();
        idx.add_with_ids(&lcg_vectors(N, DIM, 0xDECAF), &ids).unwrap();
        idx.write(&path).unwrap();
        assert!(cache_path(&path).exists());

        let loaded = IdMapIndex::load(&path).unwrap();
        assert!(seeded(loaded.inner()), "IdMapIndex load must seed inner caches");

        let queries = lcg_vectors(3, DIM, 0xC0FFEE);
        let baseline = IdMapIndex::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        assert!(!seeded(baseline.inner()));
        assert_eq!(loaded.search(&queries, 5), baseline.search(&queries, 5));

        // Removal invalidates content → write refreshes → reload seeds.
        let mut loaded = loaded;
        assert!(loaded.remove(105));
        loaded.write(&path).unwrap();
        let reloaded = IdMapIndex::load(&path).unwrap();
        assert!(seeded(reloaded.inner()));
        let baseline = IdMapIndex::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(reloaded.search(&queries, 5), baseline.search(&queries, 5));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tv_and_tvim_sidecars_do_not_collide() {
        let tv = PathBuf::from("/tmp/foo.tv");
        let tvim = PathBuf::from("/tmp/foo.tvim");
        assert_ne!(cache_path(&tv), cache_path(&tvim));
        assert!(cache_path(&tv)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with(&format!(".{BACKEND_ID}.cache")));
    }
}
