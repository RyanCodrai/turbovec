//! The planes layout (`pack::planes_for`) against the classic one.
//!
//! The layout is opt-in through the environment, read once per process, so
//! these tests switch it on for their own thread with `pack::PLANES_TEST`
//! instead. A host whose kernels cannot read the layout (x86 without
//! AVX-512 VBMI/VNNI) skips every test that builds it: the sign region is
//! packed in the host's native block layout, which is only the layout
//! these helpers address when the host can scan it.

use crate::{pack, TurboQuantIndex, BLOCK};

/// Switch the planes layout on for this thread, with the given size gate.
/// Also holds off the scalar-fallback test for as long as it lives: these
/// tests compare scores bit for bit across searches.
struct PlanesOn(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);
impl PlanesOn {
    fn new(min_vectors: usize) -> Self {
        let gate = crate::search::SCALAR_FALLBACK_GATE.read().unwrap_or_else(|e| e.into_inner());
        pack::PLANES_TEST.with(|c| c.set(Some((true, min_vectors))));
        PlanesOn(gate)
    }
}
impl Drop for PlanesOn {
    fn drop(&mut self) {
        pack::PLANES_TEST.with(|c| c.set(None));
    }
}

/// Run `f` with the layout forced off on this thread.
fn classic<R>(f: impl FnOnce() -> R) -> R {
    let prev = pack::PLANES_TEST.with(|c| c.replace(Some((false, usize::MAX))));
    let r = f();
    pack::PLANES_TEST.with(|c| c.set(prev));
    r
}

/// Whether this host has a kernel for the layout at `dim`.
fn planes_supported(dim: usize) -> bool {
    let _on = PlanesOn::new(0);
    let ok = pack::planes_for(2, dim / 4);
    if !ok {
        eprintln!("planes layout unsupported on this host; skipping");
    }
    ok
}

fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = vec![0.0f32; n * dim];
    for row in out.chunks_mut(dim) {
        let mut norm = 0.0f64;
        for x in row.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Symmetric about zero: 31 random bits over 2^30, less one.
            let v = ((s >> 33) as f64 / (1u64 << 30) as f64) - 1.0;
            *x = v as f32;
            norm += v * v;
        }
        let inv = 1.0 / (norm.sqrt() + 1e-9);
        for x in row.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
    out
}

/// Random packed bit-plane rows for `n` 2-bit vectors.
fn packed_rows(n: usize, dim: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n * 2 * (dim / 8))
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        })
        .collect()
}

const DIM: usize = 64;
const NBG: usize = DIM / 4;

/// An index over `data` with its search cache built, so the cache's layout
/// is the one in force on this thread now.
fn build(data: &[f32]) -> TurboQuantIndex {
    let mut ix = TurboQuantIndex::new(DIM, 2).unwrap();
    ix.add(data);
    let _ = ix.search(&data[..DIM], 1);
    ix
}

fn is_planes(ix: &TurboQuantIndex) -> bool {
    ix.blocked.get().is_some_and(|c| c.is_planes())
}

// ---------------------------------------------------------------- layout

#[test]
fn gather_and_spread_tables_invert_each_other() {
    for c0 in 0..=255u8 {
        for c1 in [0u8, 0x1B, 0x6C, 0xB1, 0xFF, c0.rotate_left(3)] {
            // One sign byte and one low byte cover these two code bytes.
            let sign = ((0..4).fold(0u8, |a, j| a | (((c0 >> (7 - 2 * j)) & 1) << (7 - j))))
                | ((0..4).fold(0u8, |a, j| a | (((c1 >> (7 - 2 * j)) & 1) << (3 - j))));
            let low = ((0..4).fold(0u8, |a, j| a | (((c0 >> (6 - 2 * j)) & 1) << (7 - j))))
                | ((0..4).fold(0u8, |a, j| a | (((c1 >> (6 - 2 * j)) & 1) << (3 - j))));
            assert_eq!(pack::planes_to_code_bytes(sign, low), (c0, c1), "c0={c0:#x} c1={c1:#x}");
            assert_eq!(
                pack::PLANES_COMB[((sign & 0xF0) | (low >> 4)) as usize],
                c0,
                "comb hi c0={c0:#x}"
            );
            assert_eq!(
                pack::PLANES_COMB[(((sign & 15) << 4) | (low & 15)) as usize],
                c1,
                "comb lo c1={c1:#x}"
            );
        }
    }
}

#[test]
fn seq_round_trips_through_the_two_regions() {
    if !planes_supported(DIM) {
        return;
    }
    for n in [1usize, 31, 32, 33, 77, 200] {
        let packed = packed_rows(n, DIM, 7 + n as u64);
        let seq = pack::repack_seq(&packed, n, 2, DIM);
        let (sign, low) = pack::planes_from_seq(&seq, NBG, n);
        assert_eq!(sign.len(), n.div_ceil(BLOCK) * (NBG / 2) * BLOCK);
        assert_eq!(low.len(), n * (NBG / 2));
        assert_eq!(pack::planes_to_seq(&sign, &low, NBG, n), seq, "n={n}");
    }
}

#[test]
fn repack_from_packed_matches_the_seq_route() {
    if !planes_supported(DIM) {
        return;
    }
    for n in [5usize, 64, 97] {
        let packed = packed_rows(n, DIM, 11 + n as u64);
        let (sign, low, n_blocks) = pack::planes_repack(&packed, n, DIM);
        assert_eq!(n_blocks, n.div_ceil(BLOCK));
        let seq = pack::repack_seq(&packed, n, 2, DIM);
        assert_eq!((sign, low), pack::planes_from_seq(&seq, NBG, n), "n={n}");
    }
}

#[test]
fn read_row_returns_each_vectors_code_bytes() {
    if !planes_supported(DIM) {
        return;
    }
    let n = 70;
    let packed = packed_rows(n, DIM, 3);
    let (sign, low, _) = pack::planes_repack(&packed, n, DIM);
    let flat = pack::extract_codes_flat(&packed, n, 2, DIM);
    for v in 0..n {
        assert_eq!(
            pack::planes_read_row(&sign, &low, NBG, v),
            &flat[v * NBG..(v + 1) * NBG],
            "v={v}"
        );
    }
}

#[test]
fn append_and_block_range_patch_equal_a_full_repack() {
    if !planes_supported(DIM) {
        return;
    }
    let (n0, n1) = (45usize, 60usize);
    let packed = packed_rows(n0 + n1, DIM, 19);
    let row = 2 * (DIM / 8);
    let whole = pack::planes_repack(&packed, n0 + n1, DIM);

    // Appending lanes to a cache of the first n0 rows.
    let (mut sign, mut low, _) = pack::planes_repack(&packed[..n0 * row], n0, DIM);
    pack::planes_append_lanes(&mut sign, &mut low, &packed[n0 * row..], n0, n1, DIM);
    assert_eq!((&sign, &low), (&whole.0, &whole.1), "append");

    // Rebuilding from the tail block on, as `add`'s patch path does.
    let (mut sign, mut low, _) = pack::planes_repack(&packed[..n0 * row], n0, DIM);
    let first_block = n0 / BLOCK;
    let n_blocks = (n0 + n1).div_ceil(BLOCK);
    let (sp, lp) = pack::planes_repack_block_range(&packed, n0 + n1, DIM, first_block, n_blocks);
    sign.truncate(first_block * (NBG / 2) * BLOCK);
    sign.extend_from_slice(&sp);
    low.truncate(first_block * BLOCK * (NBG / 2));
    low.extend_from_slice(&lp);
    assert_eq!((&sign, &low), (&whole.0, &whole.1), "patch");
}

#[test]
fn move_carries_a_vector_in_both_regions() {
    if !planes_supported(DIM) {
        return;
    }
    let n = 80;
    let packed = packed_rows(n, DIM, 23);
    let (mut sign, mut low, _) = pack::planes_repack(&packed, n, DIM);
    let want = pack::planes_read_row(&sign, &low, NBG, 79);
    let keep = pack::planes_read_row(&sign, &low, NBG, 11);
    pack::planes_move(&mut sign, &mut low, NBG, 79, 10);
    assert_eq!(pack::planes_read_row(&sign, &low, NBG, 10), want);
    assert_eq!(pack::planes_read_row(&sign, &low, NBG, 11), keep);
}

#[test]
fn outer_fraction_counts_codes_whose_bits_agree() {
    if !planes_supported(DIM) {
        return;
    }
    // All codes 0b11: every field outer. All codes 0b10: none.
    for (byte, want) in [(0xFFu8, 1.0f32), (0xAA, 0.0), (0x00, 1.0), (0x55, 0.0)] {
        let n = 40;
        let seq_rows = vec![byte; n * NBG];
        let n_blocks = n.div_ceil(BLOCK);
        let seq = pack::pack_blocked_sequential(n, n_blocks, NBG, n_blocks * NBG * BLOCK, &seq_rows);
        let (sign, low) = pack::planes_from_seq(&seq, NBG, n);
        assert_eq!(pack::planes_outer_frac(&sign, &low, n, NBG), want, "byte={byte:#x}");
    }
}

#[test]
fn size_gate_and_sample_thresholds() {
    {
        let _on = PlanesOn::new(100);
        if pack::planes_for(2, NBG) {
            assert!(!pack::planes_wanted(2, NBG, 0));
            assert!(!pack::planes_wanted(2, NBG, 99));
            assert!(pack::planes_wanted(2, NBG, 100));
            assert!(!pack::planes_wanted(4, NBG, 100), "4-bit never takes the layout");
            assert!(!pack::planes_wanted(2, NBG + 4, 100), "needs whole 8-group units");
        }
    }
    assert!(!classic(|| pack::planes_wanted(2, NBG, 1 << 20)));
    assert_eq!(pack::planes_min_vectors(), pack::PLANES_MIN_VECTORS);

    // The threshold sample exists only for indexes of 1024 full blocks.
    let nsg = NBG / 2;
    let small = 1023 * BLOCK + 31;
    let sign = vec![0u8; (small.div_ceil(BLOCK)) * nsg * BLOCK];
    assert!(pack::planes_sample(&sign, &vec![1.0; small], small, NBG).is_none());
    let big = 1024 * BLOCK;
    let sign: Vec<u8> = (0..big / BLOCK * nsg * BLOCK).map(|i| (i / (nsg * BLOCK)) as u8).collect();
    let scales: Vec<f32> = (0..big).map(|v| v as f32).collect();
    let (codes, s) = pack::planes_sample(&sign, &scales, big, NBG).expect("sample");
    assert_eq!(codes.len(), 48 * nsg * BLOCK);
    assert_eq!(s.len(), 48 * BLOCK);
    // Blocks are whole, strided, and carry their own scales.
    let stride = 1024 / 48;
    for i in 0..48 {
        let b = i * stride + stride / 2;
        assert_eq!(codes[i * nsg * BLOCK], b as u8, "sample block {i}");
        assert_eq!(s[i * BLOCK], (b * BLOCK) as f32, "sample scales {i}");
    }
}

#[test]
fn a_capable_host_takes_the_layout() {
    // Every other test here skips when `planes_for` says no, so one test
    // has to say when it must say yes: wherever the sign region's kernels
    // exist. (On x86 that is the vector-major layout, which needs AVX-512
    // VBMI + VNNI and can be switched off from the environment.)
    let _on = PlanesOn::new(0);
    let kernels = if cfg!(target_arch = "x86_64") {
        pack::use_vector_major()
    } else {
        cfg!(target_arch = "aarch64") && !pack::use_vm8_2bit()
    };
    assert_eq!(pack::planes_for(2, NBG), kernels);
    #[cfg(target_arch = "aarch64")]
    assert!(std::env::var_os("TURBOVEC_2BIT_VM8").is_some() || pack::planes_for(2, NBG));
}

// ---------------------------------------------------------------- index

/// `(id, score bits)` rows of a search.
fn rows(ix: &TurboQuantIndex, q: &[f32], k: usize) -> Vec<Vec<(i64, u32)>> {
    let r = ix.search(q, k);
    (0..r.nq)
        .map(|qi| {
            (0..r.k)
                .map(|j| (r.indices[qi * r.k + j], r.scores[qi * r.k + j].to_bits()))
                .collect()
        })
        .collect()
}

#[test]
fn a_shortlist_that_covers_the_index_reproduces_the_exact_scan() {
    if !planes_supported(DIM) {
        return;
    }
    // 100 vectors < the 128-candidate shortlist: every vector is rescored,
    // so ids, order and score bits must equal the classic index's.
    let data = unit_vectors(100, DIM, 1);
    let q = unit_vectors(9, DIM, 2);
    let base = classic(|| build(&data));
    let _on = PlanesOn::new(0);
    let ix = build(&data);
    assert!(is_planes(&ix) && !is_planes(&base));
    for k in [1usize, 10, 100, 500] {
        assert_eq!(rows(&ix, &q, k), rows(&base, &q, k), "batched k={k}");
        assert_eq!(rows(&ix, &q[..DIM], k), rows(&base, &q[..DIM], k), "single k={k}");
    }
}

/// Queries that each sit beside one database vector, as an embedding query
/// sits beside its neighbours. (On structureless data a top-k is decided by
/// noise-sized margins and a shortlist legitimately disagrees with the
/// exact scan about most of it.)
fn near_queries(data: &[f32], nq: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / DIM;
    let noise = unit_vectors(nq, DIM, seed);
    let mut q = Vec::with_capacity(nq * DIM);
    for qi in 0..nq {
        let row = &data[(qi * 97 + 13) % n * DIM..][..DIM];
        q.extend(row.iter().zip(&noise[qi * DIM..][..DIM]).map(|(a, b)| a + 0.3 * b));
    }
    q
}

#[test]
fn a_high_dimensional_rescore_matches_the_exact_scan_bit_for_bit() {
    // 1536 dims is 384 byte-groups: past the 256 at which the aarch64
    // kernels flush their integer sums into the float accumulator, so the
    // rescore's own flush points are exercised (at 64 dims they never are).
    const D: usize = 1536;
    if !planes_supported(D) {
        return;
    }
    let unit = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut out = vec![0.0f32; n * D];
        for row in out.chunks_mut(D) {
            let mut norm = 0.0f64;
            for x in row.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((s >> 33) as f64 / (1u64 << 30) as f64) - 1.0;
                *x = v as f32;
                norm += v * v;
            }
            let inv = 1.0 / (norm.sqrt() + 1e-9);
            for x in row.iter_mut() {
                *x = (*x as f64 * inv) as f32;
            }
        }
        out
    };
    // 120 vectors < the 128-candidate shortlist: every vector is rescored.
    let data = unit(120, 131);
    let q = unit(7, 132);
    let make = || {
        let mut ix = TurboQuantIndex::new(D, 2).unwrap();
        ix.add(&data);
        let _ = ix.search(&data[..D], 1);
        ix
    };
    let base = classic(make);
    let _on = PlanesOn::new(0);
    let ix = make();
    assert!(is_planes(&ix) && !is_planes(&base));
    for k in [1usize, 10, 120] {
        assert_eq!(rows(&ix, &q, k), rows(&base, &q, k), "batched k={k}");
        assert_eq!(rows(&ix, &q[..D], k), rows(&base, &q[..D], k), "single k={k}");
    }
}

/// Every score a planes search returns is the exact scan's score for that
/// id, in order, and each query's best match is the exact scan's.
fn assert_exact_scores(n: usize, nq: usize, k: usize, single: bool) {
    let data = unit_vectors(n, DIM, 5);
    let q = near_queries(&data, nq, 6);
    let base = classic(|| build(&data));
    let _on = PlanesOn::new(0);
    let ix = build(&data);
    assert!(is_planes(&ix));
    let (mut hit, mut total) = (0usize, 0usize);
    for qi in 0..nq {
        let one = &q[qi * DIM..(qi + 1) * DIM];
        let all = base.search(one, n);
        let exact: std::collections::HashMap<i64, u32> =
            all.indices.iter().zip(&all.scores).map(|(&i, &s)| (i, s.to_bits())).collect();
        let top: std::collections::HashSet<i64> = all.indices[..k].iter().copied().collect();
        let got = if single {
            rows(&ix, one, k).remove(0)
        } else {
            rows(&ix, &q, k).remove(qi)
        };
        assert_eq!(got.len(), k);
        assert_eq!(got[0].0, all.indices[0], "q={qi}: best match differs from the exact scan's");
        let mut seen = std::collections::HashSet::new();
        for (j, &(id, bits)) in got.iter().enumerate() {
            assert!(seen.insert(id), "duplicate id {id}");
            assert_eq!(exact[&id], bits, "q={qi} id={id}: score is not the exact scan's");
            if j > 0 {
                assert!(
                    f32::from_bits(got[j - 1].1) >= f32::from_bits(bits),
                    "q={qi}: results out of order"
                );
            }
            hit += top.contains(&id) as usize;
            total += 1;
        }
    }
    eprintln!("planes overlap n={n} k={k} single={single}: {:.4}", hit as f64 / total as f64);
}

#[test]
fn batched_planes_search_returns_exact_scores() {
    if planes_supported(DIM) {
        assert_exact_scores(5_000, 24, 10, false);
        // Large enough for the sample-seeded scan; many queries, so the
        // per-query unseeded rescan of a short one is likely exercised.
        assert_exact_scores(1_100 * BLOCK + 7, 200, 10, false);
    }
}

#[test]
fn single_query_planes_search_returns_exact_scores() {
    if planes_supported(DIM) {
        // Below the block-parallel gate: the tiled path, unseeded.
        assert_exact_scores(5_000, 12, 10, true);
        // Above it: the pooled single-query scan, seeded from the sample.
        assert_exact_scores(1_100 * BLOCK + 7, 12, 10, true);
        assert_exact_scores(1_100 * BLOCK + 7, 4, 100, true);
    }
}

#[test]
fn masked_planes_search_respects_the_mask_and_keeps_exact_scores() {
    if !planes_supported(DIM) {
        return;
    }
    let n = 3_000;
    let data = unit_vectors(n, DIM, 8);
    let q = near_queries(&data, 6, 9);
    let mask: Vec<bool> = (0..n).map(|v| v % 3 != 0).collect();
    let base = classic(|| build(&data));
    let _on = PlanesOn::new(0);
    let ix = build(&data);
    let got = ix.search_with_mask(&q, 10, Some(&mask));
    let all = base.search_with_mask(&q, n, Some(&mask));
    for qi in 0..6 {
        let exact: std::collections::HashMap<i64, u32> = (0..all.k)
            .map(|j| (all.indices[qi * all.k + j], all.scores[qi * all.k + j].to_bits()))
            .collect();
        for j in 0..got.k {
            let id = got.indices[qi * got.k + j];
            assert!(mask[id as usize], "masked-out id {id} returned");
            assert_eq!(exact[&id], got.scores[qi * got.k + j].to_bits());
        }
    }
    // A mask leaving fewer vectors than the shortlist rescoring them all.
    let few: Vec<bool> = (0..n).map(|v| v < 40).collect();
    assert_eq!(
        ix.search_with_mask(&q, 10, Some(&few)).indices,
        base.search_with_mask(&q, 10, Some(&few)).indices
    );
}

#[test]
fn bytes_do_not_depend_on_the_layout() {
    if !planes_supported(DIM) {
        return;
    }
    let data = unit_vectors(333, DIM, 12);
    let base = classic(|| build(&data));
    let want = base.to_bytes();
    let _on = PlanesOn::new(0);
    let ix = build(&data);
    assert!(is_planes(&ix));
    assert_eq!(ix.to_bytes(), want, "a planes index serializes to the classic bytes");
    // Loading under the layout lands in it and searches like a built one.
    let loaded = TurboQuantIndex::from_bytes(&want).unwrap();
    let q = unit_vectors(5, DIM, 13);
    assert_eq!(rows(&loaded, &q, 10), rows(&ix, &q, 10));
    assert!(is_planes(&loaded));
    assert_eq!(loaded.to_bytes(), want);
}

#[test]
fn mutations_keep_the_two_layouts_in_step() {
    if !planes_supported(DIM) {
        return;
    }
    let data = unit_vectors(400, DIM, 21);
    let rows_of = |a: usize, b: usize| &data[a * DIM..b * DIM];
    let mut base = classic(|| build(rows_of(0, 100)));
    let _on = PlanesOn::new(0);
    let mut ix = build(rows_of(0, 100));
    let q = unit_vectors(3, DIM, 22);
    let step = |ix: &TurboQuantIndex, base: &TurboQuantIndex, what: &str| {
        assert!(is_planes(ix), "{what}: left the planes layout");
        assert_eq!(ix.len(), base.len(), "{what}");
        assert_eq!(ix.to_bytes(), classic(|| base.to_bytes()), "{what}: bytes");
        let _ = ix.search(&q, 5);
    };
    step(&ix, &base, "build");
    for (a, b) in [(100, 131), (131, 132), (132, 260)] {
        ix.add(rows_of(a, b));
        classic(|| base.add(rows_of(a, b)));
        step(&ix, &base, "add");
    }
    for idx in [5usize, 0, 200, 256] {
        assert_eq!(ix.swap_remove(idx), classic(|| base.swap_remove(idx)));
        step(&ix, &base, "swap_remove");
    }
    // Searching (which warms the cache) between mutations, then adding
    // through the cache-only path a loaded index takes.
    let mut reloaded = TurboQuantIndex::from_bytes(&ix.to_bytes()).unwrap();
    let mut base2 = classic(|| TurboQuantIndex::from_bytes(&base.to_bytes()).unwrap());
    reloaded.add(rows_of(260, 330));
    classic(|| base2.add(rows_of(260, 330)));
    step(&reloaded, &base2, "add after load");
    assert_eq!(reloaded.swap_remove(7), classic(|| base2.swap_remove(7)));
    step(&reloaded, &base2, "swap_remove after load");
}

#[test]
fn an_index_takes_the_layout_when_it_grows_past_the_gate() {
    if !planes_supported(DIM) {
        return;
    }
    let data = unit_vectors(300, DIM, 31);
    let base = classic(|| build(&data));
    let _on = PlanesOn::new(150);
    let mut ix = build(&data[..100 * DIM]);
    let q = unit_vectors(4, DIM, 32);
    let _ = ix.search(&q, 5);
    assert!(!is_planes(&ix), "100 vectors is under the gate");
    ix.add(&data[100 * DIM..149 * DIM]);
    assert!(!is_planes(&ix), "149 vectors is under the gate");
    ix.add(&data[149 * DIM..150 * DIM]);
    assert!(is_planes(&ix), "150 vectors reaches the gate");
    ix.add(&data[150 * DIM..]);
    assert_eq!(ix.to_bytes(), classic(|| base.to_bytes()));
    // A promoted cache searches exactly like one built in the layout.
    let built = build(&data);
    assert_eq!(rows(&ix, &q, 10), rows(&built, &q, 10));
    // Shrinking back under the gate keeps the layout.
    while ix.len() > 120 {
        ix.swap_remove(0);
    }
    assert!(is_planes(&ix));
    let mut shrunk = classic(|| build(&data));
    classic(|| {
        while shrunk.len() > 120 {
            shrunk.swap_remove(0);
        }
    });
    assert_eq!(ix.to_bytes(), classic(|| shrunk.to_bytes()));
}

#[test]
fn four_bit_and_odd_geometries_stay_classic() {
    let _on = PlanesOn::new(0);
    let mut ix = TurboQuantIndex::new(DIM, 4).unwrap();
    ix.add(&unit_vectors(50, DIM, 41));
    assert!(!is_planes(&ix));
    // dim 40 -> 10 byte-groups, not a multiple of 8.
    let mut ix = TurboQuantIndex::new(40, 2).unwrap();
    ix.add(&unit_vectors(50, 40, 42));
    assert!(!is_planes(&ix));
}

#[test]
fn the_layout_holds_the_same_bytes_per_vector() {
    if !planes_supported(DIM) {
        return;
    }
    // Built in one add, and grown by many: the cache's allocations, not
    // just its lengths, so growth headroom is counted too.
    let data = unit_vectors(40_000, DIM, 51);
    let cache_bytes = |ix: &TurboQuantIndex| {
        let c = ix.blocked.get().expect("cache");
        (c.data.len() + c.low.len(), c.data.capacity() + c.low.capacity())
    };
    let grow = |ix: &mut TurboQuantIndex| {
        for chunk in data[8_000 * DIM..].chunks(1_000 * DIM) {
            ix.add(chunk);
        }
    };
    let mut base = classic(|| build(&data[..8_000 * DIM]));
    let built_classic = cache_bytes(&base);
    classic(|| grow(&mut base));
    let grown_classic = cache_bytes(&base);
    let _on = PlanesOn::new(0);
    let mut ix = build(&data[..8_000 * DIM]);
    let built = cache_bytes(&ix);
    grow(&mut ix);
    let grown = cache_bytes(&ix);
    eprintln!("cache bytes (len, capacity): built classic {built_classic:?} planes {built:?}; grown classic {grown_classic:?} planes {grown:?}");
    // The sign region rounds up to whole blocks like the code buffer it
    // replaces; the low region holds exactly one row per vector.
    assert_eq!(built.0, built_classic.0);
    assert!(grown.0 <= grown_classic.0);
    assert!(built.1 <= built_classic.1);
    assert!(
        grown.1 as f64 <= grown_classic.1 as f64 * 1.02,
        "planes cache allocates {} bytes against the classic layout's {}",
        grown.1,
        grown_classic.1
    );
}

#[test]
fn low_dot_kernels_match_the_scalar_sum() {
    use crate::search::{build_low_planes, low_dot, low_dot_scalar};
    // 64 and 320 fill whole chunks; 96 and 1568 leave 12- and 4-byte tails.
    for dim in [64usize, 96, 320, 1536, 1568] {
        let q = unit_vectors(1, dim, 61 + dim as u64);
        let planes = build_low_planes(&q, 1.0, dim);
        // The weights the masks encode, recovered coordinate by coordinate.
        let q_max = q.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        let w: Vec<i64> = q
            .iter()
            .map(|&x| {
                let m = ((x.abs() * 63.0 / q_max + 0.5) as i64).min(63);
                if x < 0.0 { -m } else { m }
            })
            .collect();
        for seed in 0..20u64 {
            let row = &packed_rows(1, dim, 71 + seed)[..dim / 8];
            let want: i64 =
                (0..dim).filter(|&i| row[i / 8] & (0x80 >> (i % 8)) != 0).map(|i| w[i]).sum();
            assert_eq!(low_dot_scalar(planes.masks(), row), want, "scalar dim={dim} seed={seed}");
            assert_eq!(low_dot(&planes, row), want, "kernel dim={dim} seed={seed}");
        }
        // Every bit set: the signed weights' own sum.
        assert_eq!(low_dot(&planes, &vec![0xFF; dim / 8]), w.iter().sum::<i64>());
        assert_eq!(low_dot(&planes, &vec![0; dim / 8]), 0);
    }
    // A zero query has no weights.
    let planes = build_low_planes(&[0.0; 64], 1.0, 64);
    assert_eq!(low_dot(&planes, &[0xFF; 8]), 0);
}
