//! Per-block TQ+ calibration: sealing, per-block search, and the
//! block-local `swap_remove` that the block extents make expressible.
//!
//! The property that motivates the whole model is
//! `a_drifting_stream_no_longer_freezes_one_global_calibration`: an
//! index fed a stream whose distribution moves commits one calibration
//! describing only the head of that stream, and quantizes everything
//! after it in a coordinate system fitted to data that does not look
//! like it. Blocks give each run of rows its own fit — measured here at
//! R@10 0.47 against 0.05 for a single fit, at 2 bits.

use turbovec::{CalibrationState, ConstructError, TurboQuantIndex, MIN_BLOCK_SIZE};

const DIM: usize = 64;
/// Above `TQPLUS_MIN_SAMPLES` (1000), so a sealing block has enough
/// rows for a real fit rather than the identity fallback.
const BS: usize = 1024;

/// Deterministic unit-norm rows. Not a great RNG; it only has to be
/// reproducible and to spread mass over every coordinate.
fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = vec![0.0f32; n * dim];
    for row in out.chunks_mut(dim) {
        let mut norm = 0.0f64;
        for x in row.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 33) as f64 / (1u64 << 30) as f64) - 1.0;
            *x = v as f32;
            norm += v * v;
        }
        let inv = 1.0 / (norm.sqrt() + 1e-12);
        for x in row.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
    out
}

/// A stream whose mean direction rotates steadily down the batch: row
/// `i` is noise plus a strong bias along `cos/sin` of an angle that
/// sweeps a quarter turn over `n` rows.
///
/// This is the shape a single global fit is worst at, and the shape real
/// ingest has whenever rows arrive clustered by topic, tenant or time.
/// Sorting i.i.d. rows by one raw coordinate does *not* reproduce it:
/// the block-Hadamard rotation smears that coordinate across all of
/// them, so the rotated marginals every block sees come out nearly
/// identical and the fits barely move.
fn drifting_rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut out = rows(n, dim, seed);
    for (i, row) in out.chunks_mut(dim).enumerate() {
        let t = i as f32 / n as f32 * std::f32::consts::FRAC_PI_2;
        let (a, b) = (t.cos(), t.sin());
        for (d, x) in row.iter_mut().enumerate() {
            let lobe = if d % 2 == 0 { a } else { b };
            *x = *x * 0.35 + lobe;
        }
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    out
}

/// Exact inner-product top-k over the uncompressed rows.
fn exact_topk(data: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let s: f32 = (0..dim).map(|d| data[i * dim + d] * query[d]).sum();
            (s, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Mean recall@k of `idx` against the exact answer over `queries`.
fn recall(idx: &TurboQuantIndex, data: &[f32], n: usize, queries: &[f32], nq: usize, k: usize) -> f64 {
    let results = idx.search(queries, k);
    let mut hits = 0usize;
    for q in 0..nq {
        let truth: std::collections::HashSet<usize> =
            exact_topk(data, n, DIM, &queries[q * DIM..(q + 1) * DIM], k)
                .into_iter()
                .collect();
        for &got in results.indices_for_query(q) {
            if truth.contains(&(got as usize)) {
                hits += 1;
            }
        }
    }
    hits as f64 / (nq * k) as f64
}

#[test]
fn block_size_must_be_a_positive_multiple_of_the_granularity() {
    // 32-row SIMD blocks and 64-slot mask words both have to start
    // fresh at a block boundary, so 64 is the only granularity at which
    // a block is searchable as a self-contained range.
    for bad in [0, 1, 63, 65, 100] {
        let err = TurboQuantIndex::with_block_size(DIM, 4, bad).unwrap_err();
        assert!(
            matches!(err, ConstructError::BlockSizeInvalid { block_size, granularity }
                     if block_size == bad && granularity == MIN_BLOCK_SIZE),
            "block_size {bad} was accepted or misreported: {err}"
        );
    }
    assert!(TurboQuantIndex::with_block_size(DIM, 4, MIN_BLOCK_SIZE).is_ok());
    assert!(TurboQuantIndex::with_block_size(DIM, 4, BS).is_ok());
}

#[test]
fn blocks_seal_when_they_fill_and_never_after() {
    let n = 3 * BS + 100;
    let data = rows(n, DIM, 0xB10C);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();

    // One row at a time across the first boundary: the split has to
    // happen inside `add`, not at the caller's batch edges.
    for i in 0..BS {
        idx.add(&data[i * DIM..(i + 1) * DIM]);
        assert_eq!(
            idx.sealed_blocks(),
            usize::from(i + 1 == BS),
            "block sealed at the wrong row ({i})"
        );
    }
    // …and the rest in one batch that spans two more boundaries.
    idx.add(&data[BS * DIM..]);
    assert_eq!(idx.sealed_blocks(), 3);
    assert_eq!(idx.len(), n);
    assert_eq!(idx.slot_capacity(), n, "no removals, so no dead rows");
    assert_eq!(idx.block_size(), Some(BS));
}

#[test]
fn each_sealed_block_gets_its_own_fit() {
    // Sorted input is what makes the fits differ by construction: each
    // block sees a different slice of the coordinate's range.
    let n = 3 * BS;
    let data = drifting_rows(n, DIM, 0x5011);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&data);
    assert_eq!(idx.sealed_blocks(), 3);

    // The open block is empty and carries the last sealed block's pair,
    // so the *stored* calibration is the third block's. What has to be
    // true is that the fit moved between blocks at all — a single global
    // fit is exactly the state where it does not.
    let mut single = TurboQuantIndex::new(DIM, 4).unwrap();
    single.add(&data);
    assert_ne!(
        idx.tqplus_shift(),
        single.tqplus_shift(),
        "the last block's fit equals the whole-index fit, so nothing was refitted per block"
    );
}

#[test]
fn a_blocked_index_matches_a_single_block_one_on_iid_rows() {
    // Blocks are a bet on non-i.i.d. input. On i.i.d. input every block
    // fits nearly the same calibration, so the bet must cost nothing.
    let n = 4 * BS;
    let nq = 40;
    let k = 10;
    let data = rows(n, DIM, 0x11D);
    let queries = rows(nq, DIM, 0x9E12);

    let mut blocked = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    blocked.add(&data);
    let mut single = TurboQuantIndex::new(DIM, 4).unwrap();
    single.add(&data);

    let r_blocked = recall(&blocked, &data, n, &queries, nq, k);
    let r_single = recall(&single, &data, n, &queries, nq, k);
    assert!(
        r_blocked > 0.8,
        "blocked recall@{k} collapsed to {r_blocked:.3} — the per-block search is not \
         scoring every block, or is merging them wrongly"
    );
    assert!(
        r_blocked >= r_single - 0.03,
        "blocked recall@{k} {r_blocked:.3} trails the single-block {r_single:.3} on i.i.d. \
         rows, where the two calibrations should be interchangeable"
    );
}

#[test]
fn a_drifting_stream_no_longer_freezes_one_global_calibration() {
    // The motivating case. The stream's mean direction sweeps a quarter
    // turn, so a single fit is taken from the head of it and every later
    // row is quantized under a calibration fitted to a different
    // distribution.
    let n = 6 * BS;
    let nq = 60;
    let k = 10;
    let data = drifting_rows(n, DIM, 0xC0FFEE);
    let queries = rows(nq, DIM, 0xF00D);

    let mut blocked = TurboQuantIndex::with_block_size(DIM, 2, BS).unwrap();
    blocked.add(&data);
    let mut single = TurboQuantIndex::new(DIM, 2).unwrap();
    single.add(&data);

    let r_blocked = recall(&blocked, &data, n, &queries, nq, k);
    let r_single = recall(&single, &data, n, &queries, nq, k);
    // Measured 0.468 against 0.050 — a single fit taken from the head of
    // this stream is barely better than chance over the tail. The margin
    // is deliberately far below that gap: what is being pinned is that
    // per-block fitting buys something large here, not the exact figure.
    assert!(
        r_blocked > r_single + 0.2,
        "on a drifting stream the blocked index ({r_blocked:.3}) did not clearly beat the \
         single-calibration one ({r_single:.3}) — per-block fitting bought nothing"
    );
}

#[test]
fn masked_search_selects_across_blocks() {
    let n = 3 * BS;
    let data = rows(n, DIM, 0x3A5C);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&data);

    // One allowed slot per block, plus one in the open block: the merge
    // has to return them all rather than the first block's alone.
    let allowed = [7usize, BS + 3, 2 * BS + 900];
    let mut mask = vec![false; idx.slot_capacity()];
    for &s in &allowed {
        mask[s] = true;
    }
    let query = rows(1, DIM, 0x77);
    let results = idx.search_with_mask(&query, 3, Some(&mask));
    assert_eq!(results.k, 3);
    let mut got: Vec<usize> = results
        .indices_for_query(0)
        .iter()
        .map(|&i| i as usize)
        .collect();
    got.sort_unstable();
    assert_eq!(got, allowed.to_vec());
}

#[test]
fn swap_remove_inside_a_sealed_block_stays_inside_it() {
    let n = 2 * BS + 500;
    let data = rows(n, DIM, 0xDEAD);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&data);
    assert_eq!(idx.sealed_blocks(), 2);

    // Remove from block 0. The filler must come from block 0's own last
    // row, not from the index's.
    let moved_from = idx.swap_remove(5);
    assert_eq!(moved_from, BS - 1, "the hole was filled from another block");
    assert_eq!(idx.len(), n - 1);
    assert_eq!(
        idx.slot_capacity(),
        n,
        "a sealed block gave its extent back, which renumbers every later slot"
    );

    // The row that moved must still be findable, at its new slot, and
    // score as itself — i.e. it was not reinterpreted under a different
    // block's calibration.
    let query = &data[(BS - 1) * DIM..BS * DIM];
    let results = idx.search(query, 1);
    assert_eq!(
        results.indices_for_query(0)[0],
        5,
        "the moved row does not score as itself at its new slot"
    );

    // …and the vacated tail row is gone for good.
    let all = idx.search(query, idx.len());
    assert!(
        !all.indices_for_query(0).contains(&((BS - 1) as i64)),
        "the dead row at the end of block 0 is still searchable"
    );
    assert_eq!(all.k, n - 1, "result width counts dead rows");
}

#[test]
fn swap_remove_in_the_open_block_gives_the_storage_back() {
    let n = BS + 10;
    let data = rows(n, DIM, 0xBEEF);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&data);

    let moved_from = idx.swap_remove(BS + 2);
    assert_eq!(moved_from, n - 1);
    assert_eq!(idx.len(), n - 1);
    assert_eq!(idx.slot_capacity(), n - 1, "the open block kept dead storage");
}

#[test]
#[should_panic(expected = "holds no vector")]
fn removing_a_dead_slot_is_a_contract_violation() {
    let n = 2 * BS;
    let data = rows(n, DIM, 0x0DD);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&data);
    idx.swap_remove(0);
    // Block 0's last slot is now dead. It is inside the storage extent,
    // so nothing but the block table can tell the caller that.
    idx.swap_remove(BS - 1);
}

#[test]
fn a_sealed_index_round_trips() {
    let dir = std::env::temp_dir().join(format!("tv-blockcal-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sealed.tv");

    // Two sealed blocks, a part-filled open block, and a hole in block 0
    // — every piece of state the block table has to carry. The open
    // block is kept short so its raw rows cost less than the codes and
    // therefore ride along; see `block_table_for_write`.
    let n = 2 * BS + 100;
    let data = rows(n, DIM, 0x5EA1);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&data);
    idx.swap_remove(4);
    assert_eq!(idx.sealed_blocks(), 2);
    assert_eq!(idx.slot_capacity(), n);
    assert_eq!(idx.len(), n - 1);

    idx.write(&path).unwrap();
    let loaded = TurboQuantIndex::load(&path).unwrap();
    assert_eq!(loaded.block_size(), Some(BS));
    assert_eq!(loaded.sealed_blocks(), 2);
    assert_eq!(loaded.len(), n - 1);
    assert_eq!(
        loaded.slot_capacity(),
        n,
        "the dead row in block 0 was compacted away, renumbering slots"
    );

    // Scores are the property that matters: each block has to come back
    // with the calibration its own rows were encoded under.
    let queries = rows(20, DIM, 0x5EA2);
    let before = idx.search(&queries, 10);
    let after = loaded.search(&queries, 10);
    assert_eq!(before, after, "a reloaded index scores differently");

    // The open block's raw rows ride along, so the reloaded index can
    // still refit and re-encode that block when it fills. Without them
    // it would seal on the provisional calibration it was carrying.
    let mut loaded = loaded;
    let sealed_pair_before = loaded.tqplus_shift().to_vec();
    loaded.add(&rows(BS - 100, DIM, 0x5EA3));
    assert_eq!(loaded.sealed_blocks(), 3);
    assert_ne!(
        loaded.tqplus_shift(),
        &sealed_pair_before[..],
        "the reloaded index sealed its open block without refitting, so the open-block \
         rows did not survive the round trip"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_default_index_calibrates_in_blocks() {
    // The default is the point of the whole change: an index nobody
    // configured still gets a fit per block rather than one taken from
    // the head of the stream.
    let idx = TurboQuantIndex::new(DIM, 4).unwrap();
    assert_eq!(idx.block_size(), Some(turbovec::DEFAULT_BLOCK_SIZE));
    assert_eq!(TurboQuantIndex::new_lazy(4).unwrap().block_size(), Some(turbovec::DEFAULT_BLOCK_SIZE));

    // Turning calibration off leaves the blocks in place — they are
    // where rows live, not just where fits happen — but there is no
    // refit for them to feed, so no raw rows are kept.
    let off = TurboQuantIndex::new_uncalibrated(DIM, 4).unwrap();
    assert_eq!(off.block_size(), Some(turbovec::DEFAULT_BLOCK_SIZE));
    assert!(!off.calibration_enabled());
}

#[test]
fn slot_liveness_tracks_the_block_table() {
    let n = 2 * BS;
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&rows(n, DIM, 0x11FE));
    assert!((0..n).all(|s| idx.slot_is_live(s)));
    assert!(!idx.slot_is_live(n), "past the extent is not live");

    idx.swap_remove(0);
    assert!(idx.slot_is_live(0), "the filler landed here");
    assert!(!idx.slot_is_live(BS - 1), "the vacated tail row is still live");
    assert!(idx.slot_is_live(BS), "a removal in block 0 killed a slot in block 1");
}

#[test]
fn health_reports_dead_rows_calibration_and_unsearchable_rows() {
    // A freshly built index is all payload bar its calibration, which is
    // two floats per coordinate per block against every row's codes.
    let n = 2 * BS;
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    idx.add(&rows(n, DIM, 0xEA17));
    // Two sealed blocks plus the (empty) open one carry three pairs of
    // `dim` floats each; everything else is codes. At the 1024-row block
    // size used here that is 2.3% of the index — at the 8192-row default
    // it is 0.3%.
    let full = idx.health();
    let pairs = 3.0;
    let codes = (n * DIM * 4 / 8) as f32;
    let expected = codes / (codes + pairs * 2.0 * DIM as f32 * 4.0);
    assert!(
        (full - expected).abs() < 1e-3,
        "a freshly built index is codes plus per-block calibration: expected {expected}, got {full}"
    );

    // Removing from a sealed block leaves the row allocated, so health
    // falls even though nothing else changed.
    for i in 0..100 {
        idx.swap_remove(i);
    }
    let after = idx.health();
    assert!(
        after < full,
        "100 dead rows did not move health ({full} -> {after})"
    );
    assert!(
        (after - (full * (n - 100) as f32 / n as f32)).abs() < 0.01,
        "health {after} does not track the live fraction of {n} rows",
    );

    // A zero-norm row is stored, counted in len(), and unreachable by
    // search — overhead by a different route, and health says so.
    let mut degenerate = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    let mut data = rows(200, DIM, 0xDE9);
    for x in data[..100 * DIM].iter_mut() {
        *x = 0.0;
    }
    degenerate.add(&data);
    assert_eq!(degenerate.len(), 200, "zero-norm rows are still stored");
    assert!(
        degenerate.health() < 0.55,
        "half the rows score against nothing, so at most half is payload, got {}",
        degenerate.health()
    );

    // Nothing allocated is nothing wasted.
    assert_eq!(TurboQuantIndex::new(DIM, 4).unwrap().health(), 1.0);
}

#[test]
fn a_drained_earlier_block_does_not_renumber_the_survivors() {
    // Blocks with no live rows are skipped by the search, and a sealed
    // block keeps its extent when it is drained — so an index can be
    // left with exactly one block to score, starting well above slot 0.
    // The single-block fast path returns the kernel's slots untouched,
    // which is only the answer when that block starts at 0.
    const SMALL: usize = MIN_BLOCK_SIZE;
    let data = rows(2 * SMALL, DIM, 0xB1);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, SMALL).unwrap();
    idx.add(&data);
    for _ in 0..SMALL {
        idx.swap_remove(0);
    }
    assert_eq!(idx.len(), SMALL);
    assert_eq!(idx.slot_capacity(), 2 * SMALL, "block 0 gave its extent back");

    // Every survivor must come back at its own slot, which is SMALL..2*SMALL.
    let results = idx.search(&data[SMALL * DIM..], 1);
    for q in 0..SMALL {
        assert_eq!(
            results.indices_for_query(q)[0] as usize,
            SMALL + q,
            "survivor {q} came back rebased to block 0's slots"
        );
    }
}

#[test]
fn a_block_below_the_sample_floor_seals_on_identity() {
    // A block is its own fit sample, so one smaller than
    // TQPLUS_MIN_SAMPLES cannot fit anything. What matters is that it
    // degrades to identity rather than to something inconsistent: the
    // pair each block declares must be the pair its rows are stored in.
    //
    // Before the fix it was inconsistent. The global warm-up kept
    // accumulating across seals, and on crossing its threshold it
    // re-encoded every stored row from slot 0 under one new global pair
    // while the sealed blocks still declared their seal-time pairs.
    const SMALL: usize = MIN_BLOCK_SIZE;
    let n = 4096;
    let data = drifting_rows(n, DIM, 0xC0FFEE);

    let mut small = TurboQuantIndex::with_block_size(DIM, 4, SMALL).unwrap();
    small.add(&data);
    assert_eq!(small.sealed_blocks(), n / SMALL);
    assert_eq!(
        small.calibration_state(),
        CalibrationState::Identity,
        "a sub-floor block size must report that it fitted nothing"
    );

    // Self-recall is the sharp test: a row must score best against
    // itself, which only holds if it is decoded in the coordinate
    // system it was encoded in.
    let hits = |ix: &TurboQuantIndex| {
        let res = ix.search(&data, 1);
        (0..n).filter(|&q| res.indices_for_query(q)[0] as usize == q).count() as f64 / n as f64
    };
    let mut uncalibrated = TurboQuantIndex::new_uncalibrated(DIM, 4).unwrap();
    uncalibrated.add(&data);
    let (small_recall, flat_recall) = (hits(&small), hits(&uncalibrated));
    assert!(
        (small_recall - flat_recall).abs() < 0.02,
        "a sub-floor block size should be indistinguishable from no calibration \
         ({small_recall:.3} vs {flat_recall:.3}); a gap means the blocks declare a \
         calibration their rows are not stored in"
    );

    // Above the floor the blocks do fit, and it shows.
    let mut big = TurboQuantIndex::with_block_size(DIM, 4, 1024).unwrap();
    big.add(&data);
    assert_eq!(big.calibration_state(), CalibrationState::Fitted);
    assert!(
        hits(&big) > small_recall + 0.5,
        "blocks above the sample floor bought nothing"
    );
}

#[test]
fn removing_from_a_sealed_block_cannot_resurrect_the_row() {
    // The warm-up buffer mirrors removals only for the open block, so
    // while it outlived a seal a removal from a sealed block left the
    // deleted row in it — and the crossing wrote that row back into a
    // live slot, evicting whatever was there. `len()` still said 1023.
    const SMALL: usize = MIN_BLOCK_SIZE;
    let n = 1024;
    let data = rows(n, DIM, 0xDEAD);
    let mut idx = TurboQuantIndex::with_block_size(DIM, 4, SMALL).unwrap();
    idx.add(&data[..600 * DIM]);
    idx.swap_remove(0);
    idx.add(&data[600 * DIM..]);
    assert_eq!(idx.len(), n - 1);

    // Query with the deleted vector. It must not be found: an exact
    // self-match scores ~1.0, and anything that close means the row is
    // still in the index.
    let res = idx.search(&data[..DIM], 1);
    assert!(
        res.scores[0] < 0.9,
        "the deleted row is still in the index — it scored {} at slot {}",
        res.scores[0],
        res.indices[0]
    );
}

#[test]
fn a_file_declaring_an_oversized_open_block_is_refused() {
    // The block table's own consistency checks bound the blocks the file
    // *lists*; the open block is the remainder, and nothing bounded it.
    // A file whose `block_size` is smaller than that remainder loaded
    // clean and then underflowed the open block's remaining capacity on
    // the next add — a panic in debug, and in release a wrap to
    // `usize::MAX` that stops the block ever sealing and sends
    // `slot_is_live` to the wrong block. There is no checksum, so a
    // corrupt file reaches this directly.
    let n = 1000;
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.add(&rows(n, DIM, 0x0F11));
    let mut bytes = idx.to_bytes();

    // The trailer's block_size word: header + codebook + codes + scales
    // + the TQ+ length word and its two arrays.
    let n_levels = 1usize << idx.bit_width();
    let off = 4 + 1 + 1 + 4 + 8
        + (n_levels - 1) * 4
        + n_levels * 4
        + idx.codes_blocked_seq().len()
        + idx.scales().len() * 4
        + 4
        + idx.tqplus_shift().len() * 4
        + idx.tqplus_scale().len() * 4;
    assert_eq!(
        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize,
        turbovec::DEFAULT_BLOCK_SIZE,
        "the block size word is not where the layout says it is",
    );

    // Declare blocks far smaller than the rows the header says are in
    // the (only, open) block.
    bytes[off..off + 4].copy_from_slice(&(MIN_BLOCK_SIZE as u32).to_le_bytes());
    let err = TurboQuantIndex::from_bytes(&bytes).expect_err("oversized open block must be refused");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("open block holds"),
        "unhelpful refusal: {err}"
    );

    // The honest case still loads: a block size that fits the rows.
    bytes[off..off + 4].copy_from_slice(&1024u32.to_le_bytes());
    assert_eq!(
        TurboQuantIndex::from_bytes(&bytes).expect("a consistent table must load").len(),
        n,
    );
}

#[test]
fn a_hole_serializes_identically_from_a_warm_and_a_cold_cache() {
    // A removal inside a sealed block leaves the row allocated, and the
    // index has two places that row lives: the packed codes and the
    // blocked cache. Serialization reads whichever is authoritative —
    // the cache when it is warm, the packed rows when it is not — so
    // both have to be cleared or the same index writes different bytes
    // depending on whether anything happened to have searched it.
    //
    // The blocked side has always been cleared (`pack::zero_lane`). The
    // packed side is cleared by `swap_remove`, and nothing else covers
    // it: the whole suite passes with that clear removed, while this
    // index serializes to 32 differing bytes — exactly one dead row at
    // dim 64 and 4 bits.
    let data = rows(2 * BS, DIM, 0xD37);

    let mut cold = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    cold.add(&data);
    cold.swap_remove(0);

    let mut warm = TurboQuantIndex::with_block_size(DIM, 4, BS).unwrap();
    warm.add(&data);
    warm.prepare(); // populates the blocked cache before the removal
    warm.swap_remove(0);

    assert_eq!(
        cold.to_bytes(),
        warm.to_bytes(),
        "an index with a dead row serializes differently once it has been searched"
    );

    // And the bytes reload to the same index either way.
    let back = TurboQuantIndex::load_from_reader(&mut &cold.to_bytes()[..]).unwrap();
    assert_eq!(back.len(), 2 * BS - 1);
    assert_eq!(back.slot_capacity(), 2 * BS);
}
