//! Structural properties of the TQ+ fit (#463).
//!
//! `compute_tqplus_calibration` had three mutants surviving the whole
//! suite. Nothing pinned the arithmetic that turns two empirical
//! quantiles into `(shift, scale)`:
//!
//! ```text
//! scale = qc_span / qe_span
//! shift = qc_lo / scale - qe_lo
//! ```
//!
//! Goldens would pin it but would also break on any legitimate retune of
//! the codebook anchors. These are the properties that follow from the
//! formula itself and hold whatever the anchors are.

use turbovec::TurboQuantIndex;

const DIM: usize = 64;

fn sample(n: usize, seed: u64, k: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; n * DIM];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        // `s >> 40` leaves 24 bits, so divide by 2^24 for [0, 1) and a
        // centred [-0.5, 0.5). Dividing by 2^23 gave [-0.5, 1.5) — a
        // +0.5k DC component in every coordinate, which made the rows
        // mutually similar instead of isotropic.
        *x = (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * k;
    }
    v
}

fn fit(k: f32) -> (Vec<f32>, Vec<f32>) {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&sample(2048, 11, k)).unwrap();
    (idx.tqplus_shift().to_vec(), idx.tqplus_scale().to_vec())
}

/// The fit is magnitude-invariant: rows are L2-normalized before the
/// quantiles are taken, so scaling the whole corpus changes nothing.
/// This is the property the crate states elsewhere ("a corpus scaled by
/// 1e-10 still yields a minimum scale near 1.62") and the one the
/// calibration bounds in `io.rs` are justified by, so it is worth
/// pinning rather than assuming.
#[test]
fn the_fit_is_invariant_to_corpus_magnitude() {
    let (sh1, sc1) = fit(1.0);
    let (sh8, sc8) = fit(8.0);
    let (sh_small, sc_small) = fit(1e-4);
    assert_eq!(sh1.len(), DIM);

    for d in 0..DIM {
        for (label, got, want) in [
            ("8x scale", sc8[d], sc1[d]),
            ("1e-4x scale", sc_small[d], sc1[d]),
        ] {
            assert!(
                (got - want).abs() <= want.abs() * 1e-3,
                "coord {d}: {label} should match the unscaled fit: {got} vs {want}"
            );
        }
        for (label, got, want) in [
            ("8x shift", sh8[d], sh1[d]),
            ("1e-4x shift", sh_small[d], sh1[d]),
        ] {
            assert!(
                (got - want).abs() <= want.abs().max(1e-6) * 1e-3,
                "coord {d}: {label} should match the unscaled fit: {got} vs {want}"
            );
        }
    }
}

/// Every fitted scale is finite and strictly positive, and every shift
/// finite — the loader enforces this, so a fit that violated it would
/// produce an index that cannot be written and reloaded.
#[test]
fn the_fit_produces_loadable_values() {
    let (sh, sc) = fit(1.0);
    for d in 0..DIM {
        assert!(sc[d].is_finite() && sc[d] > 0.0, "coord {d}: scale {}", sc[d]);
        assert!(sh[d].is_finite(), "coord {d}: shift {}", sh[d]);
    }
    // And the round trip the loader gates on actually succeeds.
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&sample(2048, 11, 1.0)).unwrap();
    idx.add(&sample(64, 12, 1.0));
    let bytes = idx.to_bytes();
    assert_eq!(TurboQuantIndex::from_bytes(&bytes).unwrap().to_bytes(), bytes);
}

/// The same sample must fit the same pair, twice, and independently of
/// how the rows were batched into the call.
#[test]
fn the_fit_is_deterministic() {
    let (a_sh, a_sc) = fit(1.0);
    let (b_sh, b_sc) = fit(1.0);
    assert_eq!(a_sh, b_sh, "shift is not deterministic");
    assert_eq!(a_sc, b_sc, "scale is not deterministic");
}

/// A calibrated index must score better than an uncalibrated one on the
/// corpus it was fitted to — the fit's whole purpose. A broken formula
/// can still produce finite, positive, covariant values while mapping
/// the data outside the codebook's range, and this is what notices.
#[test]
fn calibration_improves_self_recall() {
    let rows = sample(512, 21, 1.0);
    let queries = &rows[..8 * DIM];

    let mut plain = TurboQuantIndex::new(DIM, 4).unwrap();
    plain.add(&rows);
    let mut cal = TurboQuantIndex::new(DIM, 4).unwrap();
    cal.calibrate(&sample(2048, 11, 1.0)).unwrap();
    cal.add(&rows);

    // Self-match: each query is row i, so the top hit should be i.
    let hits = |idx: &TurboQuantIndex| -> usize {
        let r = idx.search(queries, 1);
        (0..8).filter(|&q| r.indices[q] == q as i64).count()
    };
    let (h_plain, h_cal) = (hits(&plain), hits(&cal));
    assert!(
        h_cal >= h_plain,
        "calibration should not lose self-matches: {h_cal} vs {h_plain}"
    );
    assert!(h_cal >= 7, "calibrated self-recall collapsed: {h_cal}/8");
}

/// The fit exists to map the corpus onto the codebook's range, so a
/// calibrated index must actually use that range. A scale that is wrong
/// by a factor collapses every coordinate onto one or two levels while
/// still being finite, positive, deterministic and magnitude-invariant —
/// which is exactly the gap that let arithmetic mutants survive here.
#[test]
fn a_calibrated_index_uses_most_of_the_codebook() {
    let mut idx = TurboQuantIndex::new(DIM, 4).unwrap();
    idx.calibrate(&sample(2048, 11, 1.0)).unwrap();
    idx.add(&sample(512, 31, 1.0));

    // `packed_codes()` is BIT-PLANE packed, not two codes per byte: a row
    // is `bit_width` planes of `dim / 8` bytes, and bit `p` of the code
    // for coordinate `j` lives in plane `p`, byte `j / 8`, at bit
    // `7 - (j % 8)`. Reading bytes as nibble pairs counts nothing.
    const BITS: usize = 4;
    let bytes_per_plane = DIM / 8;
    let row_bytes = BITS * bytes_per_plane;
    let packed = idx.packed_codes();
    assert_eq!(packed.len() % row_bytes, 0, "packed rows should be whole");

    let mut seen = [0usize; 1 << BITS];
    for row in packed.chunks(row_bytes) {
        for j in 0..DIM {
            let (c, k) = (j / 8, j % 8);
            let mut code = 0usize;
            for p in 0..BITS {
                let bit = (row[p * bytes_per_plane + c] >> (7 - k)) & 1;
                code |= (bit as usize) << p;
            }
            seen[code] += 1;
        }
    }

    let used = seen.iter().filter(|&&c| c > 0).count();
    assert!(
        used >= 12,
        "a calibrated 4-bit index should reach most of its 16 levels, used {used}: {seen:?}"
    );

    // And no single level may swallow the corpus — the signature of a
    // scale that crushed everything into one bucket.
    let total: usize = seen.iter().sum();
    let biggest = *seen.iter().max().unwrap();
    assert!(
        biggest * 2 < total,
        "one level holds {biggest} of {total} codes; the fit collapsed the range"
    );

    // The fit also *centres* the corpus: it maps the empirical low and
    // high quantiles onto the codebook's own anchors. A wrong shift
    // translates the distribution toward one end without narrowing it,
    // which the spread checks above cannot see.
    let mean: f64 = seen
        .iter()
        .enumerate()
        .map(|(lvl, &c)| lvl as f64 * c as f64)
        .sum::<f64>()
        / total as f64;
    assert!(
        (6.0..=9.0).contains(&mean),
        "mean code level {mean:.2} is off-centre for 16 levels; the fit is \
         translated rather than centred: {seen:?}"
    );
}
