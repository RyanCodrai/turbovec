//! Ranking must be invariant to positive uniform scaling of the query (#335).
//!
//! `search()` returns inner products, so the *scores* legitimately scale with
//! the query — but which ids come back, and in what order, must not. The
//! query-side LUT used to floor its scale on an absolute threshold, which
//! collapsed every LUT entry to 0 for small queries and destroyed the ranking
//! (measured recall@10 0.86 → 0.00 below query magnitude ~1e-11).
//!
//! Also pins the LUT cap (#332): 127 is the ceiling the NEON kernel's u8
//! pre-add of the two nibble lookups permits, on every arch, so the value is
//! asserted rather than left implicit.

use turbovec::TurboQuantIndex;

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn gauss(s: &mut u64) -> f64 {
    let u1 = splitmix(s).max(1e-12);
    let u2 = splitmix(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn unit_rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let mut tmp = vec![0.0f64; dim];
        let mut nrm = 0.0f64;
        for t in tmp.iter_mut() {
            *t = gauss(&mut s);
            nrm += *t * *t;
        }
        let inv = 1.0 / nrm.sqrt().max(1e-12);
        for (d, t) in tmp.iter().enumerate() {
            out[i * dim + d] = (t * inv) as f32;
        }
    }
    out
}

/// Query magnitudes spanning 60 orders of magnitude. The upper end stops
/// short of the documented `|value| >= 1e16` input rejection.
// The tail past 1e-30 covers what was previously an untested branch.
// `1.0 / scale` overflows f32 once `scale` drops below ~2.9e-39, and the
// builder falls back to `(1.0, 1.0)` — which is the #335 failure mode
// itself: every LUT entry rounds to 0 and the score degenerates to
// `bias` (#382).
//
// The measured breaking points are 1e-34 at d=256 and 1e-33 at d=768, so
// 1e-33 is the lowest value that holds for every dim this test covers.
// Below it the ranking really does change, and that is documented in
// docs/api.md rather than fixed: computing the reciprocal in f64 removes
// the discontinuity and buys one decade, but it also perturbs rounding
// at ordinary magnitudes — it flips an adjacent-id tie at scale 1e-8 —
// and the kernels are deliberately held numerically equivalent across
// arches (see the `max_lut = 127` note in search.rs). One decade at
// 1e-36 is not worth changing every query's scores.
const SCALES: &[f32] = &[
    1e15, 1e12, 1e8, 1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-8, 1e-10, 1e-11, 1e-14, 1e-18, 1e-22, 1e-26,
    1e-30, 1e-33,
];

fn ids_at_every_scale(dim: usize, bits: usize) {
    let (n, nq, k) = (2000usize, 64usize, 10usize);
    let db = unit_rows(n, dim, 0xC0FFEE ^ dim as u64);
    let q = unit_rows(nq, dim, 0xBADF00D ^ dim as u64);

    let mut idx = TurboQuantIndex::new(dim, bits).unwrap();
    idx.add(&db);
    idx.prepare();

    // `ORDER_TOL` is the width of the band inside which this kernel does
    // not decide an ordering (see the scale loop below). It is ~6x the
    // 1.7e-4 LUT resolution measured at dim=768 bits=2 — deliberate
    // slack, not a measurement, because the resolution varies with the
    // per-sub-table span and this test covers four configs with one
    // constant. The cost of the slack is that a genuine ranking bug
    // swapping vectors closer than 1e-3 apart would pass here; the
    // recall suites are what cover that.
    const ORDER_TOL: f32 = 1e-3;

    let base = idx.search(&q, k);
    let base_ids: Vec<Vec<i64>> = (0..nq)
        .map(|qi| base.indices_for_query(qi).to_vec())
        .collect();
    let base_scores: Vec<Vec<f32>> = (0..nq)
        .map(|qi| base.scores_for_query(qi).to_vec())
        .collect();

    // How much of each query's top-k is decidable at all.
    //
    // Whether the vector at rank k or the one at rank k+1 lands in the
    // result is LUT noise when their scores sit within `ORDER_TOL`, so
    // for those queries the top-k id *set* legitimately differs across
    // scales. Asserting it unconditionally would test the fixture, not
    // the kernel — the same disease the exact-order assertion had. This
    // fixture really does contain such queries (dim=256 bits=2 query 10:
    // rank-10 gap 6.2e-4), so rather than re-roll a seed until they
    // disappear, each query is checked over the prefix that *is* decided:
    // the leading ranks whose scores stand clear of the rank-(k+1) score
    // by more than `ORDER_TOL`.
    let base_k1 = idx.search(&q, k + 1);
    let stable: Vec<usize> = (0..nq)
        .map(|qi| {
            let row = base_k1.scores_for_query(qi);
            let span = row[0].abs().max(f32::MIN_POSITIVE);
            let cutoff = row[k];
            let mut m = (0..k)
                .take_while(|&r| (row[r] - cutoff) / span > ORDER_TOL)
                .count();
            // Clearing the rank-(k+1) score is necessary but not
            // sufficient: `cutoff + TOL` is an arbitrary level, and the
            // two ranks straddling it can sit arbitrarily close to *each
            // other* while both satisfy the condition above. If that pair
            // transposes under scaling, the scaled prefix contains rank m
            // where the base prefix had rank m-1, and the set assertion
            // fires on behaviour this test has itself declared
            // undecidable. This fixture contains such a pair (dim=256
            // bits=2 query 50, m=9, boundary gap 6.8e-4 against a 1e-3
            // tolerance). Shrinking until the boundary itself is
            // separated makes the prefix minimum clear the non-prefix
            // maximum by more than TOL, which is the sufficient condition
            // for the top-m set to be noise-stable. `m == k` needs no
            // shrink: the `take_while` already separated rank k-1 from
            // the cutoff.
            while m > 0 && m < k && (row[m - 1] - row[m]) / span <= ORDER_TOL {
                m -= 1;
            }
            m
        })
        .collect();
    // With every query decidable only at rank 0 the test would pass
    // vacuously, so require the fixture to still have teeth.
    let total: usize = stable.iter().sum();
    assert!(
        total >= nq * k / 2,
        "dim={dim} bits={bits}: only {total} of {} ranks are decidable — the fixture \
         has degenerated and the assertions below would be near-vacuous; re-roll the seed",
        nq * k
    );

    for &c in SCALES {
        let scaled: Vec<f32> = q.iter().map(|v| v * c).collect();
        let r = idx.search(&scaled, k);
        for (qi, expected) in base_ids.iter().enumerate() {
            let got = r.indices_for_query(qi);
            let scores = &base_scores[qi];
            let m = stable[qi];
            // Same ids, over the decidable prefix: that is the guarantee
            // #335 is about — a small query must not collapse the LUT and
            // destroy the result set.
            let (mut a, mut b) = (got[..m].to_vec(), expected[..m].to_vec());
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(
                a, b,
                "dim={dim} bits={bits} query {qi} returned a different id SET over its \
                 decidable prefix ({m} of {k}) at scale {c:e}"
            );
            // Order, only where the scores are far enough apart to
            // decide it. `c` is not a power of two, so `q * c` perturbs
            // every component by up to an f32 ulp; the query LUT then
            // quantizes to u8, which amplifies that to ~1e-4 relative on
            // a score. Two database vectors closer together than that
            // are genuinely not ordered by this kernel, and asserting
            // otherwise pins an accident of the fixture rather than a
            // property of the design. Measured instance: at dim=768
            // bits=2 the pair (1978, 432) sits 5.8e-5 apart relative to
            // the top score and transposes at c=1e-2.
            let span = scores[0].abs().max(f32::MIN_POSITIVE);
            for (i, (&g, &e)) in got[..m].iter().zip(expected[..m].iter()).enumerate() {
                if g == e {
                    continue;
                }
                // The set assertion above passed, so every id in
                // `got[..m]` is somewhere in `expected[..m]`.
                let ge = expected
                    .iter()
                    .position(|&x| x == g)
                    .expect("set equality over the prefix was asserted above");
                let gap = (scores[i] - scores[ge]).abs() / span;
                assert!(
                    gap <= ORDER_TOL,
                    "dim={dim} bits={bits} query {qi} reordered ids {e} and {g} at scale \
                     {c:e}, but their scores differ by {gap:e} (> {ORDER_TOL:e}) — that is a \
                     ranking change, not LUT-resolution noise"
                );
            }
        }
    }
}

#[test]
fn returned_ids_are_invariant_to_positive_query_scaling() {
    for &dim in &[256usize, 768] {
        for &bits in &[2usize, 4] {
            ids_at_every_scale(dim, bits);
        }
    }
}

/// Scaling by an exact power of two is exactly representable end to end, so
/// the u8 LUT comes out bit-identical and the scores are exactly `c` times
/// the unscaled ones. This is the sharpest form of the invariant, and it
/// pins that the scale is carried entirely by `QueryNeonLut::scale`.
#[test]
fn power_of_two_scaling_is_bit_exact() {
    let (dim, n, nq, k) = (768usize, 1000usize, 32usize, 10usize);
    let db = unit_rows(n, dim, 0x5EED);
    let q = unit_rows(nq, dim, 0x1234);

    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&db);
    idx.prepare();
    let base = idx.search(&q, k);

    for &c in &[1024.0f32, 65536.0, 1.0 / 1024.0, f32::powi(2.0, -60)] {
        let scaled: Vec<f32> = q.iter().map(|v| v * c).collect();
        let r = idx.search(&scaled, k);
        for qi in 0..nq {
            assert_eq!(r.indices_for_query(qi), base.indices_for_query(qi));
            for (a, b) in r.scores_for_query(qi).iter().zip(base.scores_for_query(qi)) {
                assert_eq!(*a, b * c, "score not exactly scaled at c={c:e}");
            }
        }
    }
}

/// The scores themselves are inner products and *must* track the query
/// magnitude linearly — invariance is a property of the ranking, not the
/// score. For non-power-of-two factors the input rounding perturbs the u8
/// LUT slightly, so the agreement is to LUT-quantization noise, not exact.
#[test]
fn scores_scale_linearly_with_the_query() {
    let (dim, n, nq, k) = (768usize, 1000usize, 32usize, 10usize);
    let db = unit_rows(n, dim, 0x5EED);
    let q = unit_rows(nq, dim, 0x1234);

    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&db);
    idx.prepare();
    let base = idx.search(&q, k);

    for &c in &[1e6f32, 1e3, 1e-3, 1e-6] {
        let scaled: Vec<f32> = q.iter().map(|v| v * c).collect();
        let r = idx.search(&scaled, k);
        for qi in 0..nq {
            for (a, b) in r
                .scores_for_query(qi)
                .iter()
                .zip(base.scores_for_query(qi))
            {
                let want = b * c;
                let denom = want.abs().max(f32::MIN_POSITIVE);
                assert!(
                    (a - want).abs() / denom < 2e-2,
                    "score {a:e} != {want:e} at scale {c:e}"
                );
            }
        }
    }
}

/// A zero query has no direction: every LUT entry is 0 and every score
/// reduces to the (zero) bias. It must still return `k` ids rather than
/// panicking or producing NaN.
#[test]
fn zero_query_is_well_formed() {
    let dim = 256usize;
    let db = unit_rows(500, dim, 0xABCD);
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    idx.add(&db);
    idx.prepare();
    let r = idx.search(&vec![0.0f32; dim], 10);
    assert_eq!(r.indices_for_query(0).len(), 10);
    assert!(r.scores_for_query(0).iter().all(|s| s.is_finite()));
}
