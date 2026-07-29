//! Lloyd-Max scalar quantizer for the Beta distribution.
//!
//! After orthogonal rotation, each coordinate of a unit vector on S^{d-1}
//! follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. This module computes optimal
//! quantization boundaries and centroids for that distribution.

use statrs::distribution::{Beta, ContinuousCDF, Continuous};
use std::collections::HashMap;
use std::sync::Mutex;

/// Memo backing [`codebook`]; `None` until the first solve is stored,
/// because `HashMap::new` is not `const`.
///
/// Locked with `try_lock` only, never `lock`, and that is a correctness
/// requirement rather than a contention tweak: [`codebook`] is on the
/// **load** path (`io::validate_codebook`), which is the first thing a
/// forked worker does. `fork` clones only the calling thread, so a child
/// inherits every mutex in whatever state it had — and a mutex held by a
/// thread that does not exist in the child is never unlocked. A blocking
/// `lock()` here would hang that child forever, which is
/// #147/#288/#321/#364's failure mode arriving through a fifth door.
///
/// `try_lock` makes that impossible: a lock the caller cannot take is
/// simply a memo miss, and a memo miss is only ever slower, never wrong,
/// because the memoised value is a pure function of `(bits, dim)`. That
/// same purity is what lets two threads race a cold shape — both solve
/// and both store the same answer.
///
/// A per-shape `OnceLock` array would be lock-free to read, but `Once`
/// is itself a blocking primitive on its cold path, so it would trade a
/// rare *slow* path for a rare *hang* — the wrong direction here — and
/// the shape space (`3 × MAX_DIM`) is far too large to reserve statically.
static MEMO: Mutex<Option<HashMap<(usize, usize), (Vec<f32>, Vec<f32>)>>> = Mutex::new(None);

/// Returns (boundaries, centroids) for the given bit width and dimension.
///
/// Crate-internal: trusts `bits ∈ {2,3,4}` and `dim >= 2`. Callers reach it
/// only after those bounds are enforced (`TurboQuantIndex::new` /
/// `from_parts`). Exposing it publicly would let a caller pass `bits` in the
/// ~32..63 range and drive an unbounded `1 << bits` allocation (DoS), or
/// `dim` 0/1 and hit a `Beta::new` panic — see the validated
/// [`from_parts`](crate::TurboQuantIndex::from_parts) boundary instead.
///
/// Memoised process-globally. The solve is 25–100 ms (200 Lloyd-Max
/// iterations, each doing `2^bits` adaptive-Simpson quadratures of a Beta
/// pdf to `epsabs=1e-14`) and the result is a pure function of the two
/// arguments, so every repeat is redundant work — building a second index
/// of the same shape, or saving, used to pay it again. See [`MEMO`] for
/// why the memo is never taken with a blocking `lock()`.
pub(crate) fn codebook(bits: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    if let Ok(memo) = MEMO.try_lock() {
        if let Some(hit) = memo.as_ref().and_then(|m| m.get(&(bits, dim))) {
            return hit.clone();
        }
    }
    // The guard is dropped before the solve: the lock is only ever held
    // across a hash lookup or a clone, so the window a fork could land
    // in is nanoseconds wide even though `try_lock` makes landing in it
    // harmless.
    let computed = lloyd_max(bits, dim, 200, 1e-12);
    if let Ok(mut memo) = MEMO.try_lock() {
        memo.get_or_insert_with(HashMap::new)
            .insert((bits, dim), computed.clone());
    }
    computed
}

/// The largest `|c_i - E[X | b_{i-1} < X < b_i]|` over the levels of
/// `centroids`, where `X ~ Beta((dim-1)/2, (dim-1)/2)` on `[-1, 1]` and the
/// `b_i` are the midpoints of consecutive centroids.
///
/// This is the Lloyd-Max fixed-point condition — the defining property of
/// the codebook — evaluated in closed form. For `X ~ Beta(a, a)` on `[0,1]`,
/// `E[X · 1{lo<X<hi}] = (a/2a)·(F_{a+1,a}(hi) - F_{a+1,a}(lo))`, so the
/// conditional mean needs four CDF evaluations per level instead of an
/// adaptive-Simpson quadrature, and no iteration at all: it verifies a
/// candidate codebook in tens of microseconds rather than re-deriving one in
/// tens of milliseconds (#357).
///
/// Beta(a, a) with `a > 1` is log-concave, so its Lloyd-Max fixed point is
/// unique — a residual near zero identifies *the* codebook for `(bits, dim)`,
/// which is what makes this a substitute for recomputation and not merely a
/// sanity check.
pub(crate) fn fixed_point_residual(dim: usize, centroids: &[f32]) -> f64 {
    let a = (dim as f64 - 1.0) / 2.0;
    let mass = Beta::new(a, a).unwrap();
    let moment = Beta::new(a + 1.0, a).unwrap();
    let n_levels = centroids.len();
    let mut edges = Vec::with_capacity(n_levels + 1);
    edges.push(-1.0f64);
    for i in 0..n_levels - 1 {
        edges.push(((centroids[i] + centroids[i + 1]) * 0.5) as f64);
    }
    edges.push(1.0);

    let mut worst = 0.0f64;
    for i in 0..n_levels {
        // Map the [-1, 1] cell onto the Beta's [0, 1] support.
        let lo = (edges[i] + 1.0) / 2.0;
        let hi = (edges[i + 1] + 1.0) / 2.0;
        let prob = mass.cdf(hi) - mass.cdf(lo);
        // An empty cell pins nothing, exactly as the solver's `prob < 1e-15`
        // branch leaves such a centroid untouched.
        if prob < 1e-15 {
            continue;
        }
        // E[x · 1{lo<x<hi}] on [-1, 1] for x = 2t - 1.
        let moment_cell = (moment.cdf(hi) - moment.cdf(lo)) - prob;
        worst = worst.max((centroids[i] as f64 - moment_cell / prob).abs());
    }
    worst
}

fn lloyd_max(bits: usize, dim: usize, max_iter: usize, tol: f64) -> (Vec<f32>, Vec<f32>) {
    let a = (dim as f64 - 1.0) / 2.0;
    // Beta(a, a) on [0, 1], shifted to [-1, 1] via loc=-1, scale=2
    let beta = Beta::new(a, a).unwrap();

    let n_levels = 1usize << bits;

    // Initialize centroids within +/- 3 std devs
    let std_dev = (2.0 * a / ((2.0 * a + 1.0) * 4.0 * a)).sqrt(); // std of Beta on [-1,1]
    let spread = 3.0 * std_dev;
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| -spread + 2.0 * spread * i as f64 / (n_levels as f64 - 1.0))
        .collect();

    for _ in 0..max_iter {
        // Boundaries = midpoints between consecutive centroids
        let boundaries: Vec<f64> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();

        let mut edges = Vec::with_capacity(n_levels + 1);
        edges.push(-1.0);
        edges.extend_from_slice(&boundaries);
        edges.push(1.0);

        let mut new_centroids = vec![0.0f64; n_levels];

        for i in 0..n_levels {
            let lo = edges[i];
            let hi = edges[i + 1];

            // CDF on [-1, 1]: transform to [0, 1] for Beta
            let cdf_lo = beta.cdf((lo + 1.0) / 2.0);
            let cdf_hi = beta.cdf((hi + 1.0) / 2.0);
            let prob = cdf_hi - cdf_lo;

            if prob < 1e-15 {
                new_centroids[i] = centroids[i];
            } else {
                // Conditional mean = integral(x * pdf(x), lo, hi) / prob
                // where pdf is on [-1, 1]: pdf_shifted(x) = beta.pdf((x+1)/2) / 2
                let mean = adaptive_simpson(
                    |x| {
                        let t = (x + 1.0) / 2.0;
                        x * beta.pdf(t) / 2.0
                    },
                    lo,
                    hi,
                    1e-14,
                    50,
                );
                new_centroids[i] = mean / prob;
            }
        }

        let max_change = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        centroids = new_centroids;

        if max_change < tol {
            break;
        }
    }

    // Round to f32 FIRST, then take midpoints in f32. Doing it the other
    // way — midpoint in f64, then one cast — makes the boundaries
    // cross-platform unstable, which is the open half of #259 and what
    // the CI fingerprint leg caught.
    //
    // The f64 iteration above is not reproducible to the last bit across
    // libms: `statrs`'s Beta cdf/pdf bottom out in `ln`/`exp`, which
    // differ by an ulp or so between glibc, Apple's libm and MSVC, and
    // the adaptive-Simpson recursion can then take a different branch.
    // At 4 bits the loop also exhausts `max_iter` without reaching `tol`
    // (200 iterations, final max change ~1e-8), so the f64 centroids are
    // only settled to ~1e-8 anyway.
    //
    // Casting a centroid to f32 absorbs all of that — f32 has ~6e-8
    // relative resolution, and the f32 centroids are empirically
    // invariant under perturbations of the pdf up to 1e-10 relative,
    // which is five orders of magnitude beyond any libm disagreement.
    // The *midpoint* was the fragile step: computed in f64 it sits a
    // fraction of an f32 ulp from a rounding boundary, so a 1e-15
    // relative perturbation of the pdf flips it — measured to flip at
    // every (bits, dim) cell tested. Averaging the already-rounded f32
    // centroids removes the knife-edge entirely: f32 add is
    // correctly rounded and `* 0.5` is exact, so the result is a pure
    // function of the two f32 centroids on every platform.
    let centroids_f32: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();
    let boundaries: Vec<f32> = (0..n_levels - 1)
        .map(|i| (centroids_f32[i] + centroids_f32[i + 1]) * 0.5)
        .collect();

    (boundaries, centroids_f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every boundary must be exactly the f32 midpoint of its two
    /// neighbouring f32 centroids.
    ///
    /// This is what makes the codebook cross-platform reproducible
    /// (#259): the f32 centroids survive the f64 iteration's libm
    /// variance, so deriving boundaries from them — rather than from the
    /// f64 centroids — makes the whole codebook a pure function of
    /// values every platform agrees on. A regression here reintroduces
    /// the divergence silently, since it only shows up as differing
    /// bytes on a different OS.
    #[test]
    fn boundaries_are_f32_midpoints_of_f32_centroids() {
        for bits in 2..=4usize {
            for dim in [8usize, 128, 200, 768, 1000, 1024, 1536, 3072] {
                let (boundaries, centroids) = codebook(bits, dim);
                assert_eq!(centroids.len(), 1 << bits);
                assert_eq!(boundaries.len(), (1 << bits) - 1);
                for i in 0..boundaries.len() {
                    let expect = (centroids[i] + centroids[i + 1]) * 0.5;
                    assert_eq!(
                        boundaries[i].to_bits(),
                        expect.to_bits(),
                        "bits={bits} dim={dim} boundary {i} is not the f32 \
                         midpoint of centroids {i} and {}",
                        i + 1,
                    );
                }
                // Ordering is what the boundary scan in the quantize
                // kernel relies on: code = count of boundaries below x.
                for w in boundaries.windows(2) {
                    assert!(w[0] < w[1], "bits={bits} dim={dim}: boundaries not ascending");
                }
                for w in centroids.windows(2) {
                    assert!(w[0] < w[1], "bits={bits} dim={dim}: centroids not ascending");
                }
            }
        }
    }
}

/// Adaptive Simpson's rule for numerical integration.
fn adaptive_simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_depth: usize) -> f64 {
    let mid = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(&f, a, b, fa, fb, fm, whole, tol, max_depth)
}

fn adaptive_simpson_rec<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> f64 {
    let mid = (a + b) / 2.0;
    let m1 = (a + mid) / 2.0;
    let m2 = (mid + b) / 2.0;
    let fm1 = f(m1);
    let fm2 = f(m2);
    let left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
    let right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
    let refined = left + right;

    if depth == 0 || (refined - whole).abs() < 15.0 * tol {
        refined + (refined - whole) / 15.0
    } else {
        adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1)
            + adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1)
    }
}

#[cfg(test)]
mod fork_safety_tests {
    use super::MEMO;

    /// A forked child inherits every mutex in the state it had at
    /// `fork`, and one held by a thread that the child does not have is
    /// never unlocked. That is exactly "the lock is held and its owner
    /// will not release it", which this reproduces in-process: an actual
    /// `fork` cannot be asserted on portably (Windows has none) and a
    /// deadlocked child can only be observed as a timeout anyway, so the
    /// inherited-lock *state* is what the test recreates.
    ///
    /// With a blocking `lock()` the spawned thread never returns and this
    /// fails on the timeout; with `try_lock` the held lock is a memo miss
    /// and the solve runs.
    #[test]
    fn codebook_does_not_block_on_a_held_memo_lock() {
        let held = MEMO.lock().unwrap_or_else(|e| e.into_inner());
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            // A shape nothing else solves, so the answer cannot already
            // be in the memo from another test.
            let (boundaries, centroids) = super::codebook(2, 1237);
            let _ = tx.send((boundaries.len(), centroids.len()));
        });
        let got = rx.recv_timeout(std::time::Duration::from_secs(30));
        drop(held);
        let _ = worker.join();
        assert_eq!(
            got.ok(),
            Some((3, 4)),
            "codebook() blocked on the memoisation mutex — a process that forks \
             while that lock is held would deadlock in the child on its first \
             load (#147/#288/#321/#364)",
        );
    }

    /// The memo must still be a memo: a repeat of a shape already solved
    /// costs a hash lookup and a clone, not another Lloyd-Max solve.
    #[test]
    fn codebook_repeats_are_served_from_the_memo() {
        let cold = std::time::Instant::now();
        let first = super::codebook(4, 1553);
        let cold = cold.elapsed();
        let warm = std::time::Instant::now();
        let second = super::codebook(4, 1553);
        let warm = warm.elapsed();
        assert_eq!(first, second);
        assert!(
            warm * 20 < cold,
            "repeat codebook() took {warm:?} against a cold solve of {cold:?} — \
             the memo is not being hit",
        );
    }
}
