//! Coarse cells over residuals: sublinear search that keeps the online
//! contract.
//!
//! An [`IvfIndex`] partitions the corpus into `nlist` cells and stores,
//! for each vector, the **residual** from its cell's centroid rather
//! than the vector itself. Search probes the `nprobe` nearest cells and
//! scores only those, so the bytes read fall with `nprobe / nlist`
//! instead of scanning everything.
//!
//! # Why residuals rather than the vectors
//!
//! Inner product decomposes exactly across the split:
//!
//! ```text
//! q · v = q · (c + r) = q · c + q · r
//! ```
//!
//! `q · c` is one dot product per probed cell, computed once and added
//! to every score in that cell; `q · r` is what the quantized scan
//! already does. So the decomposition costs nothing at query time and
//! is exact — no approximation is introduced by the split itself.
//!
//! What it buys is precision. Residuals are smaller and less spread
//! than the vectors they came from, so the same bit width resolves them
//! more finely. Cells do not only prune, they raise the recall ceiling
//! the quantizer can reach.
//!
//! A residual of exactly zero — a vector sitting on its own centroid —
//! encodes with scale 0 and scores 0 against every query, which is the
//! correct contribution here: the whole score is carried by `q · c`.
//!
//! # Why it stays online
//!
//! Centroids are fitted **once**, from the first
//! [`IvfIndex::fit_threshold`] vectors, in the background of ordinary
//! insertion; vectors that arrive before the fit are held in a buffer
//! that search scans exhaustively, so results are correct from the
//! first `add`. Nothing rebuilds when the corpus grows: a later vector
//! is assigned against the existing centroids and appended to its
//! cell. Removal stays a swap-and-pop within a cell.
//!
//! Fitting on a small early prefix is not a compromise for want of a
//! better option — on OpenAI-1536 embeddings, centroids fitted from 5%
//! of an 800k corpus scored within noise of centroids fitted from 25%
//! (0.9580 vs 0.9556 cell recall at `nprobe = 64`), with the same cell
//! balance. The partition has to be data-dependent, though: random
//! directions over the same data reach only 0.632.

use crate::{ConstructError, TurboQuantIndex};

/// Default number of buffered vectors that triggers the centroid fit.
pub const DEFAULT_FIT_THRESHOLD: usize = 40_000;

/// A partitioned index: coarse cells, residual-quantized contents.
///
/// See the [module docs](self) for the decomposition and the online
/// contract.
#[derive(Debug)]
pub struct IvfIndex {
    dim: usize,
    bit_width: usize,
    nlist: usize,
    fit_threshold: usize,
    /// `nlist * dim`, empty until the fit runs.
    centroids: Vec<f32>,
    /// One index per cell, over residuals. Empty until the fit runs.
    cells: Vec<TurboQuantIndex>,
    /// Per cell, the caller-visible id of each slot, aligned with the
    /// cell's index slots so a `swap_remove` can keep them in step.
    cell_ids: Vec<Vec<u64>>,
    /// Vectors that arrived before the fit, scanned exhaustively.
    buffer: Vec<f32>,
    buffer_ids: Vec<u64>,
    next_id: u64,
}

impl IvfIndex {
    /// A new, unfitted index over `dim`-dimensional vectors.
    ///
    /// `nlist` is the cell count to fit later; `√n` for the corpus you
    /// expect is the usual starting point, since it keeps the centroid
    /// sweep and the probed bytes growing at the same rate.
    ///
    /// Returns [`ConstructError`] on the same terms as
    /// [`TurboQuantIndex::new`]: `bit_width` must be 2, 3 or 4 and
    /// `dim` must be non-zero and within [`crate::MAX_DIM`].
    pub fn new(dim: usize, bit_width: usize, nlist: usize) -> Result<Self, ConstructError> {
        // Validate dim/bit_width by constructing one index up front and
        // discarding it: the cells are built later, and a caller should
        // not have to wait until the fit to learn the width is invalid.
        let _ = TurboQuantIndex::new(dim, bit_width)?;
        Ok(Self {
            dim,
            bit_width,
            nlist: nlist.max(1),
            fit_threshold: DEFAULT_FIT_THRESHOLD,
            centroids: Vec::new(),
            cells: Vec::new(),
            cell_ids: Vec::new(),
            buffer: Vec::new(),
            buffer_ids: Vec::new(),
            next_id: 0,
        })
    }

    /// Set how many buffered vectors trigger the centroid fit.
    ///
    /// Below `nlist` the fit cannot produce distinct centroids, so the
    /// threshold is raised to `nlist` if a smaller value is given.
    pub fn with_fit_threshold(mut self, n: usize) -> Self {
        self.fit_threshold = n.max(self.nlist);
        self
    }

    /// Vectors held, buffered and celled alike.
    pub fn len(&self) -> usize {
        self.buffer_ids.len() + self.cell_ids.iter().map(Vec::len).sum::<usize>()
    }

    /// Whether [`Self::len`] is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the centroids have been fitted.
    ///
    /// Until they are, search is exhaustive over the buffer and returns
    /// exactly what an unpartitioned index would.
    pub fn is_fitted(&self) -> bool {
        !self.centroids.is_empty()
    }

    /// The cell count actually in use — `0` before the fit.
    pub fn nlist(&self) -> usize {
        if self.is_fitted() {
            self.nlist
        } else {
            0
        }
    }

    /// Slot counts per cell, in cell order. Empty before the fit.
    ///
    /// Useful for spotting imbalance: a cell holding many times the
    /// mean makes probed traffic — and so tail latency — lumpy.
    pub fn cell_sizes(&self) -> Vec<usize> {
        self.cell_ids.iter().map(Vec::len).collect()
    }

    /// The dimensionality this index was built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The bit width residuals are stored at.
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Append `vectors` (row-major, `dim` floats each), returning the
    /// ids assigned to them.
    ///
    /// Before the fit the rows are buffered; the call that carries the
    /// buffer past [`Self::with_fit_threshold`] also fits the centroids
    /// and drains the buffer into cells. After the fit each row is
    /// assigned to its nearest centroid and stored as a residual.
    ///
    /// # Panics
    ///
    /// If `vectors.len()` is not a multiple of `dim`.
    pub fn add(&mut self, vectors: &[f32]) -> Vec<u64> {
        assert!(
            vectors.len() % self.dim == 0,
            "vector buffer length {} is not a multiple of dim {}",
            vectors.len(),
            self.dim
        );
        let n = vectors.len() / self.dim;
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(self.next_id);
            self.next_id += 1;
        }
        if self.is_fitted() {
            for (i, id) in ids.iter().enumerate() {
                let v = &vectors[i * self.dim..(i + 1) * self.dim];
                self.insert_assigned(v, *id);
            }
        } else {
            self.buffer.extend_from_slice(vectors);
            self.buffer_ids.extend_from_slice(&ids);
            if self.buffer_ids.len() >= self.fit_threshold {
                self.fit_and_drain();
            }
        }
        ids
    }

    /// Force the centroid fit now, from whatever is buffered.
    ///
    /// A no-op once fitted, or while fewer than `nlist` vectors are
    /// held — a fit cannot produce more distinct centroids than it has
    /// rows to place them on.
    pub fn fit(&mut self) {
        if !self.is_fitted() && self.buffer_ids.len() >= self.nlist {
            self.fit_and_drain();
        }
    }

    /// Assign one already-fitted vector to its cell, as a residual.
    fn insert_assigned(&mut self, v: &[f32], id: u64) {
        let c = self.nearest_centroid(v);
        let mut residual = vec![0.0f32; self.dim];
        let base = c * self.dim;
        for d in 0..self.dim {
            residual[d] = v[d] - self.centroids[base + d];
        }
        self.cells[c].add(&residual);
        self.cell_ids[c].push(id);
    }

    /// Index of the centroid with the largest inner product with `v`.
    ///
    /// Inner product rather than L2 because that is the metric the
    /// index ranks by; for the unit-norm vectors this is built for the
    /// two orders agree.
    fn nearest_centroid(&self, v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..self.nlist {
            let base = c * self.dim;
            let mut s = 0.0f32;
            for d in 0..self.dim {
                s += v[d] * self.centroids[base + d];
            }
            if s > best_score {
                best_score = s;
                best = c;
            }
        }
        best
    }

    /// Fit centroids from the buffer, then move every buffered vector
    /// into its cell as a residual.
    fn fit_and_drain(&mut self) {
        self.centroids = kmeans(&self.buffer, self.dim, self.nlist, 10);
        self.cells = (0..self.nlist)
            .map(|_| {
                TurboQuantIndex::new(self.dim, self.bit_width)
                    .expect("dim/bit_width validated in IvfIndex::new")
            })
            .collect();
        self.cell_ids = vec![Vec::new(); self.nlist];

        let buffered = std::mem::take(&mut self.buffer);
        let ids = std::mem::take(&mut self.buffer_ids);
        for (i, id) in ids.into_iter().enumerate() {
            let v = &buffered[i * self.dim..(i + 1) * self.dim];
            self.insert_assigned(v, id);
        }
    }

    /// Top-`k` for each query, probing the `nprobe` nearest cells.
    ///
    /// Returns `(scores, ids)`, row-major `nq × k`, sorted descending
    /// within each row — matching [`crate::SearchResults`]'s field
    /// order and the Python binding's return order.
    ///
    /// `nprobe >= nlist` degenerates to scanning every cell, which is
    /// the same candidate set an unpartitioned index would score, and
    /// so gives back today's recall exactly. Vectors still in the
    /// pre-fit buffer are always scanned, at any `nprobe`.
    ///
    /// # Panics
    ///
    /// If `queries.len()` is not a multiple of `dim`.
    pub fn search(&self, queries: &[f32], k: usize, nprobe: usize) -> (Vec<f32>, Vec<u64>) {
        assert!(
            queries.len() % self.dim == 0,
            "query buffer length {} is not a multiple of dim {}",
            queries.len(),
            self.dim
        );
        let nq = queries.len() / self.dim;
        let mut out_scores = Vec::with_capacity(nq * k);
        let mut out_ids = Vec::with_capacity(nq * k);

        for qi in 0..nq {
            let q = &queries[qi * self.dim..(qi + 1) * self.dim];
            let mut hits: Vec<(f32, u64)> = Vec::new();

            // Anything not yet celled is scored directly: the buffer is
            // small by construction, and skipping it would silently
            // drop the most recent arrivals.
            for (i, id) in self.buffer_ids.iter().enumerate() {
                let v = &self.buffer[i * self.dim..(i + 1) * self.dim];
                let s: f32 = (0..self.dim).map(|d| q[d] * v[d]).sum();
                hits.push((s, *id));
            }

            if self.is_fitted() {
                // Rank cells by q·c, which is both the probe order and
                // the per-cell offset — computed once, used twice.
                let mut cell_scores: Vec<(f32, usize)> = (0..self.nlist)
                    .map(|c| {
                        let base = c * self.dim;
                        let s: f32 = (0..self.dim).map(|d| q[d] * self.centroids[base + d]).sum();
                        (s, c)
                    })
                    .collect();
                cell_scores.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

                for &(offset, c) in cell_scores.iter().take(nprobe.min(self.nlist)) {
                    let n_cell = self.cell_ids[c].len();
                    if n_cell == 0 {
                        continue;
                    }
                    // Every slot in the cell, so the merge sees the same
                    // candidates an exhaustive scan would — the cell is
                    // the pruning unit, not the per-cell k.
                    let res = self.cells[c].search(q, n_cell);
                    for (s, slot) in res.scores.iter().zip(res.indices.iter()) {
                        hits.push((offset + *s, self.cell_ids[c][*slot as usize]));
                    }
                }
            }

            hits.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            hits.truncate(k);
            for (s, id) in hits {
                out_scores.push(s);
                out_ids.push(id);
            }
        }
        (out_scores, out_ids)
    }
}

/// Lloyd's algorithm on unit-normalized rows, seeded deterministically.
///
/// Deliberately plain: the fit runs once over a bounded prefix, so its
/// cost is amortized to nothing, and a fancier initialization would buy
/// cell quality that `nprobe` can also buy. Determinism matters more —
/// two builds of the same corpus must place the same vectors in the
/// same cells.
fn kmeans(data: &[f32], dim: usize, k: usize, iters: usize) -> Vec<f32> {
    let n = data.len() / dim;
    let k = k.min(n).max(1);
    let mut centroids = vec![0.0f32; k * dim];

    // Seed from evenly spaced rows: no RNG to thread through, and an
    // even stride over an arbitrarily ordered corpus is as good a
    // spread as sampling.
    let stride = (n / k).max(1);
    for c in 0..k {
        let row = (c * stride).min(n - 1);
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[row * dim..(row + 1) * dim]);
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut moved = false;
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for c in 0..k {
                let base = c * dim;
                let mut s = 0.0f32;
                for d in 0..dim {
                    s += v[d] * centroids[base + d];
                }
                if s > best_s {
                    best_s = s;
                    best = c;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                moved = true;
            }
        }

        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assign[i];
            counts[c] += 1;
            let base = c * dim;
            let v = &data[i * dim..(i + 1) * dim];
            for d in 0..dim {
                sums[base + d] += v[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // An empty cell keeps its previous centroid: dropping it
                // would renumber cells mid-fit, and re-seeding it costs
                // a pass to find a far-away row for no measured gain.
                continue;
            }
            let base = c * dim;
            let mut norm = 0.0f32;
            for d in 0..dim {
                let m = sums[base + d] / counts[c] as f32;
                centroids[base + d] = m;
                norm += m * m;
            }
            // Spherical: the metric is inner product, so only the
            // direction of a centroid affects which cell wins, and unit
            // centroids keep `q · c` comparable across cells.
            let norm = norm.sqrt();
            if norm > crate::MIN_INPUT_NORM {
                for d in 0..dim {
                    centroids[base + d] /= norm;
                }
            }
        }
        if !moved {
            break;
        }
    }
    centroids
}
