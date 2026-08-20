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

use rayon::prelude::*;

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
    /// Per cell, the largest residual norm it holds. For a unit query
    /// no member of cell `c` can score above `q·c + cell_bound[c]`, so
    /// a cell whose bound falls below the running k-th best can be
    /// skipped without reading a byte of it. Grows on insert; never
    /// shrunk by removal (a stale-high bound costs a probe, a
    /// stale-low one costs recall).
    cell_bound: Vec<f32>,
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
            cell_bound: Vec::new(),
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
            self.bulk_insert(vectors, &ids);
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

    /// Assign `vectors` in parallel, then append each cell's share in
    /// one contiguous `add` — one residual buffer and one encode pass
    /// per cell instead of one per row.
    fn bulk_insert(&mut self, vectors: &[f32], ids: &[u64]) {
        let n = ids.len();
        let dim = self.dim;
        let assigned: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| self.nearest_centroid(&vectors[i * dim..(i + 1) * dim]))
            .collect();

        // Group rows by cell, preserving arrival order within a cell so
        // ids and slots stay aligned.
        let mut rows_by_cell: Vec<Vec<usize>> = vec![Vec::new(); self.nlist];
        for (i, &c) in assigned.iter().enumerate() {
            rows_by_cell[c].push(i);
        }

        let centroids = &self.centroids;
        // Residual buffers are independent per cell: build them in
        // parallel, then do the (serial, &mut) appends.
        let residuals: Vec<(usize, Vec<f32>, f32)> = rows_by_cell
            .par_iter()
            .enumerate()
            .filter(|(_, rows)| !rows.is_empty())
            .map(|(c, rows)| {
                let base = c * dim;
                let mut buf = Vec::with_capacity(rows.len() * dim);
                let mut bound = 0.0f32;
                for &i in rows {
                    let v = &vectors[i * dim..(i + 1) * dim];
                    let start = buf.len();
                    buf.extend((0..dim).map(|d| v[d] - centroids[base + d]));
                    let norm2: f32 = buf[start..].iter().map(|x| x * x).sum();
                    bound = bound.max(norm2.sqrt());
                }
                (c, buf, bound)
            })
            .collect();
        for (c, buf, bound) in residuals {
            self.cells[c].add(&buf);
            self.cell_bound[c] = self.cell_bound[c].max(bound);
            self.cell_ids[c].extend(rows_by_cell[c].iter().map(|&i| ids[i]));
        }
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
        self.cell_bound = vec![0.0; self.nlist];
        self.cell_ids = vec![Vec::new(); self.nlist];

        let buffered = std::mem::take(&mut self.buffer);
        let ids = std::mem::take(&mut self.buffer_ids);
        self.bulk_insert(&buffered, &ids);
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
        let dim = self.dim;
        let profile = std::env::var_os("TV_IVF_PROFILE").is_some();
        let t_all = std::time::Instant::now();
        let mut t_rank = std::time::Duration::ZERO;
        let mut t_aud = std::time::Duration::ZERO;
        let mut t_scan = std::time::Duration::ZERO;
        let mut t_merge = std::time::Duration::ZERO;
        // Per-query candidate heaps, filled cell-major below.
        let mut hits: Vec<Vec<(f32, u64)>> = vec![Vec::with_capacity(k * 4); nq];

        // Anything not yet celled is scored directly: the buffer is
        // small by construction, and skipping it would silently drop
        // the most recent arrivals.
        for (qi, hit) in hits.iter_mut().enumerate() {
            let q = &queries[qi * dim..(qi + 1) * dim];
            for (i, id) in self.buffer_ids.iter().enumerate() {
                let v = &self.buffer[i * dim..(i + 1) * dim];
                let s: f32 = (0..dim).map(|d| q[d] * v[d]).sum();
                hit.push((s, *id));
            }
        }

        if self.is_fitted() {
            let nprobe = nprobe.min(self.nlist);
            // q·c for every (query, cell): the probe ranking and the
            // score offset in one pass, parallel over queries.
            let t0 = std::time::Instant::now();
            // Tiled: 8 queries share each streamed centroid row, so the
            // centroid matrix is read nq/8 times instead of nq times.
            // H1 measured the untiled form at 52% of the whole batched
            // call — a 1.1 GFLOP product hidden behind 2.2 GB of
            // re-streamed centroid traffic.
            const QT: usize = 8;
            let cell_scores: Vec<Vec<f32>> = {
                let tiles: Vec<Vec<Vec<f32>>> = (0..nq.div_ceil(QT))
                    .into_par_iter()
                    .map(|t| {
                        let q0 = t * QT;
                        let qn = QT.min(nq - q0);
                        let mut out = vec![vec![0.0f32; self.nlist]; qn];
                        for c in 0..self.nlist {
                            let cent = &self.centroids[c * dim..(c + 1) * dim];
                            for (j, row) in out.iter_mut().enumerate() {
                                let q = &queries[(q0 + j) * dim..(q0 + j + 1) * dim];
                                let mut s = 0.0f32;
                                for d in 0..dim {
                                    s += q[d] * cent[d];
                                }
                                row[c] = s;
                            }
                        }
                        out
                    })
                    .collect();
                tiles.into_iter().flatten().collect()
            };

            t_rank = t0.elapsed();
            let t0 = std::time::Instant::now();
            // Invert to cell-major: which queries probe cell c, so each
            // cell is scanned once for its whole audience and the
            // query-side preparation amortizes the way the flat batch
            // path amortizes it.
            //
            // A bound-based early exit (skip a cell when q·c + its max
            // residual norm cannot beat the running k-th best) was
            // built and measured here: recall identical, throughput
            // *lower* — the max-norm bound never fires on real
            // embeddings because one outlier residual per cell keeps
            // every bound above every floor, and the wave scheduling
            // it needs fragments the batching. Same shape as H116 in
            // the search log: removing work a bound proves unnecessary
            // saves nothing when the bound cannot prove it.
            let mut audience: Vec<Vec<usize>> = vec![Vec::new(); self.nlist];
            for qi in 0..nq {
                let mut order: Vec<usize> = (0..self.nlist).collect();
                order.sort_unstable_by(|&a, &b| cell_scores[qi][b].total_cmp(&cell_scores[qi][a]));
                for &c in order.iter().take(nprobe) {
                    audience[c].push(qi);
                }
            }

            t_aud = t0.elapsed();
            let t0 = std::time::Instant::now();
            let cell_results: Vec<Vec<(usize, f32, u64)>> = (0..self.nlist)
                .into_par_iter()
                .map(|c| {
                    let aud = &audience[c];
                    let n_cell = self.cell_ids[c].len();
                    if aud.is_empty() || n_cell == 0 {
                        return Vec::new();
                    }
                    let mut qbuf = Vec::with_capacity(aud.len() * dim);
                    for &qi in aud {
                        qbuf.extend_from_slice(&queries[qi * dim..(qi + 1) * dim]);
                    }
                    let res = self.cells[c].search(&qbuf, k.min(n_cell));
                    let mut out = Vec::with_capacity(aud.len() * res.k);
                    for (row, &qi) in aud.iter().enumerate() {
                        let offset = cell_scores[qi][c];
                        let ss = res.scores_for_query(row);
                        let ii = res.indices_for_query(row);
                        for (s, slot) in ss.iter().zip(ii.iter()) {
                            out.push((qi, offset + *s, self.cell_ids[c][*slot as usize]));
                        }
                    }
                    out
                })
                .collect();
            t_scan = t0.elapsed();
            let t0 = std::time::Instant::now();
            for triples in cell_results {
                for (qi, s, id) in triples {
                    hits[qi].push((s, id));
                }
            }
            t_merge = t0.elapsed();
        }
        if profile {
            eprintln!(
                "[ivf-profile] nq={nq} total={:?} rank={t_rank:?} audience={t_aud:?} scan={t_scan:?} merge={t_merge:?}",
                t_all.elapsed()
            );
        }

        let mut out_scores = Vec::with_capacity(nq * k);
        let mut out_ids = Vec::with_capacity(nq * k);
        for mut hit in hits {
            hit.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            hit.truncate(k);
            for (s, id) in hit {
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
        // The assignment step is the hot loop (n × k × dim), and rows
        // are independent: parallelize it, then compare against the old
        // assignment to detect convergence.
        let new_assign: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
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
                best
            })
            .collect();
        let moved = new_assign != assign;
        assign = new_assign;

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
