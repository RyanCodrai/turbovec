# How it works

Each vector is a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a simple insight: after applying a random rotation, every coordinate follows a known distribution -- regardless of the input data.

**1. Normalize.** Strip the length (norm) from each vector and store it as a single float. Now every vector is a unit direction on the hypersphere.

**2. Random rotation.** Multiply all vectors by the same random orthogonal matrix. After rotation, each coordinate independently follows a Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions. This holds for any input data -- the rotation makes the coordinate distribution predictable.

**3. Per-coordinate calibration (TQ+).** The Beta distribution from step 2 is asymptotic — at finite dimensions, individual coordinates drift from the canonical shape (especially low-bit and word-vector-style embeddings). TQ+ fits two scalars per coordinate — a shift and a scale — mapping each coordinate's empirical quantiles onto the codebook's outermost centroids. The probability level comes from the codebook, so it tracks the bit width (~0.933 at 2-bit, ~0.996 at 4-bit) rather than being fixed. The Lloyd-Max codebook then quantizes against the *target* distribution it was designed for. The fit is explicit: call `index.calibrate(sample)` once with a random, representative sample of your vectors (~1024 rows is enough — a draw of that size matches fitting on the whole corpus) before adding; afterwards the calibration is committed and reused by every add — no retraining, no rebuilds, no separate train phase. An index you never calibrate is plain TurboQuant. `index.calibration_state` reports `"uncalibrated"` or `"calibrated"`. Recall gain: up to +2.2pp at @1 on the cells that drift most (e.g. GloVe at 2-bit).

**4. Lloyd-Max scalar quantization.** Since the distribution is known, we can precompute the optimal way to bucket each coordinate. For 2-bit, that's 4 buckets; for 4-bit, 16 buckets. The [Lloyd-Max algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) finds bucket boundaries and centroids that minimize mean squared error. These are computed once from the math, not from the data.

**5. Bit-pack.** Each coordinate is now a small integer (0-3 for 2-bit, 0-15 for 4-bit). Pack these tightly into bytes. A 1536-dim vector goes from 6,144 bytes (FP32) to 384 bytes (2-bit). That's 16x compression.

**6. Length-renormalized scoring.** Scalar quantization systematically underestimates inner products — the reconstructed unit direction is a little shorter than the original. We compute one scalar per vector at encode time — the inner product of the rotated unit vector with its own centroid reconstruction — and store `||v|| / ⟨u, x̂⟩` alongside each compressed vector. The search kernel multiplies the per-candidate score by this scalar before heap insertion, turning the inner-product estimator from downward-biased into unbiased at zero search-time cost and zero extra storage. The recall gain shows up most at low bit widths, where the quantization shrinkage is largest.

Encoding cost: one extra `d`-dimensional dot product per vector to compute `⟨u, x̂⟩`. On 1M vectors at d=1536 this is sub-second of additional encode time — a one-shot price paid at ingest, not at query.

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. The scoring kernel uses SIMD intrinsics (NEON on ARM; AVX-512BW on modern x86, falling back to AVX2, then to a scalar path on pre-AVX2 CPUs) with nibble-split lookup tables for maximum throughput.

The Lloyd-Max codebook achieves distortion within a factor of 2.7x of the information-theoretic lower bound (Shannon's distortion-rate limit); the length-renormalization step removes the residual bias the Lloyd-Max codebook introduces on the inner-product estimator itself.

