"""How expressive can a codebook be while the score stays a dot product?

P15 measured a uniform 4-bit codebook at -0.021 recall against Lloyd-Max,
and that was framed as the price of making the score a plain integer dot
product. But uniform is far stronger a constraint than the dot product
actually requires, and the gap is worth re-deriving from scratch.

The score is  sum_d q[d] * C[code[d]].  For this to be computable without a
lookup table, C need not be *uniform* — it only needs to be **linear in
something we can accumulate**. Three families, in increasing expressiveness:

  uniform    C[c] = a*c + b                      1 free shape param
             -> one dot product over the raw codes.

  split      C[c] = s1*(c>>2) + s2*(c&3) + b     2 free shape params
             -> two dot products over two derived 2-bit streams.

  bitlinear  C[c] = sum_k w_k * bit_k(c) + b     4 free shape params
             -> four dot products over four *binary* streams.

Uniform is the special case w_k = a*2^k, so bitlinear can only do better.
Bitlinear is the interesting one: four free parameters is a lot of shape,
and binary streams are the cheapest possible operand.

Note on boundaries: for any FIXED set of reconstruction levels the
MSE-optimal decision boundaries are exactly the midpoints, which is what
`encode` uses throughout. So there is no separate "boundary freedom" to
exploit — the entire question is how much shape the level set can carry.
That is what this measures.

Each family is fit with a constrained Lloyd iteration: assign samples to
the nearest achievable level, then re-fit the free parameters by weighted
least squares under the family's linear constraint, and repeat. That gives
each family its best shot rather than a hand-picked parameterisation.
"""
import numpy as np

N, DIM, NQ, K, LEVELS = 20_000, 768, 200, 10, 16
BITS = 4
rng = np.random.default_rng(0)


def lloyd_max_levels(samples, levels, iters=60):
    """Unconstrained Lloyd-Max: the accuracy ceiling, needs a LUT."""
    lo, hi = np.quantile(samples, [0.001, 0.999])
    c = np.linspace(lo, hi, levels)
    for _ in range(iters):
        b = (c[1:] + c[:-1]) / 2.0
        idx = np.searchsorted(b, samples)
        for j in range(levels):
            m = idx == j
            if m.any():
                c[j] = samples[m].mean()
    return np.sort(c)


def basis_for(family):
    """Design matrix B (LEVELS x P): C = B @ theta, one row per code."""
    codes = np.arange(LEVELS)
    ones = np.ones(LEVELS)
    if family == "uniform":
        return np.column_stack([ones, codes])
    if family == "split":
        return np.column_stack([ones, codes >> 2, codes & 3])
    if family == "bitlinear":
        return np.column_stack([ones] + [(codes >> k) & 1 for k in range(BITS)])
    raise ValueError(family)


def _fit_from(samples, B, c, iters=80):
    """One constrained-Lloyd descent from a given starting level set."""
    for _ in range(iters):
        b = (c[1:] + c[:-1]) / 2.0
        idx = np.searchsorted(b, samples)
        # Per-code sufficient statistics: count and mean.
        cnt = np.bincount(idx, minlength=LEVELS).astype(np.float64)
        tot = np.bincount(idx, weights=samples, minlength=LEVELS)
        live = cnt > 0
        if live.sum() < B.shape[1]:
            break
        mean = np.zeros(LEVELS)
        mean[live] = tot[live] / cnt[live]
        # Minimising sum_j cnt_j * (mean_j - (B theta)_j)^2 over theta.
        W = np.sqrt(cnt[live])[:, None]
        theta, *_ = np.linalg.lstsq(B[live] * W, mean[live] * W[:, 0], rcond=None)
        new = B @ theta
        if np.allclose(new, c):
            c = new
            break
        c = new
    # Sort for midpoint encoding to be the nearest-level rule. Sorting
    # permutes which integer denotes which level, which is free: the
    # encoder stores the permuted code, and that code's BITS are what the
    # dot product weights. A bijective relabelling costs nothing.
    return np.sort(c)


def fit_constrained(samples, family, iters=60, restarts=60):
    """Best constrained-Lloyd fit over many starting points.

    A single descent from a uniform start converges straight back to
    uniform for every family, which would be a misleading null result —
    the uniform grid is a fixed point of the iteration, so starting there
    guarantees finding it. These families can express genuinely
    non-uniform level sets (the subset sums of four weights are
    binomially dense in the middle, which is the same qualitative shape
    Lloyd-Max wants), so the search has to be given a real chance to
    leave the uniform basin.
    """
    B = basis_for(family)
    P = B.shape[1]
    # Fitting is O(restarts * iters * |samples|); the distribution is
    # pinned long before the full 3M coordinates, so fit on a subsample
    # and let the recall evaluation below use all of the data.
    samples = samples[:: max(1, len(samples) // 200_000)]
    lo, hi = np.quantile(samples, [0.005, 0.995])
    scale = max(abs(lo), abs(hi))

    starts = []
    # 1. The uniform grid, the obvious fixed point.
    starts.append(np.linspace(lo, hi, LEVELS))
    # 2. Lloyd-Max projected onto the family — the closest this family can
    #    sit to the unconstrained optimum, as a warm start.
    lm = lloyd_max_levels(samples.copy(), LEVELS)
    theta, *_ = np.linalg.lstsq(B, lm, rcond=None)
    starts.append(np.sort(B @ theta))
    # 3. Random weight vectors, so the descent can find basins that
    #    neither of the structured starts reaches.
    r = np.random.default_rng(12345)
    for _ in range(restarts):
        th = r.normal(size=P) * scale / 4.0
        starts.append(np.sort(B @ th))

    best, best_mse = None, np.inf
    for c0 in starts:
        if not np.all(np.isfinite(c0)) or np.ptp(c0) < 1e-12:
            continue
        c = _fit_from(samples, B, c0.copy(), iters)
        mse = np.mean((c[encode(samples, c)] - samples) ** 2)
        if mse < best_mse:
            best, best_mse = c, mse
    return best


def encode(x, c):
    return np.searchsorted((c[1:] + c[:-1]) / 2.0, x).astype(np.int32)


def recall_of(c, data, queries, truth):
    recon = c[encode(data, c)]
    scores = queries @ recon.T
    top = np.argpartition(-scores, K, axis=1)[:, :K]
    return np.mean([len(set(top[r]) & set(truth[r])) / K
                    for r in range(len(queries))])


raw = rng.normal(size=(N, DIM))
raw /= np.linalg.norm(raw, axis=1, keepdims=True)
q = rng.normal(size=(NQ, DIM))
q /= np.linalg.norm(q, axis=1, keepdims=True)
truth = np.argpartition(-(q @ raw.T), K, axis=1)[:, :K]
samples = raw[:: max(1, N // 4000)].ravel().copy()

books = {
    "lloyd-max (needs LUT)": lloyd_max_levels(samples.copy(), LEVELS),
    "uniform": fit_constrained(samples, "uniform"),
    "split (2 dot products)": fit_constrained(samples, "split"),
    "bitlinear (4 binary dots)": fit_constrained(samples, "bitlinear"),
}

print(f"N={N} dim={DIM} nq={NQ} k={K} levels={LEVELS}\n")
base_r = base_m = None
for name, c in books.items():
    mse = np.mean((c[encode(samples, c)] - samples) ** 2)
    rec = recall_of(c, raw, q, truth)
    if base_r is None:
        base_r, base_m = rec, mse
    print(f"{name:<28} recall@{K}={rec:.4f} ({rec - base_r:+.4f})   "
          f"MSE={mse:.4e} ({mse / base_m:.3f}x)")

print("\nFitted level sets:")
for name, c in books.items():
    print(f"  {name:<28} {np.array2string(c, precision=3, max_line_width=200)}")
