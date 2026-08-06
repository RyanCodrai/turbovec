"""Can we keep Lloyd-Max accuracy AND get a dot-product kernel?

P17 concluded that making the score a dot product costs ~2 recall points,
because the reconstruction levels have to be uniform. That conclusion was
too strong, and this measures the variant it missed.

The requirement is not that the levels be uniform. It is that the level be
reachable from the stored nibble by something cheaper than a per-dimension
per-query table. turbovec's codebook `C` is SHARED across all dimensions,
so nibble -> level is a *fixed 16-entry permute*: query-independent,
dimension-independent, register-resident for the whole scan. Apply it to
the nibbles that already have to be unpacked, and

    score = sum_d q[d] * C[code[d]]

becomes a plain dot product over the permuted bytes — with the full
Lloyd-Max codebook intact. Nothing about the codebook changes.

The accuracy question is different from P17's and is not obviously a loss.
Today `build_query_neon_lut` quantises the *product* `q[d]*C[c]` to 7 bits
(cap 127) per (dimension, level), and the kernel sums those u8 entries.
The dot-product form instead quantises `q` and `C` to 8 bits *separately*
and accumulates the products exactly in u32. That is a different error
structure, and it could be better: no per-entry rounding of a product, and
no u16 flush.

Compares four scorers against exact float ground truth:

  float           exact q, Lloyd-Max reconstruction — the accuracy ceiling
  lut7            what ships today: per-(d,c) product quantised to 0..127
  permute-dot     q to int8, C to int8, exact integer accumulation
  permute-dot-16  the same with a 16-bit query, to separate "int8 query is
                  too coarse" from "the scheme is wrong"
"""
import numpy as np

N, DIM, NQ, K, LEVELS = 20_000, 768, 200, 10, 16
rng = np.random.default_rng(0)


def lloyd_max_levels(samples, levels, iters=60):
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


def recall(top, truth):
    return np.mean([len(set(top[r]) & set(truth[r])) / K for r in range(len(top))])


def topk(scores):
    return np.argpartition(-scores, K, axis=1)[:, :K]


raw = rng.normal(size=(N, DIM))
raw /= np.linalg.norm(raw, axis=1, keepdims=True)
q = rng.normal(size=(NQ, DIM))
q /= np.linalg.norm(q, axis=1, keepdims=True)
truth = topk(q @ raw.T)

c = lloyd_max_levels(raw[:: max(1, N // 4000)].ravel().copy(), LEVELS)
bnd = (c[1:] + c[:-1]) / 2.0
codes = np.searchsorted(bnd, raw).astype(np.int32)      # (N, DIM) nibbles

print(f"N={N} dim={DIM} nq={NQ} k={K} levels={LEVELS}\n")

# --- ceiling: exact query against the Lloyd-Max reconstruction -----------
s_float = q @ c[codes].T
r_float = recall(topk(s_float), truth)

# --- what ships today: per-(d,c) product quantised to 0..127 -------------
# Mirrors build_query_neon_lut_from_slice: per query, table[d][c] holds
# round((q[d]*c[level] - min_d) / scale) with a per-query scale, and the
# kernel sums those u8 entries. The per-dimension min is folded into a
# constant, so it does not affect ranking.
tbl = q[:, :, None] * c[None, None, :]                   # (NQ, DIM, LEVELS)
mins = tbl.min(axis=2, keepdims=True)
scale = (tbl.max(axis=2, keepdims=True) - mins).max() / 127.0
qt = np.round((tbl - mins) / scale)
s_lut = np.empty((NQ, N))
for i in range(NQ):
    s_lut[i] = qt[i][np.arange(DIM)[None, :], codes].sum(axis=1)
r_lut = recall(topk(s_lut), truth)

# --- the proposal: int8 query, int8 codebook, exact integer accumulation --
def permute_dot(qbits):
    lim = (1 << (qbits - 1)) - 1
    # Codebook to int8 once, at index-build time.
    ci = np.round(c / np.abs(c).max() * 127.0)
    # Query to `qbits` per dimension, per query (its own scale).
    qs = np.abs(q).max(axis=1, keepdims=True) / lim
    qi = np.round(q / qs)
    recon = ci[codes]                                     # (N, DIM) int8 levels
    return recall(topk(qi @ recon.T), truth)

r_p8 = permute_dot(8)
r_p16 = permute_dot(16)

print(f"float  (ceiling, needs f32 dot)   recall@{K} = {r_float:.4f}")
print(f"lut7   (ships today)              recall@{K} = {r_lut:.4f} "
      f"({r_lut - r_float:+.4f} vs ceiling)")
print(f"permute-dot, int8 query           recall@{K} = {r_p8:.4f} "
      f"({r_p8 - r_lut:+.4f} vs shipping)")
print(f"permute-dot, int16 query          recall@{K} = {r_p16:.4f} "
      f"({r_p16 - r_lut:+.4f} vs shipping)")
print()
print("The comparison that decides it is permute-dot vs lut7, since lut7")
print("is what ships. Both use the same Lloyd-Max codebook; they differ")
print("only in where the quantisation of the query side happens.")
