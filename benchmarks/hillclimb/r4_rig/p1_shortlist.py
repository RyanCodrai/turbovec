"""4-bit round 2, P1: how large a shortlist does a coarse first stage need to hold the exact 4-bit top-k?
usage: p1_shortlist.py DUMP_PREFIX DIM NDB NQ   (dump from turbovec/src/plane_probe.rs)"""
import sys, numpy as np
pre, dim, ndb, nq = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
codes = np.fromfile(pre + "_codes.u8", dtype=np.uint8).reshape(ndb, dim // 2)
# two 4-bit codes per byte; find which nibble is the first dim by checking against the exact scores below
scales = np.fromfile(pre + "_scales.f32", dtype=np.float32)
cent = np.fromfile(pre + "_centroids.f32", dtype=np.float32)
q = np.fromfile(pre + "_q.f32", dtype=np.float32).reshape(nq, dim)
bias = np.fromfile(pre + "_bias.f32", dtype=np.float32)
ids = np.fromfile(pre + "_ids.i64", dtype=np.int64).reshape(nq, -1)
sc = np.fromfile(pre + "_scores.f32", dtype=np.float32).reshape(nq, -1)
assert len(cent) == 16, len(cent)
def unpack(order):
    c = np.empty((ndb, dim), dtype=np.uint8)
    lo, hi = codes & 15, codes >> 4
    c[:, 0::2], c[:, 1::2] = (lo, hi) if order == "lo" else (hi, lo)
    return c
best = None
for order in ("lo", "hi"):
    c = unpack(order)
    v = ids[0, 0]
    est = (q[0] @ cent[c[v]] + bias[0]) * scales[v]
    err = abs(est - sc[0, 0]) / abs(sc[0, 0])
    print(f"nibble order {order}: float score {est:.5f} vs kernel {sc[0,0]:.5f} (rel err {err:.4f})")
    if best is None or err < best[0]: best = (err, order, c)
_, order, c = best
print("using order", order, "| centroids", np.round(cent, 4))
sign = c >= 8
mag = np.where(sign, c - 8, 7 - c)            # 0 = innermost level, 7 = outermost, if the codebook is symmetric
absc = np.abs(cent)
def planes_est(levels_of_mag):
    """per-dim value = sgn * levels_of_mag[mag]"""
    return np.where(sign, 1.0, -1.0).astype(np.float32) * levels_of_mag[mag].astype(np.float32)
freq = np.bincount(mag.ravel(), minlength=8) / mag.size
mag_c = np.array([(absc[8 + j] + absc[7 - j]) / 2 for j in range(8)])
m1 = (freq * mag_c).sum()                                             # sign only: one magnitude
top2 = np.array([ (freq[:4] * mag_c[:4]).sum() / freq[:4].sum() ] * 4 + [ (freq[4:] * mag_c[4:]).sum() / freq[4:].sum() ] * 4)
top3 = np.repeat([(freq[i:i+2] * mag_c[i:i+2]).sum() / freq[i:i+2].sum() for i in (0, 2, 4, 6)], 2)
# linear-in-bits fit of the magnitude (what a popcount ranking can compute): mag_c ~ a + b2*bit2 + b1*bit1 + b0*bit0
B = np.array([[1, (j >> 2) & 1, (j >> 1) & 1, j & 1] for j in range(8)], dtype=np.float64)
W = np.diag(freq)
coef = np.linalg.solve(B.T @ W @ B, B.T @ W @ mag_c)
lin = B @ coef
print("level freq", np.round(freq, 3), "| |c|", np.round(mag_c, 4), "| linear-in-bits fit", np.round(lin, 4))
ests = {"sign (1 bit)": np.full(8, m1), "top-2 bits": top2, "top-3 bits": top3, "linear-in-bits (4 bits)": lin, "exact levels (float)": mag_c}
for name, lv in ests.items():
    E = planes_est(lv)                                                # ndb x dim float32
    out = []
    for k in (1, 10, 100):
        need = np.empty(nq, dtype=np.int64)
        for s in range(0, nq, 250):
            S = (q[s:s + 250] @ E.T + bias[s:s + 250, None]) * scales[None, :]   # 250 x ndb
            order_ = np.argsort(-S, axis=1)
            rank = np.empty_like(order_); np.put_along_axis(rank, order_, np.arange(ndb)[None, :].repeat(len(order_), 0), axis=1)
            need[s:s + 250] = np.take_along_axis(rank, ids[s:s + 250, :k], axis=1).max(axis=1) + 1
        qs = np.sort(need)
        out.append(f"k={k}: median {int(np.median(need))} p99 {qs[int(nq * 0.99) - 1]} p99.9 {qs[int(nq * 0.999) - 1]} max {qs[-1]}")
    print(f"{name:26s} shortlist needed | " + " | ".join(out), flush=True)
