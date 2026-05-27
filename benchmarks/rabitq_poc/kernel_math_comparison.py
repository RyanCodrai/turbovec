"""
Math-isolated ARM vs x86 kernel comparison — pure synthetic test.

Generates random codes + random LUTs, runs four scoring variants on the
SAME inputs, compares per-vector outputs. No dependency on rotation,
centroids, or real datasets — the question we're answering ("do ARM and
x86 kernels produce the same scores for the same LUT and codes?") is
purely about kernel arithmetic.

Variants:

* `arm` — ARM NEON kernel math: per byte-group, compute `u8_sum = lo+hi`
  (capped at 254 with max_lut=127, otherwise wraps modulo 256), accumulate
  into u16, flush to f32 every FLUSH_EVERY=256 byte-groups.

* `x86_current` — AVX2 kernel math: accumulate u8 lookups directly into
  i16 lanes (FAISS even/odd-byte interleave), NO periodic flush. Per
  nibble-half sum must fit in 16 bits, so effective `max_lut <=
  65535 / n_byte_groups`. The implementation here collapses to:
  `lo_sum_u16 + hi_sum_u16` computed mod 2^16 each.

* `x86_with_flush` — hypothetical fix: same i16 accumulator BUT flushed
  to f32 every 256 byte-groups, mirroring ARM. Per-flush max sum is
  `flush_every * max_lut = 256 * 127 = 32512 <= 65535`, so this allows
  max_lut=127 at any dim.

* `exact_int` — bit-exact integer reference: pure-Python sum of LUT
  lookups in unbounded ints. No modular wrap. The "what should happen
  in real arithmetic" baseline.

Usage:
  python3 kernel_math_comparison.py [dim]  # default 3072
"""

import sys

import numpy as np


FLUSH_EVERY = 256
SEED = 42


# ─── Kernel simulations ──────────────────────────────────────────────────────

def arm_kernel_score(codes_one_vec: np.ndarray, lut_u8: np.ndarray,
                     scale: float, bias: float, vec_scale: float,
                     flush_every: int = FLUSH_EVERY) -> float:
    """NEON math: u8 sum lo+hi per byte-group → widen u16 → periodic flush."""
    dim = codes_one_vec.shape[0]
    n_byte_groups = dim // 2
    fa = float(bias)
    u16_accum = 0
    flush_idx = 0
    for g in range(n_byte_groups):
        lo = lut_u8[2 * g, codes_one_vec[2 * g]]
        hi = lut_u8[2 * g + 1, codes_one_vec[2 * g + 1]]
        u8_sum = (lo + hi) & 0xFF  # explicit u8 modular wrap
        u16_accum = (u16_accum + u8_sum) & 0xFFFF
        flush_idx += 1
        if flush_idx >= flush_every or g == n_byte_groups - 1:
            fa += scale * float(u16_accum)
            u16_accum = 0
            flush_idx = 0
    return fa * vec_scale


def x86_kernel_score(codes_one_vec: np.ndarray, lut_u8: np.ndarray,
                     scale: float, bias: float, vec_scale: float) -> float:
    """AVX2 math: lo and hi sums accumulated independently into u16 lanes,
    no flush. Each sum must fit in 16 bits; above that, modular wrap."""
    dim = codes_one_vec.shape[0]
    n_byte_groups = dim // 2
    lo_sum_u16 = 0
    hi_sum_u16 = 0
    for g in range(n_byte_groups):
        lo = lut_u8[2 * g, codes_one_vec[2 * g]]
        hi = lut_u8[2 * g + 1, codes_one_vec[2 * g + 1]]
        lo_sum_u16 = (int(lo_sum_u16) + int(lo)) & 0xFFFF
        hi_sum_u16 = (int(hi_sum_u16) + int(hi)) & 0xFFFF
    return (bias + scale * (int(lo_sum_u16) + int(hi_sum_u16))) * vec_scale


def x86_with_flush_score(codes_one_vec: np.ndarray, lut_u8: np.ndarray,
                         scale: float, bias: float, vec_scale: float,
                         flush_every: int = FLUSH_EVERY) -> float:
    """Hypothetical fix: same i16 accumulator structure, periodic flush."""
    dim = codes_one_vec.shape[0]
    n_byte_groups = dim // 2
    fa = float(bias)
    lo_sum_u16 = 0
    hi_sum_u16 = 0
    flush_idx = 0
    for g in range(n_byte_groups):
        lo = lut_u8[2 * g, codes_one_vec[2 * g]]
        hi = lut_u8[2 * g + 1, codes_one_vec[2 * g + 1]]
        lo_sum_u16 = (int(lo_sum_u16) + int(lo)) & 0xFFFF
        hi_sum_u16 = (int(hi_sum_u16) + int(hi)) & 0xFFFF
        flush_idx += 1
        if flush_idx >= flush_every or g == n_byte_groups - 1:
            fa += scale * (int(lo_sum_u16) + int(hi_sum_u16))
            lo_sum_u16 = 0
            hi_sum_u16 = 0
            flush_idx = 0
    return fa * vec_scale


def exact_int_score(codes_one_vec: np.ndarray, lut_u8: np.ndarray,
                    scale: float, bias: float, vec_scale: float) -> float:
    """Reference: unbounded integer sum, no modular wrap."""
    dim = codes_one_vec.shape[0]
    n_byte_groups = dim // 2
    total = 0
    for g in range(n_byte_groups):
        total += int(lut_u8[2 * g, codes_one_vec[2 * g]])
        total += int(lut_u8[2 * g + 1, codes_one_vec[2 * g + 1]])
    return (bias + scale * total) * vec_scale


# ─── Driver ──────────────────────────────────────────────────────────────────

def run(dim: int, n_vectors: int, max_lut: int, lut_distribution: str = "uniform"):
    n_byte_groups = dim // 2
    n_subs = dim

    print(f"\n--- dim={dim} n_byte_groups={n_byte_groups} n_subs={n_subs} max_lut={max_lut} ---")
    # x86 sum-fits-in-u16 constraint: max_lut * n_byte_groups <= 65535
    sum_cap = n_byte_groups * max_lut
    print(f"x86 per-half max sum: {n_byte_groups} * {max_lut} = {sum_cap}"
          f"  ({'FITS in u16' if sum_cap <= 65535 else 'OVERFLOWS u16 (sum mod 2^16 corrupts result)'})")
    # ARM per-flush sum: flush_every * (lo+hi cap)
    arm_per_flush = FLUSH_EVERY * min(2 * max_lut, 255)  # u8 sum capped at 255
    print(f"ARM per-flush u16 sum: min(2*max_lut, 255) * FLUSH_EVERY = "
          f"{arm_per_flush}  ({'FITS' if arm_per_flush <= 65535 else 'OVERFLOWS u16'})")

    rng = np.random.RandomState(SEED)
    # Random codes 0..15
    codes = rng.randint(0, 16, size=(n_vectors, dim), dtype=np.int8)

    # Generate a per-sub-table LUT. "uniform" = all sub-tables span similar
    # range (mimics TQ+ output), "skewed" = a few wide + many narrow (mimics
    # raw GloVe-like distribution).
    if lut_distribution == "uniform":
        # Each sub-table: values uniformly distributed up to max_lut.
        lut_u8 = rng.randint(0, max_lut + 1, size=(n_subs, 16), dtype=np.uint16)
    elif lut_distribution == "skewed":
        # 10% sub-tables span full range, 90% span ~10% of full range.
        spans = np.where(rng.uniform(size=n_subs) < 0.1, max_lut, max_lut // 10 + 1)
        lut_u8 = np.zeros((n_subs, 16), dtype=np.uint16)
        for s in range(n_subs):
            lut_u8[s] = rng.randint(0, spans[s] + 1, size=16)
    else:
        raise ValueError(lut_distribution)

    scale = 0.01  # arbitrary; doesn't affect ranking
    bias = 0.0
    vec_scales = np.full(n_vectors, 1.0)

    arm = np.zeros(n_vectors)
    x86 = np.zeros(n_vectors)
    x86_f = np.zeros(n_vectors)
    exact = np.zeros(n_vectors)
    for i in range(n_vectors):
        c = codes[i]
        arm[i] = arm_kernel_score(c, lut_u8, scale, bias, vec_scales[i])
        x86[i] = x86_kernel_score(c, lut_u8, scale, bias, vec_scales[i])
        x86_f[i] = x86_with_flush_score(c, lut_u8, scale, bias, vec_scales[i])
        exact[i] = exact_int_score(c, lut_u8, scale, bias, vec_scales[i])

    # Compare each variant to the exact reference.
    def report(name, arr):
        diff_count = int(np.sum(np.abs(arr - exact) > 1e-9))
        max_diff = float(np.max(np.abs(arr - exact)))
        ranks_arr = np.argsort(-arr)
        ranks_exact = np.argsort(-exact)
        # top-K agreement
        K = 10
        topk_arr = set(np.argpartition(-arr, K)[:K].tolist())
        topk_exact = set(np.argpartition(-exact, K)[:K].tolist())
        overlap = len(topk_arr & topk_exact)
        print(f"  {name:<16} mismatches={diff_count:>5}/{n_vectors}   "
              f"max|Δ|={max_diff:.4f}   top-{K} overlap with exact={overlap}/{K}")

    report("exact_int", exact)
    report("arm", arm)
    report("x86_current", x86)
    report("x86_with_flush", x86_f)


def main():
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 3072

    # Sweep 1: each kernel at the max_lut value where IT would currently operate.
    print("\n=== current production max_lut per arch ===")
    n_byte_groups = dim // 2
    x86_cap = min(127, 65535 // n_byte_groups // 1)  # see search.rs formula (n_byte_groups*2 in denom = n_subs)
    arm_cap = 127
    print(f"x86 cap derived from search.rs formula: min(127, 65535/{2*n_byte_groups}) "
          f"= {min(127, 65535 // (2 * n_byte_groups))}")
    print(f"ARM cap: 127")
    run(dim, n_vectors=200, max_lut=min(127, 65535 // (2 * n_byte_groups)), lut_distribution="uniform")
    # And ARM at its own cap
    run(dim, n_vectors=200, max_lut=127, lut_distribution="uniform")

    # Sweep 2: force x86 to use the ARM cap (max_lut=127) — does ARM math match
    # x86_with_flush? Does x86_current overflow?
    print("\n=== force max_lut=127 on both kernels (high precision regime) ===")
    run(dim, n_vectors=200, max_lut=127, lut_distribution="uniform")

    # Sweep 3: same at low dim where x86_current doesn't overflow.
    print("\n=== sanity check: low dim where x86_current is fine ===")
    run(dim=200, n_vectors=200, max_lut=127, lut_distribution="uniform")


if __name__ == "__main__":
    main()
