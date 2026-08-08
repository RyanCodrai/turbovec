// The 2-bit nibble-LUT scan loop, standalone, so an ablation costs two
// seconds instead of a 90-second crate build and a four-minute smoke.
//
// This is a transcription of `score_4bit_block_neon`'s inner loop as it runs
// at 2 bits: BLOCK=32 vectors, 192 byte-groups, FLUSH_EVERY=256 so there is
// exactly one batch and one flush per block. Before trusting any number here,
// check that variant 0's emitted loop still matches the shipped one —
// `objdump -d` it and compare against the 69-instruction / 4-group body in
// LOG_2bit.md. A probe that has drifted from the kernel measures nothing.
//
// Why it exists: H43 established that the arm nq=1 ST residue is inside the
// scan loop rather than the epilogue — 17.2 cycles per 4-group iteration
// where the instruction count allows 14 — and narrowing that by crate builds
// costs a hypothesis each time.
//
// The variants are ablations, not candidates. Several are deliberately wrong
// (they compute the wrong scores); they exist to price one term each.
//
//   0  exact          the shipped loop, 38.4 MB of codes
//   1  resident       identical work, one 6 KB block reread — no DRAM traffic
//   2  no-lut-load    LUT hoisted out of the group loop — prices the L1 loads
//   3  no-shift       `ushr` replaced by a second `and` — prices the 2/cycle
//                     shift pipe against the 4/cycle logical pipes
//
// Build:  cc -O3 -march=armv8.4-a -o scan_probe scan_probe.c
// Run:    ./scan_probe [variant] [n_vectors]

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK 32
#define N_GROUPS 192  // dim 768 at 2 bits: 192 code bytes per vector

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double clock_hz(void) {
    uint64_t acc = 0;
    const uint64_t n = 10000000;
    double t0 = now();
    for (uint64_t i = 0; i < n; i++) {
        __asm__ volatile("add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         : "+r"(acc)::);
    }
    return (double)n * 8.0 / (now() - t0);
}

// One block: 192 groups, 4-way unrolled, one flush. Mirrors the Rust.
//
// `variant` must constant-fold, so this is always_inline and every call site
// passes a literal. The first version took it as a runtime argument and the
// two branches left inside the group loop cost 20% — variant 0 measured 20.56
// cy/iter against the shipped kernel's 17.2, which is the tell that a probe
// is measuring itself rather than its subject.
__attribute__((always_inline)) static inline float
scan_block(const uint8_t *codes_base, const uint8_t *luts, const int variant) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const uint8x16_t mask2 = vdupq_n_u8(0x0E);
    uint16x8_t a0 = vdupq_n_u16(0), a1 = vdupq_n_u16(0);
    uint16x8_t a2 = vdupq_n_u16(0), a3 = vdupq_n_u16(0);

    uint8x16_t hoist_hi = vld1q_u8(luts), hoist_lo = vld1q_u8(luts + 16);

    for (int g = 0; g + 3 < N_GROUPS; g += 4) {
        for (int j = 0; j < 4; j++) {
            const uint8_t *lp = luts + (size_t)(g + j) * 32;
            const uint8_t *cp = codes_base + (size_t)(g + j) * BLOCK;
            uint8x16_t lut_hi, lut_lo;
            if (variant == 2) {
                lut_hi = hoist_hi;
                lut_lo = hoist_lo;
            } else {
                lut_hi = vld1q_u8(lp);
                lut_lo = vld1q_u8(lp + 16);
            }
            uint8x16_t c0 = vld1q_u8(cp), c1 = vld1q_u8(cp + 16);
            // The shifted operand feeds a 16-entry table, so any op that
            // clears the top nibble keeps the lookup in range — which is what
            // makes variant 3 a legal ablation and not a crash.
            uint8x16_t h0 = variant == 3 ? vandq_u8(c0, mask2) : vshrq_n_u8(c0, 4);
            uint8x16_t h1 = variant == 3 ? vandq_u8(c1, mask2) : vshrq_n_u8(c1, 4);
            uint8x16_t s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),
                                     vqtbl1q_u8(lut_hi, h0));
            uint8x16_t s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)),
                                     vqtbl1q_u8(lut_hi, h1));
            a0 = vaddw_u8(a0, vget_low_u8(s0));
            a1 = vaddw_high_u8(a1, s0);
            a2 = vaddw_u8(a2, vget_low_u8(s1));
            a3 = vaddw_high_u8(a3, s1);
        }
    }
    // Flush, exactly once per block at 2 bits (FLUSH_EVERY=256 > 192).
    float32x4_t f = vdupq_n_f32(0);
    uint16x8_t accs[4] = {a0, a1, a2, a3};
    for (int i = 0; i < 4; i++) {
        f = vaddq_f32(f, vcvtq_f32_u32(vmovl_u16(vget_low_u16(accs[i]))));
        f = vaddq_f32(f, vcvtq_f32_u32(vmovl_high_u16(accs[i])));
    }
    return vaddvq_f32(f);
}

int main(int argc, char **argv) {
    int variant = argc > 1 ? atoi(argv[1]) : 0;
    size_t n_vectors = argc > 2 ? (size_t)atoll(argv[2]) : 200000;
    size_t n_blocks = n_vectors / BLOCK;
    size_t block_bytes = (size_t)N_GROUPS * BLOCK;

    uint8_t *luts = aligned_alloc(64, (size_t)N_GROUPS * 32);
    // Bounded so the u8 pre-add of two table entries cannot overflow, which
    // is the same max_lut=127 constraint the shipped LUT builder honours.
    for (size_t i = 0; i < (size_t)N_GROUPS * 32; i++) {
        luts[i] = (uint8_t)(i % 64);
    }

    // Variant 1 reads one resident block forever: identical instruction
    // stream, zero DRAM traffic. The difference against variant 0 is the
    // memory term and nothing else.
    size_t alloc_blocks = variant == 1 ? 1 : n_blocks;
    uint8_t *codes = aligned_alloc(64, alloc_blocks * block_bytes);
    for (size_t i = 0; i < alloc_blocks * block_bytes; i++) {
        codes[i] = (uint8_t)(i * 31u);
    }

    double hz = clock_hz();
    float sink = 0;
// Each arm passes a literal so the ablation branches fold away entirely.
#define RUN_ALL(V)                                                            \
    for (size_t b = 0; b < n_blocks; b++) {                                   \
        sink += scan_block(codes + ((V) == 1 ? 0 : b * block_bytes), luts, V); \
    }
#define DISPATCH                                                              \
    switch (variant) {                                                        \
    case 0:                                                                   \
        RUN_ALL(0) break;                                                     \
    case 1:                                                                   \
        RUN_ALL(1) break;                                                     \
    case 2:                                                                   \
        RUN_ALL(2) break;                                                     \
    default:                                                                  \
        RUN_ALL(3) break;                                                     \
    }

    DISPATCH  // untimed: page faults and first touch are not measured

    int reps = variant == 1 ? 20 : 5;
    double best = 1e30, worst = 0.0;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        DISPATCH
        double dt = now() - t0;
        if (dt < best) {
            best = dt;
        }
        if (dt > worst) {
            worst = dt;
        }
    }

    double cycles = best * hz;
    double per_iter = cycles / (double)n_blocks / (N_GROUPS / 4.0);
    // The spread, not just the minimum. P29: this probe read 15.10 through
    // 18.59 cy/iter for unchanged code at N=200,000 across five invocations,
    // and three log entries were written from single runs inside that band.
    // A reader must see the width before quoting the centre.
    double spread = (worst - best) / best * 100.0;
    printf("variant %d  %8.3f ms   %6.2f cy/4-group-iter   %5.2f cy/group"
           "   %6.2f GB/s   spread %4.1f%%%s\n",
           variant, best * 1e3, per_iter, per_iter / 4.0,
           (double)n_blocks * block_bytes / best / 1e9, spread,
           spread > 2.0 ? "  <-- WIDE" : "");
    // `sink` must stay observable or -O3 deletes the entire scan: dropping it
    // reported 368 GB/s, 15x the memory roofline, and the loop was gone.
    printf("  (sink %.0f)\n", (double)sink);
    return 0;
}
