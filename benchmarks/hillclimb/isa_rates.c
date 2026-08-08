// Sustained issue rate, in instructions per cycle, for the NEON instructions
// the turbovec scan kernels actually contain.
//
// This exists because a stale vendor number grounded a hypothesis. Arm's
// optimization guide (and the LLVM scheduling model copied from it) prices
// NEON `TBL` at 2/cycle on Neoverse V1/V2; on Axion it issues at 4. H8 died
// on the documented figure, and every port-asymmetry idea built on that table
// row died with it. A model is only as good as its slowest-moving row, so the
// rows the kernels depend on are measured here rather than read.
//
// Method: each case issues 8 *mutually independent* instructions per unrolled
// step, four steps per loop iteration, so the measurement is throughput and
// not latency. The clock is derived in the same run from a dependent
// `add` chain, which retires at exactly 1/cycle on every core this targets —
// so the reported rates need no external knowledge of the frequency, and a
// box that clocks differently under load reports honestly.
//
// Build:  cc -O2 -march=armv8.4-a+dotprod+i8mm -o isa_rates isa_rates.c
// Run:    ./isa_rates [iterations]
//
// Read it as: a rate at or above the vector pipe count means the instruction
// is unrestricted; below it means the instruction is confined to a subset of
// the pipes, and any roofline over a loop containing it must say which.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 8 independent instances of BODY, four times, per loop iteration, timed
// three times over.
//
// Three, and both extremes reported, because single numbers from this file
// have twice been wrong in a way one number cannot show. `tbx` and `smmla`
// flip cleanly between 2.00 and 4.01 per cycle between consecutive runs of
// the same binary — exactly 2x, so not frequency drift — while every row the
// 2-bit scan actually contains (`tbl`, `and`, `ushr`, `uaddw`, `add`,
// `ucvtf`) repeats to the last digit. A reader must be able to tell those two
// situations apart, so a row whose fast and slow columns disagree is not a
// rate, it is an open question, and nothing should be grounded on it.
#define TIME_BLOCK(NAME, N, PER_ITER, SETUP, BODY)                            \
    do {                                                                      \
        SETUP;                                                                \
        double ops = (double)(N) * 4.0 * (PER_ITER);                          \
        double lo = 1e30, hi = 0.0;                                           \
        for (int rep_ = 0; rep_ < 3; rep_++) {                                \
            double t0 = now();                                                \
            for (uint64_t i = 0; i < (N); i++) {                              \
                BODY;                                                         \
                BODY;                                                         \
                BODY;                                                         \
                BODY;                                                         \
            }                                                                 \
            double dt = now() - t0;                                           \
            if (dt < lo) {                                                    \
                lo = dt;                                                      \
            }                                                                 \
            if (dt > hi) {                                                    \
                hi = dt;                                                      \
            }                                                                 \
        }                                                                     \
        results[n_results].name = NAME;                                       \
        results[n_results].ops = ops;                                         \
        results[n_results].secs = lo;                                         \
        results[n_results].slow = hi;                                         \
        n_results++;                                                          \
    } while (0)

struct row {
    const char *name;
    double ops;
    double secs;  // fastest of three
    double slow;  // slowest of three
};

// Eight instances with *different* destinations, v8-v15. Giving them one
// shared destination looks independent but is not for any accumulating form
// (`fmla`, `uadalp`, `sdot`, `smmla`, `tbx` all read their destination), and
// such a block measures latency while appearing to measure throughput — the
// first version of this file did exactly that and reported `fmla` at 0.5/cy.
#define EIGHT(PRE, POST)                                                      \
    PRE "v8" POST "\n" PRE "v9" POST "\n" PRE "v10" POST "\n" PRE "v11" POST \
        "\n" PRE "v12" POST "\n" PRE "v13" POST "\n" PRE "v14" POST "\n" PRE \
        "v15" POST "\n"

// Twenty-four destinations, v8-v31, for the forms that read their own.
//
// Eight was not enough and the tool said so: `sdot` swung 3.89 / 2.43 and
// `smmla` 2.00 / 3.78 between consecutive runs of the same binary, which is
// the signature of a measurement sitting on a dependency boundary rather
// than on a pipe limit. `TIME_BLOCK` repeats its body four times per
// iteration, so eight distinct destinations broke the chain *within* a body
// and rebuilt it *across* the repetitions — four dependent updates per
// register per iteration. This is the same error as the original
// shared-destination version, one level up, and it was invisible until the
// numbers were read twice.
//
// At 24 registers the four repetitions are 96 independent-enough ops against
// a 4-pipe machine: 24 cycles of issue against 4*latency of chain, so the
// pipes bind for any latency up to 6. Sources stay in v0-v7 so nothing here
// collides with them.
#define ACC24(PRE, POST)                                                      \
    PRE "v8" POST "\n" PRE "v9" POST "\n" PRE "v10" POST "\n" PRE "v11" POST  \
        "\n" PRE "v12" POST "\n" PRE "v13" POST "\n" PRE "v14" POST "\n" PRE  \
        "v15" POST "\n" PRE "v16" POST "\n" PRE "v17" POST "\n" PRE "v18"     \
        POST "\n" PRE "v19" POST "\n" PRE "v20" POST "\n" PRE "v21" POST      \
        "\n" PRE "v22" POST "\n" PRE "v23" POST "\n" PRE "v24" POST "\n" PRE  \
        "v25" POST "\n" PRE "v26" POST "\n" PRE "v27" POST "\n" PRE "v28"     \
        POST "\n" PRE "v29" POST "\n" PRE "v30" POST "\n" PRE "v31" POST "\n"

int main(int argc, char **argv) {
    uint64_t n = 2000000;
    if (argc > 1) {
        n = (uint64_t)atoll(argv[1]);
    }

    struct row results[32];
    int n_results = 0;

    // --- clock reference: a dependent add chain, 1 instruction per cycle ---
    TIME_BLOCK("add (dependent chain)", n, 8, uint64_t acc = 0,
               __asm__ volatile("add %0, %0, #1\n add %0, %0, #1\n"
                                "add %0, %0, #1\n add %0, %0, #1\n"
                                "add %0, %0, #1\n add %0, %0, #1\n"
                                "add %0, %0, #1\n add %0, %0, #1\n"
                                : "+r"(acc)::));
    double clock_hz = results[0].ops / results[0].secs;

    // Vector operands, confined to v0-v7 so that v8-v31 are all available as
    // destinations. Kept live across every block so nothing is folded.
    __asm__ volatile("movi v0.16b, #1\n movi v1.16b, #2\n movi v2.16b, #3\n"
                     "movi v3.16b, #4\n movi v4.16b, #5\n movi v5.16b, #6\n"
                     "movi v6.16b, #7\n movi v7.16b, #8\n"
                     :::"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");

#define VEC_CASE(NAME, PRE, POST)                                             \
    TIME_BLOCK(NAME, n, 8, (void)0,                                           \
               __asm__ volatile(EIGHT(PRE, POST)::                            \
                                : "v8", "v9", "v10", "v11", "v12", "v13",     \
                                  "v14", "v15"))

// Eight instances that all write v8, so each waits on the one before it.
// This is the shape that made the first version of this file wrong, kept
// deliberately and labelled: for any form that reads its own destination the
// reciprocal of this rate is the *latency*, and a loop whose accumulator is
// updated k times per iteration cannot beat k * latency however many pipes
// are free. Throughput alone cannot see that floor.
#define CHAIN8(PRE, POST)                                                     \
    PRE "v8" POST "\n" PRE "v8" POST "\n" PRE "v8" POST "\n" PRE "v8" POST    \
        "\n" PRE "v8" POST "\n" PRE "v8" POST "\n" PRE "v8" POST "\n" PRE     \
        "v8" POST "\n"

#define LAT_CASE(NAME, PRE, POST)                                             \
    TIME_BLOCK(NAME, n, 8, (void)0,                                           \
               __asm__ volatile(CHAIN8(PRE, POST)::: "v8"))

// Throughput for forms that read their own destination. Use this, not
// VEC_CASE, for anything accumulating — see the ACC24 comment.
#define ACC_CASE(NAME, PRE, POST)                                             \
    TIME_BLOCK(NAME, n / 3, 24, (void)0,                                      \
               __asm__ volatile(ACC24(PRE, POST)::                            \
                                : "v8", "v9", "v10", "v11", "v12", "v13",     \
                                  "v14", "v15", "v16", "v17", "v18", "v19",   \
                                  "v20", "v21", "v22", "v23", "v24", "v25",   \
                                  "v26", "v27", "v28", "v29", "v30", "v31"))

    // The four instructions the 2-bit LUT scan is made of.
    VEC_CASE("tbl  (1 table reg)", "tbl ", ".16b, {v0.16b}, v1.16b");
    ACC_CASE("uaddw", "uaddw ", ".8h, v0.8h, v1.8b");
    VEC_CASE("add  (8b)", "add ", ".16b, v0.16b, v1.16b");
    VEC_CASE("and", "and ", ".16b, v0.16b, v1.16b");
    VEC_CASE("ushr", "ushr ", ".16b, v0.16b, #4");

    // Alternatives this climb priced or rejected, kept so a future
    // hypothesis reads a measured rate instead of a published one.
    VEC_CASE("tbl  (2 table regs)", "tbl ", ".16b, {v0.16b, v1.16b}, v2.16b");
    VEC_CASE("tbl  (4 table regs)", "tbl ",
             ".16b, {v0.16b, v1.16b, v2.16b, v3.16b}, v4.16b");
    ACC_CASE("tbx", "tbx ", ".16b, {v0.16b}, v1.16b");
    ACC_CASE("uadalp", "uadalp ", ".8h, v1.16b");
    VEC_CASE("uzp1", "uzp1 ", ".16b, v0.16b, v1.16b");
    VEC_CASE("zip1", "zip1 ", ".16b, v0.16b, v1.16b");

    // The epilogue's instructions, which P20 priced at 5% of the cell.
    VEC_CASE("ucvtf", "ucvtf ", ".4s, v0.4s");
    VEC_CASE("ushll", "ushll ", ".8h, v1.8b, #0");
    ACC_CASE("fmla", "fmla ", ".4s, v0.4s, v1.4s");
    VEC_CASE("fmul", "fmul ", ".4s, v0.4s, v1.4s");

    // Dot-product forms, for the formulation questions P5/P12 closed.
    // These need `-mcpu=native` (or an explicit `+dotprod+i8mm`); the
    // default assembler target rejects them.
    ACC_CASE("sdot", "sdot ", ".4s, v0.16b, v1.16b");
    ACC_CASE("smmla", "smmla ", ".4s, v0.16b, v1.16b");

    int n_throughput = n_results;

    // Latency of the accumulating forms the scan loops chain through. A
    // kernel that updates one accumulator k times per iteration has a floor
    // of k * latency cycles no matter how wide the machine is, and the LUT
    // scan's u16 accumulators are updated twice per byte-group.
    LAT_CASE("uaddw   [latency]", "uaddw ", ".8h, v8.8h, v1.8b");
    LAT_CASE("uaddw2  [latency]", "uaddw2 ", ".8h, v8.8h, v1.16b");
    LAT_CASE("add 8b  [latency]", "add ", ".16b, v8.16b, v1.16b");
    LAT_CASE("uadalp  [latency]", "uadalp ", ".8h, v1.16b");
    LAT_CASE("fmla    [latency]", "fmla ", ".4s, v0.4s, v1.4s");
    LAT_CASE("tbl     [latency]", "tbl ", ".16b, {v0.16b}, v8.16b");

    printf("clock: %.3f GHz (dependent add chain)\n\n", clock_hz / 1e9);
    printf("%-22s %10s %10s %s\n", "instruction", "per cycle", "slowest", "");
    for (int i = 1; i < n_throughput; i++) {
        double fast = results[i].ops / results[i].secs / clock_hz;
        double slow = results[i].ops / results[i].slow / clock_hz;
        // A row that does not repeat is not a rate. Say so on the row rather
        // than in a footnote nobody reads before quoting the number.
        const char *flag = (fast - slow) > 0.05 * fast ? "  <-- UNSTABLE" : "";
        printf("%-22s %10.2f %10.2f%s\n", results[i].name, fast, slow, flag);
    }
    printf("\n%-22s %10s\n", "instruction", "cycles");
    for (int i = n_throughput; i < n_results; i++) {
        double per_cycle = results[i].ops / results[i].secs / clock_hz;
        printf("%-22s %10.2f\n", results[i].name, 1.0 / per_cycle);
    }
    return 0;
}
