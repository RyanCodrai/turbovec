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

// 8 independent instances of BODY, four times, per loop iteration.
#define TIME_BLOCK(NAME, N, PER_ITER, SETUP, BODY)                            \
    do {                                                                      \
        SETUP;                                                                \
        double t0 = now();                                                    \
        for (uint64_t i = 0; i < (N); i++) {                                  \
            BODY;                                                             \
            BODY;                                                             \
            BODY;                                                             \
            BODY;                                                             \
        }                                                                     \
        double dt = now() - t0;                                               \
        double ops = (double)(N) * 4.0 * (PER_ITER);                          \
        results[n_results].name = NAME;                                       \
        results[n_results].ops = ops;                                         \
        results[n_results].secs = dt;                                         \
        n_results++;                                                          \
    } while (0)

struct row {
    const char *name;
    double ops;
    double secs;
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

    // Vector operands. Kept live across every block so nothing is folded.
    __asm__ volatile("movi v0.16b, #1\n movi v1.16b, #2\n movi v2.16b, #3\n"
                     "movi v3.16b, #4\n movi v4.16b, #5\n movi v5.16b, #6\n"
                     "movi v6.16b, #7\n movi v7.16b, #8\n"
                     "movi v16.16b, #1\n movi v17.16b, #2\n movi v18.16b, #3\n"
                     "movi v19.16b, #4\n movi v20.16b, #5\n movi v21.16b, #6\n"
                     "movi v22.16b, #7\n movi v23.16b, #8\n"
                     "movi v24.16b, #9\n movi v25.16b, #10\n"
                     :::"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
                       "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                       "v25");

#define VEC_CASE(NAME, PRE, POST)                                             \
    TIME_BLOCK(NAME, n, 8, (void)0,                                           \
               __asm__ volatile(EIGHT(PRE, POST)::                            \
                                : "v8", "v9", "v10", "v11", "v12", "v13",     \
                                  "v14", "v15"))

    // The four instructions the 2-bit LUT scan is made of.
    VEC_CASE("tbl  (1 table reg)", "tbl ", ".16b, {v0.16b}, v1.16b");
    VEC_CASE("uaddw", "uaddw ", ".8h, v16.8h, v1.8b");
    VEC_CASE("add  (8b)", "add ", ".16b, v0.16b, v1.16b");
    VEC_CASE("and", "and ", ".16b, v0.16b, v1.16b");
    VEC_CASE("ushr", "ushr ", ".16b, v0.16b, #4");

    // Alternatives this climb priced or rejected, kept so a future
    // hypothesis reads a measured rate instead of a published one.
    VEC_CASE("tbl  (2 table regs)", "tbl ", ".16b, {v0.16b, v1.16b}, v2.16b");
    VEC_CASE("tbl  (4 table regs)", "tbl ",
             ".16b, {v0.16b, v1.16b, v2.16b, v3.16b}, v4.16b");
    VEC_CASE("tbx", "tbx ", ".16b, {v0.16b}, v1.16b");
    VEC_CASE("uadalp", "uadalp ", ".8h, v1.16b");
    VEC_CASE("uzp1", "uzp1 ", ".16b, v0.16b, v1.16b");
    VEC_CASE("zip1", "zip1 ", ".16b, v0.16b, v1.16b");

    // The epilogue's instructions, which P20 priced at 5% of the cell.
    VEC_CASE("ucvtf", "ucvtf ", ".4s, v0.4s");
    VEC_CASE("ushll", "ushll ", ".8h, v1.8b, #0");
    VEC_CASE("fmla", "fmla ", ".4s, v0.4s, v1.4s");
    VEC_CASE("fmul", "fmul ", ".4s, v0.4s, v1.4s");

    // Dot-product forms, for the formulation questions P5/P12 closed.
    // These need `-mcpu=native` (or an explicit `+dotprod+i8mm`); the
    // default assembler target rejects them.
    VEC_CASE("sdot", "sdot ", ".4s, v0.16b, v1.16b");
    VEC_CASE("smmla", "smmla ", ".4s, v0.16b, v1.16b");

    printf("clock: %.3f GHz (dependent add chain)\n\n", clock_hz / 1e9);
    printf("%-22s %10s %12s\n", "instruction", "per cycle", "ns per 1e9");
    for (int i = 1; i < n_results; i++) {
        double per_cycle = results[i].ops / results[i].secs / clock_hz;
        printf("%-22s %10.2f %12.2f\n", results[i].name, per_cycle,
               results[i].secs / results[i].ops * 1e9);
    }
    return 0;
}
