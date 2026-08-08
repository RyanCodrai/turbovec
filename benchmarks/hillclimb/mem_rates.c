// Sustained sequential-read bandwidth, in GB/s and bytes/cycle, at the
// working-set sizes the 2-bit scan actually touches.
//
// Companion to `isa_rates.c`. That file gives the *issue* ceiling of a loop;
// this one gives the *supply* ceiling. A kernel is only worth optimizing at
// the instruction level while it sits below the first and above the second —
// and until both numbers exist, "the rest is memory" is a guess. Four
// hypotheses in this climb narrowed an ARM residue by elimination without
// either number in hand.
//
// The size sweep matters more than any single figure: it is what separates
// "this cell streams from DRAM" from "this cell fits in L2 and the DRAM
// number is irrelevant to it". The 2-bit cells at N=200k, dim=768 read
// 200000 * 192 = 38.4 MB of codes per query pass, which is the size marked
// `<- 2bit cells` in the output.
//
// Method: 8 mutually independent accumulators over a sequential u64 walk, so
// the loop is limited by supply and not by a dependency chain. Each size is
// re-read enough times to run for a stable interval, and the buffer is
// pre-faulted so page faults are not counted. The clock is derived in-run
// from a dependent `add` chain (1/cycle on every core this targets), so
// bytes/cycle needs no external knowledge of the frequency.
//
// Build:  cc -O3 -pthread -o mem_rates mem_rates.c
// Run:    ./mem_rates [n_threads]
//
// Read it as: a cell whose achieved bandwidth is near the row for its own
// working-set size has no headroom left in the kernel, whatever the issue
// analysis says.

#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Derived in-run rather than read from the OS: a box that clocks down under
// an 8-thread load then reports bytes/cycle honestly instead of flattering.
//
// The method differs by arch because the obvious one does not survive both.
// On AArch64 a dependent `add` chain retires at exactly 1/cycle and the
// derived figure lands on 2.988 GHz for a 2.987 GHz Axion. The identical
// chain on Sapphire Rapids retires **4.4 adds per TSC tick** — verified with
// the final accumulator, so all 160M adds really executed — which no core
// running a serial chain can do. Whatever collapses it, the probe is not
// measuring cycles there, so x86 uses the invariant TSC instead.
//
// The TSC caveat, stated because bytes/cycle is read against it: TSC counts
// at the marked frequency (2.701 GHz here), not the actual core clock, so
// under turbo the real per-cycle figure is *lower* than printed. It is a
// conservative bound on x86 and an exact one on AArch64.
static double measure_clock(const char **how) {
#if defined(__aarch64__)
    uint64_t acc = 0;
    const uint64_t n = 20000000;
    double t0 = now();
    for (uint64_t i = 0; i < n; i++) {
        __asm__ volatile("add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         "add %0, %0, #1\n add %0, %0, #1\n"
                         : "+r"(acc)::);
    }
    *how = "dependent add chain, exact";
    return (double)n * 8.0 / (now() - t0);
#elif defined(__x86_64__)
    uint64_t c0 = __rdtsc();
    double t0 = now(), spun = 0.0;
    while (now() - t0 < 0.05) {
        spun += 1.0;  // busy interval long enough for a stable tick rate
    }
    double dt = now() - t0;
    uint64_t c1 = __rdtsc();
    if (spun < 0.0) {
        fprintf(stderr, "unreachable\n");
    }
    *how = "invariant TSC, conservative under turbo";
    return (double)(c1 - c0) / dt;
#else
#error "no clock reference for this architecture"
#endif
}

struct job {
    const uint64_t *buf;
    size_t words;
    int passes;
    int file_backed;
    uint64_t sink;
};

// Eight independent accumulators. One accumulator would measure the load-use
// latency chain and report a fraction of the true bandwidth.
static void *walk(void *arg) {
    struct job *j = (struct job *)arg;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (int p = 0; p < j->passes; p++) {
        const uint64_t *b = j->buf;
        for (size_t i = 0; i + 8 <= j->words; i += 8) {
            a0 += b[i + 0];
            a1 += b[i + 1];
            a2 += b[i + 2];
            a3 += b[i + 3];
            a4 += b[i + 4];
            a5 += b[i + 5];
            a6 += b[i + 6];
            a7 += b[i + 7];
        }
    }
    j->sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    return NULL;
}

// Each thread walks its own private buffer. Sharing one buffer across threads
// would measure how well the caches replicate a read-only line, which is a
// different question from how fast N cores can be supplied at once — and the
// scan gives every thread its own slice of the code array.
// `file` selects where the pages come from, and it is the whole reason this
// probe has a second mode. An anonymous allocation of tens of MB is eligible
// for transparent huge pages; a file-backed mapping generally is not. The
// index this climb measures is an mmap'd file, so an anon-memory ceiling can
// flatter it by the entire cost of the extra TLB walks — which is exactly the
// kind of gap that gets misread as "the kernel is ALU-bound".
static double bandwidth(size_t bytes, int threads, double target_secs,
                        int file_backed) {
    size_t words = bytes / 8;
    struct job *jobs = calloc(threads, sizeof(*jobs));
    pthread_t *tids = calloc(threads, sizeof(*tids));
    for (int t = 0; t < threads; t++) {
        uint64_t *buf = NULL;
        if (file_backed) {
            char path[64];
            snprintf(path, sizeof(path), "./.mem_rates_%d.bin", t);
            int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (fd < 0 || ftruncate(fd, (off_t)(words * 8)) != 0) {
                return 0.0;
            }
            buf = mmap(NULL, words * 8, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            unlink(path);  // stays alive through the mapping
            if (buf == MAP_FAILED) {
                return 0.0;
            }
        } else if (posix_memalign((void **)&buf, 4096, words * 8) != 0) {
            return 0.0;
        }
        for (size_t i = 0; i < words; i++) {
            buf[i] = i;  // pre-fault, so page faults are not timed
        }
        jobs[t].buf = buf;
        jobs[t].words = words;
        jobs[t].passes = 1;
        jobs[t].file_backed = file_backed;
    }

    // Calibrate the pass count on one thread so every size runs for roughly
    // the same interval: too few passes and the timer dominates, too many and
    // a 30-row sweep takes an hour.
    double t0 = now();
    walk(&jobs[0]);
    double one = now() - t0;
    int passes = (int)(target_secs / (one > 0 ? one : 1e-6));
    if (passes < 1) {
        passes = 1;
    }
    if (passes > 100000) {
        passes = 100000;
    }
    for (int t = 0; t < threads; t++) {
        jobs[t].passes = passes;
    }

    t0 = now();
    for (int t = 1; t < threads; t++) {
        pthread_create(&tids[t], NULL, walk, &jobs[t]);
    }
    walk(&jobs[0]);
    for (int t = 1; t < threads; t++) {
        pthread_join(tids[t], NULL);
    }
    double dt = now() - t0;

    uint64_t sink = 0;
    for (int t = 0; t < threads; t++) {
        sink += jobs[t].sink;
        if (file_backed) {
            munmap((void *)jobs[t].buf, words * 8);
        } else {
            free((void *)jobs[t].buf);
        }
    }
    double total = (double)bytes * (double)passes * (double)threads;
    free(jobs);
    free(tids);
    // Keep `sink` observable so the walk cannot be elided.
    if (sink == 0xdeadbeefULL) {
        fprintf(stderr, "unreachable\n");
    }
    return total / dt;
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 1;
    if (threads < 1) {
        threads = 1;
    }
    const char *how = "";
    double clock_hz = measure_clock(&how);

    // 38.4 MB is 200000 vectors * 192 code bytes: one 2-bit query pass at
    // the objective's N and dim. The rows around it exist to show which
    // cache level, if any, that size still lives in.
    const size_t sizes[] = {
        32u << 10, 512u << 10, 2u << 20,  8u << 20,  32u << 20,
        38400000u, 76800000u,  128u << 20,
    };
    const int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("clock: %.3f GHz (%s), threads: %d\n", clock_hz / 1e9, how, threads);
    FILE *thp = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if (thp) {
        char buf[128] = {0};
        if (fgets(buf, sizeof(buf), thp)) {
            printf("transparent_hugepage: %s", buf);
        }
        fclose(thp);
    }
    putchar('\n');

    printf("%14s %10s %10s %8s %12s %s\n", "working set", "anon GB/s",
           "file GB/s", "file/anon", "anon B/cyc", "");
    for (int i = 0; i < n_sizes; i++) {
        double anon = bandwidth(sizes[i], threads, 0.30, 0);
        double file = bandwidth(sizes[i], threads, 0.30, 1);
        // Per-core bytes/cycle: the figure a single-thread kernel roofline
        // needs. At >1 thread this is the aggregate divided by the count.
        double bpc = anon / clock_hz / threads;
        const char *tag = sizes[i] == 38400000u   ? "  <- 2bit cells"
                          : sizes[i] == 76800000u ? "  <- 4bit cells"
                                                  : "";
        char label[32];
        if (sizes[i] >= (1u << 20)) {
            snprintf(label, sizeof(label), "%.1f MB", sizes[i] / 1048576.0);
        } else {
            snprintf(label, sizeof(label), "%zu KB", sizes[i] >> 10);
        }
        printf("%14s %10.2f %10.2f %8.2f %12.2f %s\n", label, anon / 1e9,
               file / 1e9, file > 0 ? file / anon : 0.0, bpc, tag);
    }
    return 0;
}
