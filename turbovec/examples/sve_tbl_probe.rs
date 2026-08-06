//! Does SVE `TBL` really issue at 4/cycle on Neoverse V2 where NEON `TBL`
//! manages 2?
//!
//! The V2 optimisation guide says yes: ASIMD `TBL` with 1-2 table registers
//! is throughput 2 on pipes V01, while SVE `TBL` is throughput 4 on V — all
//! four. If true it is the only remaining route to more lookup bandwidth on
//! this core, because H25 established that `TBX`, which the same tables also
//! price at 4/cycle, cannot be used for a pure lookup: it reads its
//! destination, so every lookup pays a register initialisation.
//!
//! SVE `TBL` has no such problem — it is destination-only, like NEON `TBL`.
//! So it is a clean test of the pipe claim rather than a retest of H25.
//!
//! That claim is worth measuring rather than believing. A literature sweep
//! found it is **documentation-only**: present in the V2 and Cortex-X3
//! optimisation guides, transcribed into LLVM's V2 scheduling model by its
//! Arm author (validated only against SPEC, not per-instruction), and never
//! independently measured on V2, Cortex-X3, Graviton 4, Grace, Cortex-X4 or
//! V3. The restriction was measured to be real one generation earlier
//! (insn_bench_aarch64 on Graviton 3 / Neoverse V1), but the V2-specific
//! *asymmetry* — NEON TBL on V01 while SVE TBL escapes to all four pipes —
//! is new in V2/X3, gone again in V3, and unverified everywhere.
//!
//! This measures issue rate in isolation: eight mutually independent lookups
//! per iteration, sharing one table and one index register, so nothing is
//! serialised by a data dependency and the only limit is issue capacity.
//! Neoverse V2 implements SVE at VL=128, so a `z` register is the same 16
//! bytes as a `v` register and the comparison is like-for-like.
//!
//! Run: cargo run --release --example sve_tbl_probe

fn main() {
    #[cfg(not(target_arch = "aarch64"))]
    println!("aarch64 only");

    #[cfg(target_arch = "aarch64")]
    unsafe {
        run();
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn run() {
    use std::time::Instant;

    // Enough iterations that the loop overhead (one subs + one branch per 8
    // lookups) is small and the timer is comfortable.
    const ITERS: u64 = 200_000_000;
    const PER_ITER: u64 = 8;

    // Warm the clocks so neither measurement pays a frequency ramp.
    neon_tbl(ITERS / 10);
    sve_tbl(ITERS / 10);

    let t0 = Instant::now();
    neon_tbl(ITERS);
    let dt_neon = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    sve_tbl(ITERS);
    let dt_sve = t1.elapsed().as_secs_f64();

    let ops = (ITERS * PER_ITER) as f64;
    let neon = ops / dt_neon / 1e9;
    let sve = ops / dt_sve / 1e9;

    println!("independent lookups, no data dependencies, VL=128");
    println!();
    println!("NEON tbl v.16b, {{v.16b}}, v.16b   {neon:.2} G/s   ({dt_neon:.3} s)");
    println!("SVE  tbl z.b,   {{z.b}},   z.b     {sve:.2} G/s   ({dt_sve:.3} s)");
    println!();
    println!("SVE / NEON   x{:.3}", sve / neon);
    println!();
    println!("The guide predicts x2.00. Divide each rate by the core clock");
    println!("to read it as lookups/cycle: the guide says NEON should reach");
    println!("2 and SVE 4.");
}

/// Eight independent NEON single-table lookups per iteration.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn neon_tbl(iters: u64) {
    use std::arch::asm;
    asm!(
        // Table and indices: all-zero indices are in range, so every lookup
        // does real work and none is short-circuited.
        "movi v16.16b, #0",
        "movi v17.16b, #0",
        "2:",
        "tbl v0.16b, {{v16.16b}}, v17.16b",
        "tbl v1.16b, {{v16.16b}}, v17.16b",
        "tbl v2.16b, {{v16.16b}}, v17.16b",
        "tbl v3.16b, {{v16.16b}}, v17.16b",
        "tbl v4.16b, {{v16.16b}}, v17.16b",
        "tbl v5.16b, {{v16.16b}}, v17.16b",
        "tbl v6.16b, {{v16.16b}}, v17.16b",
        "tbl v7.16b, {{v16.16b}}, v17.16b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        n = inout(reg) iters => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v16") _, out("v17") _,
        options(nostack),
    );
}

/// The same eight lookups as SVE `TBL`. `z0-z31` alias `v0-v31`, so the
/// register clobbers are declared on the `v` names.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn sve_tbl(iters: u64) {
    use std::arch::asm;
    asm!(
        ".arch_extension sve",
        "movi v16.16b, #0",
        "movi v17.16b, #0",
        "2:",
        "tbl z0.b, {{z16.b}}, z17.b",
        "tbl z1.b, {{z16.b}}, z17.b",
        "tbl z2.b, {{z16.b}}, z17.b",
        "tbl z3.b, {{z16.b}}, z17.b",
        "tbl z4.b, {{z16.b}}, z17.b",
        "tbl z5.b, {{z16.b}}, z17.b",
        "tbl z6.b, {{z16.b}}, z17.b",
        "tbl z7.b, {{z16.b}}, z17.b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        n = inout(reg) iters => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v16") _, out("v17") _,
        options(nostack),
    );
}
