//! Cross-platform encode fingerprint.
//!
//! Prints one line per (dim, bit_width) cell, each a hash of a stage of
//! the encode pipeline for a fixed, deterministic input. Two platforms
//! that agree on every line encode identical bytes for identical
//! vectors; a platform that differs says *which stage* diverged.
//!
//! ```text
//! cargo run --release --example encode_hash
//! ```
//!
//! This is the observable half of the v5/v6 determinism claim (#259).
//! The rotation is deterministic by construction (integer permutations
//! plus f32 add/sub/scale, no FMA, golden-pinned) and the per-vector
//! norm now has a frozen reduction order, but two inputs to the encode
//! can still only be *checked* across platforms, never proved on one
//! machine:
//!
//! 1. the Lloyd-Max codebook, computed at runtime from `statrs` Beta
//!    cdf/pdf — transcendentals, so cross-libm variance is possible;
//! 2. everything downstream of it.
//!
//! Hence the split: `boundaries`, `centroids`, `calibration`, `codes`,
//! `scales`, and `file` are hashed separately, so a divergence localizes
//! instead of just saying "the bytes differ". That split is what caught
//! the codebook boundaries diverging on all three platforms while the
//! centroids agreed — see `codebook::lloyd_max`.
//!
//! The hash is FNV-1a 64, not SHA-256: this compares outputs of the same
//! code across platforms, so it needs collision resistance against
//! accident, not against an adversary — and a cryptographic hash would
//! mean a new dependency for a diagnostic.
//!
//! **The fixture and the hashing live in `tests/common/fingerprint.rs`**,
//! shared with `tests/encode_fingerprint.rs`, which pins these same rows
//! to in-tree constants. Cross-OS agreement (this example, under CI) and
//! absolute anchoring (that test) catch different failures, and sharing
//! one definition is what guarantees the two are talking about the same
//! numbers.

#[path = "../tests/common/fingerprint.rs"]
mod fingerprint;

fn main() {
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "other"
    };
    eprintln!("# arch={arch} os={} n={}", std::env::consts::OS, fingerprint::N);

    for &(dim, bits) in fingerprint::CELLS {
        println!("{}", fingerprint::fingerprint(dim, bits).line(dim, bits));
    }
}
