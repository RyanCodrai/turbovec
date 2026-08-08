//! Deterministic reproducer: a v7 commit generation is reused after a
//! crash rolls the file back, and the abandoned commit that still
//! occupies that slot can resurrect if the second attempt's header write
//! is the thing lost.
//!
//! Sequence:
//!   gen1 commits, leaving a pending redo op.
//!   gen2 attempt A materializes that op and adds a row. Crash: A's
//!     header lands, A's unit write does not, so A's delta fails and
//!     load correctly falls back to gen1. A is abandoned.
//!   The recovered index is driven forward and syncs again — as gen2,
//!     the same number, into the same slot, materializing the same op to
//!     the same bytes. Crash: this time the data lands and the header
//!     does not.
//!   Slot g%2 must not still hold attempt A's header. If it does, A's
//!     delta verifies against attempt B's identical unit bytes and load
//!     adopts A — a state that was rolled back and never observed again.
//!
//! The writer answers this by opening such a sync with a barrier that
//! destroys the rejected header before any data moves. This test models
//! that barrier as durable (it is fsynced before the data batch starts)
//! and checks the end-to-end outcome; the in-crate crash harness —
//! `a_reused_generation_cannot_resurrect_the_commit_it_overwrites` — is
//! what pins that the writer actually emits it, by tearing the real
//! barrier-separated plan at every byte.

use std::path::PathBuf;

use turbovec::TurboQuantIndex;

const DIM: usize = 64;
const BW: usize = 4;
const SECTOR: usize = 512;

fn rows(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * DIM];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    for row in v.chunks_mut(DIM) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn temp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-genreuse-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p.push("index.tv");
    p
}

/// Superblock length and header-slot stride, from the format's own
/// geometry (kind 0, calibrated so n_calib == dim).
fn geometry() -> (usize, usize) {
    const MAX_OPS: usize = 1024;
    let nl = 1usize << BW;
    let sb = 23 + (nl - 1) * 4 + nl * 4 + 4 + DIM * 8 + 4;
    let row = DIM / (8 / BW);
    let hdr = 16 + 31 * (row + 4) + 4 + MAX_OPS * (5 + 1 + row + 4) + 4 + MAX_OPS * 4 + 12 + 4;
    (sb, hdr)
}

/// First byte of the block-unit region.
fn unit0() -> usize {
    let (sb, hdr) = geometry();
    sb + 2 * hdr
}

/// Copy only those changed sectors of `after` that fall inside
/// `[lo, hi)` on top of `before` — a power cut in which the rest of the
/// batch never reached the platter.
fn land_only(before: &[u8], after: &[u8], lo: usize, hi: usize) -> Vec<u8> {
    let mut out = before.to_vec();
    let n = before.len().max(after.len()).div_ceil(SECTOR);
    for s in 0..n {
        let (a, b) = (s * SECTOR, ((s + 1) * SECTOR).min(after.len()));
        if a >= after.len() || a >= hi || b <= lo {
            continue;
        }
        if out.len() < b {
            out.resize(b, 0);
        }
        if out[a..b] != after[a..b] {
            out[a..b].copy_from_slice(&after[a..b]);
        }
    }
    out
}

#[test]
fn an_abandoned_commit_cannot_resurrect_after_its_generation_is_reused() {
    let path = temp("resurrect");
    let mut idx = TurboQuantIndex::new(DIM, BW).unwrap();
    idx.calibrate(&rows(1024, 1)).unwrap();
    idx.add(&rows(200, 2));
    idx.sync(&path).unwrap(); // gen 0, full write

    // gen 1: a removal inside a committed block rides the header as a
    // pending redo op. No unit is written, so gen 1's delta is empty and
    // it always verifies — it is the fallback for everything below.
    idx.swap_remove(10);
    idx.sync(&path).unwrap();
    let f1 = std::fs::read(&path).unwrap();
    let s1 = TurboQuantIndex::load(&path).unwrap().to_bytes();

    // gen 2, attempt A: materialize the pending op into unit 0, plus one
    // new tail row.
    let mut a = TurboQuantIndex::load(&path).unwrap();
    a.add(&rows(1, 3));
    a.sync(&path).unwrap();
    let f2a = std::fs::read(&path).unwrap();
    let s2a = TurboQuantIndex::load(&path).unwrap().to_bytes();
    assert_ne!(s2a, s1);

    // Crash A: A's header sectors land, A's unit-0 write does not.
    let crashed_a = land_only(&f1, &f2a, 0, unit0());
    std::fs::write(&path, &crashed_a).unwrap();
    let recovered = TurboQuantIndex::load(&path).expect("crash A must load");
    assert_eq!(
        recovered.to_bytes(),
        s1,
        "crash A should roll back to gen 1 — attempt A's delta names a unit that never landed"
    );

    // The application carries on from the recovered state and syncs
    // again. This sync is generation 2 for the second time.
    let before_b = std::fs::read(&path).unwrap();
    let mut b = recovered;
    b.add(&rows(3, 4));
    b.sync(&path).unwrap();
    let f2b = std::fs::read(&path).unwrap();
    let s2b = b.to_bytes();
    assert_eq!(TurboQuantIndex::load(&path).unwrap().to_bytes(), s2b);

    // Crash B: B's unit writes land, B's header write does not — but
    // B's repair barrier ran and was fsynced before either, so slot 0
    // no longer holds attempt A's header.
    let (sb, hdr) = geometry();
    let mut staged = before_b.clone();
    staged[sb..sb + hdr].fill(0);
    let crashed_b = land_only(&staged, &f2b, unit0(), usize::MAX);
    std::fs::write(&path, &crashed_b).unwrap();
    let got = TurboQuantIndex::load(&path).expect("crash B must load").to_bytes();

    assert!(
        got == s1 || got == s2b,
        "crash B resurrected an abandoned commit: recovered a state that is neither \
         the previous commit ({} rows) nor the new one ({} rows)",
        TurboQuantIndex::load(&path).unwrap().len(),
        b.len()
    );
}
