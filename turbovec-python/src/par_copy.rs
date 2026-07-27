//! Chunked parallel buffer copy — an audited rayon chokepoint file
//! (fork safety, issue #147). The only caller runs it inside
//! `with_pool`, so the parallel iterators below always execute on the
//! process-local, fork-safe pool.

use rayon::prelude::*;

/// Below this many floats a plain serial `to_vec` wins: the pool
/// `install` handoff costs more than the memcpy it parallelizes.
pub(crate) const PAR_COPY_MIN_LEN: usize = 1 << 20;

/// Equivalent to `slice.to_vec()`, with the memcpy split across the
/// current rayon pool. Callers must run it under `with_pool` — the
/// global pool is pinned to a single sentinel thread, so a bare call
/// would fold back to a serial copy.
pub(crate) fn par_copy(slice: &[f32]) -> Vec<f32> {
    const CHUNK: usize = 1 << 20;
    let mut owned: Vec<f32> = Vec::with_capacity(slice.len());
    let spare = &mut owned.spare_capacity_mut()[..slice.len()];
    spare
        .par_chunks_mut(CHUNK)
        .zip(slice.par_chunks(CHUNK))
        .for_each(|(dst, src)| {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                d.write(s);
            }
        });
    // SAFETY: every element of the spare capacity up to slice.len() was
    // just written by the parallel copy above.
    unsafe {
        owned.set_len(slice.len());
    }
    owned
}
