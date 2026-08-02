"""A one-shot bulk ``add`` must not pin its GIL-safety snapshot (#333).

Both bindings snapshot the caller's numpy buffer before releasing the GIL
and keep that buffer on the index so repeated adds reuse one warm
allocation. The shrink that hands a one-off bulk batch back was dead, so
``index.add(embeddings)`` retained a full float32 copy of the batch —
roughly 2x the input array, since the core's encode scratch held the
other copy — for the index's lifetime.

The Rust-side tests in ``snap_retention_tests`` cover the retention
*policy* across batch-size shapes, but they drive the helper directly, so
they cannot see whether ``add`` still calls it. These tests go through the
real ``add`` / ``add_with_ids`` and read the buffer back through
``_snap_capacity``, the internal accessor that exists for exactly this
(the buffer has no other Python-visible handle, and macOS RSS does not
move when it is freed, so an allocator- or RSS-level assertion would not
be a gate on every platform). Deleting either ``retain_snap`` call site
fails a test here.

Note which call reaches the binding with the whole array. The Python layer
slices an add into ``BATCH_CHUNK_SIZE`` rows for Ctrl-C latency, but only
once the TQ+ calibration has settled — the calibrating add must run whole
so slicing cannot change how a vector encodes. So the *first* bulk add on
a fresh index is delegated entire, which is both the shape the issue
reported and the only one that can pin a batch-sized buffer. The tests
assert that precondition rather than assuming it.

Sizes are chosen to be unambiguous rather than large: the buffer either
holds the whole batch or ~none of it, and the thresholds sit a factor of
four away from both outcomes, so nothing here depends on allocator
behaviour or on the exact retention target.
"""

import numpy as np
import pytest

import turbovec

# 40k x 128 float32 = 20 MiB. Comfortably above the binding's parallel-copy
# threshold and big enough that a retained copy is unmistakable, while
# staying cheap to quantize on a CI runner.
N_VECTORS = 40_000
DIM = 128
N_FLOATS = N_VECTORS * DIM


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(20250730)
    return rng.standard_normal((N_VECTORS, DIM), dtype=np.float32)


def test_one_shot_bulk_add_releases_the_snapshot(vectors):
    index = turbovec.TurboQuantIndex(dim=DIM, bit_width=4)
    assert index.calibration_state == "uncalibrated"
    index.add(vectors)
    assert len(index) == N_VECTORS
    retained = index._snap_capacity()
    assert retained < N_FLOATS // 4, (
        f"a one-shot bulk add retained {retained} float32 snapshot elements "
        f"of a {N_FLOATS}-element input ({retained * 4 / 2**20:.1f} MiB)"
    )


def test_one_shot_bulk_add_with_ids_releases_the_snapshot(vectors):
    index = turbovec.IdMapIndex(dim=DIM, bit_width=4)
    assert index.calibration_state == "uncalibrated"
    ids = np.arange(N_VECTORS, dtype=np.uint64)
    index.add_with_ids(vectors, ids)
    assert len(index) == N_VECTORS
    retained = index._snap_capacity()
    assert retained < N_FLOATS // 4, (
        f"a one-shot bulk add_with_ids retained {retained} float32 snapshot "
        f"elements of a {N_FLOATS}-element input "
        f"({retained * 4 / 2**20:.1f} MiB)"
    )


def test_repeated_same_size_adds_keep_the_snapshot_warm(vectors):
    """The release must not cost an allocation per add.

    Guards the other direction: a policy that released unconditionally
    would pass the two tests above while throwing away and re-faulting the
    whole buffer on every call. ``chunk_size=0`` keeps the whole array
    going to the binding on every add, so each call really does demand a
    batch-sized snapshot.
    """
    index = turbovec.TurboQuantIndex(dim=DIM, bit_width=4)
    for _ in range(3):
        index.add(vectors, chunk_size=0)
    retained = index._snap_capacity()
    assert retained >= N_FLOATS, (
        f"steady same-size adds dropped the warm snapshot to {retained} "
        f"elements, below the {N_FLOATS} each add needs"
    )


def test_repeated_same_size_adds_with_ids_keep_the_snapshot_warm(vectors):
    index = turbovec.IdMapIndex(dim=DIM, bit_width=4)
    for batch in range(3):
        ids = np.arange(
            batch * N_VECTORS, (batch + 1) * N_VECTORS, dtype=np.uint64
        )
        index.add_with_ids(vectors, ids, chunk_size=0)
    retained = index._snap_capacity()
    assert retained >= N_FLOATS, (
        f"steady same-size add_with_ids calls dropped the warm snapshot to "
        f"{retained} elements, below the {N_FLOATS} each add needs"
    )
