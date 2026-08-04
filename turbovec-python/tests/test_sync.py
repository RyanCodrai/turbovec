"""sync(): incremental persistence on the v7 container."""

import os

import numpy as np
import pytest
from turbovec import IdMapIndex, TurboQuantIndex

DIM = 64


def _rows(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_sync_round_trips_and_appends(tmp_path):
    path = str(tmp_path / "index.tv")
    idx = TurboQuantIndex(DIM, 4)
    idx.calibrate(_rows(1024, 1))
    idx.add(_rows(64, 2))
    idx.sync(path)
    full = os.path.getsize(path)

    loaded = TurboQuantIndex.load(path)
    assert loaded.to_bytes() == idx.to_bytes()

    idx.add(_rows(2, 3))
    idx.swap_remove(5)
    idx.sync(path)
    grown = os.path.getsize(path)
    assert grown - full < full / 2, "sync rewrote instead of appending"

    q = _rows(4, 4)
    a = TurboQuantIndex.load(path).search(q, 5)
    b = idx.search(q, 5)
    np.testing.assert_array_equal(a[1], b[1])


def test_idmap_sync_round_trips_with_ids(tmp_path):
    path = str(tmp_path / "index.tvim")
    idx = IdMapIndex(DIM, 4)
    idx.calibrate(_rows(1024, 5))
    ids = np.arange(40, dtype=np.uint64) * 3 + 1
    idx.add_with_ids(_rows(40, 6), ids)
    idx.sync(path)

    assert idx.remove(4)
    idx.add_with_ids(_rows(1, 7), np.array([999], dtype=np.uint64))
    idx.sync(path, durable=False)

    loaded = IdMapIndex.load(path)
    assert loaded.to_bytes() == idx.to_bytes()
    assert loaded.contains(999) and not loaded.contains(4)
    q = _rows(4, 8)
    np.testing.assert_array_equal(loaded.search(q, 5)[1], idx.search(q, 5)[1])

    # A loaded index keeps syncing forward.
    loaded.add_with_ids(_rows(1, 9), np.array([1000], dtype=np.uint64))
    loaded.sync(path)
    again = IdMapIndex.load(path)
    assert again.to_bytes() == loaded.to_bytes()


def test_sync_errors_name_the_path(tmp_path):
    idx = TurboQuantIndex(DIM, 4)
    idx.calibrate(_rows(1024, 10))
    idx.add(_rows(8, 11))
    missing = str(tmp_path / "no-such-dir" / "index.tv")
    with pytest.raises(OSError) as exc:
        idx.sync(missing)
    assert "no-such-dir" in str(exc.value)


def test_lazy_index_refuses_sync():
    idx = TurboQuantIndex(bit_width=4)
    with pytest.raises(OSError):
        idx.sync("/tmp/never-written.tv")
