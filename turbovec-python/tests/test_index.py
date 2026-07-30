"""Tests for TurboQuantIndex — the pyo3 binding surface.

These exercise the public Python API exposed by the Rust extension
(add, search, save/load, prepare, len, dim, bit_width), independent of
the LangChain / LlamaIndex wrappers.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from turbovec import TurboQuantIndex


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_new_reports_dim_and_bit_width():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    assert idx.dim == 128
    assert idx.bit_width == 4
    assert len(idx) == 0


@pytest.mark.parametrize("bit_width", [2, 3, 4])
def test_bit_width_options(bit_width):
    idx = TurboQuantIndex(dim=128, bit_width=bit_width)
    assert idx.bit_width == bit_width
    idx.add(unit_vectors(20, 128))
    assert len(idx) == 20


def test_add_updates_length():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(50, 128))
    assert len(idx) == 50


def test_add_is_incremental():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(20, 128, seed=1))
    idx.add(unit_vectors(30, 128, seed=2))
    assert len(idx) == 50


def test_search_shape():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(100, 128))
    scores, indices = idx.search(unit_vectors(5, 128, seed=99), k=10)
    assert scores.shape == (5, 10)
    assert indices.shape == (5, 10)


def test_search_single_query():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(100, 128))
    scores, indices = idx.search(unit_vectors(1, 128, seed=99), k=5)
    assert scores.shape == (1, 5)
    assert indices.shape == (1, 5)


def test_self_query_recall_at_1():
    vectors = unit_vectors(100, 256, seed=42)
    idx = TurboQuantIndex(dim=256, bit_width=4)
    idx.add(vectors)

    hits = 0
    for i in range(20):
        _, indices = idx.search(vectors[i:i + 1], k=1)
        if indices[0, 0] == i:
            hits += 1
    assert hits == 20, f"recall@1 failed: {hits}/20"


def test_save_load_roundtrip(tmp_path):
    vectors = unit_vectors(80, 128, seed=7)
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(vectors)
    idx.prepare()

    path = str(tmp_path / "idx.tv")
    idx.write(path)
    loaded = TurboQuantIndex.load(path)

    assert len(loaded) == 80
    assert loaded.dim == 128
    assert loaded.bit_width == 4

    q = unit_vectors(3, 128, seed=8)
    s_orig, i_orig = idx.search(q, k=10)
    s_load, i_load = loaded.search(q, k=10)
    np.testing.assert_array_equal(i_orig, i_load)
    np.testing.assert_allclose(s_orig, s_load, rtol=1e-5)


def test_load_missing_file_raises_file_not_found():
    # Issue #156: a missing file must raise FileNotFoundError (as
    # Python's open() would), not a bare OSError.
    with pytest.raises(FileNotFoundError):
        TurboQuantIndex.load("/nonexistent/path/does-not-exist.tv")


def test_write_to_missing_directory_names_the_path_and_raises_file_not_found():
    # Issue #329: write() reported a bare OSError naming no file, so a
    # batch job saving several paths couldn't tell which one failed and
    # `except FileNotFoundError:` around a write never matched.
    idx = TurboQuantIndex(dim=64, bit_width=4)
    idx.add(unit_vectors(4, 64))
    missing = "/nonexistent/path/does-not-exist.tv"
    with pytest.raises(FileNotFoundError) as exc_info:
        idx.write(missing)
    assert missing in str(exc_info.value)


def test_load_corrupt_file_stays_plain_oserror(tmp_path):
    # Pin: only the missing-file case narrows to FileNotFoundError; an
    # existing-but-corrupt file keeps raising the OSError family.
    p = tmp_path / "corrupt.tv"
    p.write_bytes(b"garbage, not a tv file")
    with pytest.raises(OSError) as exc_info:
        TurboQuantIndex.load(str(p))
    assert not isinstance(exc_info.value, FileNotFoundError)


def test_prepare_is_idempotent():
    idx = TurboQuantIndex(dim=64, bit_width=4)
    idx.add(unit_vectors(20, 64))
    idx.prepare()
    idx.prepare()
    assert len(idx) == 20


def test_batch_query_matches_individual():
    idx = TurboQuantIndex(dim=256, bit_width=4)
    vectors = unit_vectors(50, 256, seed=0)
    idx.add(vectors)

    queries = unit_vectors(5, 256, seed=99)
    _, batch_indices = idx.search(queries, k=3)

    for i in range(5):
        _, single_indices = idx.search(queries[i:i + 1], k=3)
        np.testing.assert_array_equal(
            batch_indices[i:i + 1], single_indices
        )


def test_noncontiguous_input_is_handled():
    # A strided slice of a larger array should still work.
    big = unit_vectors(100, 128)
    strided = big[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    idx = TurboQuantIndex(dim=128, bit_width=4)
    # PyO3 layer asserts contiguity; caller is expected to convert.
    # This test documents that behaviour: a contiguous copy works.
    idx.add(np.ascontiguousarray(strided))
    assert len(idx) == 50


def test_swap_remove_shrinks_length():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(10, 128))
    moved_from = idx.swap_remove(3)
    assert moved_from == 9
    assert len(idx) == 9


def test_swap_remove_last_is_no_swap():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(5, 128))
    assert idx.swap_remove(4) == 4
    assert len(idx) == 4


def test_search_after_swap_remove_reflects_new_layout():
    # Cache-invalidation regression: the vector that moves into the
    # deleted slot must be findable immediately after the delete.
    idx = TurboQuantIndex(dim=256, bit_width=4)
    vectors = unit_vectors(20, 256, seed=0)
    idx.add(vectors)

    # Prime the cache with a self-query.
    _, pre = idx.search(vectors[5:6], k=1)
    assert pre[0, 0] == 5

    # Delete slot 5 — the last vector (index 19) moves into slot 5.
    idx.swap_remove(5)
    assert len(idx) == 19

    _, post = idx.search(vectors[19:20], k=1)
    assert post[0, 0] == 5, "vector that moved into slot 5 not found there"


def test_add_with_mismatched_dim_raises_value_error():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    with pytest.raises(ValueError, match="dim mismatch"):
        idx.add(unit_vectors(3, 256))


@pytest.mark.parametrize("bad_bit_width", [0, 1, 5, 8])
def test_constructor_rejects_bad_bit_width(bad_bit_width):
    with pytest.raises(ValueError, match="bit_width"):
        TurboQuantIndex(dim=128, bit_width=bad_bit_width)


@pytest.mark.parametrize("bad_dim", [0, 1, 4, 7, 9])
def test_constructor_rejects_bad_dim(bad_dim):
    with pytest.raises(ValueError, match="dim"):
        TurboQuantIndex(dim=bad_dim, bit_width=4)


def test_search_on_empty_eager_index_returns_zero_effective_k():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    q = unit_vectors(1, 128)
    scores, indices = idx.search(q, k=3)
    assert scores.shape == (1, 0)
    assert indices.shape == (1, 0)


def test_search_k_exceeding_len_clamps_to_len():
    # k > n with 0 < n < k: the effective k is min(k, n) = 3, so the
    # returned arrays must have exactly n columns (this shape IS the
    # SearchResults.k field surfaced through the binding), with every
    # stored vector appearing exactly once, in range, per query.
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(3, 128, seed=5))

    queries = unit_vectors(2, 128, seed=6)
    scores, indices = idx.search(queries, k=10)

    assert scores.shape == (2, 3)
    assert indices.shape == (2, 3)
    for qi in range(2):
        row = indices[qi]
        assert len(set(row.tolist())) == 3, f"duplicate indices for query {qi}: {row}"
        assert ((row >= 0) & (row < 3)).all(), f"out-of-range index for query {qi}: {row}"
    assert np.isfinite(scores).all()


# ---- Wave 5: typed-exception hygiene at the binding layer ----

def test_add_noncontiguous_vectors_raises_value_error():
    # A non-C-contiguous numpy view used to surface as a Rust panic via
    # `as_slice().expect("vectors must be contiguous")`. The binding now
    # converts that to a clean ValueError pointing at the caller-side
    # fix (`np.ascontiguousarray`).
    idx = TurboQuantIndex(dim=128, bit_width=4)
    full = unit_vectors(4, 256)  # 256 cols
    sliced = full[:, ::2]  # take every other column → non-contiguous, 128 cols
    assert not sliced.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        idx.add(sliced)


def test_search_query_dim_mismatch_raises_value_error():
    # A query with the wrong number of columns previously panicked
    # inside the core's `assert_eq!(queries.len(), nq * dim)` — pyo3
    # surfaces that as a PanicException, not the ValueError users
    # naturally try to catch. The binding now validates ncols against
    # the index dim before forwarding.
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(5, 128))
    wrong_dim_query = unit_vectors(1, 64)
    with pytest.raises(ValueError, match="query dim"):
        idx.search(wrong_dim_query, k=1)


def test_search_noncontiguous_query_raises_value_error():
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(3, 128))
    full = unit_vectors(2, 256)
    sliced = full[:, ::2]
    assert not sliced.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        idx.search(sliced, k=1)


def test_swap_remove_out_of_bounds_raises_index_error():
    # Previously panicked in core's `assert!(idx < n_vectors)`. The
    # binding now raises a clean IndexError before the inner call.
    idx = TurboQuantIndex(dim=128, bit_width=4)
    idx.add(unit_vectors(3, 128))
    with pytest.raises(IndexError, match="out of range"):
        idx.swap_remove(99)


# ---- Wave 5: input validation surfaces from the Rust core ----

def test_add_rejects_nan_with_value_error():
    # The Rust core's `AddError::InvalidInputValue` is mapped through
    # the binding's `add_2d` Result → PyValueError path.
    idx = TurboQuantIndex(dim=64, bit_width=4)
    data = unit_vectors(1, 64).copy()
    data[0, 5] = np.nan
    with pytest.raises(ValueError, match="invalid input value"):
        idx.add(data)


def test_add_rejects_huge_magnitude_with_value_error():
    idx = TurboQuantIndex(dim=64, bit_width=4)
    data = unit_vectors(1, 64).copy()
    data[0, 5] = 1e20
    with pytest.raises(ValueError, match="invalid input value"):
        idx.add(data)


def test_search_with_nan_query_raises():
    # NaN in the query reaches the core via search_with_mask, which
    # now panics with a clear message; pyo3 maps Rust panics to
    # PanicException → BaseException, so the test catches broadly but
    # asserts on the message substring.
    idx = TurboQuantIndex(dim=64, bit_width=4)
    idx.add(unit_vectors(3, 64))
    q = unit_vectors(1, 64).copy()
    q[0, 0] = np.nan
    with pytest.raises(BaseException, match="invalid query value"):
        idx.search(q, k=1)


# ---- #308: a zero-row add is a true no-op on a lazy index ----


def test_zero_row_add_leaves_lazy_index_uncommitted():
    idx = TurboQuantIndex(bit_width=4)
    idx.add(np.zeros((0, 768), dtype=np.float32))
    assert idx.dim is None
    assert len(idx) == 0
    # The index is still free to commit to a different dim afterwards.
    idx.add(unit_vectors(3, 64))
    assert idx.dim == 64


def test_zero_row_add_does_not_change_serialized_bytes():
    pristine = TurboQuantIndex(bit_width=4).to_bytes()
    poked = TurboQuantIndex(bit_width=4)
    poked.add(np.zeros((0, 768), dtype=np.float32))
    assert poked.to_bytes() == pristine
    assert TurboQuantIndex.from_bytes(poked.to_bytes()).dim is None


def test_zero_row_add_still_checks_dim_once_committed():
    idx = TurboQuantIndex(dim=64, bit_width=4)
    idx.add(unit_vectors(2, 64))
    with pytest.raises(ValueError, match="dim"):
        idx.add(np.zeros((0, 128), dtype=np.float32))
    idx.add(np.zeros((0, 64), dtype=np.float32))
    assert len(idx) == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permissions")
def test_durability_shortfall_surfaces_as_a_runtime_warning(tmp_path):
    # A save whose rename succeeded but whose parent-directory fsync
    # failed has committed — it must not raise — but the durability
    # shortfall has to stay visible (#365). The core reports it through a
    # warning hook the binding points at Python's `warnings`, so it is
    # filterable, capturable by logging.captureWarnings, and assertable
    # here; an `eprintln!` from the library would be none of those.
    #
    # Provoked for real: a destination directory this process may write
    # and traverse but not open for reading, so the post-rename
    # `File::open(dir)` fails with EACCES.
    d = tmp_path / "no-read"
    d.mkdir()
    idx = TurboQuantIndex(dim=32, bit_width=4)
    idx.add(unit_vectors(64, 32))

    os.chmod(d, 0o300)
    try:
        try:
            os.listdir(d)
        except PermissionError:
            pass
        else:
            pytest.skip("directory mode not enforced (running as root?)")

        path = d / "index.tv"
        with pytest.warns(RuntimeWarning, match="power loss"):
            idx.write(str(path))
        assert path.exists(), "the rename committed, so the file must be there"
    finally:
        os.chmod(d, 0o700)


# That durability warning is emitted by the core from *inside*
# `write_with_durability`, i.e. while the binding still holds the index
# read guard — and it reaches Python through a user-replaceable
# `showwarning`. A handler that touches the same index then asks for the
# write lock while the save read-holds it: deadlock, the same defect the
# warm-up warning had (#360, whose fix left this path unguarded). Runs in
# a fresh interpreter under a hard timeout — a regression would otherwise
# hang pytest itself, and it wedges the *pool* thread that emits the
# warning, so it cannot be unwound in-process.
_REENTRANT_DURABILITY_HANDLER = r'''
import os
import stat
import sys
import tempfile
import warnings

import numpy as np
import turbovec

DIM = 32
rng = np.random.default_rng(0)
idx = turbovec.TurboQuantIndex(DIM, 4)
idx.add(rng.standard_normal((10, DIM), dtype=np.float32))
extra = rng.standard_normal((5, DIM), dtype=np.float32)

seen = []


def showwarning(message, category, filename, lineno, file=None, line=None):
    seen.append(str(message))
    idx.add(extra)          # re-enters the index the save is holding


d = tempfile.mkdtemp()
warnings.simplefilter("always")
warnings.showwarning = showwarning
os.chmod(d, stat.S_IWUSR | stat.S_IXUSR)   # writable+traversable, not readable
try:
    idx.write(os.path.join(d, "index.tv"), durable=True)
finally:
    os.chmod(d, 0o700)

assert any("power loss" in s for s in seen), seen
assert len(idx) == 20, len(idx)             # 10 + 2 handlers x 5
print("RESULT: PASS")
'''


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permissions")
def test_durability_warning_handler_may_reenter_the_index():
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _REENTRANT_DURABILITY_HANDLER],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the durability-shortfall warning DEADLOCKED: it ran user Python "
            "while the save held the index read lock, so the handler's add "
            "blocked forever on the write lock (#360)"
        )
    if "running as root" in proc.stderr or "AssertionError" in proc.stderr:
        # An unenforced directory mode means no warning fired at all;
        # the sibling test above skips for the same reason.
        pytest.skip(f"directory mode not enforced:\n{proc.stderr}")
    assert "RESULT: PASS" in proc.stdout, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
