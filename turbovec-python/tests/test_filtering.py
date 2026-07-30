"""Tests for the `mask=` and `allowlist=` filtering kwargs."""
from __future__ import annotations

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex


DIM = 128


def unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


# ------------------- TurboQuantIndex.search(mask=...) -------------------

def test_mask_none_matches_unmasked():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(100, seed=1))
    queries = unit_vectors(3, seed=2)

    s1, i1 = idx.search(queries, 10)
    s2, i2 = idx.search(queries, 10, mask=None)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


def test_mask_all_true_matches_unmasked():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(80, seed=3))
    queries = unit_vectors(2, seed=4)

    s1, i1 = idx.search(queries, 5)
    mask = np.ones(len(idx), dtype=bool)
    s2, i2 = idx.search(queries, 5, mask=mask)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


def test_mask_restricts_returned_indices():
    n = 200
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=5))
    queries = unit_vectors(4, seed=6)

    mask = np.zeros(n, dtype=bool)
    allowed = [3, 7, 19, 42, 88, 121, 150, 175, 198]
    mask[allowed] = True

    scores, indices = idx.search(queries, 5, mask=mask)
    assert scores.shape == (4, 5)
    assert indices.shape == (4, 5)
    for row in indices:
        for slot in row:
            assert slot in allowed, f"kernel returned disallowed slot {slot}"
    # Scores descending per row.
    for row in scores:
        assert np.all(np.diff(row) <= 1e-6)


def test_mask_shrinks_effective_k():
    n = 100
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=7))
    queries = unit_vectors(2, seed=8)

    mask = np.zeros(n, dtype=bool)
    mask[[5, 10, 15]] = True
    scores, indices = idx.search(queries, 10, mask=mask)
    assert scores.shape == (2, 3), "effective_k should be popcount(mask) = 3"
    assert indices.shape == (2, 3)


def test_mask_all_false_returns_empty_columns():
    n = 50
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=9))
    queries = unit_vectors(2, seed=10)

    mask = np.zeros(n, dtype=bool)
    scores, indices = idx.search(queries, 5, mask=mask)
    assert scores.shape == (2, 0)
    assert indices.shape == (2, 0)


def test_mask_wrong_length_raises():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(50, seed=11))
    queries = unit_vectors(1, seed=12)

    with pytest.raises(ValueError, match="mask length"):
        idx.search(queries, 5, mask=np.ones(10, dtype=bool))


def test_mask_must_be_bool_dtype():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(50, seed=13))
    queries = unit_vectors(1, seed=14)

    with pytest.raises((TypeError, ValueError)):
        idx.search(queries, 5, mask=np.ones(50, dtype=np.uint8))


def test_mask_matches_post_hoc_filter():
    n = 256
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=15))
    queries = unit_vectors(5, seed=16)

    mask = np.zeros(n, dtype=bool)
    mask[::3] = True
    k = 7

    unfiltered_scores, unfiltered_idx = idx.search(queries, n)
    expected_indices = []
    expected_scores = []
    for qi in range(queries.shape[0]):
        row_idx = unfiltered_idx[qi]
        row_s = unfiltered_scores[qi]
        keep = mask[row_idx]
        expected_indices.append(row_idx[keep][:k])
        expected_scores.append(row_s[keep][:k])

    masked_scores, masked_idx = idx.search(queries, k, mask=mask)
    assert masked_idx.shape == (queries.shape[0], k)
    for qi in range(queries.shape[0]):
        np.testing.assert_array_equal(masked_idx[qi], expected_indices[qi])
        # Score parity is exact when both calls go through the same code
        # path (single batched call on both sides).
        np.testing.assert_allclose(masked_scores[qi], expected_scores[qi], rtol=1e-4, atol=1e-5)


# --------- mask buffers whose bytes are not 0 or 1 (#349) ---------
#
# numpy stores `bool_` in one byte and does not constrain the value, and
# treats every non-zero byte as true. Such buffers come out of ordinary
# numpy calls, so none of this needs deliberate byte-forging. Reading one
# as Rust `bool` is undefined behavior and it mis-filtered concretely,
# up to a total filter bypass: every returned slot outside the mask and
# none of the selected ones.

MASK_ODD_BYTE_ROUTES = [
    pytest.param("uint8_view", id="uint8.view(bool)"),
    pytest.param("frombuffer", id="np.frombuffer(dtype=bool)"),
]


def _odd_byte_mask(route: str, n: int) -> np.ndarray:
    if route == "uint8_view":
        raw = np.zeros(n, dtype=np.uint8)
        raw[3] = 2
        raw[9] = 7
        raw[40] = 255
        return raw.view(bool)
    if route == "frombuffer":
        # Arbitrary bytes reinterpreted as bool — every value 2..255 in
        # here is an invalid Rust `bool`. Read-only, which the mask path
        # must also accept.
        return np.frombuffer(bytes(range(n)), dtype=bool)
    raise AssertionError(route)


@pytest.mark.parametrize("route", MASK_ODD_BYTE_ROUTES)
def test_mask_byte_other_than_zero_or_one_is_true(route):
    """A `bool_` mask whose bytes are not 0/1 filters by numpy's own
    truthiness — same result as a clean `bool` mask selecting the same
    slots, and nothing outside them."""
    n = 64
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=41))
    queries = unit_vectors(2, seed=42)

    odd = _odd_byte_mask(route, n)
    selected = np.flatnonzero(np.asarray(odd))
    assert selected.size >= 3, "the route must select something to compare"

    clean = np.zeros(n, dtype=bool)
    clean[selected] = True

    odd_scores, odd_idx = idx.search(queries, 5, mask=odd)
    clean_scores, clean_idx = idx.search(queries, 5, mask=clean)

    np.testing.assert_array_equal(odd_idx, clean_idx)
    np.testing.assert_array_equal(odd_scores, clean_scores)
    # And no slot outside the mask leaks through.
    assert set(odd_idx.ravel().tolist()) <= set(selected.tolist())


class _HijackedView(np.ndarray):
    """An `ndarray` subclass whose `view()` returns a *different* buffer."""

    def view(self, *args, **kwargs):  # noqa: D102
        hijacked = np.zeros(self.shape[0], dtype=np.uint8)
        hijacked[[1, 2, 3]] = 1
        return hijacked


class _RaisingView(np.ndarray):
    """An `ndarray` subclass whose `view()` refuses to be called."""

    def view(self, *args, **kwargs):  # noqa: D102
        raise RuntimeError("mask view() must not be called")


def test_mask_subclass_cannot_redirect_which_buffer_is_read():
    """The mask's own bytes decide the filter — never a Python method the
    caller's type controls.

    Reaching the bytes through the array's `view("uint8")` method would
    let an `ndarray` subclass overriding `view` choose which buffer the
    mask is read from, so the caller's *type* would select which slots
    are searched. That is the same leak the byte-level fix exists to
    close, so the read goes through the C-level data pointer and calls
    into Python not at all.
    """
    n = 64
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=45))
    queries = unit_vectors(2, seed=46)

    selected = [40, 41, 42]
    base = np.zeros(n, dtype=bool)
    base[selected] = True
    hijacker = base.view(_HijackedView)

    # Precondition: the override really does hand out a different buffer.
    np.testing.assert_array_equal(np.flatnonzero(hijacker.view("uint8")), [1, 2, 3])
    np.testing.assert_array_equal(
        np.flatnonzero(np.ndarray.view(hijacker, "uint8")), selected
    )

    want_scores, want_idx = idx.search(queries, 3, mask=base)
    got_scores, got_idx = idx.search(queries, 3, mask=hijacker)

    assert set(got_idx.ravel().tolist()) <= set(selected)
    np.testing.assert_array_equal(got_idx, want_idx)
    np.testing.assert_array_equal(got_scores, want_scores)


def test_mask_subclass_view_is_never_called():
    """Corollary of the above: a mask whose `view()` raises still
    searches normally, because `view()` is not on the path at all."""
    n = 50
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(n, seed=47))
    queries = unit_vectors(1, seed=48)

    base = np.zeros(n, dtype=bool)
    base[[7, 8]] = True

    want = idx.search(queries, 2, mask=base)[1]
    got = idx.search(queries, 2, mask=base.view(_RaisingView))[1]
    np.testing.assert_array_equal(got, want)


def test_mask_byte_read_still_rejects_non_contiguous():
    """The dtype and contiguity error paths survive reading the mask as
    bytes: a strided mask must still raise, not be silently copied."""
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(50, seed=43))
    queries = unit_vectors(1, seed=44)

    with pytest.raises(ValueError, match="C-contiguous"):
        idx.search(queries, 5, mask=np.zeros(100, dtype=bool)[::2])
    with pytest.raises(TypeError, match="1-D bool"):
        idx.search(queries, 5, mask=np.zeros(50, dtype=np.uint8))


# ------------------- IdMapIndex.search(allowlist=...) -------------------

def test_allowlist_none_matches_unfiltered():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    ids = np.arange(7000, 7100, dtype=np.uint64)
    idx.add_with_ids(unit_vectors(100, seed=20), ids)
    queries = unit_vectors(2, seed=21)

    s1, i1 = idx.search(queries, 10)
    s2, i2 = idx.search(queries, 10, allowlist=None)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(i1, i2)


def test_allowlist_restricts_returned_ids():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    ids = np.arange(1000, 1100, dtype=np.uint64)
    idx.add_with_ids(unit_vectors(100, seed=22), ids)
    queries = unit_vectors(3, seed=23)

    allowed = np.array([1003, 1010, 1042, 1077, 1099], dtype=np.uint64)
    scores, returned = idx.search(queries, 10, allowlist=allowed)
    assert scores.shape == (3, len(allowed))
    assert returned.shape == (3, len(allowed))
    for row in returned:
        for rid in row:
            assert rid in allowed


def test_allowlist_empty_raises_value_error():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    idx.add_with_ids(
        unit_vectors(5, seed=24),
        np.array([1, 2, 3, 4, 5], dtype=np.uint64),
    )
    queries = unit_vectors(1, seed=25)
    with pytest.raises(ValueError, match="allowlist is empty"):
        idx.search(queries, 3, allowlist=np.array([], dtype=np.uint64))


def test_allowlist_unknown_id_raises_key_error():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    idx.add_with_ids(
        unit_vectors(5, seed=26),
        np.array([1, 2, 3, 4, 5], dtype=np.uint64),
    )
    queries = unit_vectors(1, seed=27)
    with pytest.raises(KeyError, match="not present"):
        idx.search(queries, 3, allowlist=np.array([2, 999], dtype=np.uint64))


def test_allowlist_kwarg_is_keyword_only():
    idx = IdMapIndex(dim=DIM, bit_width=4)
    ids = np.arange(100, 110, dtype=np.uint64)
    idx.add_with_ids(unit_vectors(10, seed=28), ids)
    queries = unit_vectors(1, seed=29)
    # Positional third arg should be rejected by PyO3 signature.
    with pytest.raises(TypeError):
        idx.search(queries, 3, ids)


def test_mask_kwarg_is_keyword_only():
    idx = TurboQuantIndex(dim=DIM, bit_width=4)
    idx.add(unit_vectors(10, seed=30))
    queries = unit_vectors(1, seed=31)
    mask = np.ones(10, dtype=bool)
    with pytest.raises(TypeError):
        idx.search(queries, 3, mask)
