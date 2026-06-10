"""Unit tests for the shared in-batch duplicate-resolution helper."""
from __future__ import annotations

import pytest

from turbovec._dedup import DuplicatePolicy, resolve_duplicates


def test_keep_all_returns_every_index():
    assert resolve_duplicates(["a", "a", "b"], DuplicatePolicy.KEEP_ALL) == [0, 1, 2]


def test_keep_last_collapses_to_last_occurrence():
    # a@0,a@2 -> keep 2; b@1 -> keep 1. Ascending order.
    assert resolve_duplicates(["a", "b", "a"], DuplicatePolicy.KEEP_LAST) == [1, 2]


def test_keep_first_collapses_to_first_occurrence():
    assert resolve_duplicates(["a", "b", "a"], DuplicatePolicy.KEEP_FIRST) == [0, 1]


def test_reject_raises_on_duplicate():
    with pytest.raises(ValueError, match="duplicate id in batch"):
        resolve_duplicates(["a", "b", "a"], DuplicatePolicy.REJECT)


def test_reject_passes_through_when_unique():
    assert resolve_duplicates(["a", "b", "c"], DuplicatePolicy.REJECT) == [0, 1, 2]


def test_empty_batch():
    for policy in DuplicatePolicy:
        assert resolve_duplicates([], policy) == []


def test_no_duplicates_preserves_order_for_all_policies():
    keys = ["x", "y", "z"]
    for policy in DuplicatePolicy:
        assert resolve_duplicates(keys, policy) == [0, 1, 2]
