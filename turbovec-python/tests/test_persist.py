"""Tests for the shared persistence consistency helpers."""
from __future__ import annotations

import pytest

from turbovec._persist import check_persisted_handles, check_sidecar_keysets


def test_check_sidecar_keysets_mixed_type_ids_raise_value_error():
    # A hand-corrupted side-car can hold mixed-type ids — JSON arrays
    # (e.g. llama_index's node_id_to_u64 pairs) survive parsing with an
    # int among the strings. Building the error's id sample must not
    # itself blow up sorting unorderable types: the promise is a
    # ValueError, never a TypeError.
    with pytest.raises(ValueError, match="out of sync"):
        check_sidecar_keysets(
            ["a", 1],
            [],
            what="node",
            mapping_name="node_id_to_u64",
            sidecar_name="nodes",
        )
    # Same for the extraneous direction.
    with pytest.raises(ValueError, match="out of sync"):
        check_sidecar_keysets(
            [],
            ["a", 1],
            what="node",
            mapping_name="node_id_to_u64",
            sidecar_name="nodes",
        )


class _FakeIndex:
    """Minimal stand-in exposing the `len`/`contains` surface
    `check_persisted_handles` uses."""

    def __init__(self, handles):
        self._handles = set(int(h) for h in handles)

    def __len__(self):
        return len(self._handles)

    def contains(self, h):
        return int(h) in self._handles


def test_check_persisted_handles_rejects_rewound_watermark():
    # Issue #321: the watermark is the one field the corruption check
    # forgot. Below the largest live handle it reissues them on the next
    # write, bricking the store with a leaked internal id.
    index = _FakeIndex([1, 2, 3])
    with pytest.raises(ValueError, match="next_u64"):
        check_persisted_handles(index, [1, 2, 3], what="document", next_u64=0)
    with pytest.raises(ValueError, match="next_u64"):
        check_persisted_handles(index, [1, 2, 3], what="document", next_u64=2)


def test_check_persisted_handles_accepts_sound_watermark():
    index = _FakeIndex([1, 2, 3])
    check_persisted_handles(index, [1, 2, 3], what="document", next_u64=3)
    check_persisted_handles(index, [1, 2, 3], what="document", next_u64=99)
    # Omitted watermark keeps the pre-#321 behaviour (callers that don't
    # have it).
    check_persisted_handles(index, [1, 2, 3], what="document")
    # Empty store: any watermark is sound.
    check_persisted_handles(_FakeIndex([]), [], what="document", next_u64=0)
