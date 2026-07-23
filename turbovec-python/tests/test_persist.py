"""Tests for the shared persistence consistency helpers."""
from __future__ import annotations

import pytest

from turbovec._persist import check_sidecar_keysets


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
