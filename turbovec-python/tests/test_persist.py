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


def test_atomic_save_concurrent_same_process_does_not_corrupt(tmp_path):
    # #316: two store objects saving to the same directory from one
    # process used to derive identical `.tmp.{pid}` temp names — they
    # interleaved writes into one temp file, os.replace'd each other's
    # partial output, and each `finally` unlinked the other's in-flight
    # temp (FileNotFoundError escaping the save, or a permanently
    # mismatched index/side-car pair on disk).
    import json
    import threading

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    def make_index(n, seed):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, 32)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        idx = IdMapIndex(dim=32, bit_width=4)
        idx.add_with_ids(v, np.arange(n, dtype=np.uint64))
        return idx

    stores = [(make_index(5, 0), list(range(5))), (make_index(300, 1), list(range(300)))]
    index_path = tmp_path / "index.tvim"
    sidecar_path = tmp_path / "docstore.json"
    errors = []
    barrier = threading.Barrier(len(stores))

    def save_loop(index, payload):
        try:
            barrier.wait()
            for _ in range(25):
                atomic_save(index, index_path, payload, sidecar_path)
        except Exception as e:  # noqa: BLE001 — recorded for the assert
            errors.append(e)

    threads = [threading.Thread(target=save_loop, args=s) for s in stores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent saves raised: {errors!r}"
    # Each artifact must individually be a complete write from one of
    # the two stores — never interleaved bytes.
    loaded = IdMapIndex.load(str(index_path))
    assert len(loaded) in (5, 300)
    payload = json.loads(sidecar_path.read_text())
    assert payload in (stores[0][1], stores[1][1])
    # No temp strays.
    strays = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert strays == []
