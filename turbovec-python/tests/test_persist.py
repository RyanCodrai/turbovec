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


def test_tmp_path_fits_name_max_for_long_destinations():
    # #299/#355: the temp suffix grew from ~10 to ~24 bytes, so a legal
    # destination filename of ~232-255 bytes produced a temp name past
    # NAME_MAX and the save failed with ENAMETOOLONG — for names that had
    # saved fine before. The base is truncated to fit; the destination
    # name itself is untouched.
    import os

    from turbovec._persist import _tmp_path

    for n in (10, 200, 232, 245, 255):
        dest = os.path.join("/tmp", "x" * n + ".json")
        tmp = _tmp_path(dest)
        base = os.path.basename(tmp)
        assert len(base.encode()) <= 255, f"{n}: temp name is {len(base.encode())} bytes"
        assert ".tmp." in base, "temp must stay recognizable to the sweep"
    # Distinct per call even after truncation.
    dest = os.path.join("/tmp", "y" * 250 + ".json")
    assert _tmp_path(dest) != _tmp_path(dest)


def test_atomic_save_round_trips_a_long_sidecar_name(tmp_path):
    # End-to-end: the whole save path must work for a destination whose
    # name is close to NAME_MAX (#355).
    import json

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    rng = np.random.default_rng(0)
    v = rng.standard_normal((4, 32)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx = IdMapIndex(dim=32, bit_width=4)
    idx.add_with_ids(v, np.arange(4, dtype=np.uint64))

    stem = "s" * 200
    index_path = tmp_path / f"{stem}.tvim"
    sidecar_path = tmp_path / f"{stem}.json"
    atomic_save(idx, index_path, {"docs": [1, 2, 3]}, sidecar_path)

    assert len(IdMapIndex.load(str(index_path))) == 4
    assert json.loads(sidecar_path.read_text()) == {"docs": [1, 2, 3]}
    assert [p.name for p in tmp_path.iterdir() if ".tmp." in p.name] == []


# ---- #350: the side-car must be faithful, portable JSON ---------------
#
# `json.dumps` accepts two things it cannot round-trip, and silently:
# non-str mapping keys (stringified, so `1` and `"1"` merge and one entry
# is lost) and NaN/Infinity (bare tokens RFC 8259 forbids — jq rewrites
# them to null, serde_json/JSON.parse reject the file). Both now fail
# loudly at save time, before any file is touched.


class _WriteRecordingIndex:
    """Records whether `atomic_save` ever got as far as writing."""

    def __init__(self):
        self.wrote = False

    def write(self, path):  # pragma: no cover - must never run in these tests
        self.wrote = True
        open(path, "wb").close()


@pytest.mark.parametrize(
    "bad_key",
    [1, 2020, True, None, 3.5],
    ids=["int", "year-int", "bool", "none", "float"],
)
def test_atomic_save_rejects_non_str_metadata_keys(tmp_path, bad_key):
    from turbovec._persist import atomic_save

    index = _WriteRecordingIndex()
    payload = {"docs": {"a": {"metadata": {bad_key: "x"}}}}
    with pytest.raises(TypeError) as exc:
        atomic_save(
            index, tmp_path / "i.tvim", payload, tmp_path / "s.json"
        )
    # The message must name the offending key and where it lives, so the
    # fix is mechanical.
    assert repr(bad_key) in str(exc.value)
    assert "['docs']['a']['metadata']" in str(exc.value)
    assert "not str" in str(exc.value)
    # Fail-before-touching-files: nothing was written, not even a temp.
    assert not index.wrote
    assert list(tmp_path.iterdir()) == []


def test_atomic_save_rejects_colliding_int_and_str_keys(tmp_path):
    # The exact loss from #350: in-memory `{1: "int-one", "1": "str-one"}`
    # used to land on disk as `{"1": "str-one"}` with save() returning
    # success — the int-keyed entry gone, undetectably.
    from turbovec._persist import atomic_save

    payload = {"docs": {"a": {"metadata": {1: "int-one", "1": "str-one"}}}}
    with pytest.raises(TypeError):
        atomic_save(
            _WriteRecordingIndex(), tmp_path / "i.tvim", payload, tmp_path / "s.json"
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_value, token",
    [
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
    ],
    ids=["nan", "inf", "-inf"],
)
def test_atomic_save_rejects_non_finite_floats(tmp_path, bad_value, token):
    from turbovec._persist import atomic_save

    index = _WriteRecordingIndex()
    payload = {"docs": {"a": {"metadata": {"score": bad_value}}}}
    with pytest.raises(ValueError) as exc:
        atomic_save(
            index, tmp_path / "i.tvim", payload, tmp_path / "s.json"
        )
    assert token in str(exc.value)
    assert "['docs']['a']['metadata']['score']" in str(exc.value)
    assert not index.wrote
    assert list(tmp_path.iterdir()) == []


def test_atomic_save_finds_bad_values_nested_in_lists(tmp_path):
    from turbovec._persist import atomic_save

    with pytest.raises(ValueError, match="NaN"):
        atomic_save(
            _WriteRecordingIndex(),
            tmp_path / "i.tvim",
            {"docs": [{"metadata": {"xs": [1.0, float("nan")]}}]},
            tmp_path / "s.json",
        )
    with pytest.raises(TypeError, match="not str"):
        atomic_save(
            _WriteRecordingIndex(),
            tmp_path / "i.tvim",
            {"docs": [{"metadata": {"nested": {7: "v"}}}]},
            tmp_path / "s.json",
        )
    assert list(tmp_path.iterdir()) == []


def test_check_json_faithful_terminates_on_cyclic_metadata():
    # Metadata is user-supplied; a self-referential container must not
    # spin the validator forever. json.dumps rejects it afterwards, which
    # is the pre-existing (and correct) behaviour for a cycle.
    from turbovec._persist import _check_json_faithful

    cycle: dict = {"a": 1}
    cycle["self"] = cycle
    _check_json_faithful({"docs": cycle})  # must return, not hang

    shared = {"k": "v"}
    _check_json_faithful({"docs": {"x": shared, "y": shared}})


def test_check_json_faithful_survives_deep_nesting():
    # An iterative walk, so nesting deeper than Python's recursion limit
    # is a job for json.dumps' own guard, not a RecursionError from us.
    from turbovec._persist import _check_json_faithful

    deep: object = "leaf"
    for _ in range(5000):
        deep = {"n": deep}
    _check_json_faithful(deep)


def test_atomic_save_still_accepts_faithful_payloads(tmp_path):
    # Guard against over-rejection: the values the side-car is documented
    # to carry must keep working.
    import json

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    v = np.eye(4, 32, dtype=np.float32)
    idx = IdMapIndex(dim=32, bit_width=4)
    idx.add_with_ids(v, np.arange(4, dtype=np.uint64))

    payload = {
        "schema_version": 2,
        "docs": {"a": {"metadata": {"n": None, "f": 1.5, "b": True, "": "empty"}}},
        "pairs": [["id", 7], ["id2", 8]],  # int *values* stay fine
        "big": 2**70,
        "unicode": "\U0001f600 é",
    }
    atomic_save(idx, tmp_path / "i.tvim", payload, tmp_path / "s.json")
    assert json.loads((tmp_path / "s.json").read_text()) == payload
