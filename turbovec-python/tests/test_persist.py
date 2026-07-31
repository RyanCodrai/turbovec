"""Tests for the shared persistence consistency helpers."""
from __future__ import annotations

import os
from unittest import mock

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
    import traceback

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
        except Exception:  # noqa: BLE001 — recorded for the assert
            # Keep the *traceback*, not just the exception object. When
            # this failed intermittently on Windows CI (#415) the log
            # held only `PermissionError(13, 'Access is denied')`, which
            # named neither the failing call nor the winerror behind it
            # — two different raise sites in atomic_save produce that
            # identical repr. The frame is what makes a rare failure
            # actionable from a CI log alone.
            errors.append(traceback.format_exc())

    threads = [threading.Thread(target=save_loop, args=s) for s in stores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, "concurrent saves raised:\n" + "\n".join(errors)
    # Each artifact must individually be a complete write from one of
    # the two stores — never interleaved bytes.
    loaded = IdMapIndex.load(str(index_path))
    assert len(loaded) in (5, 300)
    payload = json.loads(sidecar_path.read_text())
    assert payload in (stores[0][1], stores[1][1])
    # No temp strays.
    strays = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert strays == []


def _win_oserror(winerror, strerror="Access is denied"):
    """An OSError shaped like the one Windows raises for ``winerror``.

    On Windows, build the real thing — the four-argument OSError form
    sets the ``winerror`` attribute and maps to the matching subclass.
    Off Windows there is no such attribute, so fabricate it. That
    fabrication is the *only* simulated part of these tests: the retry
    helper reads ``winerror`` with ``getattr``, so a hand-built exception
    drives the real control flow, which is otherwise unreachable from a
    POSIX machine.
    """
    if os.name == "nt":  # pragma: no cover - Windows-only path
        return OSError(13, strerror, None, winerror)
    exc = PermissionError(13, strerror)
    exc.winerror = winerror
    return exc


def test_win_oserror_helper_reproduces_the_reported_repr():
    # The CI repr in #415 was `PermissionError(13, 'Access is denied')`.
    # OSError.args is the (errno, strerror) pair on every platform, so
    # the winerror never appears in the repr — which is exactly why that
    # log could not distinguish ERROR_ACCESS_DENIED (5) from
    # ERROR_SHARING_VIOLATION (32) by eye. The strerror is what pins it:
    # 'Access is denied' is the text of 5, and 32 reads differently.
    assert repr(_win_oserror(5)) == "PermissionError(13, 'Access is denied')"


@pytest.fixture
def as_windows(monkeypatch):
    """Drive the Windows-only retry path from any platform.

    Flips ``_persist``'s own platform flag and neutralizes the backoff
    sleeps. Deliberately *not* done by patching ``os.name``: pathlib
    reads that to pick its flavour, so setting it hands WindowsPath
    objects to everything else running, pytest's own reporting included.
    """
    from turbovec import _persist

    monkeypatch.setattr(_persist, "_IS_WINDOWS", True)
    delays = []
    monkeypatch.setattr(_persist.time, "sleep", delays.append)
    return delays


@pytest.mark.parametrize("winerror", [5, 32])
def test_with_windows_retry_retries_transient_failures(as_windows, winerror):
    # #415: a rename onto a destination another save is concurrently
    # replacing fails transiently — with ERROR_SHARING_VIOLATION (32)
    # while a competing handle lacks FILE_SHARE_DELETE, and with
    # ERROR_ACCESS_DENIED (5) while the superseded destination sits in
    # the delete-pending state Windows leaves it in until the last handle
    # closes. Both clear on their own within microseconds. The retry
    # covered only 32, so the 5 escaped to the caller as a failed save.
    from turbovec import _persist

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 4:
            raise _win_oserror(winerror)
        return "done"

    assert _persist._with_windows_retry(flaky) == "done"
    assert len(calls) == 4


# ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, ERROR_WRITE_PROTECT,
# ERROR_DIRECTORY_NOT_SUPPORTED, ERROR_PRIVILEGE_NOT_HELD — none of
# these clear by waiting.
@pytest.mark.parametrize("winerror", [2, 3, 19, 267, 1314])
def test_with_windows_retry_reraises_permanent_failures(as_windows, winerror):
    # The guard that stops this decaying into retry-everything. A
    # permanent condition — a read-only destination, a directory in the
    # way, a missing privilege — must surface on the *first* attempt, not
    # after a third of a second of pointless sleeping. Widening the
    # whitelist to a catch-all still raises, so only the call count
    # catches it; that is what this asserts.
    from turbovec import _persist

    calls = []

    def always_fails():
        calls.append(1)
        raise _win_oserror(winerror, "nope")

    with pytest.raises(OSError):
        _persist._with_windows_retry(always_fails)
    assert len(calls) == 1, "a permanent Windows failure must not be retried"


def test_with_windows_retry_gives_up_after_the_attempt_budget(as_windows):
    # A transient code that never clears is a failure, not a hang: the
    # helper raises the last error after a bounded number of attempts,
    # with the backoff doubling from 1ms to a 64ms cap.
    from turbovec import _persist

    delays = as_windows
    calls = []

    def always_denied():
        calls.append(1)
        raise _win_oserror(5)

    with pytest.raises(OSError, match="Access is denied"):
        _persist._with_windows_retry(always_denied)
    assert len(calls) == _persist._RETRY_ATTEMPTS
    # One sleep between attempts, none after the last.
    assert len(delays) == _persist._RETRY_ATTEMPTS - 1
    assert delays[:4] == [0.001, 0.002, 0.004, 0.008]
    assert max(delays) == 0.064


def test_with_windows_retry_does_not_retry_off_windows():
    # POSIX rename and unlink are defined against open files, so none of
    # these conditions exist there: a failure is real and immediate.
    from turbovec import _persist

    assert not _persist._IS_WINDOWS, "this test asserts the POSIX branch"
    calls = []

    def always_fails():
        calls.append(1)
        raise PermissionError(13, "Permission denied")

    with pytest.raises(PermissionError):
        _persist._with_windows_retry(always_fails)
    assert len(calls) == 1


def test_replace_atomic_routes_through_the_retry(tmp_path, as_windows):
    # Wiring check: _replace_atomic must go through the helper above,
    # and must still perform a real rename.
    from turbovec import _persist

    src = tmp_path / "src"
    src.write_text("payload")
    dst = tmp_path / "dst"
    real_replace = os.replace
    calls = []

    def flaky_replace(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise _win_oserror(5)
        real_replace(a, b)

    with mock.patch.object(_persist.os, "replace", flaky_replace):
        _persist._replace_atomic(str(src), str(dst))
    assert len(calls) == 3
    assert dst.read_text() == "payload"
    assert not src.exists()


def test_unlink_best_effort_swallows_a_real_permission_error(tmp_path):
    # Not simulated: a file inside a directory with the write bit off
    # genuinely cannot be unlinked. Cleanup must absorb that.
    from turbovec import _persist

    victim = tmp_path / "locked" / "temp"
    victim.parent.mkdir()
    victim.write_text("x")
    victim.parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            os.unlink(str(victim))  # the real failure this absorbs
        _persist._unlink_best_effort(str(victim))  # must not raise
    finally:
        victim.parent.chmod(0o700)


def test_atomic_save_cleanup_failure_does_not_fail_a_completed_save(tmp_path):
    # #415, second raise site. The temp unlink runs in a `finally` and is
    # documented best-effort, but it only swallowed FileNotFoundError. On
    # Windows an antivirus or indexer that opened the freshly-created
    # temp to scan it makes unlink fail with ERROR_ACCESS_DENIED — the
    # same PermissionError(13, 'Access is denied') — which turned a save
    # that had already landed on disk into an error the caller could
    # neither act on nor distinguish from a lost write.
    import json

    import numpy as np

    from turbovec import IdMapIndex, _persist

    idx = IdMapIndex(dim=8, bit_width=4)
    idx.add_with_ids(np.eye(8, dtype=np.float32), np.arange(8, dtype=np.uint64))
    index_path = tmp_path / "index.tvim"
    sidecar_path = tmp_path / "docstore.json"

    real_unlink = os.unlink

    def denied_unlink(path):
        # Remove it where it still exists, then fail the way Windows
        # does. Not short-circuiting on FileNotFoundError matters: by the
        # time cleanup runs, a successful save has already renamed both
        # temps away, and swallowing only FileNotFoundError (what this
        # used to do) would then never see the PermissionError at all.
        try:
            real_unlink(path)
        except FileNotFoundError:
            pass
        raise PermissionError(13, "Access is denied")

    with mock.patch.object(_persist.os, "unlink", denied_unlink):
        _persist.atomic_save(idx, index_path, ["a", "b"], sidecar_path)

    assert len(IdMapIndex.load(str(index_path))) == 8
    assert json.loads(sidecar_path.read_text()) == ["a", "b"]


def test_atomic_save_cleanup_failure_does_not_mask_the_real_error(tmp_path):
    # The other half: swallowing in the `finally` must not hide why the
    # save failed. The swallow is scoped to one unlink of one temp path
    # this call created, so the body's own exception still propagates
    # unchanged.
    import numpy as np

    from turbovec import IdMapIndex, _persist

    idx = IdMapIndex(dim=8, bit_width=4)
    idx.add_with_ids(np.eye(8, dtype=np.float32), np.arange(8, dtype=np.uint64))

    def boom(src, dst):
        raise RuntimeError("the actual failure")

    def denied_unlink(path):
        raise PermissionError(13, "Access is denied")

    with mock.patch.object(_persist, "_replace_atomic", boom), mock.patch.object(
        _persist.os, "unlink", denied_unlink
    ):
        with pytest.raises(RuntimeError, match="the actual failure"):
            _persist.atomic_save(
                idx, tmp_path / "index.tvim", ["a"], tmp_path / "docstore.json"
            )


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
        "unicode": "\U0001f600 é",
    }
    atomic_save(idx, tmp_path / "i.tvim", payload, tmp_path / "s.json")
    assert json.loads((tmp_path / "s.json").read_text()) == payload


# The guard's scope is deliberately narrower than "everything JSON
# round-trips imperfectly": it covers values whose JSON form *loses data*
# (a collided key) or *is not JSON* (a bare NaN token). Total, documented
# type narrowings are out of scope. These two tests pin that boundary so
# it cannot drift silently in either direction.


def test_tuples_are_accepted_and_narrow_to_lists(tmp_path):
    # Out of scope by decision, not oversight. Unlike a collided key,
    # tuple -> list loses no element and applies uniformly to every tuple;
    # the stores' own payloads are built from `dict.items()` pairs
    # (llama_index's `node_id_to_u64`), and both langchain.py's dump and
    # llama_index.py's persist document the coercion in-line. Rejecting
    # tuples would break the writers before it helped any caller.
    import json

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    idx = IdMapIndex(dim=32, bit_width=4)
    idx.add_with_ids(np.eye(2, 32, dtype=np.float32), np.arange(2, dtype=np.uint64))

    payload = {"pairs": [("id", 7)], "docs": {"a": {"bbox": (10, 20, 30, 40)}}}
    atomic_save(idx, tmp_path / "i.tvim", payload, tmp_path / "s.json")
    on_disk = json.loads((tmp_path / "s.json").read_text())
    assert on_disk == {"pairs": [["id", 7]], "docs": {"a": {"bbox": [10, 20, 30, 40]}}}

    # ...and a bad value *inside* a tuple is still caught: the walk
    # descends tuples, it just does not reject them.
    with pytest.raises(ValueError, match="NaN"):
        atomic_save(
            _WriteRecordingIndex(),
            tmp_path / "j.tvim",
            {"docs": ({"score": float("nan")},)},
            tmp_path / "t.json",
        )


def test_ints_wider_than_2_53_are_accepted_and_exact_in_python(tmp_path):
    # Also out of scope by decision. 9007199254740993 is the smallest
    # integer a double cannot hold — `JSON.parse` silently returns ...992
    # — but Python writes and reads it exactly and an arbitrary-precision
    # integer literal is valid RFC 8259, so the imprecision belongs to the
    # reader, not the file. Rejecting these would break the legitimate
    # int64 ids the stores carry. (2**70 would not test this: powers of
    # two survive a double round-trip, so the value must be one that does
    # not.)
    import json

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    idx = IdMapIndex(dim=32, bit_width=4)
    idx.add_with_ids(np.eye(2, 32, dtype=np.float32), np.arange(2, dtype=np.uint64))

    lossy_as_double = 9007199254740993  # 2**53 + 1
    assert int(float(lossy_as_double)) != lossy_as_double  # a double loses it
    payload = {"docs": {"a": {"metadata": {"id64": lossy_as_double, "max": 2**64 - 1}}}}
    atomic_save(idx, tmp_path / "i.tvim", payload, tmp_path / "s.json")
    assert json.loads((tmp_path / "s.json").read_text()) == payload


# --- #350: rename durability and the schema_version type gate ------------
#
# Two residuals from the side-car durability review. The payload-fidelity
# items from the same issue (non-str keys, NaN/Infinity) are covered above.


@pytest.mark.skipif(
    os.name == "nt", reason="no directory-fsync equivalent on Windows"
)
def test_atomic_save_fsyncs_the_containing_directory(tmp_path, monkeypatch):
    # `os.replace` publishes the new name by updating the *directory*.
    # fsyncing the two temp files only makes their contents durable; the
    # directory entry that names them can still be in cache when `save()`
    # returns, so a power loss afterwards loses a save that reported
    # success.
    #
    # The effect is unobservable from userspace — the rename is visible
    # either way, and only a crash separates the two worlds — so this
    # asserts on the syscall rather than on the outcome: it wraps
    # `os.fsync` and records which of the fds handed to it are
    # directories. Without the fix nothing but the two regular files is
    # ever fsynced and `synced_dirs` is empty.
    import os as os_mod
    import stat

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    real_fsync = os_mod.fsync
    synced_dirs = []

    def recording_fsync(fd):
        if stat.S_ISDIR(os_mod.fstat(fd).st_mode):
            synced_dirs.append(os_mod.fstat(fd).st_ino)
        return real_fsync(fd)

    monkeypatch.setattr(os_mod, "fsync", recording_fsync)

    idx = IdMapIndex(dim=32, bit_width=4)
    idx.add_with_ids(np.eye(2, 32, dtype=np.float32), np.arange(2, dtype=np.uint64))
    atomic_save(idx, tmp_path / "i.tvim", {"docs": {}}, tmp_path / "s.json")

    assert os_mod.stat(tmp_path).st_ino in synced_dirs, (
        "atomic_save fsynced no directory, so the renames it just made are "
        "not durable"
    )


def test_atomic_save_fsyncs_both_directories_when_the_pair_is_split(
    tmp_path, monkeypatch
):
    # The index and its side-car normally share a directory, but the
    # signature does not require it. One fsync of "the" directory would
    # leave the other rename unpublished.
    if os.name == "nt":  # pragma: no cover - Windows-only path
        pytest.skip("no directory-fsync equivalent on Windows")
    import os as os_mod
    import stat

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    real_fsync = os_mod.fsync
    synced_dirs = []

    def recording_fsync(fd):
        if stat.S_ISDIR(os_mod.fstat(fd).st_mode):
            synced_dirs.append(os_mod.fstat(fd).st_ino)
        return real_fsync(fd)

    monkeypatch.setattr(os_mod, "fsync", recording_fsync)

    index_dir = tmp_path / "index"
    sidecar_dir = tmp_path / "sidecar"
    index_dir.mkdir()
    sidecar_dir.mkdir()

    idx = IdMapIndex(dim=32, bit_width=4)
    idx.add_with_ids(np.eye(2, 32, dtype=np.float32), np.arange(2, dtype=np.uint64))
    atomic_save(idx, index_dir / "i.tvim", {"docs": {}}, sidecar_dir / "s.json")

    assert os_mod.stat(index_dir).st_ino in synced_dirs
    assert os_mod.stat(sidecar_dir).st_ino in synced_dirs


def test_atomic_save_survives_a_directory_that_cannot_be_fsynced(
    tmp_path, monkeypatch
):
    # The rename has already succeeded by the time the directory fsync
    # runs, so a filesystem that refuses fsync on a directory fd must not
    # turn a completed save into a raised exception.
    import os as os_mod

    import numpy as np

    from turbovec import IdMapIndex
    from turbovec._persist import atomic_save

    real_fsync = os_mod.fsync

    def refusing_fsync(fd):
        import stat

        if stat.S_ISDIR(os_mod.fstat(fd).st_mode):
            raise OSError(22, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(os_mod, "fsync", refusing_fsync)

    idx = IdMapIndex(dim=32, bit_width=4)
    idx.add_with_ids(np.eye(2, 32, dtype=np.float32), np.arange(2, dtype=np.uint64))
    atomic_save(idx, tmp_path / "i.tvim", {"docs": {}}, tmp_path / "s.json")
    assert (tmp_path / "i.tvim").exists()
    assert (tmp_path / "s.json").exists()


@pytest.mark.parametrize(
    "version",
    [2.0, True, 1.0],
    ids=["float-2.0", "bool-True", "float-1.0"],
)
def test_check_schema_version_rejects_numbers_that_merely_equal_a_version(version):
    # `version not in compat` compares with `==`, which crosses numeric
    # types: `2.0 == 2` and `True == 1`. A JS producer emits `2.0`
    # naturally — JSON has one number type. A schema version is an
    # identifier, not a quantity, so the type has to match as well.
    from turbovec._persist import check_schema_version

    assert version in (1, 2)  # the equality that made this slip through
    with pytest.raises(ValueError, match="schema version"):
        check_schema_version(version, (1, 2), prefix="store.json has schema version")


@pytest.mark.parametrize("version", [1, 2])
def test_check_schema_version_accepts_the_real_versions(version):
    from turbovec._persist import check_schema_version

    check_schema_version(version, (1, 2), prefix="store.json has schema version")


def test_check_schema_version_still_rejects_unknown_and_non_numeric():
    from turbovec._persist import check_schema_version

    for bad in (99, "2", None, 0):
        with pytest.raises(ValueError, match="accepts versions"):
            check_schema_version(bad, (1, 2), prefix="store.json has schema version")
