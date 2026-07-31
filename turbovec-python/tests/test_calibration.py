"""TQ+ calibration lifecycle as seen from Python (#107/#284/#285/#303/#317).

An index warms up until it has seen 1000 vectors: the raw rows are
buffered and re-encoded once the threshold is crossed, so ingesting in
small batches ends up with the same fitted calibration a single bulk add
would produce.
"""

import copy
import os
import pickle
import sys
import warnings

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex
from turbovec._persist import atomic_save

DIM = 64


def _rand(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _recall_at_10(idx, data, queries):
    scores, indices = idx.search(queries, 10)
    exact = np.argsort(-(queries @ data.T), axis=1)[:, :10]
    hits = sum(len(set(indices[q]) & set(exact[q])) for q in range(len(queries)))
    return hits / (len(queries) * 10)


def test_calibration_state_transitions():
    idx = TurboQuantIndex(DIM, 4)
    assert idx.calibration_state == "warming_up"
    idx.add(_rand(500, seed=1))
    assert idx.calibration_state == "warming_up"
    idx.add(_rand(600, seed=2))
    assert idx.calibration_state == "fitted"

    # A file carries no warm-up buffer, so an index saved mid-warm-up
    # comes back committed to identity for good.
    warm = TurboQuantIndex(DIM, 4)
    warm.add(_rand(100, seed=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blob = warm.to_bytes()
    assert TurboQuantIndex.from_bytes(blob).calibration_state == "identity"


def test_id_map_calibration_state():
    idx = IdMapIndex(DIM, 4)
    assert idx.calibration_state == "warming_up"
    idx.add_with_ids(_rand(1200, seed=4), np.arange(1200, dtype=np.uint64))
    assert idx.calibration_state == "fitted"


def test_small_batch_ingestion_matches_bulk_recall():
    # #317: batches of 500 used to freeze identity calibration and lose
    # the TQ+ recall gain for the index's whole life.
    data = _rand(3000, seed=5)
    queries = _rand(50, seed=6)

    bulk = TurboQuantIndex(DIM, 2)
    bulk.add(data)

    batched = TurboQuantIndex(DIM, 2)
    for start in range(0, len(data), 500):
        batched.add(data[start : start + 500])

    assert len(batched) == len(bulk)
    assert batched.calibration_state == "fitted"
    assert _recall_at_10(batched, data, queries) >= _recall_at_10(bulk, data, queries) - 0.05


def test_saving_before_calibration_is_fitted_freezes_identity(tmp_path):
    # Saving mid-warm-up is the one way into the permanently-unfitted
    # state, so it is what the (one-shot per index) RuntimeWarning
    # flags; the durable signal is the reloaded index's state.
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(10, seed=7))
    assert idx.calibration_state == "warming_up"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        idx.write(str(tmp_path / "warm.tv"))
    reloaded = TurboQuantIndex.load(str(tmp_path / "warm.tv"))
    assert reloaded.calibration_state == "identity"
    assert len(reloaded) == 10


def test_drain_to_empty_then_add_keeps_calibration(tmp_path):
    # #284
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(1500, seed=8))
    assert idx.calibration_state == "fitted"
    while len(idx):
        idx.swap_remove(len(idx) - 1)
    idx.add(_rand(1500, seed=9))
    assert idx.calibration_state == "fitted"


def test_drained_warmup_index_survives_a_copy(tmp_path):
    # #418: draining a *warming-up* index to zero leaves the non-empty
    # identity pair its sub-threshold add committed. The live index stays
    # recoverable, but every serialization route — write/load, to_bytes,
    # and the copy.copy / pickle that every integration store's
    # __getstate__ takes — used to come back committed to identity, with
    # zero vectors and no way to ever fit a calibration again.
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(500, seed=418))
    assert idx.calibration_state == "warming_up"
    while len(idx):
        idx.swap_remove(len(idx) - 1)
    assert len(idx) == 0
    assert idx.calibration_state == "warming_up"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        idx.write(str(tmp_path / "drained.tv"))
        blob = idx.to_bytes()
        shallow = copy.copy(idx)
        unpickled = pickle.loads(pickle.dumps(idx))

    for name, back in (
        ("load", TurboQuantIndex.load(str(tmp_path / "drained.tv"))),
        ("from_bytes", TurboQuantIndex.from_bytes(blob)),
        ("copy.copy", shallow),
        ("pickle", unpickled),
    ):
        assert len(back) == 0, name
        assert back.calibration_state == "warming_up", name

    # The point of restoring warm-up: the next corpus gets a real fit.
    revived = TurboQuantIndex.from_bytes(blob)
    revived.add(_rand(1500, seed=419))
    assert revived.calibration_state == "fitted"


def test_drained_fitted_index_keeps_its_calibration_across_a_copy():
    # #284's half of the contract, which #418's fix must not disturb: a
    # drained *fitted* index carries a real fit in its trailer, not
    # identity, so it stays fitted through the same round trip.
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(1500, seed=420))
    assert idx.calibration_state == "fitted"
    while len(idx):
        idx.swap_remove(len(idx) - 1)
    back = TurboQuantIndex.from_bytes(idx.to_bytes())
    assert len(back) == 0
    assert back.calibration_state == "fitted"


def test_drained_warmup_id_map_survives_a_copy():
    # IdMapIndex is what the four integration stores hold, so this is the
    # shape "delete every document, then persist/copy the store" produces.
    idx = IdMapIndex(DIM, 4)
    ids = np.arange(500, dtype=np.uint64)
    idx.add_with_ids(_rand(500, seed=421), ids)
    assert idx.calibration_state == "warming_up"
    for i in ids:
        assert idx.remove(int(i))
    assert len(idx) == 0

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        back = copy.copy(idx)
    assert back.calibration_state == "warming_up"
    back.add_with_ids(_rand(1500, seed=422), np.arange(1500, dtype=np.uint64))
    assert back.calibration_state == "fitted"


def test_add_after_load_is_searchable():
    # #303: a loaded index plus a later add must keep every vector
    # reachable, not just counted by len().
    fresh = _rand(2000, seed=11)
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(1500, seed=10))
    idx = TurboQuantIndex.from_bytes(idx.to_bytes())
    idx.add(fresh)
    assert len(idx) == 3500
    _, indices = idx.search(fresh[:100], 1)
    hits = sum(1 for q in range(100) if indices[q][0] == 1500 + q)
    assert hits > 50, f"only {hits}/100 newly added vectors are reachable"


@pytest.mark.parametrize("cls", [TurboQuantIndex, IdMapIndex])
def test_calibration_state_is_read_only(cls):
    idx = cls(DIM, 4)
    with pytest.raises(AttributeError):
        idx.calibration_state = "fitted"


# The warm-up save warning calls `warnings.warn`, which dispatches through
# the filter chain into a user-replaceable `showwarning`. That is arbitrary
# Python, so it must not run while the binding holds the index lock: a
# handler that touches the same index would request the write lock on a
# thread already holding a read guard and wedge the interpreter (#360).
# Runs in a fresh interpreter with a hard timeout — a regression would
# hang pytest itself.
_REENTRANT_WARN_HANDLER = r'''
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
    idx.add(extra)          # re-enters the very index being serialized
    idx.swap_remove(0)


warnings.simplefilter("always")
warnings.showwarning = showwarning
payload = idx.to_bytes()
assert seen, "the warming-up save did not warn"
assert len(idx) == 14, len(idx)
assert len(payload) > 0
print("RESULT: PASS")
'''


def test_warning_handler_may_reenter_the_index():
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _REENTRANT_WARN_HANDLER],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the warm-up save warning DEADLOCKED: `warnings.warn` ran while the "
            "binding held the index read lock, so the handler's add blocked "
            "forever on the write lock (#360)"
        )
    assert "RESULT: PASS" in proc.stdout, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _warming(n=10, seed=100):
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_rand(n, seed=seed))
    assert idx.calibration_state == "warming_up"
    return idx


def test_warmup_warning_is_one_shot_per_index_not_per_process():
    # #360/#366: the latch used to be a process-global AtomicBool, so a
    # service holding one small store per tenant warned for the first
    # tenant and stayed silent for every later one — each losing TQ+
    # identically. Every index has to speak for itself.
    first, second = _warming(seed=101), _warming(seed=102)
    for idx in (first, second):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            idx.to_bytes()
        assert [w for w in caught if w.category is RuntimeWarning], (
            "a warming-up index did not warn on serialization — the one-shot "
            "latch is shared between indexes"
        )
    # It is still one-shot *within* one index, so a save loop cannot flood.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first.to_bytes()
        first.to_bytes()
    assert not [w for w in caught if w.category is RuntimeWarning], (
        "the same index warned twice"
    )


def test_error_filter_does_not_burn_the_warmup_latch():
    # #360: the latch was consumed *before* the warn was attempted, so a
    # filter that turned the warning into an error consumed it on a
    # warning the caller never received as a warning — and the index went
    # quiet for good. It must only be consumed by a warn that returned.
    idx = _warming(seed=103)
    unraisable = []
    prev_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # The save is already committed by the time the warn runs, so
            # the error is routed to sys.unraisablehook rather than
            # failing the serialization.
            idx.to_bytes()
    finally:
        sys.unraisablehook = prev_hook
    assert any(
        isinstance(u.exc_value, RuntimeWarning) for u in unraisable
    ), f"the error filter did not raise out of the warn: {unraisable}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        idx.to_bytes()
    assert [w for w in caught if w.category is RuntimeWarning], (
        "the index went silent after a warnings-as-errors save consumed the "
        "one-shot latch on a warning that was never delivered"
    )


def test_warmup_warning_is_attributed_to_the_caller_not_turbovec(tmp_path):
    # #366: `warnings.warn` from a Rust frame credits the nearest *Python*
    # frame, and every integration store saves through
    # `turbovec._persist.atomic_save` — so the warning pointed at a
    # turbovec internal the user never wrote, and keyed
    # `__warningregistry__` there too.
    idx = _warming(seed=104)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        atomic_save(idx, tmp_path / "warm.tv", {}, tmp_path / "warm.json")
    runtime = [w for w in caught if w.category is RuntimeWarning]
    assert runtime, "saving through atomic_save did not warn"
    assert os.path.basename(runtime[0].filename) == os.path.basename(__file__), (
        f"the warning is attributed to {runtime[0].filename}, not the caller's "
        f"own file — a stacklevel of 1 credits turbovec's own _persist module"
    )

    # A direct `write` from user code is still attributed to that call.
    direct = _warming(seed=105)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        direct.write(str(tmp_path / "direct.tv"))
    runtime = [w for w in caught if w.category is RuntimeWarning]
    assert runtime, "a direct write did not warn"
    assert os.path.basename(runtime[0].filename) == os.path.basename(__file__)


def test_warmup_warning_names_serialization_not_only_saving():
    # #366: the message also fires from `to_bytes`, which is the path
    # `pickle` and `copy.copy` take on every integration store, where
    # "saving an index" points the reader at a save call that does not
    # exist.
    idx = _warming(seed=106)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        idx.to_bytes()
    message = str([w for w in caught if w.category is RuntimeWarning][0].message)
    assert "serializing an index" in message, message
    assert "copying" in message, message


def test_id_map_warmup_warning_has_its_own_latch(tmp_path):
    a, b = IdMapIndex(DIM, 4), IdMapIndex(DIM, 4)
    for i, idx in enumerate((a, b)):
        idx.add_with_ids(_rand(10, seed=200 + i), np.arange(10, dtype=np.uint64))
        assert idx.calibration_state == "warming_up"
    for i, idx in enumerate((a, b)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            idx.write(str(tmp_path / f"warm{i}.tvim"))
        assert [w for w in caught if w.category is RuntimeWarning], (
            f"IdMapIndex {i} did not warn — the latch is shared"
        )
