"""TQ+ calibration lifecycle as seen from Python (#107/#284/#285/#303/#317).

An index warms up until it has seen 1000 vectors: the raw rows are
buffered and re-encoded once the threshold is crossed, so ingesting in
small batches ends up with the same fitted calibration a single bulk add
would produce.
"""

import warnings

import numpy as np
import pytest

from turbovec import IdMapIndex, TurboQuantIndex

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
    # state, so it is what the (one-shot, process-wide) RuntimeWarning
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
