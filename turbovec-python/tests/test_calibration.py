"""TQ+ calibration as seen from Python.

A calibration comes from exactly one place: the sample the caller hands
to ``calibrate``. The index never fits one on its own — an index that is
never calibrated is plain TurboQuant, order-independent by construction,
and reports ``"uncalibrated"``. ``calibrate`` may be called at any time,
including on a populated index, which re-encodes every stored row under
the new pair.
"""

import copy
import pickle

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
    assert idx.calibration_state == "uncalibrated"

    # Adds of any size never change the state — 1000 was the old
    # implicit-fit threshold, so straddle it deliberately.
    idx.add(_rand(999, seed=1))
    assert idx.calibration_state == "uncalibrated"
    idx.add(_rand(500, seed=2))
    assert idx.calibration_state == "uncalibrated"

    # The one transition.
    idx.calibrate(_rand(1024, seed=3))
    assert idx.calibration_state == "calibrated"

    # And it sticks: adds and removes do not unfit.
    idx.add(_rand(10, seed=4))
    idx.swap_remove(0)
    assert idx.calibration_state == "calibrated"


def test_id_map_calibration_state():
    idx = IdMapIndex(DIM, 4)
    assert idx.calibration_state == "uncalibrated"
    idx.calibrate(_rand(1024, seed=5))
    assert idx.calibration_state == "calibrated"
    idx.add_with_ids(_rand(50, seed=6), np.arange(50, dtype=np.uint64))
    assert idx.calibration_state == "calibrated"


def test_calibrate_then_add_beats_or_matches_uncalibrated():
    data = _rand(5000, seed=7)
    queries = _rand(64, seed=8)

    plain = TurboQuantIndex(DIM, 2)
    plain.add(data)
    plain_recall = _recall_at_10(plain, data, queries)

    calibrated = TurboQuantIndex(DIM, 2)
    calibrated.calibrate(data[np.random.default_rng(9).permutation(5000)[:1024]])
    calibrated.add(data)
    calibrated_recall = _recall_at_10(calibrated, data, queries)

    # On i.i.d. Gaussian data the TQ+ gain is modest; the invariant worth
    # pinning from Python is that calibration never makes things worse
    # than materially so.
    assert calibrated_recall >= plain_recall - 0.02, (
        f"calibrated {calibrated_recall:.3f} vs uncalibrated {plain_recall:.3f}"
    )


def test_refit_on_a_populated_index_keeps_rows_searchable():
    data = _rand(2000, seed=10)
    idx = TurboQuantIndex(DIM, 4)
    idx.add(data)
    assert idx.calibration_state == "uncalibrated"

    idx.calibrate(_rand(1024, seed=11))
    assert idx.calibration_state == "calibrated"
    assert len(idx) == 2000

    # Every probed row still finds itself.
    for row in (0, 7, 1999):
        _, indices = idx.search(data[row : row + 1], 1)
        assert indices[0, 0] == row

    # Refitting under the identical sample reproduces the codes exactly
    # (the fixed point the refit rests on); the per-vector scales may
    # move by one ulp, so pin behaviour rather than bytes: every search
    # answer is unchanged.
    q = _rand(32, seed=99)
    before = idx.search(q, 10)
    idx.calibrate(_rand(1024, seed=11))
    after = idx.search(q, 10)
    assert np.array_equal(before[1], after[1])


def test_id_map_refit_keeps_ids_resolving():
    data = _rand(500, seed=12)
    ids = (np.arange(500, dtype=np.uint64) * 7) + 3
    idx = IdMapIndex(DIM, 4)
    idx.add_with_ids(data, ids)
    idx.calibrate(_rand(1024, seed=13))
    assert idx.calibration_state == "calibrated"
    for row in (0, 123, 499):
        _, found = idx.search(data[row : row + 1], 1)
        assert found[0, 0] == ids[row]


def test_calibrate_errors_are_value_errors():
    idx = TurboQuantIndex(DIM, 4)
    # One row is below the fit floor.
    with pytest.raises(ValueError):
        idx.calibrate(_rand(1, seed=14))
    # Wrong dim on a committed index.
    wide = np.zeros((1024, DIM * 2), np.float32) + 0.5
    with pytest.raises(ValueError):
        idx.calibrate(wide)
    # Non-finite sample.
    bad = _rand(1024, seed=15)
    bad[3, 5] = np.nan
    with pytest.raises(ValueError):
        idx.calibrate(bad)
    # None of the rejections committed anything.
    assert idx.calibration_state == "uncalibrated"


def test_both_states_round_trip_through_persistence(tmp_path):
    data = _rand(1500, seed=16)

    plain = TurboQuantIndex(DIM, 4)
    plain.add(data)
    p = tmp_path / "plain.tv"
    plain.write(str(p))
    assert TurboQuantIndex.load(str(p)).calibration_state == "uncalibrated"
    assert pickle.loads(pickle.dumps(plain)).calibration_state == "uncalibrated"

    fitted = TurboQuantIndex(DIM, 4)
    fitted.calibrate(_rand(1024, seed=17))
    fitted.add(data)
    f = tmp_path / "fitted.tv"
    fitted.write(str(f))
    loaded = TurboQuantIndex.load(str(f))
    assert loaded.calibration_state == "calibrated"
    assert copy.deepcopy(fitted).calibration_state == "calibrated"
    # The reloaded index answers identically.
    q = _rand(8, seed=18)
    assert np.array_equal(loaded.search(q, 5)[1], fitted.search(q, 5)[1])


def test_drain_to_empty_keeps_the_calibration():
    idx = TurboQuantIndex(DIM, 4)
    idx.calibrate(_rand(1024, seed=19))
    idx.add(_rand(20, seed=20))
    while len(idx):
        idx.swap_remove(len(idx) - 1)
    assert idx.calibration_state == "calibrated"
    clone = pickle.loads(pickle.dumps(idx))
    assert clone.calibration_state == "calibrated"


def test_uncalibrated_ingestion_is_batch_shape_independent():
    data = _rand(3000, seed=21)
    bulk = TurboQuantIndex(DIM, 4)
    bulk.add(data)
    dripped = TurboQuantIndex(DIM, 4)
    for start in range(0, 3000, 250):
        dripped.add(data[start : start + 250])
    assert bulk.to_bytes() == dripped.to_bytes()


def test_calibrated_ingestion_is_batch_shape_independent():
    data = _rand(3000, seed=22)
    sample = _rand(1024, seed=23)
    bulk = TurboQuantIndex(DIM, 4)
    bulk.calibrate(sample)
    bulk.add(data)
    dripped = TurboQuantIndex(DIM, 4)
    dripped.calibrate(sample)
    for start in range(0, 3000, 250):
        dripped.add(data[start : start + 250])
    assert bulk.to_bytes() == dripped.to_bytes()


@pytest.mark.parametrize("cls", [TurboQuantIndex, IdMapIndex])
def test_calibration_state_is_read_only(cls):
    idx = cls(DIM, 4)
    with pytest.raises(AttributeError):
        idx.calibration_state = "calibrated"
