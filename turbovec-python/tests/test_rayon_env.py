"""RAYON_NUM_THREADS handling at module init (issue #158).

Before the fix, a value above the OS thread limit (``ulimit -u``) made
rayon's lazy global-pool construction fail with EAGAIN, surfacing as an
uncatchable ``PanicException`` ("The global thread pool has not been
initialized") on the first ``add``/``search``. The module now clamps an
explicit request to 4x the available parallelism at import time.

The rayon pool is process-global and built at most once, so every
scenario runs in a fresh subprocess.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

# The probe value must exceed the clamp (4x cores) for the clamp path to
# engage; on any realistic machine/CI runner it does.
_HUGE = "100000"
_CAP = 4 * (os.cpu_count() or 1)

_ADD_SEARCH_PROBE = """
import numpy as np, turbovec
rng = np.random.default_rng(7)
idx = turbovec.TurboQuantIndex(dim=16)
idx.add(rng.random((64, 16), dtype=np.float32))
scores, ids = idx.search(rng.random((2, 16), dtype=np.float32), 3)
print("OK", ids.tolist())
"""

_WARNINGS_PROBE = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import turbovec
print([str(w.message) for w in caught])
"""


def _run(probe: str, rayon_num_threads: str | None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "RAYON_NUM_THREADS"}
    if rayon_num_threads is not None:
        env["RAYON_NUM_THREADS"] = rayon_num_threads
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_huge_rayon_num_threads_does_not_panic():
    # Clean behavior contract: either the clamped pool works, or a
    # clear error names RAYON_NUM_THREADS — never a PanicException.
    # (Whether the *unfixed* build panics here depends on the OS thread
    # limit; the fixed build never attempts the oversized spawn at all.)
    res = _run(_ADD_SEARCH_PROBE, _HUGE)
    assert "PanicException" not in res.stderr, res.stderr
    assert "panicked" not in res.stderr, res.stderr
    if res.returncode == 0:
        assert "OK" in res.stdout
    else:
        assert "RAYON_NUM_THREADS" in (res.stderr + res.stdout)


@pytest.mark.skipif(
    _CAP >= int(_HUGE),
    reason="machine so wide the probe value is under the clamp",
)
def test_huge_rayon_num_threads_warns_naming_the_variable():
    res = _run(_WARNINGS_PROBE, _HUGE)
    assert res.returncode == 0, res.stderr
    assert "RAYON_NUM_THREADS" in res.stdout, res.stdout


def test_small_rayon_num_threads_matches_unset_results():
    # Pin: values rayon could always honor stay byte-identical with the
    # eager-init path (same thread count rayon's lazy path would pick).
    base = _run(_ADD_SEARCH_PROBE, None)
    two = _run(_ADD_SEARCH_PROBE, "2")
    assert base.returncode == 0, base.stderr
    assert two.returncode == 0, two.stderr
    assert base.stdout == two.stdout


def test_unset_and_zero_emit_no_warning():
    # Pin: unset and "0" (rayon's "auto") take the untouched lazy path.
    for value in (None, "0"):
        res = _run(_WARNINGS_PROBE, value)
        assert res.returncode == 0, res.stderr
        assert "RAYON_NUM_THREADS" not in res.stdout, (value, res.stdout)
