"""Fork safety (issue #147).

rayon's thread pool does not survive ``fork()``: its worker threads live
only in the parent, so the first parallel op a forked child runs injects
work into a worker-less registry and blocks on the latch forever. That is
the default failure mode of ``multiprocessing`` (fork start method — the
Linux default through 3.13), gunicorn ``--preload``, Celery prefork, and
PyTorch ``DataLoader(num_workers>0)``.

The binding routes every rayon-using kernel entry through a process-local,
fork-aware pool (``with_pool`` in ``lib.rs``): a forked child detects the
fork (via ``os.register_at_fork`` / ``pthread_atfork``) and transparently
rebuilds the pool, so its parallel ops run on live workers.

**These tests are only meaningful on Linux.** Fork/pool interaction is
glibc/futex-specific; macOS uses psynch pthreads and defaults
``multiprocessing`` to spawn. The suite runs on all platforms (spawn and
the no-op cases are portable) but the fork-specific cases are the gate and
must pass on the Linux CI leg. See the PR for the oldest-glibc
(manylinux2014 / glibc 2.17) caveat.

Every scenario runs in a **fresh subprocess** with a hard wall-clock
timeout: a regression makes the child hang, the subprocess is killed, and
the test FAILS — the pytest process itself never hangs.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

_HAS_FORK = hasattr(os, "fork")
_IS_LINUX = sys.platform.startswith("linux")

# Self-contained harness executed in a fresh interpreter per scenario. It
# selects a scenario from SCENARIO and prints exactly one terminal line:
#   RESULT: PASS <detail>   or   RESULT: FAIL <reason>
# Every child wait is bounded (select + SIGKILL) so the harness cannot hang
# even if the fix regresses; the outer subprocess timeout is the backstop.
_HARNESS = r'''
import os, sys, select, pickle, time

DIM = 96
DEADLINE = float(os.environ.get("CHILD_DEADLINE", "12"))


def _np():
    import numpy as np
    return np


def make_vecs(n, seed):
    np = _np()
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def build_index(n=5000, seed=0):
    import turbovec
    idx = turbovec.TurboQuantIndex(dim=DIM)
    idx.add(make_vecs(n, seed))
    return idx


def run_forked(fn, deadline=DEADLINE):
    """Run fn() in a forked child; return ('ok', pickled_result) | ('hang',) |
    ('err', text). Bounded by deadline with SIGKILL so the parent never hangs."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        try:
            payload = pickle.dumps(("ok", fn()))
        except BaseException as e:  # noqa: BLE001
            payload = pickle.dumps(("err", "%s: %s" % (type(e).__name__, e)))
        try:
            os.write(w, payload)
        finally:
            os.close(w)
            os._exit(0)
    os.close(w)
    ready, _, _ = select.select([r], [], [], deadline)
    if not ready:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        os.close(r)
        return ("hang",)
    buf = b""
    while True:
        chunk = os.read(r, 1 << 20)
        if not chunk:
            break
        buf += chunk
    os.waitpid(pid, 0)
    os.close(r)
    return pickle.loads(buf)


def emit(ok, detail):
    print("RESULT: %s %s" % ("PASS" if ok else "FAIL", detail), flush=True)


# --------------------------------------------------------------------------
def scenario_probe():
    """Primed parent; a fresh fork per op. Every op must succeed in the child:
    the split boundary (nq/n > 1) is exactly where the baseline hangs."""
    np = _np()
    idx = build_index()
    q1 = make_vecs(1, 1); q64 = make_vecs(64, 2); q512 = make_vecs(512, 3)
    ops = [
        ("search nq=1", lambda: int(idx.search(q1, k=5)[0].shape[0])),
        ("search nq=64", lambda: int(idx.search(q64, k=5)[0].shape[0])),
        ("search nq=512", lambda: int(idx.search(q512, k=5)[0].shape[0])),
        ("add 1", lambda: (idx.add(q1), "ok")[1]),
        ("add 64", lambda: (idx.add(q64), "ok")[1]),
        ("fresh-index add 64", lambda: (build_index(64, 9), "ok")[1]),
    ]
    for name, fn in ops:
        res = run_forked(fn)
        if res[0] != "ok":
            return emit(False, "child op %r -> %r" % (name, res))
    emit(True, "all 6 child ops ok")


def scenario_correctness():
    """Inherited-index search must be bit-identical parent vs child vs
    grandchild (the search path has no thread-count-dependent FP reduction)."""
    np = _np()
    idx = build_index(10000, 7)
    q = make_vecs(32, 11)
    p_scores, p_ids = idx.search(q, k=10)

    def child_search():
        s, i = idx.search(q, k=10)
        return s, i

    def grandchild_search():
        idx.search(q, k=10)  # child rebuilds its pool first
        return run_forked(child_search, deadline=DEADLINE)

    r = run_forked(child_search)
    if r[0] != "ok":
        return emit(False, "child search -> %r" % (r,))
    s, i = r[1]
    if not (np.array_equal(s, p_scores) and np.array_equal(i, p_ids)):
        return emit(False, "child results differ from parent")

    r = run_forked(grandchild_search)
    if r[0] != "ok":
        return emit(False, "grandchild outer -> %r" % (r,))
    inner = r[1]
    if inner[0] != "ok":
        return emit(False, "grandchild search -> %r" % (inner,))
    gs, gi = inner[1]
    if not (np.array_equal(gs, p_scores) and np.array_equal(gi, p_ids)):
        return emit(False, "grandchild results differ from parent")

    s2, i2 = idx.search(q, k=10)  # parent still healthy afterward
    if not (np.array_equal(s2, p_scores) and np.array_equal(i2, p_ids)):
        return emit(False, "parent unstable after forks")
    emit(True, "parent/child/grandchild bit-identical; parent stable")


def scenario_osfork_fresh():
    """os.fork; child builds a brand-new index (the baseline b2 HANG)."""
    build_index()  # prime the parent

    def child():
        f = build_index(200, 5)
        return int(f.search(make_vecs(1, 6), k=5)[0].shape[0])

    r = run_forked(child)
    emit(r[0] == "ok", "child fresh add+search -> %r" % (r,))


def scenario_fork_before_use():
    """Fork after import but before any turbovec op; child does add+search."""
    import turbovec  # noqa: F401

    def child():
        return int(build_index(200, 4).search(make_vecs(1, 4), k=5)[0].shape[0])

    r = run_forked(child)
    emit(r[0] == "ok", "child -> %r" % (r,))


def scenario_eager_env():
    """RAYON_NUM_THREADS is set (the #158 eager-init path); parent only
    imports, then forks. Baseline scenario (e) HANGs here."""
    assert os.environ.get("RAYON_NUM_THREADS"), "driver must set RAYON_NUM_THREADS"
    import turbovec  # noqa: F401

    def child():
        return int(build_index(200, 4).search(make_vecs(1, 4), k=5)[0].shape[0])

    r = run_forked(child)
    emit(r[0] == "ok", "child (RAYON_NUM_THREADS set) -> %r" % (r,))


def scenario_child_nq1_after_rebuild():
    """In a forked child, once a splitting op has rebuilt the pool,
    `in_forked_child()` is false again and single-query searches go back to
    the inline path (against the child's own live pool / the dead inherited
    sentinel). That inline nq==1 must still work and return the same neighbors
    as the parent — this covers the 'nq==1 inline after rebuild in a child'
    path that the guarded bypass does NOT force through install."""
    np = _np()
    idx = build_index(5000, 7)
    q1 = make_vecs(1, 3)
    parent_s, parent_i = idx.search(q1, k=5)  # parent inline nq==1

    def child():
        idx.search(make_vecs(64, 2), k=5)  # nq=64 -> rebuilds the child pool
        # now in_forked_child() is false again; this nq==1 takes the inline path
        s, i = idx.search(q1, k=5)
        return s, i

    r = run_forked(child)
    if r[0] != "ok":
        return emit(False, "child -> %r" % (r,))
    s, i = r[1]
    if not np.array_equal(i, parent_i):
        return emit(False, "post-rebuild inline nq==1 neighbors differ from parent")
    if not np.allclose(s, parent_s, rtol=1e-4, atol=1e-4):
        return emit(False, "post-rebuild inline nq==1 scores diverge from parent")
    emit(True, "child inline nq==1 after rebuild ok and matches parent")


def scenario_gunicorn_proxy():
    """gunicorn --preload proxy: build the index in the 'master', fork a
    'worker', and have the worker serve a batch search (nq=8) AND an add —
    both of which wedge a forked worker on the baseline."""
    np = _np()
    master = build_index(20000, 3)
    n0 = len(master)
    q8 = make_vecs(8, 21)

    def worker():
        s, i = master.search(q8, k=5)  # batch search (nq>1) in the worker
        master.add(make_vecs(100, 22))  # write in the worker
        return int(i.shape[0]), len(master)

    r = run_forked(worker)
    if r[0] != "ok":
        return emit(False, "worker -> %r" % (r,))
    nq, n1 = r[1]
    ok = nq == 8 and n1 == n0 + 100
    emit(ok, "worker batch+add ok (nq=%d, n %d->%d)" % (nq, n0, n1))


def scenario_mp(method, inherited):
    import multiprocessing as mp
    if method not in mp.get_all_start_methods():
        return emit(True, "start method %r unavailable; skipped" % method)
    ctx = mp.get_context(method)
    q = ctx.Queue()
    if inherited:
        globals()["_SHARED"] = build_index()  # inherited via fork only
        p = ctx.Process(target=_mp_inherited, args=(q,))
    else:
        p = ctx.Process(target=_mp_fresh, args=(q,))
    p.start()
    p.join(DEADLINE)
    if p.is_alive():
        p.kill(); p.join()
        return emit(False, "%s worker hang" % method)
    try:
        msg = q.get_nowait()
    except Exception:
        msg = "exitcode=%s" % p.exitcode
    emit(not str(msg).startswith("err:"), "%s worker -> %s" % (method, msg))


def _mp_inherited(q):
    try:
        idx = globals()["_SHARED"]
        q.put("ok:%d" % idx.search(make_vecs(64, 1), k=5)[0].shape[0])
    except BaseException as e:  # noqa: BLE001
        q.put("err:%s: %s" % (type(e).__name__, e))


def _mp_fresh(q):
    try:
        f = build_index(200, 5)
        q.put("ok:%d" % f.search(make_vecs(1, 6), k=5)[0].shape[0])
    except BaseException as e:  # noqa: BLE001
        q.put("err:%s: %s" % (type(e).__name__, e))


SCENARIOS = {
    "probe": scenario_probe,
    "correctness": scenario_correctness,
    "osfork_fresh": scenario_osfork_fresh,
    "fork_before_use": scenario_fork_before_use,
    "eager_env": scenario_eager_env,
    "gunicorn_proxy": scenario_gunicorn_proxy,
    "child_nq1_after_rebuild": scenario_child_nq1_after_rebuild,
    "mp_fork_inherited": lambda: scenario_mp("fork", True),
    "mp_fork_fresh": lambda: scenario_mp("fork", False),
    "mp_spawn_fresh": lambda: scenario_mp("spawn", False),
}

if __name__ == "__main__":
    SCENARIOS[os.environ["SCENARIO"]]()
'''


def _run(scenario: str, *, env: dict | None = None, timeout: int = 40) -> str:
    """Run one harness scenario in a fresh interpreter; return its stdout.

    A hang shows up as a subprocess ``TimeoutExpired`` -> the child is killed
    and the test fails loudly, rather than the pytest process wedging.
    """
    full_env = dict(os.environ)
    full_env["SCENARIO"] = scenario
    full_env.setdefault("CHILD_DEADLINE", str(max(5, timeout - 8)))
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _HARNESS],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        out = out.decode() if isinstance(out, bytes) else out
        pytest.fail(
            f"scenario {scenario!r} HUNG (killed after {timeout}s) — a forked "
            f"child deadlocked in rayon. Partial output:\n{out}"
        )
    assert "RESULT: PASS" in proc.stdout, (
        f"scenario {scenario!r} did not pass.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc.stdout


# --- multiprocessing (portable: spawn everywhere, fork where available) ----

def test_mp_spawn_fresh():
    """spawn workers get a clean interpreter — must always work."""
    _run("mp_spawn_fresh")


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_mp_fork_inherited():
    """fork worker searching an inherited (preloaded) index."""
    _run("mp_fork_inherited")


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_mp_fork_fresh():
    """fork worker building a fresh index — baseline a2 HANG."""
    _run("mp_fork_fresh")


# --- os.fork scenarios (the gate; fork-only) -------------------------------

@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_child_probe_ops():
    """nq=1/64/512 search, add 1/64, fresh-index add — all must survive fork."""
    _run("probe")


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_inherited_index_bit_identical():
    """Inherited-index search results identical across parent/child/grandchild."""
    _run("correctness")


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_osfork_fresh_index():
    """os.fork; child builds a brand-new index — baseline b2 HANG."""
    _run("osfork_fresh")


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_fork_before_first_use():
    """Fork before the parent's first turbovec op; child add+search."""
    _run("fork_before_use")


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_eager_init_rayon_env_path():
    """RAYON_NUM_THREADS set + import-only parent + fork — baseline (e) HANG."""
    _run("eager_env", env={"RAYON_NUM_THREADS": "4"})


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_gunicorn_preload_proxy():
    """gunicorn --preload proxy: forked worker serves a batch search AND an
    add — the flagship pattern that only the pool rebuild makes work."""
    _run("gunicorn_proxy")


@pytest.mark.skipif(not _HAS_FORK, reason="fork() unavailable on this platform")
def test_child_nq1_inline_after_rebuild():
    """A child's single-query search AFTER its pool has been rebuilt takes the
    inline path again and must stay correct (review finding F5-adjacent)."""
    _run("child_nq1_after_rebuild")


# --- chokepoint completeness (review finding F5) ---------------------------

# rayon parallel-iterator / entry tokens, matched in call form so prose
# mentions in comments do not trip the guard. Deliberately broad so it
# catches *future* rayon surface too, not just the calls used today:
#   - `.par_<anything>(`  -> every ParallelIterator/ParallelSlice adaptor
#     (par_iter, par_chunks_exact_mut, par_rchunks, par_split,
#      par_sort_by_cached_key, par_bridge, par_extend, ...)
#   - `into_par_iter(`    -> method OR UFCS `IntoParallelIterator::into_par_iter(`
#   - `rayon::<lower>(`   -> free functions (join, scope, spawn, broadcast,
#     in_place_scope, ...); PascalCase paths like `rayon::ThreadPoolBuilder`
#     are intentionally NOT matched (they are pool construction, not work
#     submission, and the fork-safe pool is built through them on purpose).
_RAYON_CALL = re.compile(r"\.par_[a-z_]*\s*\(|\binto_par_iter\s*\(|\brayon::[a-z_]+\s*\(")

# The ONLY core source files allowed to contain rayon parallelism. Every
# site in them is reachable exclusively through the six `with_pool`-wrapped
# Python entries (add, add_with_ids, search x2, prepare x2). If a rayon call
# appears in any other file, a new un-chokepointed parallel site has been
# introduced: route it through `with_pool` and, if it legitimately belongs
# in a new file, add that file here in the same change.
_RAYON_ALLOWED_FILES = {"encode.rs", "search.rs"}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def test_rayon_chokepoint_completeness():
    """Fail if a rayon parallel site appears outside the audited chokepoint
    files — the standing backstop for chokepoint completeness (F5)."""
    root = _repo_root()
    src_dirs = [root / "turbovec" / "src", root / "turbovec-python" / "src"]
    offenders: dict[str, list[str]] = {}
    scanned = 0
    for d in src_dirs:
        assert d.is_dir(), f"expected source dir missing: {d}"
        for f in sorted(d.rglob("*.rs")):
            scanned += 1
            hits = []
            for lineno, raw in enumerate(f.read_text().splitlines(), 1):
                code = raw.split("//", 1)[0]  # drop line comments
                if _RAYON_CALL.search(code):
                    hits.append(f"{lineno}: {raw.strip()}")
            if hits and f.name not in _RAYON_ALLOWED_FILES:
                offenders[str(f.relative_to(root))] = hits
    assert scanned > 0, "no Rust sources scanned — path layout changed?"
    assert not offenders, (
        "rayon parallel site(s) found outside the fork-safe chokepoint "
        f"({sorted(_RAYON_ALLOWED_FILES)}). Route each through `with_pool` "
        "and update _RAYON_ALLOWED_FILES if a new file is intended:\n"
        + "\n".join(f"  {f}\n    " + "\n    ".join(h) for f, h in offenders.items())
    )


def test_audited_files_still_have_rayon():
    """Guard against silent drift the other way: if the audited files stop
    containing rayon calls, the allowlist is stale and must be revisited."""
    root = _repo_root()
    for name in _RAYON_ALLOWED_FILES:
        matches = list((root / "turbovec" / "src").rglob(name))
        assert matches, f"audited file {name} not found"
        text = matches[0].read_text()
        code = "\n".join(ln.split("//", 1)[0] for ln in text.splitlines())
        assert _RAYON_CALL.search(code), (
            f"{name} no longer contains rayon calls; the chokepoint allowlist "
            "is stale — re-audit and update _RAYON_ALLOWED_FILES."
        )


# --- nq==1 no-split assumption (review findings F8/F9) ---------------------

def test_nq1_inline_matches_pooled():
    """The nq==1 inline fast path must return the SAME nearest neighbors as
    the pooled multi-query path. This guards the (rayon-internal,
    undocumented) 'a single query never splits into the pool' assumption the
    inline bypass relies on: if a future rayon reshaped splitting so the
    single-query path diverged, this fails instead of drifting silently
    (F8/F9).

    Neighbor IDs must be bit-identical. Scores are compared with a tolerance:
    turbovec's SIMD-blocked scoring groups queries differently at nq==1 vs
    nq>1, so the last ~ULP of the float32 score legitimately differs by batch
    size — a pre-existing kernel property, independent of the pool path."""
    import numpy as np
    import turbovec

    rng = np.random.default_rng(123)
    idx = turbovec.TurboQuantIndex(dim=48)
    idx.add(rng.standard_normal((2000, 48)).astype(np.float32))
    queries = rng.standard_normal((5, 48)).astype(np.float32)

    # Pooled path: all 5 queries at once (nq>1 -> install).
    batch_scores, batch_ids = idx.search(queries, k=7)
    # Inline path: each query on its own (nq==1 -> inline bypass).
    for j in range(queries.shape[0]):
        s, i = idx.search(queries[j : j + 1], k=7)
        assert np.array_equal(i, batch_ids[j : j + 1]), f"neighbor ids differ at q{j}"
        assert np.allclose(s, batch_scores[j : j + 1], rtol=1e-4, atol=1e-4), (
            f"scores diverge beyond float32 SIMD-blocking noise at q{j}"
        )


@pytest.mark.skipif(not _IS_LINUX, reason="thread census via /proc is Linux-only")
def test_nq1_does_not_grow_thread_pool():
    """Direct no-split check: repeated single-query searches must not spawn
    new OS threads. If nq==1 started entering/expanding a pool, the process
    thread count would jump (the D8-report symptom the sentinel prevents)."""
    import numpy as np
    import turbovec

    def n_threads() -> int:
        return len(list(pathlib.Path("/proc/self/task").iterdir()))

    rng = np.random.default_rng(7)
    idx = turbovec.TurboQuantIndex(dim=32)
    idx.add(rng.standard_normal((3000, 32)).astype(np.float32))  # builds the pool
    q = rng.standard_normal((1, 32)).astype(np.float32)
    idx.search(q, k=5)  # settle any one-time lazy init
    before = n_threads()
    for _ in range(200):
        idx.search(q, k=5)
    after = n_threads()
    assert after <= before, f"thread count grew {before} -> {after} on nq==1 searches"
