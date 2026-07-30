"""Python object model of the two native index classes (#340, #345).

Covers what an index looks like to generic Python machinery rather than
to a caller who already knows the API: where the class says it lives,
whether it survives `pickle` / `copy` / a `spawn` process boundary,
whether it can be weakly referenced, and whether `inspect.signature`
reports the parameters the methods actually accept.

`pickle` / `copy` of the four *integration stores* is covered by
test_pickle_copy.py; this file is about the bare index.
"""

from __future__ import annotations

import copy
import inspect
import pickle
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest

import turbovec
from turbovec import IdMapIndex, TurboQuantIndex

DIM = 32
# Above the 1000-vector TQ+ sample threshold, so `calibration_state` is
# `fitted` and serialization carries a real calibration (a below-threshold
# index is a separate, documented case — see the warm-up test below).
FITTED = 1200


def _vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _tq(n: int = FITTED, seed: int = 1) -> TurboQuantIndex:
    idx = TurboQuantIndex(DIM, 4)
    idx.add(_vectors(n, seed))
    return idx


def _idmap(n: int = FITTED, seed: int = 2) -> IdMapIndex:
    idx = IdMapIndex(DIM, 4)
    idx.add_with_ids(_vectors(n, seed), np.arange(n, dtype=np.uint64))
    return idx


def _builders():
    return [pytest.param(_tq, id="TurboQuantIndex"), pytest.param(_idmap, id="IdMapIndex")]


# --------------------------------------------------------------------
# __module__ (#340)
# --------------------------------------------------------------------


@pytest.mark.parametrize("cls", [TurboQuantIndex, IdMapIndex])
def test_class_names_its_real_module(cls):
    """`__module__` must name the extension module the class lives in.

    It read `builtins` before, so anything recording
    `f"{cls.__module__}.{cls.__name__}"` — a framework `to_dict`, a
    spawn payload, a Sphinx cross-reference — stored a path that
    resolves nowhere.
    """
    assert cls.__module__ == "turbovec._turbovec"
    # And the recorded path actually resolves back to the class.
    import importlib

    module = importlib.import_module(cls.__module__)
    assert getattr(module, cls.__name__) is cls


@pytest.mark.parametrize("cls", [TurboQuantIndex, IdMapIndex])
def test_class_reference_is_picklable(cls):
    """A *class reference* (not an instance) must round-trip, which is
    what `multiprocessing` spawn payloads and task graphs carry."""
    assert pickle.loads(pickle.dumps(cls)) is cls


@pytest.mark.parametrize("cls", [TurboQuantIndex, IdMapIndex])
def test_repr_agrees_with_module_path(cls):
    """`__repr__` has always hard-coded `turbovec.<Class>`; the class's
    own module path must not contradict it."""
    idx = cls(DIM, 4)
    assert repr(idx).startswith(f"turbovec.{cls.__name__}(")
    assert cls.__module__.startswith("turbovec.")


# --------------------------------------------------------------------
# pickle / copy (#340)
# --------------------------------------------------------------------


def _roundtrips():
    return [
        pytest.param(lambda o: pickle.loads(pickle.dumps(o)), id="pickle"),
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
    ]


@pytest.mark.parametrize("build", _builders())
@pytest.mark.parametrize("roundtrip", _roundtrips())
def test_index_survives_pickle_and_copy(build, roundtrip):
    """`__reduce__` makes an index crossable by pickle and copy, so it
    can reach a `spawn` child (the default start method on macOS and
    Windows) and a container holding one can be deep-copied."""
    original = build()
    queries = _vectors(4, seed=99)
    want_scores, want_ids = original.search(queries, 5)

    clone = roundtrip(original)

    assert type(clone) is type(original)
    assert len(clone) == len(original)
    assert clone.dim == original.dim
    assert clone.bit_width == original.bit_width
    assert clone.calibration_state == original.calibration_state
    got_scores, got_ids = clone.search(queries, 5)
    np.testing.assert_array_equal(got_ids, want_ids)
    np.testing.assert_array_equal(got_scores, want_scores)


@pytest.mark.parametrize("build", _builders())
@pytest.mark.parametrize("roundtrip", _roundtrips())
def test_copy_is_independent_of_the_original(build, roundtrip):
    """Every clone owns its own index — a mutation must not bleed across
    (the failure mode #149 hit on the integration stores)."""
    original = build()
    clone = roundtrip(original)
    assert clone is not original

    before = len(original)
    if isinstance(clone, TurboQuantIndex):
        clone.add(_vectors(10, seed=77))
    else:
        clone.add_with_ids(
            _vectors(10, seed=77),
            np.arange(before, before + 10, dtype=np.uint64),
        )
    assert len(original) == before
    assert len(clone) == before + 10


@pytest.mark.parametrize("build", _builders())
def test_reduce_goes_through_the_public_bytes_pair(build):
    """`__reduce__` reduces to `from_bytes(to_bytes())` — the documented
    serialization path — rather than a private back door."""
    original = build()
    factory, args = original.__reduce__()

    assert factory == type(original).from_bytes
    (payload,) = args
    assert isinstance(payload, bytes)
    assert payload == original.to_bytes()
    rebuilt = factory(*args)
    assert type(rebuilt) is type(original)
    assert len(rebuilt) == len(original)


def test_pickling_a_warming_up_index_still_warns():
    """Pickle inherits the `to_bytes` persistence contract exactly,
    including the warning that a below-threshold index reloads committed
    to identity calibration (#366).

    `__reduce__` must not route around that warning: a silent downgrade
    is the whole hazard.

    Run in a subprocess so the assertion does not depend on the ambient
    filter configuration: a bare interpreter is immune to whatever
    `filterwarnings` pytest has installed and to any `-W` on the session.

    Note it is NOT needed to dodge the per-index latch (#360) or CPython's
    `__warningregistry__`. The latch is per-index, so this fresh index
    warns whatever earlier tests saved; and the assertion runs under
    `catch_warnings` + `simplefilter("always")`, which bumps
    `warnings._filters_version` and so makes `warn_explicit` clear the
    caller module's registry before consulting it. Neither can suppress
    this delivery.
    """
    script = textwrap.dedent(
        """
        import pickle, warnings
        import numpy as np
        from turbovec import TurboQuantIndex

        idx = TurboQuantIndex(32, 4)
        idx.add(np.zeros((16, 32), np.float32) + 0.5)
        assert idx.calibration_state == "warming_up"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clone = pickle.loads(pickle.dumps(idx))
        assert len(clone) == 16, len(clone)
        messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert any("TQ+ calibration" in m for m in messages), messages
        print("OK")
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout


# --------------------------------------------------------------------
# weakref (#340)
# --------------------------------------------------------------------


@pytest.mark.parametrize("build", _builders())
def test_index_is_weak_referenceable(build):
    """An index must be usable as a `WeakValueDictionary` value — the
    standard way to key a per-tenant cache without pinning the memory."""
    idx = build(n=64)
    ref = weakref.ref(idx)
    assert ref() is idx

    cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    cache["tenant"] = idx
    assert cache["tenant"] is idx

    del idx
    assert ref() is None
    assert "tenant" not in cache


# --------------------------------------------------------------------
# inspect.signature (#340)
# --------------------------------------------------------------------


def _chunked_methods():
    tq = TurboQuantIndex(DIM, 4)
    idmap = IdMapIndex(DIM, 4)
    return [
        pytest.param(tq.search, id="TurboQuantIndex.search"),
        pytest.param(tq.add, id="TurboQuantIndex.add"),
        pytest.param(idmap.search, id="IdMapIndex.search"),
        pytest.param(idmap.add_with_ids, id="IdMapIndex.add_with_ids"),
    ]


@pytest.mark.parametrize("method", _chunked_methods())
def test_signature_exposes_chunk_size(method):
    """`inspect.signature` must report the documented `chunk_size`.

    The chunking wrappers set `__wrapped__`, which `inspect.signature`
    follows by default, so it reported the *native* signature and hid
    `chunk_size` — making every signature-driven caller
    (`pydantic.validate_call`, framework tool-arg introspection, CLI
    adapters) reject the documented parameter.
    """
    sig = inspect.signature(method)
    assert "chunk_size" in sig.parameters
    param = sig.parameters["chunk_size"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


@pytest.mark.parametrize("method", _chunked_methods())
def test_signature_binds_what_the_method_accepts(method):
    """The reported signature must accept exactly the calls that work at
    runtime — `chunk_size` bound, an unknown kwarg still rejected."""
    sig = inspect.signature(method)
    positional = [np.zeros((1, DIM), np.float32)]
    if "k" in sig.parameters:
        positional.append(3)
    if "ids" in sig.parameters:
        positional.append(np.zeros(1, dtype=np.uint64))

    sig.bind(*positional, chunk_size=10)
    sig.bind(*positional, chunk_size=0)
    with pytest.raises(TypeError):
        sig.bind(*positional, definitely_not_a_parameter=1)


@pytest.mark.parametrize("method", _chunked_methods())
def test_signature_keeps_the_native_parameters(method):
    """Adding `chunk_size` must not drop or reorder the native
    parameters, and `chunk_size` must come last."""
    sig = inspect.signature(method)
    native = inspect.signature(method.__wrapped__)
    native_names = [n for n in native.parameters if n != "self"]
    assert list(sig.parameters) == [*native_names, "chunk_size"]


@pytest.mark.parametrize("method", _chunked_methods())
def test_wrapper_presents_itself_as_the_public_method(method):
    """`help()` must render the public method, not the internal
    `_make_search.<locals>.search` closure."""
    assert "<locals>" not in method.__qualname__
    assert method.__qualname__ == method.__wrapped__.__qualname__
    assert method.__module__ == "turbovec._turbovec"
    assert method.__doc__ == method.__wrapped__.__doc__


# --------------------------------------------------------------------
# BATCH_CHUNK_SIZE coercion (#345)
# --------------------------------------------------------------------


@pytest.fixture
def restore_batch_chunk_size():
    old = turbovec.BATCH_CHUNK_SIZE
    try:
        yield
    finally:
        turbovec.BATCH_CHUNK_SIZE = old


def test_float_batch_chunk_size_default_is_coerced(restore_batch_chunk_size):
    """The module docstring documents the slice size as coerced with
    `int(...)`; that must hold for the `turbovec.BATCH_CHUNK_SIZE`
    default too, not only an explicit `chunk_size=` (#345).

    Before, a float assigned to the documented knob surfaced as a
    `TypeError` from inside `range()` on an add with nothing wrong with
    it.
    """
    idx = _tq()  # calibration settled, so the add actually chunks
    turbovec.BATCH_CHUNK_SIZE = 500.7

    idx.add(_vectors(600, seed=31))
    assert len(idx) == FITTED + 600

    # Truncation, matching an explicit float argument.
    from turbovec import _interruptible

    assert _interruptible._default_chunk_size() == 500


def test_float_batch_chunk_size_default_chunks_identically(restore_batch_chunk_size):
    """Coercion must truncate to the same slice size an explicit float
    argument produces — same results, not merely no exception."""
    data = _vectors(600, seed=32)

    reference = _tq()
    reference.add(data, chunk_size=500.7)

    turbovec.BATCH_CHUNK_SIZE = 500.7
    from_default = _tq()
    from_default.add(data)

    queries = _vectors(3, seed=33)
    ref_scores, ref_ids = reference.search(queries, 10, chunk_size=0)
    got_scores, got_ids = from_default.search(queries, 10, chunk_size=0)
    np.testing.assert_array_equal(got_ids, ref_ids)
    np.testing.assert_array_equal(got_scores, ref_scores)
