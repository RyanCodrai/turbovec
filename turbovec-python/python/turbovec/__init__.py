from ._turbovec import IdMapIndex, TurboQuantIndex

#: Default batch slice size for the interruptible-chunking wrappers over
#: ``search`` / ``add`` / ``add_with_ids`` (issue #216). Batches with more
#: rows than this are processed one slice at a time so a queued Ctrl-C is
#: serviced between slices instead of at the end of the call. Set to ``0``
#: (globally, or per call via ``chunk_size=0``) to disable chunking. See
#: ``turbovec._interruptible`` for the full contract.
#:
#: 4096 rows keeps the between-slice Ctrl-C latency in single-digit
#: milliseconds at every supported dim/bit-width while paying the
#: per-slice overhead (snapshot + pool handoff, ~1 ms) a quarter as
#: often as the previous 1000-row default — measured as the difference
#: between 0.11 s and ~0.05 s for a 100k x 768d bulk add. Now that
#: every add chunks (an add never fits a calibration, so slicing is
#: always byte-exact), bulk loads pay this overhead too, and the
#: default has to respect them.
BATCH_CHUNK_SIZE = 4096

from . import _interruptible as _interruptible  # noqa: E402

_interruptible.install()

__all__ = ["BATCH_CHUNK_SIZE", "IdMapIndex", "TurboQuantIndex", "__version__"]


def __getattr__(name: str) -> str:
    # PEP 562: resolve __version__ lazily on first access. Importing
    # importlib.metadata costs ~20 ms — an order of magnitude more than
    # the rest of `import turbovec` — so it must not run at import time.
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            v = version("turbovec")
        except PackageNotFoundError:
            # Source tree without installed dist metadata (e.g. the
            # extension built in place); an obviously-dev marker.
            v = "0.0.0.dev0"
        globals()["__version__"] = v  # cache: __getattr__ never fires again
        return v
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    # PEP 562 companion: advertise the lazy attribute before its first
    # access (the set-union keeps it single once cached in globals()).
    return sorted(set(globals()) | {"__version__"})
