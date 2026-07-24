from ._turbovec import IdMapIndex, TurboQuantIndex

__all__ = ["IdMapIndex", "TurboQuantIndex", "__version__"]


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
