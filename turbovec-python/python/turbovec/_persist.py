"""Shared persistence consistency checks for the framework integrations.

Each wrapper persists two artifacts: the binary ``.tvim`` index and a JSON
side-car holding the handle -> document/node/text payload maps. At query
time the wrapper resolves an index-returned u64 handle through that side-car
map. If the two files are out of sync — a partial copy, a stale backup, a
hand-edited or tampered side-car — an index handle won't resolve and the
wrapper would raise an opaque ``KeyError`` deep inside a query.

``check_persisted_handles`` turns that into a clean ``ValueError`` at load
time. ``IdMapIndex`` exposes only ``__len__`` and ``contains``; that's
sufficient: if the side-car's handle set and the index have equal size and
every side-car handle is present in the index, the two are a bijection (no
index handle can be missing from the side-car).
"""
from __future__ import annotations

from typing import Iterable


def check_persisted_handles(index, handles: Iterable[int], *, what: str = "entry") -> None:
    """Validate that the side-car's handle set matches the loaded index.

    Args:
        index: the loaded ``IdMapIndex`` (uses ``len`` and ``contains``).
        handles: the u64 handles the side-car maps can resolve.
        what: noun for error messages (e.g. "document", "node").

    Raises:
        ValueError: if the side-car has duplicate handles, a different count
            than the index, or a handle the index doesn't contain.
    """
    handle_list = [int(h) for h in handles]
    n_index = len(index)

    if len(set(handle_list)) != len(handle_list):
        raise ValueError(
            f"persisted store is corrupt: duplicate {what} handles in the side-car"
        )
    if len(handle_list) != n_index:
        raise ValueError(
            f"persisted store is inconsistent with its index: side-car has "
            f"{len(handle_list)} {what} handle(s) but the index holds {n_index}. "
            f"The .tvim index and its JSON side-car are out of sync."
        )
    for h in handle_list:
        if not index.contains(h):
            raise ValueError(
                f"persisted store is inconsistent with its index: {what} handle "
                f"{h} is not present in the index. The .tvim index and its JSON "
                f"side-car are out of sync."
            )


__all__ = ["check_persisted_handles"]
