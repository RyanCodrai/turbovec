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

Some side-cars additionally hold *two* structures keyed by the same string
ids (e.g. an id -> handle map plus an id -> payload map). Those can desync
independently of the index — a hand-edited or partially-written side-car
where one map lost an entry still passes the handle check and would raise
an opaque ``KeyError`` mid-query. ``check_sidecar_keysets`` turns that into
the same clean ``ValueError`` at load time.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable


def _tmp_path(path: str) -> str:
    """Pid-suffixed sibling temp-file name in the same directory as
    ``path``."""
    return f"{path}.tmp.{os.getpid()}"


def atomic_save(index, index_path, payload: Any, sidecar_path) -> None:
    """Atomically persist an index + JSON side-car pair — the shared
    write path for all four integrations' save methods.

    The failure-ordering guarantees:

    1. ``payload`` is JSON-serialized fully in memory *first*, so a
       non-serializable value (a set or ndarray in document metadata)
       raises ``TypeError`` before any file is touched.
    2. Both artifacts are written to sibling temp files in the
       destination directory, flushed and fsynced, then moved into place
       with ``os.replace`` (atomic on POSIX). A failure or crash before
       the first replace leaves a previous store at these paths intact.
    3. On failure the temp files are removed (best effort).

    The one remaining non-atomic window is between the two ``replace``
    calls: a hard crash exactly there leaves a new index beside the old
    side-car. The LangChain, LlamaIndex, and Haystack load paths detect
    that mismatch via ``check_persisted_handles`` and raise a clean
    ``ValueError`` instead of returning silently corrupted data; the agno
    load path gains the same check with the side-car keyset validation
    work (#178).

    Args:
        index: the ``IdMapIndex`` to persist (uses ``index.write``).
        index_path: destination for the binary ``.tvim`` file.
        payload: JSON-serializable side-car payload.
        sidecar_path: destination for the JSON side-car.
    """
    payload_str = json.dumps(payload)  # fail before touching any file

    index_path = os.fspath(index_path)
    sidecar_path = os.fspath(sidecar_path)
    index_tmp = _tmp_path(index_path)
    sidecar_tmp = _tmp_path(sidecar_path)
    try:
        # The Rust binding owns the index file handle, so fsync via a
        # reopened read descriptor (fsync flushes the inode, not the fd).
        index.write(index_tmp)
        with open(index_tmp, "rb") as f:
            os.fsync(f.fileno())
        with open(sidecar_tmp, "w") as f:
            f.write(payload_str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(index_tmp, index_path)
        os.replace(sidecar_tmp, sidecar_path)
    finally:
        for tmp in (index_tmp, sidecar_tmp):
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass


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


def check_sidecar_keysets(
    mapping_keys: Iterable,
    sidecar_keys: Iterable,
    *,
    what: str = "entry",
    mapping_name: str = "id map",
    sidecar_name: str = "payload map",
) -> None:
    """Validate that two side-car structures keyed by the same ids agree.

    Args:
        mapping_keys: ids resolvable through the id -> handle map.
        sidecar_keys: ids present in the id -> payload map.
        what: noun for error messages (e.g. "document", "node").
        mapping_name: side-car field name of the id -> handle map.
        sidecar_name: side-car field name of the id -> payload map.

    Raises:
        ValueError: if either map holds an id the other lacks.
    """
    mapping_set = set(mapping_keys)
    sidecar_set = set(sidecar_keys)
    if mapping_set == sidecar_set:
        return
    missing = mapping_set - sidecar_set
    if missing:
        # key=repr: a hand-corrupted side-car can hold mixed-type ids
        # (JSON arrays survive parsing with e.g. an int among strings),
        # which plain sorted() would turn into a TypeError instead of
        # the promised ValueError.
        sample = ", ".join(repr(k) for k in sorted(missing, key=repr)[:3])
        raise ValueError(
            f"persisted store is corrupt: {len(missing)} {what} id(s) present "
            f"in `{mapping_name}` but missing from `{sidecar_name}` "
            f"(e.g. {sample}). The JSON side-car's maps are out of sync."
        )
    extraneous = sidecar_set - mapping_set
    sample = ", ".join(repr(k) for k in sorted(extraneous, key=repr)[:3])
    raise ValueError(
        f"persisted store is corrupt: {len(extraneous)} {what} id(s) present "
        f"in `{sidecar_name}` but missing from `{mapping_name}` "
        f"(e.g. {sample}). The JSON side-car's maps are out of sync."
    )


__all__ = ["check_persisted_handles", "check_sidecar_keysets"]
