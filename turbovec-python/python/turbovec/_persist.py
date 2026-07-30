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

import itertools
import json
import math
import os
import secrets
import time
from typing import Any, Iterable, Optional

# Mirrors the Rust writer's TMP_SEQ (turbovec/src/io.rs): a pid suffix
# alone collides when two store objects in one process save to the same
# directory — they interleave writes into one temp file and each
# ``finally`` unlinks the other's in-flight temp (#316). ``count().
# __next__`` is atomic under the GIL/free-threading lock.
_TMP_SEQ = itertools.count()


# NAME_MAX on every filesystem we target (ext4, APFS, NTFS component).
_TMP_NAME_MAX = 255


def _tmp_path(path: str) -> str:
    """Sibling temp-file name ``<path>.tmp.{pid}.{seq}.{rand}`` in the
    same directory as ``path`` — unique per save, even across concurrent
    saves from one process.

    Mirrors the Rust writer's ``tmp_sibling``: when the destination's own
    filename would push the sibling past NAME_MAX, the *base* portion of
    the temp name is truncated to fit. Without this a legal destination
    name of ~232-255 bytes saves fine but its temp does not, so the save
    fails with ENAMETOOLONG (#299/#355). The destination name itself is
    never touched — the temp only has to be unique and recognizable.
    """
    directory, base = os.path.split(path)
    suffix = f".tmp.{os.getpid()}.{next(_TMP_SEQ)}.{secrets.token_hex(4)}"
    encoded = base.encode()
    budget = _TMP_NAME_MAX - len(suffix.encode())
    if len(encoded) > budget:
        # Cut on a character boundary so the name stays valid text.
        base = encoded[: max(budget, 0)].decode(errors="ignore")
    return os.path.join(directory, base + suffix)


def _replace_atomic(src: str, dst: str) -> None:
    """``os.replace`` with a short retry on Windows sharing violations.

    On Windows the rename fails with ERROR_SHARING_VIOLATION (winerror 32)
    while any other handle to the destination lacks FILE_SHARE_DELETE —
    CPython's own ``open()`` qualifies, as do antivirus and indexer scans.
    The Rust writer already retries (``rename_atomic``); the Python side
    needs the same posture or the integrations still hit #313 (#355).
    """
    if os.name != "nt":
        os.replace(src, dst)
        return
    delay = 0.001
    for _ in range(10):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:  # pragma: no cover - Windows-only path
            if getattr(exc, "winerror", None) != 32:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.064)
    os.replace(src, dst)


def _fsync_dir(directory: str) -> None:
    """fsync ``directory``, so a rename completed inside it survives a
    power loss (#350).

    ``os.replace`` publishes the new name by updating the *directory*, not
    the file. fsyncing the temp file only guarantees its contents; without
    an fsync of the containing directory, POSIX permits the directory
    entry itself to still be in cache when ``save()`` returns, so a crash
    afterwards can leave the old name — or no name — in place. The Rust
    writer already does this (#281); the Python side did not.

    Windows has no directory-fsync equivalent — ``os.open`` on a directory
    fails with EACCES, and ``FlushFileBuffers`` (what ``os.fsync`` wraps)
    is not defined for directory handles — so the call is skipped there.
    NTFS metadata journalling covers the same ground.

    Errors are swallowed. The rename has already succeeded at this point,
    so failing the save would report a durability shortfall as a lost
    write; some filesystems (and every non-POSIX one) legitimately refuse
    ``fsync`` on a directory fd.
    """
    if os.name == "nt":  # pragma: no cover - Windows-only path
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - unreadable parent directory
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - filesystem refuses dir fsync
        pass
    finally:
        os.close(fd)


def check_schema_version(version: Any, compat: Iterable[int], *, prefix: str) -> None:
    """Gate a side-car's ``schema_version`` field.

    The obvious spelling, ``version not in compat``, accepts more than it
    looks like it does: Python's ``==`` crosses numeric types, so ``2.0``
    and ``True`` compare equal to ``2`` and ``1`` (#350). A side-car
    written by a JavaScript producer naturally carries ``2.0`` — JSON has
    one number type and ``JSON.stringify(2.0)`` is only ``"2"`` by luck of
    the value being integral. A version field is an identifier, not a
    quantity, so the type has to match too: this requires an ``int``, and
    ``bool`` is excluded even though it is a subclass of ``int``.

    Args:
        version: the raw value read from the side-car.
        compat: the schema versions this build accepts.
        prefix: message lead-in, e.g. ``"docstore.json has schema version"``.

    Raises:
        ValueError: if ``version`` is not an ``int``, or is not in ``compat``.
    """
    if type(version) is not int or version not in compat:
        raise ValueError(
            f"{prefix} {version}; this turbovec accepts versions {list(compat)}"
        )


def _crumb_path(entry) -> str:
    """Rebuild ``payload['docs']['a']['metadata'][1]`` from a stack entry.

    Only called on the failure path — see ``_check_json_faithful`` for why
    the walk carries parent links instead of prebuilt path strings.
    """
    keys = []
    while entry is not None:
        _obj, parent, key = entry
        if parent is not None:
            keys.append(f"[{key!r}]" if isinstance(key, str) else f"[{key}]")
        entry = parent
    return "payload" + "".join(reversed(keys))


def _check_json_faithful(payload: Any) -> None:
    """Reject payloads whose JSON form would lose data or not be portable
    JSON (#350).

    ``json.dumps`` accepts two things it writes *destructively*, silently,
    and irreversibly:

    1. **Non-string mapping keys.** JSON object keys are strings, so
       ``json.dumps`` stringifies ``int``/``float``/``bool``/``None`` keys.
       ``{1: "a", "1": "b"}`` becomes ``{"1": "b"}`` — the int-keyed entry
       is *gone*, ``save()`` returns success, and the loss is invisible
       until someone reads the data back. ``True``/``1`` and ``2020``/
       ``"2020"`` collide the same way.
    2. **Non-finite floats.** ``allow_nan`` defaults to True, emitting bare
       ``NaN``/``Infinity`` tokens that RFC 8259 forbids. Python round-trips
       them, so the damage only shows outside Python: ``JSON.parse`` and
       ``serde_json`` reject the file outright, and ``jq .`` silently
       rewrites ``NaN`` to ``null`` — corrupting values in a side-car this
       project documents as plain, inspectable JSON.

    **Scope — exactly these two, and deliberately not "everything JSON
    round-trips imperfectly".** The guard covers values whose JSON form
    either *loses data* (a collided key: two entries in, one out, and no
    way to tell which) or *is not JSON at all* (a bare ``NaN`` token).
    Total, well-known type narrowings are out of scope and are documented
    at the call sites instead:

    - ``tuple`` -> ``list``. Every element survives; only the type narrows,
      the mapping is total and one-way for every tuple alike, and the
      side-car's own payloads are built from ``dict.items()`` pairs.
      ``langchain.py``'s dump and ``llama_index.py``'s ``node_id_to_u64``
      both already document the coercion in-line.
    - ``int`` wider than 2**53. Python writes and reads these exactly, and
      an arbitrary-precision integer literal *is* valid RFC 8259 — the
      imprecision lives in double-based readers (``JSON.parse`` turns
      ``9007199254740993`` into ``...992``), not in the file. Rejecting
      them would break the legitimate int64 ids the stores are expected to
      carry, so they are accepted and the reader caveat is documented.

    So the enforced contract is: **a save whose side-car would lose data
    or would not be portable JSON fails loudly before any file is
    touched.** That extends the posture ``atomic_save`` already documents
    for sets and ndarrays to the two cases where it silently did not hold.

    Both are rejected rather than coerced. Coercion is what causes the
    damage — stringifying keys is exactly the step that merges ``1`` into
    ``"1"``, and mapping NaN to ``null`` (what jq does) turns a "score was
    NaN" into "score was absent". Neither can be undone at load time, and
    neither is detectable by the handle/keyset checks, so the only place
    to be loud is the write.

    **This is a breaking change for one of the two cases.** Non-str keys
    were already lossy on reload — those saves never worked, they only
    reported success. NaN/Infinity metadata is different: it round-tripped
    correctly through turbovec's own ``save``/``load``, because Python's
    ``json`` both writes and reads the non-standard tokens. Such a store
    now raises ``ValueError`` at save time. That is intended — the file it
    used to write is not JSON, and any non-Python consumer either rejects
    it or (jq) quietly corrupts it — but it does break working code.
    Callers with legitimately non-finite metadata should sanitize before
    saving (``None`` for "no value", or a sentinel that is a real number).
    The error names the exact path to the offending entry.

    Raises:
        TypeError: if any mapping key anywhere in ``payload`` is not a str.
        ValueError: if any float anywhere in ``payload`` is NaN or Infinity.
    """
    # Iterative with a visited set: metadata may nest arbitrarily deep
    # (recursion would blow the stack before json.dumps' own guard fires)
    # and may contain shared or cyclic containers. Recording container
    # identity keeps a cycle from spinning forever; a shared subtree is
    # still validated, just once.
    #
    # Each entry is (obj, parent_entry, key_in_parent) and doubles as its
    # children's parent link, so the walk costs one tuple per node and
    # builds no path strings. Eagerly formatting `f"{path}[{key!r}]"` for
    # every node cost ~18% of the whole validation pass on a 200k-doc
    # payload, all of it to produce strings that are thrown away unless a
    # save fails. `_crumb_path` reconstructs the path from the links on
    # the failure path only.
    root = (payload, None, None)
    stack = [root]
    seen: set[int] = set()
    while stack:
        entry = stack.pop()
        obj = entry[0]
        if isinstance(obj, dict):
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"side-car key {key!r} at "
                        f"{_crumb_path(entry)} is {type(key).__name__}, not "
                        f"str. JSON object keys are strings, so writing it "
                        f"would stringify the key and silently merge it with "
                        f"any existing {str(key)!r} key, losing data on "
                        f"reload. Convert the key to a str before saving."
                    )
                stack.append((value, entry, key))
        elif isinstance(obj, (list, tuple)):
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            for i, value in enumerate(obj):
                stack.append((value, entry, i))
        elif isinstance(obj, float) and not math.isfinite(obj):
            if math.isnan(obj):
                token = "NaN"
            else:
                token = "Infinity" if obj > 0 else "-Infinity"
            raise ValueError(
                f"side-car value at {_crumb_path(entry)} is {obj!r}, "
                f"which JSON cannot represent: it would be written as a bare "
                f"{token} token that RFC 8259 forbids. Other JSON readers "
                f"reject the file (serde_json, JSON.parse) or silently "
                f"rewrite the value to null (jq). Replace it with None or a "
                f"finite number before saving."
            )


def atomic_save(index, index_path, payload: Any, sidecar_path) -> None:
    """Atomically persist an index + JSON side-car pair — the shared
    write path for all four integrations' save methods.

    The failure-ordering guarantees:

    1. ``payload`` is validated and JSON-serialized fully in memory
       *first*, so a value whose JSON form would lose data or not be
       portable JSON raises before any file is touched: a non-serializable
       value (a set or ndarray in document metadata) or a non-str mapping
       key raises ``TypeError``, and a NaN/Infinity float raises
       ``ValueError``. See ``_check_json_faithful`` for the exact scope,
       for why the last two are rejected rather than coerced, and for
       what is deliberately *not* covered (#350). Validation walks the
       whole payload, so it costs roughly as much again as the
       ``json.dumps`` it precedes.
    2. Both artifacts are written to sibling temp files in the
       destination directory, flushed and fsynced, then moved into place
       with ``os.replace`` (atomic on POSIX), and the containing
       directory is fsynced so the renames themselves are durable — a
       rename publishes a name in the *directory*, so without that fsync
       a crash after a successful return could still lose it (#350). A failure or crash before
       the first replace leaves a previous store at these paths intact.
    3. On failure the temp files are removed (best effort).

    The one remaining non-atomic window is between the two ``replace``
    calls (and the directory fsync that follows them): a hard crash
    there leaves a new index beside the old side-car. The LangChain, LlamaIndex, and Haystack load paths detect
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
    # Fail before touching any file. The walk catches what json.dumps
    # accepts but writes destructively (non-str keys, non-finite floats)
    # and is the thing that produces the useful message. `allow_nan=False`
    # is defence in depth, not a second mechanism: json's encoder handles
    # exactly the container and scalar types the walk descends, so there
    # is no known float it reaches that the walk does not. It is here so
    # that a future encoder change (or a `default=` hook added to this
    # call) cannot re-open the hole silently — a NaN slipping past the
    # walk would raise here rather than land on disk.
    _check_json_faithful(payload)
    payload_str = json.dumps(payload, allow_nan=False)

    index_path = os.fspath(index_path)
    sidecar_path = os.fspath(sidecar_path)
    index_tmp = _tmp_path(index_path)
    sidecar_tmp = _tmp_path(sidecar_path)
    try:
        # The Rust binding owns the index file handle, so fsync via a
        # reopened descriptor (fsync flushes the inode, not the fd). The
        # handle must be writable: on Windows, fsync calls _commit, which
        # rejects read-only descriptors with EBADF.
        index.write(index_tmp)
        with open(index_tmp, "rb+") as f:
            os.fsync(f.fileno())
        # "x" (O_CREAT|O_EXCL) refuses a pre-existing file or planted
        # symlink at the temp name instead of writing through it.
        with open(sidecar_tmp, "x") as f:
            f.write(payload_str)
            f.flush()
            os.fsync(f.fileno())
        _replace_atomic(index_tmp, index_path)
        _replace_atomic(sidecar_tmp, sidecar_path)
        # The renames above are already visible to readers, but are not on
        # stable storage until their parent directory is synced. Both
        # destinations normally share a directory; fsync each distinct one
        # so a store split across two directories is covered too.
        for directory in dict.fromkeys(
            (os.path.dirname(index_path) or ".", os.path.dirname(sidecar_path) or ".")
        ):
            _fsync_dir(directory)
    finally:
        for tmp in (index_tmp, sidecar_tmp):
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass


def check_persisted_handles(
    index,
    handles: Iterable[int],
    *,
    what: str = "entry",
    next_u64: Optional[int] = None,
) -> None:
    """Validate that the side-car's handle set matches the loaded index.

    Args:
        index: the loaded ``IdMapIndex`` (uses ``len`` and ``contains``).
        handles: the u64 handles the side-car maps can resolve.
        what: noun for error messages (e.g. "document", "node").
        next_u64: the side-car's handle watermark, if the caller has it.
            Handles are issued by pre-incrementing it, so it must be at
            least the largest handle in use; a smaller value reissues live
            handles on the next write (issue #321).

    Raises:
        ValueError: if the side-car has duplicate handles, a different count
            than the index, a handle the index doesn't contain, or a
            watermark below the largest handle in use.
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
                f"persisted store is inconsistent with its index: a {what} in "
                f"the side-car has no vector in the index (internal record id "
                f"{h}). The .tvim index and its JSON side-car are out of sync."
            )
    if next_u64 is not None and handle_list and int(next_u64) < max(handle_list):
        raise ValueError(
            f"persisted store is corrupt: the handle watermark next_u64="
            f"{int(next_u64)} is below the largest {what} handle in use "
            f"({max(handle_list)}). Loading it would reissue live handles "
            f"on the next write."
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


__all__ = ["check_persisted_handles", "check_schema_version", "check_sidecar_keysets"]
