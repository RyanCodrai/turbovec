"""Binary I/O for ``.tv`` / ``.tvim`` files on the MLX backend.

Pure-Python parsing/serialization — no Rust trip — because the format
is a tiny struct header + flat little-endian arrays. Round-trips
byte-exactly with ``turbovec/src/io.rs``.

File formats
============

``.tv``::

    +-------------------------+
    | bit_width   u8          |
    | dim         u32 LE      |
    | n_vectors   u32 LE      |
    +-------------------------+
    | packed_codes            |  (dim/8) * bit_width * n_vectors bytes
    +-------------------------+
    | norms                   |  n_vectors * f32 LE
    +-------------------------+

``.tvim``::

    +-------------------------+
    | magic   b"TVIM"  4 B    |
    | version u8 = 1          |
    +-------------------------+
    | core payload (same .tv) |
    +-------------------------+
    | slot_to_id              |  n_vectors * u64 LE
    +-------------------------+
"""
from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


_TVIM_MAGIC = b"TVIM"
_TVIM_VERSION = 1
_HEADER_FMT = "<BII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _bytes_per_vec(dim: int, bit_width: int) -> int:
    return (dim // 8) * bit_width


def write_tv(
    path: str,
    dim: int,
    bit_width: int,
    n_vectors: int,
    packed_codes: np.ndarray,
    norms: np.ndarray,
) -> None:
    """Write a ``.tv`` file. ``packed_codes`` is shape
    ``(n_vectors, bytes_per_vec)`` ``uint8``; ``norms`` is shape
    ``(n_vectors,)`` ``float32``.
    """
    with open(path, "wb") as f:
        f.write(struct.pack(_HEADER_FMT, bit_width, dim, n_vectors))
        if n_vectors:
            f.write(np.ascontiguousarray(packed_codes, dtype=np.uint8).tobytes())
            f.write(np.ascontiguousarray(norms, dtype="<f4").tobytes())


def load_tv(path: str) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
    """Read a ``.tv`` file.

    Returns ``(bit_width, dim, n_vectors, packed_codes, norms)``.
    """
    with open(path, "rb") as f:
        header = f.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE:
            raise ValueError(f"file too short: missing {_HEADER_SIZE}-byte header")
        bit_width, dim, n_vectors = struct.unpack(_HEADER_FMT, header)
        bpv = _bytes_per_vec(dim, bit_width)
        packed_size = bpv * n_vectors

        packed_bytes = f.read(packed_size)
        if len(packed_bytes) != packed_size:
            raise ValueError(
                f"truncated packed_codes: expected {packed_size} bytes, "
                f"got {len(packed_bytes)}"
            )
        norms_bytes = f.read(n_vectors * 4)
        if len(norms_bytes) != n_vectors * 4:
            raise ValueError(
                f"truncated norms: expected {n_vectors * 4} bytes, "
                f"got {len(norms_bytes)}"
            )

    packed = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(n_vectors, bpv).copy()
    norms = np.frombuffer(norms_bytes, dtype="<f4").astype(np.float32, copy=True)
    return bit_width, dim, n_vectors, packed, norms


def write_tvim(
    path: str,
    dim: int,
    bit_width: int,
    n_vectors: int,
    packed_codes: np.ndarray,
    norms: np.ndarray,
    slot_to_id: np.ndarray,
) -> None:
    """Write a ``.tvim`` file. ``slot_to_id`` is shape ``(n_vectors,)``
    ``uint64``.
    """
    if slot_to_id.shape != (n_vectors,):
        raise ValueError(
            f"slot_to_id shape {slot_to_id.shape} != ({n_vectors},)"
        )
    with open(path, "wb") as f:
        f.write(_TVIM_MAGIC)
        f.write(bytes([_TVIM_VERSION]))
        f.write(struct.pack(_HEADER_FMT, bit_width, dim, n_vectors))
        if n_vectors:
            f.write(np.ascontiguousarray(packed_codes, dtype=np.uint8).tobytes())
            f.write(np.ascontiguousarray(norms, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(slot_to_id, dtype="<u8").tobytes())


def load_tvim(
    path: str,
) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Read a ``.tvim`` file.

    Returns ``(bit_width, dim, n_vectors, packed_codes, norms, slot_to_id)``.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _TVIM_MAGIC:
            raise ValueError(f"not a TVIM file: wrong magic {magic!r}")
        version_byte = f.read(1)
        if not version_byte or version_byte[0] != _TVIM_VERSION:
            raise ValueError(
                f"unsupported TVIM version: {version_byte!r}"
            )
        header = f.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE:
            raise ValueError(f"file too short: missing {_HEADER_SIZE}-byte header")
        bit_width, dim, n_vectors = struct.unpack(_HEADER_FMT, header)
        bpv = _bytes_per_vec(dim, bit_width)
        packed_size = bpv * n_vectors

        packed_bytes = f.read(packed_size)
        if len(packed_bytes) != packed_size:
            raise ValueError(
                f"truncated packed_codes: expected {packed_size} bytes, "
                f"got {len(packed_bytes)}"
            )
        norms_bytes = f.read(n_vectors * 4)
        if len(norms_bytes) != n_vectors * 4:
            raise ValueError(
                f"truncated norms: expected {n_vectors * 4} bytes, "
                f"got {len(norms_bytes)}"
            )
        ids_bytes = f.read(n_vectors * 8)
        if len(ids_bytes) != n_vectors * 8:
            raise ValueError(
                f"truncated slot_to_id: expected {n_vectors * 8} bytes, "
                f"got {len(ids_bytes)}"
            )

    packed = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(n_vectors, bpv).copy()
    norms = np.frombuffer(norms_bytes, dtype="<f4").astype(np.float32, copy=True)
    slot_to_id = np.frombuffer(ids_bytes, dtype="<u8").astype(np.uint64, copy=True)
    return bit_width, dim, n_vectors, packed, norms, slot_to_id
