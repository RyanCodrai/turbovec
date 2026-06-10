"""Shared in-batch duplicate resolution for the framework integrations.

Each upstream library resolves a repeated id *within a single write* its own
way, and every turbovec wrapper must match its upstream to stay a true
drop-in:

- LangChain's ``InMemoryVectorStore`` overwrites on a repeated key  → KEEP_LAST
- LlamaIndex rejects duplicate ``node_id`` in a batch               → REJECT
- agno's LanceDb is append-only and keeps every row                 → KEEP_ALL
- Haystack exposes a runtime ``DuplicatePolicy`` (FAIL/SKIP/OVERWRITE).
  Its resolution is *stateful* (it dedups against the existing store as well
  as the batch, with deferred issue-#89 removal), so it does not reduce to
  the pure in-batch function here and keeps its own logic; this enum still
  documents the mapping (OVERWRITE→KEEP_LAST, SKIP→KEEP_FIRST, FAIL→REJECT).

The shared piece is the in-batch resolution only: given one key per item,
return the indices to keep. Each wrapper still owns its key extraction and
its cross-store upsert/removal.
"""
from __future__ import annotations

import enum
from typing import Hashable, List, Sequence


class DuplicatePolicy(enum.Enum):
    """How to resolve items that share a key within a single batch."""

    KEEP_LAST = "keep_last"
    """One item per key; the last occurrence wins (dict-overwrite semantics)."""

    KEEP_FIRST = "keep_first"
    """One item per key; the first occurrence wins."""

    REJECT = "reject"
    """Raise ``ValueError`` if any key repeats; otherwise keep everything."""

    KEEP_ALL = "keep_all"
    """No deduplication; items with duplicate keys all survive."""


def resolve_duplicates(
    keys: Sequence[Hashable], policy: DuplicatePolicy
) -> List[int]:
    """Return, in ascending order, the batch indices to keep under ``policy``.

    The returned indices index into ``keys`` (and any parallel arrays the
    caller holds). For KEEP_ALL and REJECT the result is ``0..len(keys)``;
    for KEEP_LAST/KEEP_FIRST it collapses to one index per distinct key.

    Raises:
        ValueError: under REJECT, if any key occurs more than once.
    """
    if policy is DuplicatePolicy.KEEP_ALL:
        return list(range(len(keys)))
    if policy is DuplicatePolicy.REJECT:
        seen: set = set()
        for k in keys:
            if k in seen:
                raise ValueError(f"duplicate id in batch: {k!r}")
            seen.add(k)
        return list(range(len(keys)))
    # KEEP_LAST / KEEP_FIRST collapse to one index per key.
    chosen: dict = {}
    for i, k in enumerate(keys):
        if policy is DuplicatePolicy.KEEP_LAST or k not in chosen:
            chosen[k] = i
    return sorted(chosen.values())


__all__ = ["DuplicatePolicy", "resolve_duplicates"]
