"""Shared similarity-mode helpers for the framework integration stores.

Each integration store supports two similarity modes, fixed at
construction and recorded in the persisted side-car:

- ``"cosine"`` (default): document vectors are L2-normalized before they
  reach the quantized index and query vectors are L2-normalized before
  search, so the engine's raw inner product is true cosine similarity in
  ``[-1, 1]`` and the stores' relevance mappings / threshold paths hold
  for embeddings of any magnitude.
- ``"dot_product"``: vectors are stored and queried as-is; scores are raw
  inner products and ranking is magnitude-aware. Score magnitudes are
  dataset-relative, so absolute score thresholds need calibrating per
  embedder/dataset.

Zero vectors cannot be normalized (the norm is 0); they are kept as-is
in both modes — a zero vector scores 0 against everything, matching the
reference stores (LangChain's ``InMemoryVectorStore`` maps the resulting
NaN cosine to 0.0; Haystack's ``InMemoryDocumentStore`` substitutes a
norm of 1, which leaves the zero dot product intact).
"""

from __future__ import annotations

import numpy as np

COSINE = "cosine"
DOT_PRODUCT = "dot_product"
_VALID_MODES = (COSINE, DOT_PRODUCT)


def validate_similarity(value: str, *, param: str = "similarity") -> str:
    """Return ``value`` if it names a supported similarity mode, else
    raise a ``ValueError`` naming the parameter and the valid options."""
    if value not in _VALID_MODES:
        raise ValueError(
            f"{param} must be one of {list(_VALID_MODES)}, got {value!r}"
        )
    return value


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Return a float32 copy of the 2D batch ``vectors`` with every row
    L2-normalized. Rows with zero norm are kept as-is (see module
    docstring). Pure computation — safe to run outside store locks."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Substitute 1.0 for zero norms so zero rows pass through unchanged
    # instead of dividing by zero.
    out = vectors / np.where(norms == 0.0, 1.0, norms)
    return np.ascontiguousarray(out, dtype=np.float32)


__all__ = [
    "COSINE",
    "DOT_PRODUCT",
    "validate_similarity",
    "l2_normalize_rows",
]
