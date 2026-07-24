use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

fn not_contiguous_err(kind: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{kind} must be C-contiguous; call np.ascontiguousarray(...) first",
    ))
}

/// Name of a Python object's type, for error messages.
fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

/// Describe an argument for an array-mismatch error: "2-D float64" for
/// anything ndarray-like, otherwise the plain type name (e.g. "list").
fn array_desc(obj: &Bound<'_, PyAny>) -> String {
    let ndim = obj
        .getattr("ndim")
        .ok()
        .and_then(|n| n.extract::<usize>().ok());
    let dtype = obj
        .getattr("dtype")
        .ok()
        .and_then(|d| d.str().ok())
        .map(|s| s.to_string());
    match (ndim, dtype) {
        (Some(n), Some(d)) => format!("{n}-D {d}"),
        _ => type_name(obj),
    }
}

fn array_type_err(name: &str, expected: &str, obj: &Bound<'_, PyAny>) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be a {expected} array, got {}",
        array_desc(obj),
    ))
}

/// Extract a 2-D float32 array, replacing pyo3's opaque downcast error
/// ("'ndarray' object cannot be cast as 'ndarray'") with one that names
/// the argument and states expected vs got dtype/ndim.
fn extract_f32_2d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f32>> {
    obj.extract()
        .map_err(|_| array_type_err(name, "2-D float32", obj))
}

/// Extract a 1-D uint64 array; see [`extract_f32_2d`].
fn extract_u64_1d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, u64>> {
    obj.extract()
        .map_err(|_| array_type_err(name, "1-D uint64", obj))
}

/// Extract a 1-D bool array; see [`extract_f32_2d`].
fn extract_bool_1d<'py>(
    name: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, bool>> {
    obj.extract()
        .map_err(|_| array_type_err(name, "1-D bool", obj))
}

/// Whether `obj` is integer-valued: a Python `int` of any magnitude, or
/// any object (numpy scalar, `__index__` implementor) that converts to
/// one. Used to pick a range error over a type error once the fast-path
/// fixed-width extraction has failed.
fn is_py_int(obj: &Bound<'_, PyAny>) -> bool {
    obj.extract::<i128>().is_ok() || obj.is_instance_of::<pyo3::types::PyInt>()
}

/// `str()` of an argument, for error messages.
fn int_repr(obj: &Bound<'_, PyAny>) -> String {
    obj.str()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<unprintable>".to_string())
}

/// Extract a non-negative count/size argument (`dim`, `bit_width`, `k`)
/// as `usize`. Values in the unsigned 64-bit range pass through untouched
/// (over-large-but-representable values stay subject to each method's own
/// range rules, e.g. `k` clamping and the core's `dim` cap). Integers
/// outside that range raise a `ValueError` naming the argument instead of
/// pyo3's bare `OverflowError`; non-integers raise `TypeError`.
fn extract_size(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(v) = obj.extract::<usize>() {
        return Ok(v);
    }
    if is_py_int(obj) {
        let msg = if obj.lt(0).unwrap_or(false) {
            format!(
                "{name} must be a non-negative integer, got {}",
                int_repr(obj)
            )
        } else {
            format!(
                "{name} must fit in an unsigned 64-bit integer, got {}",
                int_repr(obj)
            )
        };
        return Err(pyo3::exceptions::PyValueError::new_err(msg));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be an integer, got {}",
        type_name(obj),
    )))
}

/// Extract an external id for membership-style calls (`contains`,
/// `__contains__`, `remove`). Integers outside the `u64` range can never
/// be present in the index, so they yield `None` ("absent") rather than
/// pyo3's bare `OverflowError`; non-integers raise `TypeError`.
fn extract_membership_id(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if let Ok(v) = obj.extract::<u64>() {
        return Ok(Some(v));
    }
    if is_py_int(obj) {
        return Ok(None);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be an integer, got {}",
        type_name(obj),
    )))
}

/// Map a numpy shape error from reassembling search results into a typed
/// RuntimeError. The result dimensions are derived from the core's own
/// output, so this never fires today — but a future change to result shaping
/// would otherwise surface as an uncatchable panic instead of a catchable
/// exception.
fn shape_err(e: numpy::ndarray::ShapeError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "internal error: malformed search result shape: {e}"
    ))
}

/// Reject NaN / Inf / overflow-magnitude query coordinates with a typed
/// `ValueError`. The core `search` panics on invalid values (its documented
/// Rust contract), which would otherwise surface to Python as an uncatchable
/// `PanicException`. `add` already maps the same condition to `ValueError`;
/// this keeps `search` consistent.
fn validate_queries(values: &[f32], dim: usize) -> PyResult<()> {
    if let Some((vi, ci, v)) = turbovec_core::first_invalid_coord(values, dim) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid query value at query {vi}, coord {ci}: {v} \
             (must be finite and |value| < 1e16)",
        )));
    }
    Ok(())
}

/// Map an `io::Error` from `load` to Python: `ErrorKind::NotFound`
/// becomes `FileNotFoundError` (so `except FileNotFoundError:` works,
/// matching what the integrations' Python-side `open()` of their JSON
/// side-cars raises for a missing path); every other kind — including
/// permission errors — stays plain `OSError`, as before. The path is
/// appended because the io::Error alone doesn't name the file.
fn load_err(path: &str, e: std::io::Error) -> PyErr {
    let msg = format!("{e}: {path}");
    if e.kind() == std::io::ErrorKind::NotFound {
        pyo3::exceptions::PyFileNotFoundError::new_err(msg)
    } else {
        pyo3::exceptions::PyIOError::new_err(msg)
    }
}

/// Read-lock an index, recovering from poisoning. A panic that escaped
/// a previous call has already surfaced to Python as a PanicException;
/// before the GIL-release change such a panic likewise left the object
/// reachable, so poisoning must not turn every later call into a panic.
fn lock_read<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Write-lock an index; see [`lock_read`] for the poisoning rationale.
fn lock_write<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// Locking discipline (both pyclasses): the pyclass is `frozen`, so pyo3
// performs no runtime borrow-checking of its own and every method takes
// `&self`; the inner index sits behind an `RwLock` instead. Concurrent
// searches share the read lock and run in parallel; a write blocks until
// it is alone, then succeeds — the same observable outcome as when the
// GIL serialized every call, minus the serialization of reads.
//
// Every method acquires the lock INSIDE `py.detach(..)` — even trivial
// O(1) calls like `__len__` — so a thread blocked on the lock is never
// holding the GIL (a GIL-held `len()` landing during an in-flight write
// would otherwise stall every Python thread for the write's duration),
// and every index-dependent check runs under the same guard as the core
// call it protects (the pre-GIL-release code got that atomicity for
// free from the GIL). A guard is only ever released from code that does
// not need the GIL to finish, so no lock/GIL deadlock cycle exists.

#[pyclass(frozen)]
struct TurboQuantIndex {
    inner: std::sync::RwLock<turbovec_core::TurboQuantIndex>,
}

#[pymethods]
impl TurboQuantIndex {
    /// Construct an index. `dim` is optional: when omitted, the
    /// underlying quantized index is created lazily on the first
    /// `add` call, picking up the dimensionality from the input
    /// array's shape. `bit_width` defaults to 4.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=None))]
    fn new(dim: Option<&Bound<'_, PyAny>>, bit_width: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let bit_width = match bit_width {
            Some(b) => extract_size("bit_width", b)?,
            None => 4,
        };
        let inner = match dim {
            Some(d) => turbovec_core::TurboQuantIndex::new(extract_size("dim", d)?, bit_width),
            None => turbovec_core::TurboQuantIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
        })
    }

    fn add(&self, py: Python<'_>, vectors: &Bound<'_, PyAny>) -> PyResult<()> {
        let vectors = extract_f32_2d("vectors", vectors)?;
        let arr = vectors.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("vectors"))?;
        // Snapshot the numpy buffer before releasing the GIL (`detach`):
        // once released, another Python thread may write to the (possibly
        // writable) source array, and rust-numpy's borrow flags cannot
        // prevent Python-side writes. The copy is O(n·dim) against the
        // quantization kernel, so it is cheap by comparison.
        let owned = slice.to_vec();
        // `add_2d` handles both eager (dim must match) and lazy (locks
        // dim on first call) cases.
        py.detach(|| lock_write(&self.inner).add_2d(&owned, dim))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Run a top-`k` search against the index.
    ///
    /// `mask`, when given, is a bool array of length `len(self)`. Only slots
    /// with `mask[i] == True` contribute to the returned top-`k`. The
    /// returned result count per query is `min(k, mask.sum())`.
    #[pyo3(signature = (queries, k, *, mask=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: &Bound<'py, PyAny>,
        mask: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
        let queries = extract_f32_2d("queries", queries)?;
        let k = extract_size("k", k)?;
        let mask = mask.map(|m| extract_bool_1d("mask", m)).transpose()?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let ncols = arr.ncols();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        // Snapshot the query (and mask) buffers before releasing the
        // GIL: once released, another Python thread may write to the
        // source arrays mid-search. Validation runs on the snapshot so
        // the searched data is exactly the data that was validated.
        let q_owned = q_slice.to_vec();
        let mask_owned: Option<Vec<bool>> = match mask.as_ref().map(|m| m.as_array()).as_ref() {
            Some(m_arr) => Some(
                m_arr
                    .as_slice()
                    .ok_or_else(|| not_contiguous_err("mask"))?
                    .to_vec(),
            ),
            None => None,
        };

        // Index-dependent checks run under the same read guard as the
        // kernel, so a concurrent writer cannot invalidate them between
        // check and search (pre-GIL-release, holding the GIL made the
        // whole call atomic).
        let results = py.detach(|| {
            let inner = lock_read(&self.inner);
            // Reject wrong-dim queries cleanly. Previously the inner
            // `assert_eq!(queries.len(), nq * dim)` would fire as a Rust
            // panic and surface to Python as a PanicException, not the
            // ValueError users expect for input-shape mismatch.
            if let Some(idx_dim) = inner.dim_opt() {
                if ncols != idx_dim {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query dim {ncols} does not match index dim {idx_dim}",
                    )));
                }
            }
            validate_queries(&q_owned, ncols)?;
            if let Some(m) = mask_owned.as_deref() {
                let expected = inner.len();
                if m.len() != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mask length {} does not match index size {}",
                        m.len(),
                        expected,
                    )));
                }
            }
            Ok(inner.search_with_mask(&q_owned, k, mask_owned.as_deref()))
        })?;
        let effective_k = results.k;

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.scores)
            .map_err(shape_err)?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.indices)
            .map_err(shape_err)?
            .into_pyarray(py);

        Ok((scores, indices))
    }

    fn write(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| lock_read(&self.inner).write(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    #[classmethod]
    fn load(cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = cls
            .py()
            .detach(|| turbovec_core::TurboQuantIndex::load(path))
            .map_err(|e| load_err(path, e))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
        })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self, py: Python<'_>) {
        py.detach(|| lock_read(&self.inner).prepare());
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    ///
    /// The last vector moves into the deleted slot — order is not
    /// preserved. Returns the old index of the moved vector; equals `idx`
    /// when `idx` was already the last element.
    ///
    /// Raises ``IndexError`` if ``idx`` is out of range. Negative indices
    /// are out of range: Python-style indexing from the end is not
    /// supported.
    fn swap_remove(&self, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = idx.extract::<usize>() {
            // Bounds check and removal share one write guard, so a
            // concurrent writer cannot shrink the index in between.
            let removed = py.detach(|| {
                let mut inner = lock_write(&self.inner);
                let len = inner.len();
                if i < len {
                    Ok(inner.swap_remove(i))
                } else {
                    Err(len)
                }
            });
            match removed {
                Ok(moved) => return Ok(moved),
                Err(len) => {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "index {} out of range for index of length {len}",
                        int_repr(idx),
                    )))
                }
            }
        }
        if !is_py_int(idx) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "idx must be an integer, got {}",
                type_name(idx),
            )));
        }
        // Any integer that isn't a valid slot — negative, or of any
        // magnitude past the end — is out of range.
        let len = py.detach(|| lock_read(&self.inner).len());
        Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "index {} out of range for index of length {len}",
            int_repr(idx),
        )))
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).len())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let (dim, bit_width, len) = py.detach(|| {
            let inner = lock_read(&self.inner);
            (inner.dim_opt(), inner.bit_width(), inner.len())
        });
        let dim = dim.map_or_else(|| "None".to_string(), |d| d.to_string());
        format!("turbovec.TurboQuantIndex(dim={dim}, bit_width={bit_width}, n_vectors={len})")
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily (no ``dim=``) and hasn't seen an add yet;
    /// otherwise an ``int``.
    #[getter]
    fn dim(&self, py: Python<'_>) -> Option<usize> {
        py.detach(|| lock_read(&self.inner).dim_opt())
    }

    #[getter]
    fn bit_width(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).bit_width())
    }
}

#[pyclass(frozen)]
struct IdMapIndex {
    inner: std::sync::RwLock<turbovec_core::IdMapIndex>,
}

#[pymethods]
impl IdMapIndex {
    /// Construct an id-mapped index. `dim` is optional: when omitted,
    /// the underlying quantized index is created lazily on the first
    /// `add_with_ids` call, picking up dim from the input array shape.
    /// `bit_width` defaults to 4.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=None))]
    fn new(dim: Option<&Bound<'_, PyAny>>, bit_width: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let bit_width = match bit_width {
            Some(b) => extract_size("bit_width", b)?,
            None => 4,
        };
        let inner = match dim {
            Some(d) => turbovec_core::IdMapIndex::new(extract_size("dim", d)?, bit_width),
            None => turbovec_core::IdMapIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
        })
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`.
    ///
    /// `ids` must be a 1-D array of `uint64` with length equal to
    /// `vectors.shape[0]`. Raises `ValueError` if any id is already
    /// present or if the lengths don't match. On a lazy index, this
    /// call commits the dimensionality from `vectors.shape[1]`.
    fn add_with_ids(
        &self,
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        ids: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let vectors = extract_f32_2d("vectors", vectors)?;
        let ids = extract_u64_1d("ids", ids)?;
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        // Snapshot both numpy buffers before releasing the GIL (another
        // Python thread may write to them once it is released); the core
        // then validates and quantizes the snapshot.
        let v_owned = v_slice.to_vec();
        let i_owned = i_slice.to_vec();
        py.detach(|| lock_write(&self.inner).add_with_ids_2d(&v_owned, dim, &i_owned))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise. Integers outside the `uint64` range are
    /// never present, so they return `False`.
    fn remove(&self, py: Python<'_>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => py.detach(|| lock_write(&self.inner).remove(v)),
            None => false,
        })
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// `allowlist`, when given, is a `uint64` array of external ids; the
    /// returned top-`k` is restricted to ids in this list. The allowlist
    /// is deduplicated: the returned result count per query is
    /// `min(k, number of unique ids in allowlist)`, so repeated ids
    /// don't widen the result.
    ///
    /// Returns `(scores, ids)` as `(nq, effective_k)` arrays, `ids` typed
    /// `uint64`. Raises `ValueError` for an empty allowlist and `KeyError`
    /// if any allowlist id is not present in the index.
    #[pyo3(signature = (queries, k, *, allowlist=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: &Bound<'py, PyAny>,
        allowlist: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let queries = extract_f32_2d("queries", queries)?;
        let k = extract_size("k", k)?;
        let allowlist = allowlist
            .map(|a| extract_u64_1d("allowlist", a))
            .transpose()?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let ncols = arr.ncols();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        // Snapshot the query (and allowlist) buffers before releasing
        // the GIL: once released, another Python thread may write to the
        // source arrays mid-search. Validation runs on the snapshot so
        // the searched data is exactly the data that was validated.
        let q_owned = q_slice.to_vec();

        let allow_arr = allowlist.as_ref().map(|a| a.as_array());
        let allow_owned: Option<Vec<u64>> = match allow_arr.as_ref() {
            Some(a_arr) => {
                if a_arr.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "allowlist is empty",
                    ));
                }
                Some(
                    a_arr
                        .as_slice()
                        .ok_or_else(|| not_contiguous_err("allowlist"))?
                        .to_vec(),
                )
            }
            None => None,
        };

        // Index-dependent checks (dim, allowlist membership) run under
        // the same read guard as the kernel, so a concurrent writer
        // cannot invalidate them between check and search. `len` is
        // captured under the same guard for the nq == 0 shape contract
        // below.
        let (scores, ids, len_at_search) = py.detach(|| {
            let inner = lock_read(&self.inner);
            if let Some(idx_dim) = inner.dim_opt() {
                if ncols != idx_dim {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "query dim {ncols} does not match index dim {idx_dim}",
                    )));
                }
            }
            validate_queries(&q_owned, ncols)?;
            if let Some(allow) = allow_owned.as_deref() {
                let mut unknown: Vec<u64> = Vec::new();
                for &id in allow {
                    if !inner.contains(id) {
                        if unknown.len() < 5 {
                            unknown.push(id);
                        } else {
                            unknown.push(id);
                            break;
                        }
                    }
                }
                if !unknown.is_empty() {
                    let preview: Vec<u64> = unknown.iter().take(5).copied().collect();
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "allowlist contains id(s) not present in index: {:?}{}",
                        preview,
                        if unknown.len() > 5 { ", ..." } else { "" },
                    )));
                }
            }
            let (scores, ids) = inner.search_with_allowlist(&q_owned, k, allow_owned.as_deref());
            Ok((scores, ids, inner.len()))
        })?;
        // For empty queries (nq=0), match TurboQuantIndex's shape
        // contract: effective_k is `min(k, n_vectors, n_allowed)`. The
        // kernel dedups the allowlist via a packed bool mask for nq>0,
        // so we have to dedup here too — otherwise `allowlist=[1, 1, 1]`
        // returns shape `(0, 3)` for empty queries but `(N, 1)` for
        // non-empty queries, a silent shape divergence.
        let effective_k = if nq == 0 {
            let n_allowed = match allow_owned.as_deref() {
                Some(s) => {
                    let mut seen: std::collections::HashSet<u64> =
                        std::collections::HashSet::with_capacity(s.len());
                    s.iter().filter(|id| seen.insert(**id)).count()
                }
                None => len_at_search,
            };
            k.min(len_at_search).min(n_allowed)
        } else {
            scores.len() / nq
        };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .map_err(shape_err)?
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .map_err(shape_err)?
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    /// Return `True` if external id `id` is present. Integers outside the
    /// `uint64` range are never present, so they return `False`.
    fn contains(&self, py: Python<'_>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => py.detach(|| lock_read(&self.inner).contains(v)),
            None => false,
        })
    }

    fn prepare(&self, py: Python<'_>) {
        py.detach(|| lock_read(&self.inner).prepare());
    }

    /// Serialize the index and id-map side-tables to a `.tvim` file.
    fn write(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| lock_read(&self.inner).write(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Load an `IdMapIndex` from a `.tvim` file previously written by
    /// [`IdMapIndex.write`].
    #[classmethod]
    fn load(cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = cls
            .py()
            .detach(|| turbovec_core::IdMapIndex::load(path))
            .map_err(|e| load_err(path, e))?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
        })
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).len())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let (dim, bit_width, len) = py.detach(|| {
            let inner = lock_read(&self.inner);
            (inner.dim_opt(), inner.bit_width(), inner.len())
        });
        let dim = dim.map_or_else(|| "None".to_string(), |d| d.to_string());
        format!("turbovec.IdMapIndex(dim={dim}, bit_width={bit_width}, n_vectors={len})")
    }

    fn __contains__(&self, py: Python<'_>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => py.detach(|| lock_read(&self.inner).contains(v)),
            None => false,
        })
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily and hasn't seen an add yet; otherwise ``int``.
    #[getter]
    fn dim(&self, py: Python<'_>) -> Option<usize> {
        py.detach(|| lock_read(&self.inner).dim_opt())
    }

    #[getter]
    fn bit_width(&self, py: Python<'_>) -> usize {
        py.detach(|| lock_read(&self.inner).bit_width())
    }
}

/// Cap applied to an explicit `RAYON_NUM_THREADS` request. Threads
/// beyond a small multiple of the hardware parallelism add no
/// throughput for turbovec's CPU-bound kernels, and a request past the
/// OS thread limit (`ulimit -u`) makes rayon's lazy pool construction
/// panic — surfacing as an opaque, uncatchable `PanicException` on the
/// first `add`/`search` (issue #158). 4x leaves room for deliberate
/// oversubscription; 1024 is the fallback when the hardware
/// parallelism cannot be determined.
fn rayon_thread_cap() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_mul(4))
        .unwrap_or(1024)
}

/// Build the global rayon pool eagerly at module init when
/// `RAYON_NUM_THREADS` is set, clamping over-large values to
/// [`rayon_thread_cap`] (with a `RuntimeWarning` naming the variable).
///
/// When the variable is unset — or holds a value rayon itself would
/// ignore (unparseable) or treat as "auto" (`0`) — this does nothing,
/// so rayon's lazy auto-sized initialization is preserved exactly.
/// Values at or under the cap eagerly build the pool with the same
/// thread count rayon's lazy path would have chosen, so results are
/// unchanged. Pool-build failure never fails the import.
fn init_rayon_pool(py: Python<'_>) -> PyResult<()> {
    let Ok(raw) = std::env::var("RAYON_NUM_THREADS") else {
        return Ok(());
    };
    // Mirror rayon's own parsing (plain `usize::from_str`, no trim):
    // anything rayon would ignore, we ignore; 0 means "auto" to rayon.
    let Ok(requested) = raw.parse::<usize>() else {
        return Ok(());
    };
    if requested == 0 {
        return Ok(());
    }
    let cap = rayon_thread_cap();
    let n = requested.min(cap);
    if n < requested {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (
                format!(
                    "RAYON_NUM_THREADS={requested} exceeds turbovec's thread cap \
                     of {cap} (4x available parallelism); capping at {cap} threads",
                ),
                py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            ),
        )?;
    }
    if rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .is_err()
    {
        // Either another extension already initialized the global pool
        // (harmless — its pool wins) or thread spawn failed even at the
        // clamped count. Retry once at the hardware default; if that
        // also fails, leave rayon to its lazy init. Import never fails.
        let default_n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(default_n)
            .build_global();
    }
    Ok(())
}

#[pymodule]
fn _turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_rayon_pool(m.py())?;
    m.add_class::<TurboQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    Ok(())
}
