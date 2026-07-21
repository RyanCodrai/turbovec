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

#[pyclass]
struct TurboQuantIndex {
    inner: turbovec_core::TurboQuantIndex,
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
        Ok(Self { inner })
    }

    fn add(&mut self, vectors: &Bound<'_, PyAny>) -> PyResult<()> {
        let vectors = extract_f32_2d("vectors", vectors)?;
        let arr = vectors.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("vectors"))?;
        // `add_2d` handles both eager (dim must match) and lazy (locks
        // dim on first call) cases.
        self.inner
            .add_2d(slice, dim)
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
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        // Reject wrong-dim queries cleanly. Previously the inner
        // `assert_eq!(queries.len(), nq * dim)` would fire as a Rust
        // panic and surface to Python as a PanicException, not the
        // ValueError users expect for input-shape mismatch.
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }
        validate_queries(q_slice, arr.ncols())?;

        let mask_arr = mask.as_ref().map(|m| m.as_array());
        let mask_slice: Option<&[bool]> = match mask_arr.as_ref() {
            Some(m_arr) => {
                let expected = self.inner.len();
                if m_arr.len() != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mask length {} does not match index size {}",
                        m_arr.len(),
                        expected,
                    )));
                }
                Some(m_arr.as_slice().ok_or_else(|| not_contiguous_err("mask"))?)
            }
            None => None,
        };

        let results = self.inner.search_with_mask(q_slice, k, mask_slice);
        let effective_k = results.k;

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.scores)
            .map_err(shape_err)?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.indices)
            .map_err(shape_err)?
            .into_pyarray(py);

        Ok((scores, indices))
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::TurboQuantIndex::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self) {
        self.inner.prepare();
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
    fn swap_remove(&mut self, idx: &Bound<'_, PyAny>) -> PyResult<usize> {
        let len = self.inner.len();
        if let Ok(i) = idx.extract::<usize>() {
            if i < len {
                return Ok(self.inner.swap_remove(i));
            }
        } else if !is_py_int(idx) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "idx must be an integer, got {}",
                type_name(idx),
            )));
        }
        // Any integer that isn't a valid slot — negative, or of any
        // magnitude past the end — is out of range.
        Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "index {} out of range for index of length {len}",
            int_repr(idx),
        )))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.TurboQuantIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily (no ``dim=``) and hasn't seen an add yet;
    /// otherwise an ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pyclass]
struct IdMapIndex {
    inner: turbovec_core::IdMapIndex,
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
        Ok(Self { inner })
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`.
    ///
    /// `ids` must be a 1-D array of `uint64` with length equal to
    /// `vectors.shape[0]`. Raises `ValueError` if any id is already
    /// present or if the lengths don't match. On a lazy index, this
    /// call commits the dimensionality from `vectors.shape[1]`.
    fn add_with_ids(&mut self, vectors: &Bound<'_, PyAny>, ids: &Bound<'_, PyAny>) -> PyResult<()> {
        let vectors = extract_f32_2d("vectors", vectors)?;
        let ids = extract_u64_1d("ids", ids)?;
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        self.inner
            .add_with_ids_2d(v_slice, dim, i_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise. Integers outside the `uint64` range are
    /// never present, so they return `False`.
    fn remove(&mut self, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => self.inner.remove(v),
            None => false,
        })
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// `allowlist`, when given, is a `uint64` array of external ids; the
    /// returned top-`k` is restricted to ids in this list. The returned
    /// result count per query is `min(k, len(allowlist))` (after
    /// de-duplication).
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
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }
        validate_queries(q_slice, arr.ncols())?;

        let allow_arr = allowlist.as_ref().map(|a| a.as_array());
        let allow_slice: Option<&[u64]> = match allow_arr.as_ref() {
            Some(a_arr) => {
                if a_arr.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "allowlist is empty",
                    ));
                }
                let slice = a_arr
                    .as_slice()
                    .ok_or_else(|| not_contiguous_err("allowlist"))?;
                let mut unknown: Vec<u64> = Vec::new();
                for &id in slice {
                    if !self.inner.contains(id) {
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
                Some(slice)
            }
            None => None,
        };

        let (scores, ids) = self.inner.search_with_allowlist(q_slice, k, allow_slice);
        // For empty queries (nq=0), match TurboQuantIndex's shape
        // contract: effective_k is `min(k, n_vectors, n_allowed)`. The
        // kernel dedups the allowlist via a packed bool mask for nq>0,
        // so we have to dedup here too — otherwise `allowlist=[1, 1, 1]`
        // returns shape `(0, 3)` for empty queries but `(N, 1)` for
        // non-empty queries, a silent shape divergence.
        let effective_k = if nq == 0 {
            let n_allowed = match allow_slice {
                Some(s) => {
                    let mut seen: std::collections::HashSet<u64> =
                        std::collections::HashSet::with_capacity(s.len());
                    s.iter().filter(|id| seen.insert(**id)).count()
                }
                None => self.inner.len(),
            };
            k.min(self.inner.len()).min(n_allowed)
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
    fn contains(&self, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => self.inner.contains(v),
            None => false,
        })
    }

    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Serialize the index and id-map side-tables to a `.tvim` file.
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Load an `IdMapIndex` from a `.tvim` file previously written by
    /// [`IdMapIndex.write`].
    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::IdMapIndex::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbovec.IdMapIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    fn __contains__(&self, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            Some(v) => self.inner.contains(v),
            None => false,
        })
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily and hasn't seen an add yet; otherwise ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pymodule]
fn _turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    Ok(())
}
