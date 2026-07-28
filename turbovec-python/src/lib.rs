use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

mod par_copy;
use par_copy::{par_copy_into, PAR_COPY_MIN_LEN};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};

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

/// Extract a bytes-like argument (`bytes` / `bytearray`) into an owned
/// buffer; see [`extract_f32_2d`] for the error-naming rationale. Owned
/// because the buffer must cross a GIL release (`py.detach`), where a
/// borrowed Python buffer would be mutable out from under us.
fn extract_bytes(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(b) = obj.cast::<pyo3::types::PyByteArray>() {
        return Ok(b.to_vec());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{name} must be a bytes-like object (bytes or bytearray), got {}",
        type_name(obj),
    )))
}

/// Map an `io::Error` from `from_bytes` to Python. Unlike [`load_err`]
/// there is no path to blame, and the input is an in-memory value —
/// so corrupt bytes raise `ValueError` (like other malformed-value
/// arguments) rather than `OSError`.
fn from_bytes_err(e: std::io::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
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

/// Write-lock for O(1) ops called while attached to the GIL: try the
/// lock without blocking (the uncontended fast path — cheaper than a
/// GIL detach round-trip), and only when another thread holds the lock
/// (e.g. a multi-second bulk add) detach before waiting, so a brief
/// removal never stalls every Python thread behind a long writer.
fn lock_write_gil_aware<'l, T: Send + Sync>(
    py: Python<'_>,
    lock: &'l std::sync::RwLock<T>,
) -> std::sync::RwLockWriteGuard<'l, T> {
    loop {
        match lock.try_write() {
            Ok(guard) => return guard,
            Err(std::sync::TryLockError::Poisoned(p)) => return p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                // A guard cannot cross the GIL release, so wait for the
                // lock detached (acquire and immediately drop), then
                // retry attached. Another thread may slip in between —
                // the loop just waits again; each iteration makes
                // progress once the long writer finishes.
                py.detach(|| drop(lock_write(lock)));
            }
        }
    }
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
    /// Reusable GIL-safety snapshot buffer for `add`. Purely scratch:
    /// contents are meaningless between calls; kept so repeated adds
    /// reuse one warm allocation instead of paying a fresh multi-MB
    /// mmap + page-fault walk per call. Taken (mem::take) for the
    /// duration of one add and put back after, so concurrent adds
    /// simply fall back to a fresh buffer.
    snap: std::sync::Mutex<Vec<f32>>,
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
            snap: std::sync::Mutex::new(Vec::new()),
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
        // quantization kernel — but done serially it was the largest
        // single-threaded span left on the add path, so split it across
        // the rayon pool (same pool, and therefore the same
        // RAYON_NUM_THREADS clamp, as the encode kernels). The copy runs
        // under `with_pool` while the GIL is still held — the sentinel
        // global pool is single-threaded, so a bare par_iter would fold
        // serial. No index lock is taken inside the closure (see the
        // `with_pool` invariant). Small inputs skip the pool handoff
        // entirely: a single-vector add must not pay an `install` for a
        // 6 KB memcpy.
        // Reuse the per-index snapshot buffer so repeated adds keep one
        // warm allocation (a fresh multi-MB buffer per call costs a
        // page-fault walk). Taken out of the mutex for the duration of
        // this add; a concurrent add simply starts from an empty
        // buffer.
        let mut owned = std::mem::take(&mut *self.snap.lock().expect("snap lock poisoned"));
        if slice.len() < PAR_COPY_MIN_LEN {
            owned.clear();
            owned.extend_from_slice(slice);
        } else {
            with_pool(|| par_copy_into(slice, &mut owned))?;
        }
        // `add_2d` handles both eager (dim must match) and lazy (locks
        // dim on first call) cases.
        //
        // Single-row adds take the same inline bypass as nq==1 search:
        // every rayon bridge in a one-row encode (normalize, rotate,
        // quantize, and the sub-chunk validation scan) has length 1 and
        // folds on the calling thread, so skipping the pool `install`
        // handoff cannot change results — it only removes the per-call
        // latency. The forked-child guard mirrors `with_pool_if`.
        let n_rows = if dim == 0 { 0 } else { slice.len() / dim };
        py.detach(|| {
            let mut guard = lock_write(&self.inner);
            let inner = &mut *guard;
            with_pool_if(n_rows > 1, || inner.add_2d(&owned, dim))
        })?
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        *self.snap.lock().expect("snap lock poisoned") = owned;
        Ok(())
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
        // whole call atomic). The guard is taken on the calling thread —
        // never inside `with_pool` — see the invariant on `with_pool`.
        // nq > 1 splits over queries and runs in the fork-safe pool; nq == 1
        // never splits and runs inline unless we are in a forked child (see
        // `with_pool_if`).
        let results = py.detach(|| -> PyResult<_> {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
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
            with_pool_if(nq > 1, || {
                inner.search_with_mask(&q_owned, k, mask_owned.as_deref())
            })
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
            snap: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Serialize the index to ``bytes`` in the ``.tv`` format —
    /// byte-identical to the file ``write(path)`` produces. Pairs with
    /// ``from_bytes`` for in-memory persistence (caches, databases,
    /// pickling) without a filesystem round-trip.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        // Serialize under the read lock with the GIL released (the
        // payload scales with the index size); wrap into a PyBytes
        // only once back under the GIL.
        let buf = py.detach(|| lock_read(&self.inner).to_bytes());
        PyBytes::new(py, &buf)
    }

    /// Deserialize an index from ``bytes`` produced by ``to_bytes`` (or
    /// read out of a ``.tv`` file). Applies exactly the same validation
    /// as ``load`` — corrupt or drifted payloads raise ``ValueError``.
    #[classmethod]
    fn from_bytes(cls: &Bound<PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Snapshot the buffer before releasing the GIL (a bytearray can
        // be mutated by another Python thread once it is released).
        let owned = extract_bytes("data", data)?;
        let inner = cls
            .py()
            .detach(|| turbovec_core::TurboQuantIndex::from_bytes(&owned))
            .map_err(from_bytes_err)?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
            with_pool(|| inner.prepare())
        })
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
            //
            // No GIL detach on the uncontended path: the removal is
            // O(1), cheaper than the detach round-trip. When another
            // thread holds the write lock (a long bulk add), the
            // gil-aware helper detaches before waiting so other Python
            // threads keep running.
            let removed = {
                let mut inner = lock_write_gil_aware(py, &self.inner);
                let len = inner.len();
                if i < len {
                    Ok(inner.swap_remove(i))
                } else {
                    Err(len)
                }
            };
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
    /// See `TurboQuantIndex::snap`.
    snap: std::sync::Mutex<Vec<f32>>,
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
            snap: std::sync::Mutex::new(Vec::new()),
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
        // then validates and quantizes the snapshot. The vector snapshot
        // reuses the per-index buffer (see `TurboQuantIndex::add`).
        let mut v_owned =
            std::mem::take(&mut *self.snap.lock().expect("snap lock poisoned"));
        if v_slice.len() < PAR_COPY_MIN_LEN || !pool_idle() {
            // Small input, or the pool is busy with a long batch —
            // bounded serial copy while attached (see TurboQuantIndex::add).
            v_owned.clear();
            v_owned.extend_from_slice(v_slice);
        } else {
            with_pool(|| par_copy_into(v_slice, &mut v_owned))?;
        }
        let i_owned = i_slice.to_vec();
        py.detach(|| {
            let mut guard = lock_write(&self.inner);
            let inner = &mut *guard;
            with_pool(|| inner.add_with_ids_2d(&v_owned, dim, &i_owned))
        })?
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Same shrink policy as TurboQuantIndex::add.
        let mut v_owned = v_owned;
        if v_owned.capacity() > 4 * v_slice.len() {
            v_owned.shrink_to(v_slice.len());
        }
        *self.snap.lock().expect("snap lock poisoned") = v_owned;
        Ok(())
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise. Integers outside the `uint64` range are
    /// never present, so they return `False`.
    fn remove(&self, py: Python<'_>, id: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(match extract_membership_id("id", id)? {
            // No GIL detach on the uncontended path: the removal is O(1)
            // (two hash-map updates and a swap), far cheaper than the
            // detach round-trip. When another thread holds the write
            // lock (a long bulk add), the gil-aware helper detaches
            // before waiting so other Python threads keep running.
            Some(v) => lock_write_gil_aware(py, &self.inner).remove(v),
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
        // cannot invalidate them between check and search. The guard is
        // taken on the calling thread — never inside `with_pool` — see
        // the invariant on `with_pool`. `len` is captured under the same
        // guard for the nq == 0 shape contract below.
        let (scores, ids, len_at_search) = py.detach(|| -> PyResult<_> {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
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
            let (scores, ids) = with_pool_if(nq > 1, || {
                inner.search_with_allowlist(&q_owned, k, allow_owned.as_deref())
            })?;
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

    fn prepare(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let guard = lock_read(&self.inner);
            let inner = &*guard;
            with_pool(|| inner.prepare())
        })
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
            snap: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Serialize the index (and its id-map side-tables) to ``bytes`` in
    /// the ``.tvim`` format — byte-identical to the file ``write(path)``
    /// produces. Pairs with ``from_bytes`` for in-memory persistence
    /// (caches, databases, pickling) without a filesystem round-trip.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        // Serialize under the read lock with the GIL released (the
        // payload scales with the index size); wrap into a PyBytes
        // only once back under the GIL.
        let buf = py.detach(|| lock_read(&self.inner).to_bytes());
        PyBytes::new(py, &buf)
    }

    /// Deserialize an index from ``bytes`` produced by ``to_bytes`` (or
    /// read out of a ``.tvim`` file). Applies exactly the same
    /// validation as ``load`` — corrupt or drifted payloads raise
    /// ``ValueError``.
    #[classmethod]
    fn from_bytes(cls: &Bound<PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Snapshot the buffer before releasing the GIL (a bytearray can
        // be mutated by another Python thread once it is released).
        let owned = extract_bytes("data", data)?;
        let inner = cls
            .py()
            .detach(|| turbovec_core::IdMapIndex::from_bytes(&owned))
            .map_err(from_bytes_err)?;
        Ok(Self {
            inner: std::sync::RwLock::new(inner),
            snap: std::sync::Mutex::new(Vec::new()),
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

// ===========================================================================
// Fork safety (issue #147)
// ===========================================================================
//
// rayon's global thread pool does not survive `fork()`: the worker threads
// live only in the parent, so the first parallel op a forked child runs
// injects work into a registry whose workers are gone and blocks on the
// latch forever. This is the default failure mode of `multiprocessing`
// (fork start method, the Linux default through 3.13), gunicorn
// `--preload`, Celery prefork, and PyTorch `DataLoader(num_workers>0)`.
//
// Fix: route every rayon-using kernel entry through `with_pool`, which
// installs a *process-local* `ThreadPool`. A forked child detects the fork
// and transparently rebuilds that pool, so its parallel ops run on live
// workers instead of the parent's dead ones.
//
// Fork detection (defence in depth, most authoritative first):
//   1. Python's `os.register_at_fork(after_in_child=...)` bumps
//      FORK_GENERATION. This is the authoritative signal and fires for
//      every CPython-mediated fork (os.fork, multiprocessing, billiard).
//      It is immune to the glibc `getpid()` userspace cache that would
//      otherwise let a child on manylinux2014 (glibc 2.17, cache not
//      removed until 2.25) observe the parent's PID and skip the rebuild —
//      the exact silent hang this fix targets (review finding F1).
//   2. A `pthread_atfork` child handler (Unix) also bumps FORK_GENERATION,
//      catching forks issued by non-Python native code that never runs the
//      Python hook.
//   3. A `getpid()` secondary confirm in `current_pool`: even if both
//      handlers somehow miss, a changed PID forces a rebuild on any modern
//      glibc where getpid is a live syscall.
//
// The nq==1 inline fast path (see `with_pool_if`) never enters any pool, so
// a missed detection there is still safe: length-1 parallel bridges fold on
// the calling thread. Only splitting workloads (nq>1 search, add, prepare)
// enter a pool, and those all go through `current_pool`'s generation+PID
// gate.

/// Bumped by every fork-detecting handler. A child's copy is CoW-inherited
/// from the parent then incremented, so within a fork lineage the child's
/// generation is always strictly greater than the pool cell built by its
/// parent — the comparison in `current_pool` cannot false-negative.
static FORK_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Generation the currently-published pool was built at, mirrored out of
/// the `PoolCell` so the nq==1 bypass can decide "are we in a fresh child?"
/// with two relaxed atomic loads and no pointer dereference. `u64::MAX`
/// means "no pool built yet".
static POOL_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// The process-local pool plus the identity of the process/generation that
/// built it.
struct PoolCell {
    /// FORK_GENERATION value when this pool was built.
    gen: u64,
    /// PID that built this pool (secondary fork confirm, review F1).
    pid: u32,
    /// Thread count this pool was built with. A forked child rebuilds at
    /// the *parent's* recorded count rather than re-reading
    /// `available_parallelism`, so a child under a different cgroup CPU
    /// quota / `taskset` cannot silently rebuild at a different thread
    /// count (which would be a fresh source of the #206 rotation/encode
    /// thread-count sensitivity — review finding F2).
    threads: usize,
    pool: rayon::ThreadPool,
}

/// The live pool cell, or null before the first build.
///
/// Leak-not-drop is *structural* here: cells are `Box::leak`ed and this
/// pointer is only ever overwritten, never used to free. A `ThreadPool`
/// inherited across fork must never be dropped — `ThreadPool::drop`
/// terminates the registry and wakes workers, deadlocking on pthread
/// mutexes inherited in an undefined state (observed on macOS via `sample`:
/// `Arc::drop_slow` -> `Sleep::wake_specific_thread` ->
/// `_pthread_mutex_firstfit_lock_slow`, hung forever). Never freeing any
/// cell sidesteps that entirely and bounds the leak to <=1 pool per forked
/// process (a process's generation changes at most once per fork it
/// performs; a parent that forks N children never leaks, each child leaks
/// exactly the parent pool it inherited — a few KB, CoW-cheap).
static POOL_PTR: std::sync::atomic::AtomicPtr<PoolCell> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Record that we are now running in a forked child (called from the fork
/// handlers). Only a single atomic increment — async-signal-safe, so it is
/// legal from a `pthread_atfork` child handler.
fn note_fork_in_child() {
    FORK_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// pthread_atfork child callback (Unix). Must do nothing but async-signal-
/// safe work.
#[cfg(unix)]
extern "C" fn atfork_child_handler() {
    note_fork_in_child();
}

/// Python-callable shim registered via `os.register_at_fork`.
#[pyfunction]
fn _note_fork_in_child() {
    note_fork_in_child();
}

/// Thread count for the process-local pool: `RAYON_NUM_THREADS` (clamped to
/// [`rayon_thread_cap`], mirroring the eager-init rules) when set, else the
/// hardware default rayon itself would choose.
fn desired_threads() -> usize {
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(n) if n > 0 => n.min(rayon_thread_cap()),
        _ => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    }
}

/// Build a process-local pool of `threads` workers. A build failure (thread
/// spawn refused, e.g. a resource-exhausted process) is retried once at a
/// single thread; if even that fails the error is returned so callers can
/// surface a catchable `RuntimeError` (never an uncatchable panic) or, at
/// import, degrade gracefully.
fn try_build_pool(threads: usize) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .or_else(|_| rayon::ThreadPoolBuilder::new().num_threads(1).build())
}

/// A pool-build failure surfaced to Python. `PyRuntimeError` subclasses
/// `Exception`, so ordinary `except Exception:` handling catches it (unlike a
/// Rust panic, which arrives as an uncatchable `PanicException`).
fn pool_build_error(threads: usize, e: &rayon::ThreadPoolBuildError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "turbovec: could not create its thread pool ({e}). The process may be \
         out of thread resources (see `ulimit -u`, cgroup pids limits, or a \
         too-large RAYON_NUM_THREADS={threads}). If this is a forked worker, \
         use the 'spawn' start method or fork before the parent's first \
         turbovec operation."
    ))
}

/// Return the pool for the current process, rebuilding it if we have forked
/// since it was built. Fast path: one acquire load + a generation/PID
/// compare (no lock), so concurrent readers under #208's shared read lock
/// never serialize here. Returns `Err` only when a fresh pool cannot be
/// built at all (see `try_build_pool`).
fn current_pool() -> PyResult<&'static PoolCell> {
    use std::sync::atomic::Ordering;
    let gen = FORK_GENERATION.load(Ordering::Acquire);
    let pid = std::process::id();
    let p = POOL_PTR.load(Ordering::Acquire);
    if !p.is_null() {
        // SAFETY: cells are `Box::leak`ed and never freed, so any non-null
        // pointer we ever store is valid for the process lifetime.
        let cell = unsafe { &*p };
        if cell.gen == gen && cell.pid == pid {
            return Ok(cell);
        }
    }
    rebuild_pool(gen, pid)
}

/// Cold path: build and publish a fresh cell. Lock-free (a CAS publish), so
/// it cannot itself deadlock even if a fork races another thread's rebuild.
#[cold]
fn rebuild_pool(gen: u64, pid: u32) -> PyResult<&'static PoolCell> {
    use std::sync::atomic::Ordering;
    // Reuse the parent's recorded thread count across a fork (F2); only the
    // very first build in a process (null pointer) reads the environment.
    let prev = POOL_PTR.load(Ordering::Acquire);
    let threads = if prev.is_null() {
        desired_threads()
    } else {
        // SAFETY: see current_pool — leaked, never freed.
        unsafe { &*prev }.threads
    };
    let pool = try_build_pool(threads).map_err(|e| pool_build_error(threads, &e))?;
    let cell: &'static PoolCell = Box::leak(Box::new(PoolCell {
        gen,
        pid,
        threads,
        pool,
    }));
    loop {
        let cur = POOL_PTR.load(Ordering::Acquire);
        if !cur.is_null() {
            // SAFETY: leaked, never freed.
            let c = unsafe { &*cur };
            if c.gen == gen && c.pid == pid {
                // Another thread already published an equivalent cell (a
                // startup race — a fresh fork has only the calling thread,
                // so this is only reachable at process start). Adopt theirs;
                // our freshly built `cell` leaks. Its workers are alive in
                // this process but idle forever: bounded and rare.
                return Ok(c);
            }
        }
        match POOL_PTR.compare_exchange(
            cur,
            cell as *const PoolCell as *mut PoolCell,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                POOL_GENERATION.store(gen, Ordering::Release);
                return Ok(cell);
            }
            // Lost the race; retry. `cell` stays valid (leaked).
            Err(_) => continue,
        }
    }
}

/// True when we appear to be in a forked child that has not yet rebuilt its
/// pool — used to route the nq==1 fast path through `install` so the child
/// rebuilds its pool promptly on first use. Two relaxed loads, no getpid, no
/// dereference: this is on the hot single-query search path.
fn in_forked_child() -> bool {
    use std::sync::atomic::Ordering;
    FORK_GENERATION.load(Ordering::Acquire) != POOL_GENERATION.load(Ordering::Acquire)
}

/// Run `f` inside the process-local, fork-safe pool so every `par_iter`
/// beneath it uses that pool instead of rayon's global (fork-unsafe) one.
///
/// INVARIANT: `f` must not acquire the index `RwLock` (or any lock another
/// `with_pool` closure can hold). `install` runs `f` on a pool WORKER
/// thread, and a worker parked at a `join` point work-steals other injected
/// closures — so a lock taken inside `f` can be re-entered on a thread that
/// already holds it (read-while-holding-write across two stolen ops),
/// deadlocking permanently. Call sites therefore lock on the calling thread
/// (which blocks on rayon's latch and never steals) and move only a plain
/// `&T` / `&mut T` into `f`.
fn with_pool<R: Send>(f: impl FnOnce() -> R + Send) -> PyResult<R> {
    POOL_IN_FLIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let result = current_pool().map(|c| c.pool.install(f));
    POOL_IN_FLIGHT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    result
}

/// Number of `with_pool` jobs currently running or queued. Used by
/// GIL-attached callers (the add snapshot copy) to avoid queueing behind
/// a long-running batch while holding the GIL: when the pool is busy
/// they fall back to bounded serial work instead of waiting. The check
/// is advisory — a race simply means one snapshot copies serially (or
/// briefly queues), never an error.
static POOL_IN_FLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// True when no `with_pool` job is currently in flight.
fn pool_idle() -> bool {
    POOL_IN_FLIGHT.load(std::sync::atomic::Ordering::Relaxed) == 0
}

/// Hybrid dispatch. `parallel == false` (single-query search) runs inline on
/// the calling thread with no ~70 us pool handoff — every rayon site in the
/// search path parallelizes over queries, so a single query never splits.
///
/// The inline bypass is taken only when we are NOT in a fresh forked child,
/// so a child's first op routes through `install` and rebuilds its pool
/// promptly. Note that once a child HAS rebuilt, `in_forked_child` is false
/// again and its single-query searches go inline just like the parent's —
/// which is still safe, because a length-1 parallel iterator folds on the
/// calling thread and never enters (the child's own live, or the dead
/// inherited) pool. Inline single-query correctness thus rests on rayon's
/// no-split-at-length-1 behavior in every process; `test_nq1_*` guards it and
/// the rayon pin (see Cargo.toml) holds the version it was measured on.
fn with_pool_if<R: Send>(parallel: bool, f: impl FnOnce() -> R + Send) -> PyResult<R> {
    if parallel || in_forked_child() {
        with_pool(f)
    } else {
        Ok(f())
    }
}

/// Pin this extension's rayon *global* pool to a single sentinel thread. All
/// real parallelism runs in the process-local pool via `with_pool`; the
/// global registry is touched only by the nq==1 inline path, which reads its
/// thread count but never injects work. Without this, that read lazily
/// builds a full-size global pool — a whole redundant set of idle workers.
/// turbovec's rayon-core is statically linked and private to this extension,
/// so the sentinel cannot affect other rayon users (polars, tokenizers) in
/// the process.
fn pin_global_pool_sentinel() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
}

/// Register the fork-detection handlers. The Python hook is authoritative;
/// pthread_atfork is a Unix belt-and-suspenders for non-Python forks.
fn register_fork_handlers(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(unix)]
    // SAFETY: pthread_atfork with a child handler that only does an atomic
    // increment (async-signal-safe). Registered once at module init.
    unsafe {
        libc::pthread_atfork(None, None, Some(atfork_child_handler));
    }
    // os.register_at_fork exists on platforms with fork (Unix); on Windows
    // it is absent and there is no fork, so skipping it is correct.
    let os = py.import("os")?;
    if let Ok(register) = os.getattr("register_at_fork") {
        let func = module.getattr("_note_fork_in_child")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("after_in_child", func)?;
        register.call((), Some(&kwargs))?;
    }
    Ok(())
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
    // Eagerly build the process-*local* pool (issue #147). The pre-#147
    // code built rayon's *global* pool here, which is exactly what poisoned
    // a child forked after import when `RAYON_NUM_THREADS` was set: the dead
    // global pool was inherited and the child's first parallel op hung. The
    // local pool is fork-safe — a child rebuilds it on first use — and it
    // honors `RAYON_NUM_THREADS` via `desired_threads` (`n` is validated and
    // clamped above only to drive the warning; `desired_threads` re-derives
    // the same count).
    //
    // A build failure here is deliberately SWALLOWED so import never fails
    // (preserving #158's "import never fails" guarantee): if the pool cannot
    // be built now, nothing is published and the next kernel call retries via
    // `current_pool`, raising a catchable `RuntimeError` then if it still
    // fails — at call time, not import time.
    let _ = n;
    let _ = current_pool();
    Ok(())
}

#[pymodule]
fn _turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pin the 1-thread global sentinel BEFORE anything can touch rayon, so
    // the nq==1 inline path never lazily builds a full global pool.
    pin_global_pool_sentinel();
    m.add_function(wrap_pyfunction!(_note_fork_in_child, m)?)?;
    register_fork_handlers(m.py(), m)?;
    init_rayon_pool(m.py())?;
    m.add_class::<TurboQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    Ok(())
}
