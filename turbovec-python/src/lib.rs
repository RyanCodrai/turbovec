use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass]
struct TurboQuantIndex {
    inner: turbovec_core::TurboQuantIndex,
}

#[pymethods]
impl TurboQuantIndex {
    #[new]
    fn new(dim: usize, bit_width: usize) -> Self {
        Self {
            inner: turbovec_core::TurboQuantIndex::new(dim, bit_width),
        }
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) {
        let arr = vectors.as_array();
        let slice = arr.as_slice().expect("vectors must be contiguous");
        self.inner.add(slice);
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>) {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().expect("queries must be contiguous");
        let results = self.inner.search(slice, k);

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .unwrap()
            .into_pyarray(py);

        (scores, indices)
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner.write(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::TurboQuantIndex::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })?;
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
    fn swap_remove(&mut self, idx: usize) -> usize {
        self.inner.swap_remove(idx)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
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
    #[new]
    fn new(dim: usize, bit_width: usize) -> Self {
        Self {
            inner: turbovec_core::IdMapIndex::new(dim, bit_width),
        }
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`.
    ///
    /// `ids` must be a 1-D array of `uint64` with length equal to
    /// `vectors.shape[0]`. Raises if any id is already present or if the
    /// lengths don't match.
    fn add_with_ids(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) {
        let v = vectors.as_array();
        let v_slice = v.as_slice().expect("vectors must be contiguous");
        let i = ids.as_array();
        let i_slice = i.as_slice().expect("ids must be contiguous");
        self.inner.add_with_ids(v_slice, i_slice);
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise.
    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// Returns `(scores, ids)` as `(nq, k)` arrays, `ids` typed `uint64`.
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>) {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().expect("queries must be contiguous");
        let (scores, ids) = self.inner.search(slice, k);
        let effective_k = if nq == 0 { k } else { scores.len() / nq };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .unwrap()
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .unwrap()
            .into_pyarray(py);
        (scores_arr, ids_arr)
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Serialize the index and id-map side-tables to a `.tvim` file.
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner.write(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })
    }

    /// Load an `IdMapIndex` from a `.tvim` file previously written by
    /// [`IdMapIndex.write`].
    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::IdMapIndex::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
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
