use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
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

    /// Remove the vector at `idx` (swap-and-pop).
    ///
    /// Returns the old index of the moved vector; equals `idx` when
    /// `idx` was already the last element.
    fn delete(&mut self, idx: usize) -> usize {
        self.inner.delete(idx)
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

#[pymodule]
fn turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantIndex>()?;
    Ok(())
}
