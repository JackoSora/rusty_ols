use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
mod ols_module;
use ols_module::ols::OLS;

#[pymodule]
fn ols(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOLS>()?;
    Ok(())
}

#[pyclass]
pub struct PyOLS {
    inner: OLS,
}

#[pymethods]
impl PyOLS {
    #[new]
    fn new(num_features: usize) -> Self {
        PyOLS {
            inner: OLS::new(num_features),
        }
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        let n_samples = x.len();
        let n_features = x[0].len();

        let x_flat: Vec<f64> = x.into_iter().flatten().collect();
        let x_matrix = DMatrix::from_row_slice(n_samples, n_features, &x_flat);
        let y_vector = DVector::from_vec(y);

        self.inner
            .fit(&x_matrix, &y_vector)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let n_samples = x.len();
        let n_features = x[0].len();

        let x_flat: Vec<f64> = x.into_iter().flatten().collect();
        let x_matrix = DMatrix::from_row_slice(n_samples, n_features, &x_flat);

        let predictions = self
            .inner
            .predict(&x_matrix)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        // Convert back to Python list
        Ok(predictions.as_slice().to_vec())
    }

    fn get_weights(&self) -> Vec<f64> {
        self.inner.get_weights().as_slice().to_vec()
    }

    fn get_bias(&self) -> f64 {
        self.inner.get_bias()
    }

    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    fn __repr__(&self) -> String {
        format!(
            "OLS(weights={:?}, bias={}, is_fitted={})",
            self.inner.weights, self.inner.bias, self.inner.is_fitted
        )
    }
}
