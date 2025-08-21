use nalgebra::{DMatrix, DVector};

#[derive(Debug)]
pub struct OLS {
    pub weights: DVector<f64>,
    pub bias: f64,
    pub is_fitted: bool,
}

impl OLS {
    pub fn new(num_features: usize) -> Self {
        Self {
            weights: DVector::zeros(num_features),
            bias: 0.0,
            is_fitted: false,
        }
    }

    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<(), String> {
        if x.nrows() != y.len() {
            return Err(format!(
                "Dimension mismatch: X has {} rows, y has {} elements",
                x.nrows(),
                y.len()
            ));
        }

        let x_t = x.transpose();
        let x_t_x = &x_t * x;

        let x_t_x_inv = x_t_x.try_inverse()
            .ok_or("Matrix X^T X is singular (not invertible). This usually means features are linearly dependent.")?;

        let x_t_y = &x_t * y;
        self.weights = x_t_x_inv * x_t_y;

        self.bias = 0.0;

        self.is_fitted = true;
        Ok(())
    }

    pub fn predict(&self, x: &DMatrix<f64>) -> Result<DVector<f64>, String> {
        if !self.is_fitted {
            return Err("Model must be fitted before making predictions".to_string());
        }

        if x.ncols() != self.weights.len() {
            return Err(format!(
                "Feature dimension mismatch: X has {} columns, model expects {} features",
                x.ncols(),
                self.weights.len()
            ));
        }

        let predictions = x * &self.weights;
        Ok(predictions)
    }

    pub fn get_weights(&self) -> &DVector<f64> {
        &self.weights
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }
}
