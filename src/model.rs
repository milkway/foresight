//! The two traits every forecasting method implements.

use crate::series::Series;

/// Named numeric parameters of a fitted model, for display and audit trails.
pub type Params = Vec<(String, f64)>;

/// A forecasting method, before seeing any data.
///
/// Fitting is separate from forecasting so a fitted model can be inspected
/// and asked for any horizon without being estimated again. Models are plain
/// configuration, shared between the threads of a backtest.
pub trait Model: Send + Sync {
    /// Short identifier, e.g. `holt_winters`.
    fn name(&self) -> String;

    /// One-line description of the method.
    fn description(&self) -> String;

    /// Estimates the model. `None` when the series is too short or otherwise
    /// unsuitable — never a made-up fit.
    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>>;

    /// Fits and forecasts the `h` periods after the last observation.
    fn forecast(&self, y: Series<'_>, h: usize) -> Option<Vec<f64>> {
        Some(self.fit(y)?.forecast(h))
    }
}

/// A model estimated on a series.
pub trait Fitted {
    /// Point forecasts for the `h` periods after the last observation.
    fn forecast(&self, h: usize) -> Vec<f64>;

    /// Estimated parameters.
    fn params(&self) -> Params {
        Vec::new()
    }
}
