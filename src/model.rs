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

    /// Fits and forecasts the `h` periods after the last observation. `None`
    /// also when a forecast is not finite: regressors that do not reach the
    /// horizon, or a transformation that cannot be brought back.
    fn forecast(&self, y: Series<'_>, h: usize) -> Option<Vec<f64>> {
        let forecast = self.fit(y)?.forecast(h);
        forecast.iter().all(|v| v.is_finite()).then_some(forecast)
    }
}

/// A model estimated on a series. It is plain data: it can be kept, sent to
/// another thread and asked for forecasts from several at once.
pub trait Fitted: Send + Sync {
    /// Point forecasts for the `h` periods after the last observation. They
    /// are not finite where the model cannot say: past the rows of its
    /// regressors, for instance. [`Model::forecast`] checks that.
    fn forecast(&self, h: usize) -> Vec<f64>;

    /// Estimated parameters.
    fn params(&self) -> Params {
        Vec::new()
    }
}
