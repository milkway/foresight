//! Forecasting by decomposition: take the seasonal patterns out, forecast
//! what is left, put them back.

use crate::decompose::Mstl;
use crate::model::{Fitted, Model, Params};
use crate::series::Series;

/// Runs a model on the seasonally adjusted series and adds the seasonal
/// patterns of the last cycle to its forecasts.
///
/// The decomposition is [`Mstl`], so the series may have several seasonal
/// periods; by default the period is the one of the series. The model inside
/// sees a series without seasonality. For patterns that grow with the level,
/// wrap the whole in [`Transformed`](crate::transform::Transformed).
///
/// ```
/// use foresight::{models::{Decomposed, Drift}, Model, Series};
///
/// let y: Vec<f64> = (0..60).map(|t| 20.0 + t as f64 + [6.0, -2.0, -4.0, 0.0][t % 4]).collect();
/// let forecast = Decomposed::new(Drift).forecast(Series::quarterly(&y, 0), 4).unwrap();
/// assert!((forecast[0] - (20.0 + 60.0 + 6.0)).abs() < 0.5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposed<M> {
    model: M,
    periods: Option<Vec<usize>>,
    robust: bool,
}

impl<M: Model> Decomposed<M> {
    pub fn new(model: M) -> Self {
        Decomposed {
            model,
            periods: None,
            robust: false,
        }
    }

    /// Seasonal periods to take out, instead of the period of the series.
    pub fn periods(mut self, periods: &[usize]) -> Self {
        self.periods = Some(periods.to_vec());
        self
    }

    /// Down-weights outliers in the decomposition.
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }
}

struct DecomposedFit {
    inner: Box<dyn Fitted>,
    /// The last cycle of each seasonal pattern.
    cycles: Vec<Vec<f64>>,
}

impl Fitted for DecomposedFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        self.inner
            .forecast(h)
            .into_iter()
            .enumerate()
            .map(|(k, v)| v + self.cycles.iter().map(|c| c[k % c.len()]).sum::<f64>())
            .collect()
    }

    fn params(&self) -> Params {
        let mut p = self.inner.params();
        p.push(("seasonal_periods".into(), self.cycles.len() as f64));
        p
    }
}

impl<M: Model> Model for Decomposed<M> {
    fn name(&self) -> String {
        format!("stl_{}", self.model.name())
    }

    fn description(&self) -> String {
        format!(
            "{}, on the series seasonally adjusted by STL",
            self.model.description()
        )
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let periods = match &self.periods {
            Some(p) => p.clone(),
            None => vec![y.period()],
        };
        if periods.iter().all(|p| *p < 2) {
            // nothing to take out
            return self.model.fit(y);
        }
        let n = y.len();
        let d = Mstl::new(&periods)
            .robust(self.robust)
            .decompose(y.values())?;
        let adjusted = d.seasonally_adjusted();
        let inner = self.model.fit(Series::non_seasonal(&adjusted))?;
        let cycles = d
            .periods
            .iter()
            .zip(&d.seasonal)
            .map(|(p, s)| s[n - p..].to_vec())
            .collect();
        Some(Box::new(DecomposedFit { inner, cycles }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Drift, Naive};
    use crate::testing::noisy_seasonal_growth;
    use crate::transform::Transformed;

    #[test]
    fn follows_trend_and_seasonality_on_the_log_scale() {
        let all = noisy_seasonal_growth(132, 0.04);
        let model = Transformed::log(Decomposed::new(Drift));
        assert_eq!(model.name(), "log_stl_drift");
        let p = model.forecast(Series::monthly(&all[..120], 0), 12).unwrap();
        for (k, v) in p.iter().enumerate() {
            assert!((v / all[120 + k] - 1.0).abs() < 0.06, "h={}", k + 1);
        }
    }

    #[test]
    fn the_forecast_is_the_inner_forecast_plus_the_last_cycle() {
        let y = noisy_seasonal_growth(96, 0.05);
        let series = Series::monthly(&y, 0);
        let d = Mstl::new(&[12]).decompose(&y).unwrap();
        let adjusted = d.seasonally_adjusted();
        let fit = Decomposed::new(Naive).fit(series).unwrap();
        let p = fit.forecast(14);
        for (k, v) in p.iter().enumerate() {
            let expected = adjusted[95] + d.seasonal[0][84 + k % 12];
            assert!((v - expected).abs() < 1e-9);
        }
        assert!(fit
            .params()
            .contains(&("seasonal_periods".to_string(), 1.0)));
    }

    #[test]
    fn without_seasonality_the_model_runs_as_it_is() {
        let y: Vec<f64> = (0..30).map(|t| 5.0 + 2.0 * t as f64).collect();
        let series = Series::non_seasonal(&y);
        assert_eq!(
            Decomposed::new(Drift).forecast(series, 3),
            Drift.forecast(series, 3)
        );
        // a period that does not fit twice in the series cannot be taken out
        assert!(Decomposed::new(Drift)
            .forecast(Series::new(&y, 24), 3)
            .is_none());
    }
}
