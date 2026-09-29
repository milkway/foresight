//! Box-Cox transformations, and a wrapper that runs any model on the
//! transformed scale.

use crate::model::{Fitted, Model, Params};
use crate::optimize::golden;
use crate::series::Series;

/// The Box-Cox family: log for λ = 0, (y^λ − 1)/λ otherwise. Defined for
/// positive values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxCox {
    lambda: f64,
}

impl BoxCox {
    pub fn new(lambda: f64) -> Self {
        BoxCox { lambda }
    }

    /// The logarithm (λ = 0).
    pub fn log() -> Self {
        BoxCox { lambda: 0.0 }
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn apply(&self, y: f64) -> f64 {
        if self.lambda == 0.0 {
            y.ln()
        } else {
            (y.powf(self.lambda) - 1.0) / self.lambda
        }
    }

    /// Back to the original scale. This is the median of the forecast
    /// distribution, not its mean.
    pub fn invert(&self, z: f64) -> f64 {
        if self.lambda == 0.0 {
            z.exp()
        } else {
            (self.lambda * z + 1.0).max(0.0).powf(1.0 / self.lambda)
        }
    }

    /// λ in [−1, 2] that makes the variability most even across the cycles of
    /// the series (Guerrero, 1993): it minimises the coefficient of variation
    /// of sd / mean^(1 − λ) over blocks of one period (two observations when
    /// there is no seasonality). `None` without at least two blocks of
    /// positive values.
    pub fn guerrero(y: Series<'_>) -> Option<Self> {
        if !y.is_positive() {
            return None;
        }
        let size = y.period().max(2);
        let blocks = y.len() / size;
        if blocks < 2 {
            return None;
        }
        let v = &y.values()[y.len() - blocks * size..];
        let stats: Vec<(f64, f64)> = v
            .chunks(size)
            .map(|b| {
                let mean = b.iter().sum::<f64>() / size as f64;
                let var =
                    b.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (size - 1) as f64;
                (mean, var.sqrt())
            })
            .collect();
        let cv = |lambda: f64| {
            let r: Vec<f64> = stats
                .iter()
                .map(|(mean, sd)| sd / mean.powf(1.0 - lambda))
                .collect();
            let mean = r.iter().sum::<f64>() / r.len() as f64;
            let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (r.len() - 1) as f64;
            let value = var.sqrt() / mean;
            if value.is_finite() {
                value
            } else {
                f64::INFINITY
            }
        };
        Some(BoxCox::new(golden(cv, -1.0, 2.0)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Choice {
    Fixed(BoxCox),
    Guerrero,
}

/// Runs a model on Box-Cox transformed values and brings the forecasts back
/// to the original scale. The series must be positive.
///
/// ```
/// use foresight::{models::Arima, transform::Transformed, Model, Series};
///
/// let y: Vec<f64> = (0..60).map(|t| 100.0 * 1.01f64.powi(t)).collect();
/// let model = Transformed::log(Arima::new(0, 1, 0).constant(true));
/// let forecast = model.forecast(Series::non_seasonal(&y), 3).unwrap();
/// assert!((forecast[0] / (100.0 * 1.01f64.powi(60)) - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct Transformed<M> {
    model: M,
    choice: Choice,
}

impl<M: Model> Transformed<M> {
    /// The model on the log scale.
    pub fn log(model: M) -> Self {
        Self::box_cox(model, 0.0)
    }

    /// The model on the Box-Cox scale with the given λ.
    pub fn box_cox(model: M, lambda: f64) -> Self {
        Transformed {
            model,
            choice: Choice::Fixed(BoxCox::new(lambda)),
        }
    }

    /// The model on the Box-Cox scale with λ chosen by
    /// [`BoxCox::guerrero`] at each fit.
    pub fn auto(model: M) -> Self {
        Transformed {
            model,
            choice: Choice::Guerrero,
        }
    }
}

struct TransformedFit {
    inner: Box<dyn Fitted>,
    transform: BoxCox,
}

impl Fitted for TransformedFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        self.inner
            .forecast(h)
            .into_iter()
            .map(|z| self.transform.invert(z))
            .collect()
    }

    fn params(&self) -> Params {
        let mut p = self.inner.params();
        p.push(("lambda".into(), self.transform.lambda()));
        p
    }
}

impl<M: Model> Model for Transformed<M> {
    fn name(&self) -> String {
        match self.choice {
            Choice::Fixed(t) if t.lambda() == 0.0 => format!("log_{}", self.model.name()),
            _ => format!("boxcox_{}", self.model.name()),
        }
    }

    fn description(&self) -> String {
        match self.choice {
            Choice::Fixed(t) if t.lambda() == 0.0 => {
                format!("{}, on the log scale", self.model.description())
            }
            Choice::Fixed(t) => format!(
                "{}, on the Box-Cox scale (λ = {})",
                self.model.description(),
                t.lambda()
            ),
            Choice::Guerrero => format!(
                "{}, on the Box-Cox scale (λ by Guerrero's method)",
                self.model.description()
            ),
        }
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        if !y.is_positive() {
            return None;
        }
        let transform = match self.choice {
            Choice::Fixed(t) => t,
            Choice::Guerrero => BoxCox::guerrero(y)?,
        };
        let z: Vec<f64> = y.values().iter().map(|v| transform.apply(*v)).collect();
        let inner = self.model.fit(y.with_values(&z)?)?;
        Some(Box::new(TransformedFit { inner, transform }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Drift, Naive};
    use crate::testing::noisy_seasonal_growth;

    #[test]
    fn transformation_and_inverse_undo_each_other() {
        for lambda in [-1.0, -0.5, 0.0, 0.3, 1.0, 2.0] {
            let t = BoxCox::new(lambda);
            for y in [0.5, 1.0, 7.0, 1e6] {
                assert!((t.invert(t.apply(y)) / y - 1.0).abs() < 1e-9, "λ={lambda}");
            }
        }
        assert_eq!(BoxCox::log().apply(1.0), 0.0);
    }

    #[test]
    fn drift_on_the_log_scale_is_geometric_growth() {
        let y: Vec<f64> = (0..40).map(|t| 5.0 * 1.03f64.powi(t)).collect();
        let model = Transformed::log(Drift);
        assert_eq!(model.name(), "log_drift");
        let fit = model.fit(Series::non_seasonal(&y)).unwrap();
        for (k, v) in fit.forecast(5).iter().enumerate() {
            assert!((v / (5.0 * 1.03f64.powi(40 + k as i32)) - 1.0).abs() < 1e-9);
        }
        assert!(fit.params().contains(&("lambda".to_string(), 0.0)));
        assert!(Transformed::log(Naive)
            .fit(Series::non_seasonal(&[1.0, -1.0]))
            .is_none());
    }

    #[test]
    fn guerrero_picks_a_low_lambda_for_multiplicative_data() {
        // variability proportional to the level: close to the log
        let y = noisy_seasonal_growth(240, 0.2);
        let t = BoxCox::guerrero(Series::new(&y, 12)).unwrap();
        assert!(t.lambda() < 0.5, "λ = {}", t.lambda());
        assert!(BoxCox::guerrero(Series::new(&y[..12], 12)).is_none());
        let model = Transformed::auto(Naive);
        assert_eq!(model.name(), "boxcox_naive");
        assert!(model.forecast(Series::new(&y, 12), 2).is_some());
    }
}
