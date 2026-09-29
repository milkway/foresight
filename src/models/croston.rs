//! Intermittent demand: series where most periods have no demand at all.

use crate::model::{Fitted, Model, Params};
use crate::optimize::golden;
use crate::series::Series;

/// How the demand rate is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Intermittent {
    /// Croston (1972): the size of the demands and the interval between them
    /// are smoothed separately; the rate is size ÷ interval.
    #[default]
    Croston,
    /// Syntetos & Boylan (2005): Croston's rate times 1 − α/2, which removes
    /// its upward bias.
    Sba,
    /// Teunter, Syntetos & Babai (2011): the probability of a demand is
    /// smoothed at every period, so the rate falls while nothing is sold.
    Tsb,
}

/// Forecasts of the demand per period for intermittent series.
///
/// The values must not be negative. The forecast is the same for every
/// horizon: the rate at which demand is expected to arrive.
///
/// ```
/// use foresight::{models::Croston, Model, Series};
///
/// let demand = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 3.0];
/// let forecast = Croston::new().forecast(Series::non_seasonal(&demand), 3).unwrap();
/// assert!(forecast[0] > 0.0 && forecast[0] < 3.0);
/// assert_eq!(forecast[0], forecast[2]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Croston {
    variant: Intermittent,
    alpha: Option<f64>,
    beta: Option<f64>,
    optimise: bool,
}

impl Croston {
    /// Croston's method with smoothing 0.1.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn variant(mut self, variant: Intermittent) -> Self {
        self.variant = variant;
        self
    }

    /// Smoothing of the demand sizes (and of the intervals), between 0 and 1.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Smoothing of the probability of demand, for [`Intermittent::Tsb`].
    /// Default: the same as `alpha`.
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = Some(beta);
        self
    }

    /// Chooses the smoothing that minimises the squared error of the rate
    /// against what was demanded, instead of a fixed value.
    pub fn optimised(mut self) -> Self {
        self.optimise = true;
        self
    }

    /// The rate after the last observation, for given smoothing parameters,
    /// and the mean squared error of the rate as a one-step forecast.
    fn run(&self, y: &[f64], alpha: f64, beta: f64) -> Option<(f64, f64, f64, f64)> {
        let first = y.iter().position(|v| *v > 0.0)?;
        let mut size = y[first];
        let (mut interval, mut probability) = ((first + 1) as f64, 1.0 / (first + 1) as f64);
        let mut since = 0.0;
        let (mut squares, mut count) = (0.0, 0usize);
        let rate = |size: f64, interval: f64, probability: f64| match self.variant {
            Intermittent::Croston => size / interval,
            Intermittent::Sba => (1.0 - alpha / 2.0) * size / interval,
            Intermittent::Tsb => probability * size,
        };
        for &v in &y[first + 1..] {
            let e = v - rate(size, interval, probability);
            squares += e * e;
            count += 1;
            since += 1.0;
            if v > 0.0 {
                size += alpha * (v - size);
                interval += alpha * (since - interval);
                probability += beta * (1.0 - probability);
                since = 0.0;
            } else {
                probability *= 1.0 - beta;
            }
        }
        let mse = if count > 0 {
            squares / count as f64
        } else {
            0.0
        };
        Some((size, interval, probability, mse))
    }
}

struct CrostonFit {
    rate: f64,
    alpha: f64,
    beta: f64,
    size: f64,
    interval: f64,
    probability: f64,
    variant: Intermittent,
}

impl Fitted for CrostonFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        vec![self.rate; h]
    }

    fn params(&self) -> Params {
        let mut p: Params = vec![
            ("alpha".into(), self.alpha),
            ("demand_size".into(), self.size),
        ];
        if self.variant == Intermittent::Tsb {
            p.push(("beta".into(), self.beta));
            p.push(("demand_probability".into(), self.probability));
        } else {
            p.push(("demand_interval".into(), self.interval));
        }
        p
    }
}

impl Model for Croston {
    fn name(&self) -> String {
        match self.variant {
            Intermittent::Croston => "croston",
            Intermittent::Sba => "croston_sba",
            Intermittent::Tsb => "croston_tsb",
        }
        .into()
    }

    fn description(&self) -> String {
        match self.variant {
            Intermittent::Croston => {
                "Croston: demand size ÷ interval between demands, each smoothed"
            }
            Intermittent::Sba => "Croston with the Syntetos-Boylan correction of bias",
            Intermittent::Tsb => {
                "Teunter-Syntetos-Babai: demand size × probability of demand, each smoothed"
            }
        }
        .into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let v = y.values();
        if v.is_empty() || v.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return None;
        }
        let inside = |a: f64| a > 0.0 && a <= 1.0;
        let alpha = if self.optimise {
            golden(
                |a| {
                    self.run(v, a, self.beta.unwrap_or(a))
                        .map_or(f64::INFINITY, |r| r.3)
                },
                0.01,
                0.99,
            )
        } else {
            self.alpha.unwrap_or(0.1)
        };
        let beta = self.beta.unwrap_or(alpha);
        if !inside(alpha) || !inside(beta) {
            return None;
        }
        // no demand at all: nothing is expected
        let Some((size, interval, probability, _)) = self.run(v, alpha, beta) else {
            return Some(Box::new(CrostonFit {
                rate: 0.0,
                alpha,
                beta,
                size: 0.0,
                interval: f64::INFINITY,
                probability: 0.0,
                variant: self.variant,
            }));
        };
        let rate = match self.variant {
            Intermittent::Croston => size / interval,
            Intermittent::Sba => (1.0 - alpha / 2.0) * size / interval,
            Intermittent::Tsb => probability * size,
        };
        Some(Box::new(CrostonFit {
            rate,
            alpha,
            beta,
            size,
            interval,
            probability,
            variant: self.variant,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEMAND: [f64; 16] = [
        0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 3.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0,
    ];

    #[test]
    fn croston_is_size_over_interval() {
        // sizes 2, 4, 3, 5, 2 and intervals 2, 3, 4, 2, 3, smoothed with 0.5
        let fit = Croston::new()
            .alpha(0.5)
            .fit(Series::non_seasonal(&DEMAND))
            .unwrap();
        let size = [4.0, 3.0, 5.0, 2.0]
            .iter()
            .fold(2.0, |l, v| l + 0.5 * (v - l));
        let interval = [3.0, 4.0, 2.0, 3.0]
            .iter()
            .fold(2.0, |l, v| l + 0.5 * (v - l));
        assert!((fit.forecast(1)[0] - size / interval).abs() < 1e-12);
        let sba = Croston::new()
            .variant(Intermittent::Sba)
            .alpha(0.5)
            .forecast(Series::non_seasonal(&DEMAND), 2)
            .unwrap();
        assert!((sba[1] - 0.75 * size / interval).abs() < 1e-12);
    }

    #[test]
    fn tsb_lets_the_rate_fall_while_nothing_is_sold() {
        let mut long = DEMAND.to_vec();
        let model = Croston::new().variant(Intermittent::Tsb).alpha(0.2);
        let before = model.forecast(Series::non_seasonal(&long), 1).unwrap()[0];
        long.extend([0.0; 12]);
        let after = model.forecast(Series::non_seasonal(&long), 1).unwrap()[0];
        assert!(after < 0.2 * before);
        // Croston's rate does not move without a demand
        let croston = Croston::new().alpha(0.2);
        assert_eq!(
            croston.forecast(Series::non_seasonal(&long), 1),
            croston.forecast(Series::non_seasonal(&DEMAND), 1)
        );
    }

    #[test]
    fn optimised_smoothing_is_no_worse_than_the_default() {
        let fixed = Croston::new();
        let tuned = Croston::new().optimised();
        let error = |m: &Croston, a: f64| m.run(&DEMAND, a, a).unwrap().3;
        let fit = tuned.fit(Series::non_seasonal(&DEMAND)).unwrap();
        let alpha = fit.params()[0].1;
        assert!((0.01..=0.99).contains(&alpha));
        assert!(error(&tuned, alpha) <= error(&fixed, 0.1) + 1e-12);
    }

    #[test]
    fn edge_cases() {
        let none = Croston::new()
            .forecast(Series::non_seasonal(&[0.0; 8]), 2)
            .unwrap();
        assert_eq!(none, vec![0.0, 0.0]);
        assert!(Croston::new()
            .fit(Series::non_seasonal(&[1.0, -1.0]))
            .is_none());
        assert!(Croston::new().fit(Series::non_seasonal(&[])).is_none());
        assert!(Croston::new()
            .alpha(1.5)
            .fit(Series::non_seasonal(&DEMAND))
            .is_none());
        assert_eq!(
            Croston::new().variant(Intermittent::Tsb).name(),
            "croston_tsb"
        );
    }
}
