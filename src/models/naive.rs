//! Benchmarks: the yardsticks any other model has to beat.

use crate::model::{Fitted, Model, Params};
use crate::series::Series;

// ------------------------------------------------------------------ mean

/// Every forecast is the mean of the history.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mean;

struct Constant(f64);

impl Fitted for Constant {
    fn forecast(&self, h: usize) -> Vec<f64> {
        vec![self.0; h]
    }
}

impl Model for Mean {
    fn name(&self) -> String {
        "mean".into()
    }

    fn description(&self) -> String {
        "Mean of the history".into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        if y.is_empty() || !y.is_finite() {
            return None;
        }
        Some(Box::new(Constant(
            y.values().iter().sum::<f64>() / y.len() as f64,
        )))
    }
}

// ------------------------------------------------------------------ naive

/// Every forecast is the last observation (random walk).
#[derive(Debug, Clone, Copy, Default)]
pub struct Naive;

impl Model for Naive {
    fn name(&self) -> String {
        "naive".into()
    }

    fn description(&self) -> String {
        "Naive: the last observation".into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        if !y.is_finite() {
            return None;
        }
        Some(Box::new(Constant(*y.values().last()?)))
    }
}

// ------------------------------------------------------------------ drift

/// Random walk with drift: the last observation plus the average change per
/// period over the history.
#[derive(Debug, Clone, Copy, Default)]
pub struct Drift;

struct DriftFit {
    last: f64,
    slope: f64,
}

impl Fitted for DriftFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        (1..=h).map(|k| self.last + k as f64 * self.slope).collect()
    }

    fn params(&self) -> Params {
        vec![("drift".into(), self.slope)]
    }
}

impl Model for Drift {
    fn name(&self) -> String {
        "drift".into()
    }

    fn description(&self) -> String {
        "Random walk with drift: last observation plus the average change".into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let v = y.values();
        if v.len() < 2 || !y.is_finite() {
            return None;
        }
        let last = v[v.len() - 1];
        Some(Box::new(DriftFit {
            last,
            slope: (last - v[0]) / (v.len() - 1) as f64,
        }))
    }
}

// ------------------------------------------------------------------ seasonal naive

/// The same season of the last cycle, optionally scaled by recent growth.
///
/// With growth, ŷ(T+k) = y(T+k−m) × g, where g is the sum of the last cycle
/// over the sum of the one before; forecasts beyond one cycle compound g.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeasonalNaive {
    growth: bool,
}

impl SeasonalNaive {
    /// Plain seasonal naive: the same season of the last cycle.
    pub fn new() -> Self {
        SeasonalNaive { growth: false }
    }

    /// Seasonal naive scaled by the growth between the last two cycles.
    pub fn with_growth() -> Self {
        SeasonalNaive { growth: true }
    }
}

struct SeasonalNaiveFit {
    last_cycle: Vec<f64>,
    growth: Option<f64>,
}

impl Fitted for SeasonalNaiveFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        let m = self.last_cycle.len();
        let g = self.growth.unwrap_or(1.0);
        (1..=h)
            .map(|k| {
                let cycles = (k - 1) / m + 1;
                self.last_cycle[(k - 1) % m] * g.powi(cycles as i32)
            })
            .collect()
    }

    fn params(&self) -> Params {
        self.growth
            .map(|g| vec![("growth".into(), g)])
            .unwrap_or_default()
    }
}

impl Model for SeasonalNaive {
    fn name(&self) -> String {
        if self.growth {
            "seasonal_naive_growth".into()
        } else {
            "seasonal_naive".into()
        }
    }

    fn description(&self) -> String {
        if self.growth {
            "Seasonal naive with growth: same season of the last cycle × growth between the last two cycles".into()
        } else {
            "Seasonal naive: same season of the last cycle".into()
        }
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let (v, m, n) = (y.values(), y.period(), y.len());
        if !y.is_finite() || n < m {
            return None;
        }
        let growth = if self.growth {
            if n < 2 * m {
                return None;
            }
            let last: f64 = v[n - m..].iter().sum();
            let before: f64 = v[n - 2 * m..n - m].iter().sum();
            if before <= 0.0 || last <= 0.0 {
                return None;
            }
            Some(last / before)
        } else {
            None
        };
        Some(Box::new(SeasonalNaiveFit {
            last_cycle: v[n - m..].to_vec(),
            growth,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_naive_and_drift() {
        let v = [2.0, 4.0, 6.0, 8.0];
        let y = Series::non_seasonal(&v);
        assert_eq!(Mean.forecast(y, 2), Some(vec![5.0, 5.0]));
        assert_eq!(Naive.forecast(y, 2), Some(vec![8.0, 8.0]));
        assert_eq!(Drift.forecast(y, 3), Some(vec![10.0, 12.0, 14.0]));
        assert!(Naive.forecast(Series::non_seasonal(&[]), 1).is_none());
        assert!(Drift.forecast(Series::non_seasonal(&[1.0]), 1).is_none());
        assert_eq!(Naive.forecast(y, 0), Some(vec![]));
    }

    #[test]
    fn seasonal_naive_repeats_the_last_cycle() {
        let v = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let y = Series::new(&v, 4);
        let p = SeasonalNaive::new().forecast(y, 6).unwrap();
        assert_eq!(p, vec![10.0, 20.0, 30.0, 40.0, 10.0, 20.0]);
    }

    #[test]
    fn seasonal_naive_with_growth_scales_the_same_season() {
        let v: Vec<f64> = (0..24).map(|t| if t < 12 { 10.0 } else { 11.0 }).collect();
        let y = Series::new(&v, 12);
        let p = SeasonalNaive::with_growth().forecast(y, 13).unwrap();
        assert!((p[0] - 12.1).abs() < 1e-9); // 11 × 1.1
        assert!((p[12] - 11.0 * 1.1 * 1.1).abs() < 1e-9);
        assert!(SeasonalNaive::with_growth()
            .forecast(y.head(20), 1)
            .is_none());
    }
}
