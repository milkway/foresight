//! Automatic choice of the ARIMA orders (Hyndman & Khandakar, 2008).

use super::{Arima, ArimaFit};
use crate::diagnostics::{difference, ndiffs, nsdiffs};
use crate::model::{Fitted, Model};
use crate::series::Series;

/// Information criterion that ranks the models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Criterion {
    /// AIC corrected for small samples.
    #[default]
    Aicc,
    Aic,
    Bic,
}

/// Seasonal ARIMA with orders chosen from the data.
///
/// The differences come first: a seasonal difference when the seasonality is
/// strong ([`nsdiffs`]), then as many ordinary differences as the KPSS test
/// asks for ([`ndiffs`]). The remaining orders are found by a stepwise search:
/// starting from four simple models, the best one is varied — each order up or
/// down by one, alone and in pairs, with and without a constant — for as long
/// as the information criterion improves. Models with roots too close to the
/// unit circle are discarded.
///
/// As a [`Model`], the orders are chosen again at every fit, so a backtest
/// judges the whole procedure, not one lucky specification.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoArima {
    pub max_p: usize,
    pub max_q: usize,
    pub max_seasonal_p: usize,
    pub max_seasonal_q: usize,
    pub max_d: usize,
    pub max_seasonal_d: usize,
    /// Fixes the number of ordinary differences instead of testing.
    pub d: Option<usize>,
    /// Fixes the number of seasonal differences instead of testing.
    pub seasonal_d: Option<usize>,
    pub criterion: Criterion,
    /// Most models the search will fit.
    pub max_models: usize,
}

impl Default for AutoArima {
    fn default() -> Self {
        AutoArima {
            max_p: 5,
            max_q: 5,
            max_seasonal_p: 2,
            max_seasonal_q: 2,
            max_d: 2,
            max_seasonal_d: 1,
            d: None,
            seasonal_d: None,
            criterion: Criterion::Aicc,
            max_models: 94,
        }
    }
}

/// (p, q, P, Q, constant)
type Key = (usize, usize, usize, usize, bool);

impl AutoArima {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Fixes the ordinary and seasonal differences.
    pub fn differences(mut self, d: usize, seasonal_d: usize) -> Self {
        (self.d, self.seasonal_d) = (Some(d), Some(seasonal_d));
        self
    }

    /// Limits of the search: p, q, seasonal P and seasonal Q.
    pub fn max_orders(mut self, p: usize, q: usize, sp: usize, sq: usize) -> Self {
        (self.max_p, self.max_q) = (p, q);
        (self.max_seasonal_p, self.max_seasonal_q) = (sp, sq);
        self
    }

    fn score(&self, fit: &ArimaFit) -> f64 {
        match self.criterion {
            Criterion::Aicc => fit.aicc,
            Criterion::Aic => fit.aic,
            Criterion::Bic => fit.bic,
        }
    }

    /// Chooses the orders and returns the estimated model.
    pub fn select(&self, y: Series<'_>) -> Option<ArimaFit> {
        if !y.is_finite() || y.is_empty() {
            return None;
        }
        let (v, m) = (y.values(), y.period());
        let seasonal = m > 1;
        let constant_series = v.iter().all(|x| *x == v[0]);
        let sd = if !seasonal || constant_series {
            0
        } else {
            self.seasonal_d
                .unwrap_or_else(|| nsdiffs(v, m).min(self.max_seasonal_d))
        };
        let d = if constant_series {
            0
        } else {
            self.d.unwrap_or_else(|| {
                let mut w = v.to_vec();
                for _ in 0..sd {
                    w = difference(&w, m);
                }
                ndiffs(&w, self.max_d)
            })
        };
        let with_constant = d + sd <= 1;

        let mut tried: Vec<Key> = Vec::new();
        let mut best: Option<(f64, Key, ArimaFit)> = None;
        let mut consider = |key: Key, best: &mut Option<(f64, Key, ArimaFit)>| -> bool {
            let (p, q, sp, sq, constant) = key;
            if p > self.max_p
                || q > self.max_q
                || sp > self.max_seasonal_p
                || sq > self.max_seasonal_q
                || (constant && !with_constant)
                || (!seasonal && (sp > 0 || sq > 0))
                || tried.contains(&key)
                || tried.len() >= self.max_models
            {
                return false;
            }
            tried.push(key);
            let model = Arima::new(p, d, q).seasonal(sp, sd, sq).constant(constant);
            let near = best.as_ref().map(|(_, _, fit)| fit);
            let Some(fit) = model.estimate_from(y, near) else {
                return false;
            };
            let score = self.score(&fit);
            if !score.is_finite() || !fit.is_well_behaved(1.01) {
                return false;
            }
            if best.as_ref().is_none_or(|(s, _, _)| score < *s) {
                *best = Some((score, key, fit));
                return true;
            }
            false
        };

        let s = usize::from(seasonal);
        for (p, q, sp, sq) in [(2, 2, s, s), (0, 0, 0, 0), (1, 0, s, 0), (0, 1, 0, s)] {
            consider((p, q, sp, sq, with_constant), &mut best);
        }
        if with_constant {
            consider((0, 0, 0, 0, false), &mut best);
        }
        loop {
            let (_, (p, q, sp, sq, c), _) = best.as_ref()?.clone();
            let up = |x: usize| Some(x + 1);
            let down = |x: usize| x.checked_sub(1);
            let mut neighbours: Vec<Key> = Vec::new();
            for step in [down, up] {
                if let Some(x) = step(p) {
                    neighbours.push((x, q, sp, sq, c));
                }
                if let Some(x) = step(q) {
                    neighbours.push((p, x, sp, sq, c));
                }
                if let Some(x) = step(sp) {
                    neighbours.push((p, q, x, sq, c));
                }
                if let Some(x) = step(sq) {
                    neighbours.push((p, q, sp, x, c));
                }
                if let (Some(x), Some(z)) = (step(p), step(q)) {
                    neighbours.push((x, z, sp, sq, c));
                }
                if let (Some(x), Some(z)) = (step(sp), step(sq)) {
                    neighbours.push((p, q, x, z, c));
                }
            }
            neighbours.push((p, q, sp, sq, !c));
            if !neighbours.into_iter().any(|key| consider(key, &mut best)) {
                break;
            }
        }
        // the winner once more from the usual starting points, in case the
        // search settled on a lesser peak of its likelihood
        let (score, _, fit) = best?;
        let spec = Arima::new(fit.spec.p, d, fit.spec.q)
            .seasonal(fit.spec.sp, sd, fit.spec.sq)
            .constant(fit.constant.is_some());
        match spec.estimate(y) {
            Some(again) if self.score(&again) < score && again.is_well_behaved(1.01) => Some(again),
            _ => Some(fit),
        }
    }
}

impl Model for AutoArima {
    fn name(&self) -> String {
        "auto_arima".into()
    }

    fn description(&self) -> String {
        "ARIMA with differences chosen by tests and orders by stepwise search on the information criterion".into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        Some(Box::new(self.select(y)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    fn noise(n: usize) -> Vec<f64> {
        noisy_seasonal_growth(n, 1.0)
            .iter()
            .zip(seasonal_growth(n))
            .map(|(a, b)| a / b - 1.0)
            .collect()
    }

    #[test]
    fn white_noise_needs_no_differences_and_few_terms() {
        let e = noise(150);
        let fit = AutoArima::new().select(Series::non_seasonal(&e)).unwrap();
        let ((p, d, q), seasonal) = (fit.order(), fit.seasonal_order());
        assert_eq!(d, 0);
        assert_eq!(seasonal, (0, 0, 0));
        assert!(p + q <= 2, "ARIMA({p},{d},{q})");
    }

    #[test]
    fn a_random_walk_is_differenced_once() {
        let e = noise(150);
        let mut level = 0.0;
        let y: Vec<f64> = e
            .iter()
            .map(|v| {
                level += v;
                level
            })
            .collect();
        let fit = AutoArima::new().select(Series::non_seasonal(&y)).unwrap();
        assert_eq!(fit.order().1, 1);
    }

    #[test]
    fn seasonal_data_gets_a_seasonal_difference_and_beats_the_fixed_start() {
        let log: Vec<f64> = noisy_seasonal_growth(96, 0.04)
            .iter()
            .map(|v| v.ln())
            .collect();
        let y = Series::monthly(&log, 0);
        let auto = AutoArima::new().max_orders(2, 2, 1, 1);
        let fit = auto.select(y).unwrap();
        assert_eq!(fit.seasonal_order().1, 1);
        let (d, sd) = (fit.order().1, 1);
        let start = Arima::new(0, d, 0).seasonal(0, sd, 0).estimate(y).unwrap();
        assert!(fit.aicc <= start.aicc);
        assert_eq!(auto.forecast(y, 6).unwrap().len(), 6);
    }

    #[test]
    fn fixed_differences_and_other_criteria_are_respected() {
        let e = noise(120);
        let fit = AutoArima::new()
            .differences(1, 0)
            .criterion(Criterion::Bic)
            .select(Series::non_seasonal(&e))
            .unwrap();
        assert_eq!(fit.order().1, 1);
        assert!(AutoArima::new().select(Series::non_seasonal(&[])).is_none());
    }
}
