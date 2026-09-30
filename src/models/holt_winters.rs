//! Holt-Winters exponential smoothing with multiplicative seasonality and a
//! damped additive trend.

use crate::model::{Fitted, Model, Params};
use crate::series::Series;

const ALPHA: [f64; 19] = [
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
    0.9, 0.95,
];
const BETA: [f64; 5] = [0.0, 0.02, 0.05, 0.1, 0.2];
const GAMMA: [f64; 8] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8];
const PHI: [f64; 4] = [0.9, 0.95, 0.98, 1.0];

/// Level, damped additive trend (φ) and multiplicative seasonal indices.
///
/// α, β, γ and φ come from a grid search (3,040 combinations) minimising the
/// sum of squared relative one-step-ahead errors on the training data, the
/// first cycle left out as warm-up. Needs a seasonal period of at least 2,
/// positive values and three full cycles.
#[derive(Debug, Clone, Copy, Default)]
pub struct HoltWinters;

#[derive(Clone)]
struct State {
    level: f64,
    trend: f64,
    seasonal: Vec<f64>,
}

/// Classical starting values: level = mean of the first cycle; trend =
/// difference between the means of the second and first cycles ÷ period;
/// seasonal index = average, over the first two cycles, of y ÷ cycle mean.
fn initial(y: &[f64], m: usize) -> State {
    let m1 = y[..m].iter().sum::<f64>() / m as f64;
    let m2 = y[m..2 * m].iter().sum::<f64>() / m as f64;
    State {
        level: m1,
        trend: (m2 - m1) / m as f64,
        seasonal: (0..m).map(|j| (y[j] / m1 + y[m + j] / m2) / 2.0).collect(),
    }
}

/// Runs the recursions over `y` from the state `e`; returns the sum of squared
/// relative one-step errors from the second cycle on, leaving the final state
/// in `e`.
fn filter(y: &[f64], m: usize, e: &mut State, a: f64, b: f64, g: f64, phi: f64) -> Option<f64> {
    let mut sse = 0.0;
    for (t, &obs) in y.iter().enumerate() {
        let s = t % m;
        let base = e.level + phi * e.trend;
        let pred = base * e.seasonal[s];
        if t >= m {
            let r = (obs - pred) / obs;
            sse += r * r;
        }
        let level = a * obs / e.seasonal[s] + (1.0 - a) * base;
        if !level.is_finite() || level <= 0.0 {
            return None;
        }
        e.trend = b * (level - e.level) + (1.0 - b) * phi * e.trend;
        e.seasonal[s] = g * obs / level + (1.0 - g) * e.seasonal[s];
        e.level = level;
    }
    sse.is_finite().then_some(sse)
}

struct HoltWintersFit {
    alpha: f64,
    beta: f64,
    gamma: f64,
    phi: f64,
    sse: f64,
    state: State,
    n: usize,
}

impl Fitted for HoltWintersFit {
    /// ŷ(T+k) = (L + (φ + φ² + … + φᵏ)·B) × S(season of T+k).
    fn forecast(&self, h: usize) -> Vec<f64> {
        let m = self.state.seasonal.len();
        let (mut damp, mut pow) = (0.0, 1.0);
        (1..=h)
            .map(|k| {
                pow *= self.phi;
                damp += pow;
                let t = self.n - 1 + k;
                ((self.state.level + damp * self.state.trend) * self.state.seasonal[t % m]).max(0.0)
            })
            .collect()
    }

    fn params(&self) -> Params {
        vec![
            ("alpha".into(), self.alpha),
            ("beta".into(), self.beta),
            ("gamma".into(), self.gamma),
            ("phi".into(), self.phi),
            ("sse_relative".into(), self.sse),
        ]
    }
}

impl Model for HoltWinters {
    fn name(&self) -> String {
        "holt_winters".into()
    }

    fn description(&self) -> String {
        "Holt-Winters, multiplicative seasonality and damped trend (grid search on one-step error)"
            .into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let (v, m) = (y.values(), y.period());
        if m < 2 || v.len() < 3 * m || !y.is_positive() {
            return None;
        }
        let start = initial(v, m);
        let mut scratch = start.clone();
        let mut best: Option<(f64, [f64; 4])> = None;
        for &a in &ALPHA {
            for &b in &BETA {
                for &g in &GAMMA {
                    for &phi in &PHI {
                        scratch.clone_from(&start);
                        if let Some(sse) = filter(v, m, &mut scratch, a, b, g, phi) {
                            if best.map_or(true, |(s, _)| sse < s) {
                                best = Some((sse, [a, b, g, phi]));
                            }
                        }
                    }
                }
            }
        }
        let (sse, [alpha, beta, gamma, phi]) = best?;
        let mut state = start;
        filter(v, m, &mut state, alpha, beta, gamma, phi)?;
        Some(Box::new(HoltWintersFit {
            alpha,
            beta,
            gamma,
            phi,
            sse,
            state,
            n: v.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::seasonal_growth;

    #[test]
    fn recovers_a_known_seasonal_pattern() {
        let all = seasonal_growth(132);
        let y = Series::new(&all[..120], 12);
        let fit = HoltWinters.fit(y).expect("fit");
        let p = fit.forecast(12);
        for (k, v) in p.iter().enumerate() {
            let actual = all[120 + k];
            let err = (v / actual - 1.0).abs();
            assert!(err < 0.01, "h={} error {err:.4} ({v} × {actual})", k + 1);
        }
        // December is the peak and February the trough, as in the generator
        let (dec, feb) = (p[11], p[1]);
        assert!(p.iter().all(|v| *v <= dec + 1e-9) && p.iter().all(|v| *v >= feb - 1e-9));
        assert!(fit.params().iter().any(|(k, _)| *k == "alpha"));
    }

    #[test]
    fn refuses_short_non_seasonal_or_non_positive_series() {
        let all = seasonal_growth(60);
        assert!(HoltWinters.fit(Series::new(&all[..30], 12)).is_none());
        assert!(HoltWinters.fit(Series::non_seasonal(&all)).is_none());
        let mut z = all.clone();
        z[10] = 0.0;
        assert!(HoltWinters.fit(Series::new(&z, 12)).is_none());
    }

    #[test]
    fn works_with_quarterly_data() {
        let idx = [0.9, 1.0, 0.8, 1.3];
        let all: Vec<f64> = (0..48)
            .map(|t| 50.0 * 1.01f64.powi(t) * idx[t as usize % 4])
            .collect();
        let p = HoltWinters.forecast(Series::new(&all[..40], 4), 8).unwrap();
        for (k, v) in p.iter().enumerate() {
            assert!((v / all[40 + k] - 1.0).abs() < 0.02, "h={}", k + 1);
        }
    }
}
