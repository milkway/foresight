//! The Theta method (Assimakopoulos & Nikolopoulos, 2000), in the form shown
//! by Hyndman & Billah (2003) to be simple exponential smoothing with drift.

use crate::diagnostics::acf;
use crate::linalg::line;
use crate::model::{Fitted, Model, Params};
use crate::series::Series;

/// Simple exponential smoothing plus half the slope of the linear trend.
///
/// Seasonal series are first tested for seasonality (autocorrelation at the
/// seasonal lag, 90% level); when the test passes and the values are positive,
/// the series is seasonally adjusted by classical multiplicative decomposition
/// and the forecasts are re-seasonalised. The smoothing parameter and the
/// initial level minimise the sum of squared one-step errors.
#[derive(Debug, Clone, Copy, Default)]
pub struct Theta;

struct ThetaFit {
    level: f64,
    alpha: f64,
    initial_level: f64,
    /// Half the slope of the linear trend.
    drift: f64,
    n: usize,
    /// Seasonal indices by season, and the season of the first forecast.
    seasonal: Option<(Vec<f64>, usize)>,
}

impl Fitted for ThetaFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        let a = self.alpha;
        let carry = (1.0 - (1.0 - a).powi(self.n as i32)) / a;
        (1..=h)
            .map(|k| {
                let v = self.level + self.drift * ((k - 1) as f64 + carry);
                match &self.seasonal {
                    Some((idx, first)) => v * idx[(first + k - 1) % idx.len()],
                    None => v,
                }
            })
            .collect()
    }

    fn params(&self) -> Params {
        vec![
            ("alpha".into(), self.alpha),
            ("initial_level".into(), self.initial_level),
            ("drift".into(), self.drift),
            (
                "seasonal".into(),
                f64::from(u8::from(self.seasonal.is_some())),
            ),
        ]
    }
}

/// True when the autocorrelation at the seasonal lag is significant at 90%
/// (Bartlett's standard error).
fn is_seasonal(y: &[f64], m: usize) -> bool {
    let r = acf(y, m);
    if r.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let others: f64 = r[..m - 1].iter().map(|v| v * v).sum();
    let limit = 1.645 * ((1.0 + 2.0 * others) / y.len() as f64).sqrt();
    r[m - 1].abs() > limit
}

/// Seasonal indices (by position in the cycle, position 0 = `y[0]`) from the
/// classical multiplicative decomposition: ratios to a centred moving average,
/// averaged by position and scaled to mean 1.
fn seasonal_indices(y: &[f64], m: usize) -> Option<Vec<f64>> {
    let n = y.len();
    let half = m / 2;
    let mut sum = vec![0.0; m];
    let mut count = vec![0usize; m];
    for t in half..n.saturating_sub(half) {
        let trend = if m.is_multiple_of(2) {
            let inner: f64 = y[t + 1 - half..t + half].iter().sum();
            (0.5 * y[t - half] + inner + 0.5 * y[t + half]) / m as f64
        } else {
            y[t - half..=t + half].iter().sum::<f64>() / m as f64
        };
        sum[t % m] += y[t] / trend;
        count[t % m] += 1;
    }
    if count.contains(&0) {
        return None;
    }
    let raw: Vec<f64> = sum.iter().zip(&count).map(|(s, c)| s / *c as f64).collect();
    let mean = raw.iter().sum::<f64>() / m as f64;
    let idx: Vec<f64> = raw.iter().map(|v| v / mean).collect();
    idx.iter()
        .all(|v| v.is_finite() && *v > 1e-4)
        .then_some(idx)
}

/// Simple exponential smoothing for a given α with the initial level that
/// minimises the sum of squared one-step errors (the errors are linear in the
/// initial level, so it has a closed form). Returns (sse, initial, final level).
fn ses(y: &[f64], alpha: f64) -> (f64, f64, f64) {
    // level before y[t] = u + w·l0; error = (y[t] − u) − w·l0
    let (mut u, mut w) = (0.0, 1.0);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &v in y {
        sxy += w * (v - u);
        sxx += w * w;
        u = alpha * v + (1.0 - alpha) * u;
        w *= 1.0 - alpha;
    }
    let l0 = sxy / sxx;
    let (mut level, mut sse) = (l0, 0.0);
    for &v in y {
        let e = v - level;
        sse += e * e;
        level += alpha * e;
    }
    (sse, l0, level)
}

/// α in [0.01, 0.99]: a grid of 99 points refined by golden section around
/// the best one.
fn best_alpha(y: &[f64]) -> f64 {
    let sse = |a: f64| ses(y, a).0;
    let mut best = (f64::INFINITY, 0.5);
    for i in 1..=99 {
        let a = f64::from(i) / 100.0;
        let s = sse(a);
        if s < best.0 {
            best = (s, a);
        }
    }
    let (mut lo, mut hi) = ((best.1 - 0.01).max(0.0001), (best.1 + 0.01).min(0.9999));
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..40 {
        let (a, b) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if sse(a) < sse(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    (lo + hi) / 2.0
}

impl Model for Theta {
    fn name(&self) -> String {
        "theta".into()
    }

    fn description(&self) -> String {
        "Theta method: exponential smoothing with drift, on seasonally adjusted data when seasonal"
            .into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let (v, m, n) = (y.values(), y.period(), y.len());
        if n < 3 || !y.is_finite() {
            return None;
        }
        let constant = v.iter().all(|x| *x == v[0]);
        let indices = (m > 1 && n > 2 * m && !constant && y.is_positive() && is_seasonal(v, m))
            .then(|| seasonal_indices(v, m))
            .flatten();
        let adjusted: Vec<f64> = match &indices {
            Some(idx) => v.iter().enumerate().map(|(t, x)| x / idx[t % m]).collect(),
            None => v.to_vec(),
        };
        let alpha = best_alpha(&adjusted);
        let (_, initial_level, level) = ses(&adjusted, alpha);
        let (_, slope) = line(&adjusted)?;
        (level.is_finite() && initial_level.is_finite()).then(|| {
            Box::new(ThetaFit {
                level,
                alpha,
                initial_level,
                drift: slope / 2.0,
                n,
                seasonal: indices.map(|idx| (idx, n % m)),
            }) as Box<dyn Fitted>
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    #[test]
    fn a_straight_line_is_forecast_with_half_its_slope_plus_smoothing() {
        // on an exact line the level follows the data (α → 1) and the drift is
        // half the slope: forecasts keep rising, at half the pace
        let y: Vec<f64> = (0..50).map(|t| 10.0 + 2.0 * f64::from(t)).collect();
        let fit = Theta.fit(Series::non_seasonal(&y)).unwrap();
        let p = fit.forecast(3);
        assert!((p[1] - p[0] - 1.0).abs() < 1e-9 && (p[2] - p[1] - 1.0).abs() < 1e-9);
        assert!(p[0] > y[49] && p[0] < y[49] + 2.0);
    }

    #[test]
    fn seasonal_data_is_adjusted_and_reseasonalised() {
        let all = seasonal_growth(132);
        let fit = Theta.fit(Series::new(&all[..120], 12)).unwrap();
        assert!(fit.params().contains(&("seasonal".to_string(), 1.0)));
        let p = fit.forecast(12);
        for (k, v) in p.iter().enumerate() {
            assert!((v / all[120 + k] - 1.0).abs() < 0.05, "h={}", k + 1);
        }
        // the peak stays in December and the trough in February
        assert!(p.iter().all(|v| *v <= p[11]) && p.iter().all(|v| *v >= p[1]));
    }

    #[test]
    fn noise_without_seasonality_is_not_adjusted() {
        let y: Vec<f64> = noisy_seasonal_growth(120, 0.3)
            .iter()
            .zip(seasonal_growth(120))
            .map(|(noisy, clean)| noisy / clean * 100.0)
            .collect();
        let fit = Theta.fit(Series::new(&y, 12)).unwrap();
        assert!(fit.params().contains(&("seasonal".to_string(), 0.0)));
    }

    #[test]
    fn initial_level_has_the_closed_form_minimum() {
        let y = noisy_seasonal_growth(60, 0.2);
        let (sse, l0, _) = ses(&y, 0.3);
        // moving the initial level either way only makes it worse
        for delta in [-1.0, 1.0] {
            let mut level = l0 + delta;
            let mut other = 0.0;
            for v in &y {
                let e = v - level;
                other += e * e;
                level += 0.3 * e;
            }
            assert!(other > sse);
        }
    }
}
