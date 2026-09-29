//! Log-linear regression: trend plus seasonal dummies, optionally on deflated
//! values.

use crate::linalg::solve;
use crate::model::{Fitted, Model, Params};
use crate::series::Series;

/// log(y/I) = a + b·t + seasonal dummies, by least squares over the last
/// `window` observations, where I is a price index (1 without a deflator).
///
/// The back-transformation is corrected for bias with Duan's smearing factor
/// (the mean of exp(residual)). With a deflator, real forecasts are
/// re-inflated by the index growth over the last cycle, held constant over the
/// horizon. Needs positive values and three full cycles.
#[derive(Debug, Clone, Default)]
pub struct LogLinear {
    window: Option<usize>,
    deflator: Option<Vec<f64>>,
}

impl LogLinear {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of most recent observations used in the fit. Default: six
    /// cycles, at least 24 — enough to catch the recent trend without dragging
    /// old breaks along.
    pub fn window(mut self, n: usize) -> Self {
        self.window = Some(n);
        self
    }

    /// Price index aligned with the ORIGINAL data: `index[i]` deflates the
    /// observation whose absolute position is `i` (see
    /// [`Series::index`]). Only positions inside the training data are read,
    /// so handing the full index to a backtest does not leak the future.
    pub fn deflated_by(mut self, index: Vec<f64>) -> Self {
        self.deflator = Some(index);
        self
    }
}

struct LogLinearFit {
    /// Real (deflated) forecasts are `exp(row(t)·beta) × duan`.
    beta: Vec<f64>,
    duan: f64,
    window: usize,
    period: usize,
    /// Relative position and season of the first forecast.
    n: usize,
    first_season: usize,
    center: f64,
    /// Last index value and index growth over the last cycle.
    prices: Option<(f64, f64)>,
}

fn row(t: usize, season: usize, center: f64, m: usize) -> Vec<f64> {
    // columns: 1, t (in cycles, centred on the window), m−1 dummies
    let mut x = vec![0.0; m + 1];
    x[0] = 1.0;
    x[1] = (t as f64 - center) / m as f64;
    if season > 0 {
        x[1 + season] = 1.0;
    }
    x
}

fn dot(x: &[f64], beta: &[f64]) -> f64 {
    x.iter().zip(beta).map(|(x, b)| x * b).sum()
}

impl Fitted for LogLinearFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        let m = self.period;
        let (base, inflation) = self.prices.unwrap_or((1.0, 1.0));
        (1..=h)
            .map(|k| {
                let t = self.n - 1 + k;
                let season = (self.first_season + k - 1) % m;
                let real = dot(&row(t, season, self.center, m), &self.beta).exp() * self.duan;
                real * base * inflation.powf(k as f64 / m as f64)
            })
            .collect()
    }

    fn params(&self) -> Params {
        let mut p = vec![
            ("window".into(), self.window as f64),
            (
                "trend_growth_pct".into(),
                (self.beta[1].exp() - 1.0) * 100.0,
            ),
            ("duan_factor".into(), self.duan),
        ];
        if let Some((_, inflation)) = self.prices {
            p.push(("inflation_pct".into(), (inflation - 1.0) * 100.0));
        }
        p
    }
}

impl Model for LogLinear {
    fn name(&self) -> String {
        if self.deflator.is_some() {
            "log_linear_deflated".into()
        } else {
            "log_linear".into()
        }
    }

    fn description(&self) -> String {
        if self.deflator.is_some() {
            "Log-linear regression at constant prices, re-inflated by the index growth of the last cycle".into()
        } else {
            "Log-linear regression: trend plus seasonal dummies".into()
        }
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let (v, m, n) = (y.values(), y.period(), y.len());
        if !y.is_positive() {
            return None;
        }
        let w = n.min(self.window.unwrap_or((6 * m).max(24)));
        if w < 3 * m || w < m + 2 {
            return None;
        }
        let first = n - w;
        let index = match &self.deflator {
            Some(i) => {
                let used = i.get(y.index(first)..y.index(n))?;
                if used.iter().any(|p| !p.is_finite() || *p <= 0.0) {
                    return None;
                }
                Some(used)
            }
            None => None,
        };
        let price = |t: usize| index.map_or(1.0, |i| i[t - first]);

        let k = m + 1;
        let center = (first + n - 1) as f64 / 2.0;
        let mut xtx = vec![vec![0.0; k]; k];
        let mut xty = vec![0.0; k];
        for (t, &obs) in v.iter().enumerate().skip(first) {
            let x = row(t, y.season(t), center, m);
            let target = (obs / price(t)).ln();
            for a in 0..k {
                xty[a] += x[a] * target;
                for b in 0..k {
                    xtx[a][b] += x[a] * x[b];
                }
            }
        }
        let beta = solve(xtx, xty)?;
        let duan = (first..n)
            .map(|t| ((v[t] / price(t)).ln() - dot(&row(t, y.season(t), center, m), &beta)).exp())
            .sum::<f64>()
            / w as f64;
        Some(Box::new(LogLinearFit {
            beta,
            duan,
            window: w,
            period: m,
            n,
            first_season: y.season(n),
            center,
            prices: index.map(|i| (i[w - 1], i[w - 1] / i[w - 1 - m])),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::seasonal_growth;

    #[test]
    fn recovers_trend_and_seasonality() {
        let all = seasonal_growth(108);
        let fit = LogLinear::new().fit(Series::new(&all[..96], 12)).unwrap();
        for (k, v) in fit.forecast(12).iter().enumerate() {
            assert!((v / all[96 + k] - 1.0).abs() < 1e-6);
        }
        let params = fit.params();
        let growth = params
            .iter()
            .find(|(k, _)| *k == "trend_growth_pct")
            .unwrap()
            .1;
        assert!((growth - (1.005f64.powi(12) - 1.0) * 100.0).abs() < 1e-6);
        assert_eq!(params[0], ("window".to_string(), 72.0));
    }

    #[test]
    fn deflated_forecasts_are_reinflated_by_the_last_cycle() {
        // constant real values with seasonality; prices rise 0.4% a period
        let real: Vec<f64> = seasonal_growth(108)
            .iter()
            .enumerate()
            .map(|(t, v)| v / 1.005f64.powi(t as i32))
            .collect();
        let index: Vec<f64> = (0..108).map(|t| 1.004f64.powi(t)).collect();
        let nominal: Vec<f64> = real.iter().zip(&index).map(|(r, i)| r * i).collect();
        let model = LogLinear::new().deflated_by(index.clone());
        let p = model.forecast(Series::new(&nominal[..96], 12), 12).unwrap();
        for (k, v) in p.iter().enumerate() {
            assert!((v / nominal[96 + k] - 1.0).abs() < 1e-6, "h={}", k + 1);
        }
        // a sliced series still reads the index at the right positions
        let p2 = model
            .forecast(Series::new(&nominal, 12).slice(10..96), 12)
            .unwrap();
        for (k, v) in p2.iter().enumerate() {
            assert!((v / nominal[96 + k] - 1.0).abs() < 1e-6, "h={}", k + 1);
        }
        // an index shorter than the training data gives no forecast
        let short = LogLinear::new().deflated_by(index[..50].to_vec());
        assert!(short
            .forecast(Series::new(&nominal[..96], 12), 12)
            .is_none());
    }

    #[test]
    fn non_seasonal_series_get_a_plain_log_trend() {
        let y: Vec<f64> = (0..40).map(|t| 10.0 * 1.02f64.powi(t)).collect();
        let p = LogLinear::new()
            .forecast(Series::non_seasonal(&y[..30]), 10)
            .unwrap();
        for (k, v) in p.iter().enumerate() {
            assert!((v / y[30 + k] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn refuses_short_or_non_positive_series() {
        let all = seasonal_growth(60);
        assert!(LogLinear::new().fit(Series::new(&all[..30], 12)).is_none());
        let mut z = all;
        z[10] = -1.0;
        assert!(LogLinear::new().fit(Series::new(&z, 12)).is_none());
    }
}
