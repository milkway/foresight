//! ARIMA and seasonal ARIMA models, estimated by exact maximum likelihood.

mod arma;
mod auto;

pub use auto::{AutoArima, Criterion};

use crate::model::{Fitted, Model, Params};
use crate::optimize::{nelder_mead, Effort};
use crate::series::Series;
use arma::{expand, roots_outside, stationary, Arma};

/// ARIMA(p, d, q)(P, D, Q) with the seasonal period of the series.
///
/// The series is differenced (d ordinary and D seasonal differences) and an
/// ARMA model is fitted to the result by exact Gaussian maximum likelihood,
/// computed with the innovations algorithm. The search runs over partial
/// autocorrelations, so the estimates are always stationary and invertible.
///
/// For series whose swings grow with the level, fit on the log scale with
/// [`Transformed`](crate::transform::Transformed).
///
/// ```
/// use foresight::{models::Arima, Model, Series};
///
/// let y: Vec<f64> = (0..48).map(|t| 10.0 + t as f64 + [2.0, -1.0, 0.5, -1.5][t % 4]).collect();
/// let model = Arima::new(0, 1, 0).seasonal(0, 1, 0);
/// let forecast = model.forecast(Series::quarterly(&y, 0), 4).unwrap();
/// assert!((forecast[0] - (10.0 + 48.0 + 2.0)).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Arima {
    pub(crate) p: usize,
    pub(crate) d: usize,
    pub(crate) q: usize,
    pub(crate) sp: usize,
    pub(crate) sd: usize,
    pub(crate) sq: usize,
    pub(crate) constant: Option<bool>,
}

impl Arima {
    /// Non-seasonal ARIMA(p, d, q).
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Arima {
            p,
            d,
            q,
            sp: 0,
            sd: 0,
            sq: 0,
            constant: None,
        }
    }

    /// Adds the seasonal part (P, D, Q). Ignored for series without
    /// seasonality.
    pub fn seasonal(mut self, p: usize, d: usize, q: usize) -> Self {
        (self.sp, self.sd, self.sq) = (p, d, q);
        self
    }

    /// Whether to estimate a constant: the mean when the series is not
    /// differenced, the drift when it is differenced once. By default only
    /// the mean is estimated. With two or more differences there is never a
    /// constant.
    pub fn constant(mut self, include: bool) -> Self {
        self.constant = Some(include);
        self
    }

    /// The "airline model", ARIMA(0,1,1)(0,1,1): a good default for seasonal
    /// series, usually on the log scale.
    pub fn airline() -> Self {
        Arima::new(0, 1, 1).seasonal(0, 1, 1)
    }

    fn differences(&self) -> usize {
        self.d + self.sd
    }

    fn has_constant(&self) -> bool {
        match self.differences() {
            0 => self.constant.unwrap_or(true),
            1 => self.constant.unwrap_or(false),
            _ => false,
        }
    }

    /// The model without the seasonal part, for series of period 1.
    fn for_period(mut self, m: usize) -> Self {
        if m < 2 {
            (self.sp, self.sd, self.sq) = (0, 0, 0);
        }
        self
    }

    /// Fits the model and returns everything that was estimated.
    ///
    /// The likelihood of a mixed model can have more than one peak, so the
    /// search starts from three points and keeps the best.
    pub fn estimate(&self, y: Series<'_>) -> Option<ArimaFit> {
        self.estimate_from(y, None)
    }

    /// Unconstrained starting values with the autoregressive partial
    /// autocorrelations at `ar` and the moving average ones at `ma`.
    fn start(&self, ar: f64, ma: f64, constant: bool) -> Vec<f64> {
        let mut u = Vec::new();
        u.extend(std::iter::repeat_n(ar.atanh(), self.p));
        u.extend(std::iter::repeat_n(ma.atanh(), self.q));
        u.extend(std::iter::repeat_n(ar.atanh(), self.sp));
        u.extend(std::iter::repeat_n(ma.atanh(), self.sq));
        u.extend(std::iter::repeat_n(0.0, usize::from(constant)));
        u
    }

    /// Starting values taken from a fitted model of other orders: what both
    /// have in common is kept and the rest starts at zero, which describes
    /// the same process.
    fn start_near(&self, other: &ArimaFit, constant: bool) -> Vec<f64> {
        let o = &other.spec;
        let blocks = [
            (self.p, o.p),
            (self.q, o.q),
            (self.sp, o.sp),
            (self.sq, o.sq),
        ];
        let mut u = Vec::new();
        let mut at = 0;
        for (mine, theirs) in blocks {
            for i in 0..mine {
                u.push(if i < theirs {
                    other.unconstrained[at + i]
                } else {
                    0.0
                });
            }
            at += theirs;
        }
        if constant {
            u.push(other.unconstrained.get(at).copied().unwrap_or(0.0));
        }
        u
    }

    /// [`estimate`](Arima::estimate), or a quicker search from a neighbouring
    /// model when one is given — enough to compare models.
    pub(crate) fn estimate_from(&self, y: Series<'_>, near: Option<&ArimaFit>) -> Option<ArimaFit> {
        let m = y.period();
        let spec = self.for_period(m);
        if !y.is_finite() {
            return None;
        }
        let delta = differencing(spec.d, spec.sd, m);
        let lost = delta.len() - 1;
        let v = y.values();
        if v.len() <= lost {
            return None;
        }
        let w: Vec<f64> = (lost..v.len())
            .map(|t| delta.iter().enumerate().map(|(k, c)| c * v[t - k]).sum())
            .collect();
        let n = w.len();
        let coefficients = spec.p + spec.q + spec.sp + spec.sq;
        let longest = (spec.p + spec.sp * m).max(spec.q + spec.sq * m);
        if n < longest + coefficients + 8 {
            return None;
        }
        let mean = w.iter().sum::<f64>() / n as f64;
        let spread = (w.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64).sqrt();
        if spread.is_nan() {
            return None;
        }
        let constant = spec.has_constant();
        // the constant moves in units of its standard error
        let unit = if spread > 0.0 {
            spread / (n as f64).sqrt()
        } else {
            1.0
        };

        let build = |u: &[f64]| -> (Parts, f64) {
            let (ar, rest) = u.split_at(spec.p);
            let (ma, rest) = rest.split_at(spec.q);
            let (sar, rest) = rest.split_at(spec.sp);
            let (sma, rest) = rest.split_at(spec.sq);
            let parts = Parts {
                ar: stationary(ar),
                ma: stationary(ma).iter().map(|c| -c).collect(),
                sar: stationary(sar),
                sma: stationary(sma).iter().map(|c| -c).collect(),
            };
            let mu = match rest.first() {
                Some(c) => mean + c * unit,
                None => 0.0,
            };
            (parts, mu)
        };
        let evaluate = |u: &[f64]| -> Option<(f64, f64, f64, Vec<f64>)> {
            let (parts, mu) = build(u);
            let arma = parts.arma(m);
            let x: Vec<f64> = w.iter().map(|v| v - mu).collect();
            let inn = arma.innovations(n - 1)?;
            let f = arma.filter(&x, &inn);
            let sigma2 = f.sum_squares / n as f64;
            if !sigma2.is_finite() || !f.log_det.is_finite() {
                return None;
            }
            Some((sigma2, f.log_det, mu, f.errors))
        };
        // −2 log likelihood with the variance concentrated out, up to a constant
        let objective = |u: &[f64]| -> f64 {
            match evaluate(u) {
                Some((sigma2, log_det, _, _)) if sigma2 > 0.0 => n as f64 * sigma2.ln() + log_det,
                // a perfect fit: nothing left to explain
                Some(_) => -1e300,
                None => f64::INFINITY,
            }
        };

        let (starts, effort) = match near {
            Some(other) => (vec![spec.start_near(other, constant)], Effort::QUICK),
            None if coefficients > 1 => (
                vec![
                    spec.start(0.0, 0.0, constant),
                    spec.start(0.5, 0.5, constant),
                    spec.start(-0.5, -0.5, constant),
                ],
                Effort::THOROUGH,
            ),
            None => (vec![spec.start(0.0, 0.0, constant)], Effort::THOROUGH),
        };
        let (u, _) = starts
            .iter()
            .map(|s| nelder_mead(&objective, s, 0.5, effort))
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        let (sigma2, log_det, mu, errors) = evaluate(&u)?;
        let (parts, _) = build(&u);
        let nf = n as f64;
        let k = (coefficients + usize::from(constant) + 1) as f64;
        let log_likelihood = if sigma2 > 0.0 {
            -0.5 * (nf * (2.0 * std::f64::consts::PI * sigma2).ln() + log_det + nf)
        } else {
            f64::INFINITY
        };
        let aic = -2.0 * log_likelihood + 2.0 * k;
        let aicc = if nf - k - 1.0 > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (nf - k - 1.0)
        } else {
            f64::INFINITY
        };
        Some(ArimaFit {
            spec,
            period: m,
            arma: parts.arma(m),
            ar: parts.ar,
            ma: parts.ma,
            seasonal_ar: parts.sar,
            seasonal_ma: parts.sma,
            constant: constant.then_some(mu),
            sigma2,
            log_likelihood,
            aic,
            aicc,
            bic: -2.0 * log_likelihood + k * nf.ln(),
            residuals: errors,
            unconstrained: u,
            centred: w.iter().map(|v| v - mu).collect(),
            tail: v[v.len() - lost..].to_vec(),
            delta,
        })
    }
}

struct Parts {
    ar: Vec<f64>,
    ma: Vec<f64>,
    sar: Vec<f64>,
    sma: Vec<f64>,
}

impl Parts {
    /// The seasonal and non-seasonal polynomials multiplied out.
    fn arma(&self, m: usize) -> Arma {
        let negate = |v: &[f64]| -> Vec<f64> { v.iter().map(|c| -c).collect() };
        Arma {
            phi: negate(&expand(&negate(&self.ar), &negate(&self.sar), m)),
            theta: expand(&self.ma, &self.sma, m),
        }
    }
}

/// Coefficients of (1 − B)ᵈ (1 − Bᵐ)ᴰ, from the power 0.
fn differencing(d: usize, sd: usize, m: usize) -> Vec<f64> {
    fn times(lag: usize, c: &mut Vec<f64>) {
        let mut next = c.clone();
        next.resize(c.len() + lag, 0.0);
        for (i, v) in c.iter().enumerate() {
            next[i + lag] -= v;
        }
        *c = next;
    }
    let mut c = vec![1.0];
    for _ in 0..d {
        times(1, &mut c);
    }
    for _ in 0..sd {
        times(m, &mut c);
    }
    c
}

/// An estimated ARIMA model.
#[derive(Debug, Clone, PartialEq)]
pub struct ArimaFit {
    spec: Arima,
    period: usize,
    arma: Arma,
    /// φ₁, φ₂, …: X(t) = φ₁X(t−1) + … + Z(t) + θ₁Z(t−1) + …
    pub ar: Vec<f64>,
    /// θ₁, θ₂, …
    pub ma: Vec<f64>,
    /// Φ₁, Φ₂, … at the seasonal lags.
    pub seasonal_ar: Vec<f64>,
    /// Θ₁, Θ₂, … at the seasonal lags.
    pub seasonal_ma: Vec<f64>,
    /// Mean of the differenced series, when estimated.
    pub constant: Option<f64>,
    /// Innovation variance (maximum likelihood, not corrected for degrees of
    /// freedom).
    pub sigma2: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub aicc: f64,
    pub bic: f64,
    /// One-step prediction errors of the differenced series.
    pub residuals: Vec<f64>,
    centred: Vec<f64>,
    /// The estimates on the scale of the search.
    unconstrained: Vec<f64>,
    /// Last observations of the original series, to undo the differences.
    tail: Vec<f64>,
    delta: Vec<f64>,
}

impl ArimaFit {
    /// The orders (p, d, q).
    pub fn order(&self) -> (usize, usize, usize) {
        (self.spec.p, self.spec.d, self.spec.q)
    }

    /// The seasonal orders (P, D, Q).
    pub fn seasonal_order(&self) -> (usize, usize, usize) {
        (self.spec.sp, self.spec.sd, self.spec.sq)
    }

    pub fn period(&self) -> usize {
        self.period
    }

    /// Whether every root of the four polynomials is farther from the unit
    /// circle than `margin` (e.g. 1.01).
    pub fn is_well_behaved(&self, margin: f64) -> bool {
        let negate = |v: &[f64]| -> Vec<f64> { v.iter().map(|c| -c).collect() };
        roots_outside(&self.ar, margin)
            && roots_outside(&self.seasonal_ar, margin)
            && roots_outside(&negate(&self.ma), margin)
            && roots_outside(&negate(&self.seasonal_ma), margin)
    }

    /// Variance of the forecast error `h` periods ahead, on the scale the
    /// model was fitted on.
    pub fn forecast_variance(&self, h: usize) -> Vec<f64> {
        // ψ weights of the model with the differences put back
        let negate = |v: &[f64]| -> Vec<f64> { v.iter().map(|c| -c).collect() };
        let mut ar = vec![1.0];
        ar.extend(negate(&self.arma.phi));
        let mut full = vec![0.0; ar.len() + self.delta.len() - 1];
        for (i, a) in ar.iter().enumerate() {
            for (j, b) in self.delta.iter().enumerate() {
                full[i + j] += a * b;
            }
        }
        let integrated = Arma {
            phi: negate(&full[1..]),
            theta: self.arma.theta.clone(),
        };
        let psi = integrated.psi(h.saturating_sub(1));
        let mut total = 0.0;
        (0..h)
            .map(|k| {
                total += psi[k] * psi[k];
                total * self.sigma2
            })
            .collect()
    }
}

fn label(differences: usize) -> &'static str {
    if differences == 0 {
        "mean"
    } else {
        "drift"
    }
}

impl Fitted for ArimaFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        let n = self.centred.len();
        let differenced = match self.arma.innovations(n + h.saturating_sub(1)) {
            Some(inn) => self.arma.forecast(&self.centred, &self.residuals, &inn, h),
            None => vec![0.0; h],
        };
        let mu = self.constant.unwrap_or(0.0);
        let lost = self.delta.len() - 1;
        // undo the differences: y(t) = w(t) − Σ δₖ y(t − k)
        let mut path = self.tail.clone();
        for w in differenced {
            let back: f64 = (1..=lost)
                .map(|k| self.delta[k] * path[path.len() - k])
                .sum();
            path.push(w + mu - back);
        }
        path.split_off(lost)
    }

    fn params(&self) -> Params {
        let mut p: Params = Vec::new();
        let mut push = |prefix: &str, values: &[f64]| {
            for (i, v) in values.iter().enumerate() {
                p.push((format!("{prefix}{}", i + 1), *v));
            }
        };
        push("ar", &self.ar);
        push("ma", &self.ma);
        push("sar", &self.seasonal_ar);
        push("sma", &self.seasonal_ma);
        if let Some(c) = self.constant {
            // per period of the original series
            let scale = (self.period as f64).powi(self.spec.sd as i32);
            p.push((label(self.spec.differences()).into(), c / scale));
        }
        for (name, value) in [
            ("p", self.spec.p),
            ("d", self.spec.d),
            ("q", self.spec.q),
            ("P", self.spec.sp),
            ("D", self.spec.sd),
            ("Q", self.spec.sq),
        ] {
            p.push((name.into(), value as f64));
        }
        p.push(("sigma2".into(), self.sigma2));
        p.push(("log_likelihood".into(), self.log_likelihood));
        p.push(("aicc".into(), self.aicc));
        p
    }
}

impl Model for Arima {
    fn name(&self) -> String {
        let seasonal = if (self.sp, self.sd, self.sq) == (0, 0, 0) {
            String::new()
        } else {
            format!("_{}{}{}", self.sp, self.sd, self.sq)
        };
        format!("arima_{}{}{}{seasonal}", self.p, self.d, self.q)
    }

    fn description(&self) -> String {
        let seasonal = if (self.sp, self.sd, self.sq) == (0, 0, 0) {
            String::new()
        } else {
            format!("({},{},{})", self.sp, self.sd, self.sq)
        };
        format!(
            "ARIMA({},{},{}){seasonal} by exact maximum likelihood",
            self.p, self.d, self.q
        )
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        Some(Box::new(self.estimate(y)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    /// Deterministic noise in ±0.5.
    fn noise(n: usize) -> Vec<f64> {
        noisy_seasonal_growth(n, 1.0)
            .iter()
            .zip(seasonal_growth(n))
            .map(|(a, b)| a / b - 1.0)
            .collect()
    }

    #[test]
    fn differencing_polynomials() {
        assert_eq!(differencing(0, 0, 12), vec![1.0]);
        assert_eq!(differencing(1, 0, 12), vec![1.0, -1.0]);
        assert_eq!(differencing(2, 0, 12), vec![1.0, -2.0, 1.0]);
        // (1 − B)(1 − B⁴) = 1 − B − B⁴ + B⁵
        assert_eq!(differencing(1, 1, 4), vec![1.0, -1.0, 0.0, 0.0, -1.0, 1.0]);
    }

    #[test]
    fn recovers_the_coefficient_of_a_simulated_ar1() {
        let e = noise(400);
        let mut x = vec![0.0; 400];
        for t in 1..400 {
            x[t] = 0.7 * x[t - 1] + e[t];
        }
        let fit = Arima::new(1, 0, 0)
            .estimate(Series::non_seasonal(&x))
            .unwrap();
        assert!((fit.ar[0] - 0.7).abs() < 0.08, "φ = {}", fit.ar[0]);
        // uniform noise in ±0.5 has variance 1/12
        assert!(
            (fit.sigma2 - 1.0 / 12.0).abs() < 0.02,
            "σ² = {}",
            fit.sigma2
        );
        assert!(fit.constant.is_some());
        assert_eq!(fit.residuals.len(), 400);
        assert!(fit.is_well_behaved(1.01));
    }

    #[test]
    fn recovers_the_coefficient_of_a_simulated_ma1() {
        let e = noise(400);
        let x: Vec<f64> = (1..400).map(|t| e[t] - 0.5 * e[t - 1]).collect();
        let fit = Arima::new(0, 0, 1)
            .constant(false)
            .estimate(Series::non_seasonal(&x))
            .unwrap();
        assert!((fit.ma[0] + 0.5).abs() < 0.1, "θ = {}", fit.ma[0]);
        assert!(fit.constant.is_none());
    }

    #[test]
    fn a_random_walk_with_drift_is_the_drift_model() {
        let e = noise(80);
        let mut level = 10.0;
        let y: Vec<f64> = e
            .iter()
            .map(|v| {
                level += 0.4 + v;
                level
            })
            .collect();
        let series = Series::non_seasonal(&y);
        let arima = Arima::new(0, 1, 0)
            .constant(true)
            .forecast(series, 5)
            .unwrap();
        let drift = crate::models::Drift.forecast(series, 5).unwrap();
        for (a, b) in arima.iter().zip(&drift) {
            assert!((a - b).abs() < 1e-5, "{a} × {b}");
        }
        // without the constant it is the naive forecast
        let naive = Arima::new(0, 1, 0).forecast(series, 2).unwrap();
        assert_eq!(naive, vec![y[79], y[79]]);
    }

    #[test]
    fn the_airline_model_follows_trend_and_seasonality_on_the_log_scale() {
        let all = noisy_seasonal_growth(132, 0.04);
        let log: Vec<f64> = all.iter().map(|v| v.ln()).collect();
        let fit = Arima::airline()
            .estimate(Series::monthly(&log[..120], 0))
            .unwrap();
        assert_eq!(fit.order(), (0, 1, 1));
        assert_eq!(fit.seasonal_order(), (0, 1, 1));
        for (k, z) in fit.forecast(12).iter().enumerate() {
            assert!((z.exp() / all[120 + k] - 1.0).abs() < 0.05, "h={}", k + 1);
        }
        let variance = fit.forecast_variance(12);
        assert!((variance[0] - fit.sigma2).abs() < 1e-12);
        assert!(variance.windows(2).all(|w| w[1] >= w[0]));
        let names: Vec<String> = fit.params().into_iter().map(|(k, _)| k).collect();
        assert!(names.contains(&"ma1".to_string()) && names.contains(&"sma1".to_string()));
    }

    #[test]
    fn seasonal_terms_are_dropped_without_a_period_and_short_series_refused() {
        let e = noise(100);
        let fit = Arima::airline().estimate(Series::non_seasonal(&e)).unwrap();
        assert_eq!(fit.seasonal_order(), (0, 0, 0));
        assert_eq!(Arima::airline().name(), "arima_011_011");
        assert!(Arima::airline()
            .estimate(Series::monthly(&e[..30], 0))
            .is_none());
        assert!(Arima::new(1, 0, 0)
            .estimate(Series::non_seasonal(&[1.0, f64::NAN, 2.0]))
            .is_none());
    }
}
