//! Exponential smoothing in state space form (Hyndman, Koehler, Snyder &
//! Grose, 2002; Hyndman, Koehler, Ord & Snyder, 2008).

use super::Criterion;
use crate::model::{Fitted, Model, Params};
use crate::optimize::{nelder_mead, Effort};
use crate::series::Series;

/// How the error enters the observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// y = μ + ε
    Additive,
    /// y = μ (1 + ε); needs positive values.
    Multiplicative,
}

/// Trend component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trend {
    None,
    Additive,
    /// Additive, flattening out over the horizon.
    Damped,
}

/// Seasonal component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Season {
    None,
    Additive,
    /// Needs positive values.
    Multiplicative,
}

/// One member of the ETS family: an error, a trend and a seasonal component.
///
/// Simple exponential smoothing is `Ets::new(Additive, None, None)`, Holt's
/// linear method `(Additive, Additive, None)`, the multiplicative Holt-Winters
/// method `(Multiplicative, Additive, Multiplicative)`. The smoothing
/// parameters and the initial states are estimated together by maximum
/// likelihood.
///
/// ```
/// use foresight::{models::Ets, Model, Series};
///
/// let y: Vec<f64> = (0..48).map(|t| 50.0 + t as f64 + [4.0, -2.0, 1.0, -3.0][t % 4]).collect();
/// let model = Ets::from_code("AAA").unwrap();
/// let forecast = model.forecast(Series::quarterly(&y, 0), 4).unwrap();
/// assert!((forecast[0] - (50.0 + 48.0 + 4.0)).abs() < 0.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ets {
    pub error: ErrorKind,
    pub trend: Trend,
    pub season: Season,
}

const ALPHA: (f64, f64) = (1e-4, 0.9999);
const BETA_MIN: f64 = 1e-4;
const GAMMA_MIN: f64 = 1e-4;
const PHI: (f64, f64) = (0.8, 0.98);

fn logistic(u: f64) -> f64 {
    1.0 / (1.0 + (-u).exp())
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// Parameters and initial states of a model.
#[derive(Debug, Clone, PartialEq)]
struct State {
    alpha: f64,
    beta: f64,
    gamma: f64,
    phi: f64,
    level: f64,
    trend: f64,
    /// By position in the cycle; position 0 is the first observation.
    seasonal: Vec<f64>,
}

/// What one pass over the data gives.
struct Pass {
    /// ε of each observation (relative for multiplicative errors).
    errors: Vec<f64>,
    fitted: Vec<f64>,
    /// n ln Σε² + 2 Σ ln|μ| (the last term only for multiplicative errors).
    criterion: f64,
    /// States after the last observation.
    last: State,
}

impl Ets {
    pub fn new(error: ErrorKind, trend: Trend, season: Season) -> Self {
        Ets {
            error,
            trend,
            season,
        }
    }

    /// From the usual three-letter code, error–trend–season, with `Ad` for a
    /// damped trend: `"ANN"`, `"AAdN"`, `"MAM"`…
    pub fn from_code(code: &str) -> Option<Self> {
        let c: Vec<char> = code.chars().collect();
        let (error, rest) = match c.split_first()? {
            ('A', rest) => (ErrorKind::Additive, rest),
            ('M', rest) => (ErrorKind::Multiplicative, rest),
            _ => return None,
        };
        let (trend, rest) = match rest {
            ['A', 'd', rest @ ..] => (Trend::Damped, rest),
            ['A', rest @ ..] => (Trend::Additive, rest),
            ['N', rest @ ..] => (Trend::None, rest),
            _ => return None,
        };
        let season = match rest {
            ['N'] => Season::None,
            ['A'] => Season::Additive,
            ['M'] => Season::Multiplicative,
            _ => return None,
        };
        Some(Ets::new(error, trend, season))
    }

    /// The three-letter code.
    pub fn code(&self) -> String {
        let e = match self.error {
            ErrorKind::Additive => "A",
            ErrorKind::Multiplicative => "M",
        };
        let t = match self.trend {
            Trend::None => "N",
            Trend::Additive => "A",
            Trend::Damped => "Ad",
        };
        let s = match self.season {
            Season::None => "N",
            Season::Additive => "A",
            Season::Multiplicative => "M",
        };
        format!("{e}{t}{s}")
    }

    fn has_trend(&self) -> bool {
        self.trend != Trend::None
    }

    /// Without the seasonal component, for series of period 1.
    fn for_period(mut self, m: usize) -> Self {
        if m < 2 {
            self.season = Season::None;
        }
        self
    }

    /// Estimated parameters: smoothing parameters, initial states and the
    /// variance.
    fn parameters(&self, m: usize) -> usize {
        let smoothing = 1
            + usize::from(self.has_trend())
            + usize::from(self.trend == Trend::Damped)
            + usize::from(self.season != Season::None);
        let states = 1
            + usize::from(self.has_trend())
            + if self.season == Season::None {
                0
            } else {
                m - 1
            };
        smoothing + states + 1
    }

    /// Runs the recursions over `y` from the given states.
    fn pass(&self, y: &[f64], start: &State) -> Option<Pass> {
        let m = start.seasonal.len().max(1);
        let mut s = start.clone();
        let mut errors = Vec::with_capacity(y.len());
        let mut fitted = Vec::with_capacity(y.len());
        let (mut squares, mut logs) = (0.0, 0.0);
        for (t, &obs) in y.iter().enumerate() {
            let at = t % m;
            let q = s.level + s.phi * s.trend;
            let season = match self.season {
                Season::None => 0.0,
                _ => s.seasonal[at],
            };
            let mu = match self.season {
                Season::None => q,
                Season::Additive => q + season,
                Season::Multiplicative => q * season,
            };
            if !mu.is_finite() {
                return None;
            }
            let e = match self.error {
                ErrorKind::Additive => obs - mu,
                ErrorKind::Multiplicative => {
                    if mu <= 0.0 {
                        return None;
                    }
                    logs += mu.ln();
                    (obs - mu) / mu
                }
            };
            // what a unit of error does to the level, the trend and the season
            let (to_level, to_season) = match (self.error, self.season) {
                (ErrorKind::Additive, Season::None | Season::Additive) => (1.0, 1.0),
                (ErrorKind::Additive, Season::Multiplicative) => (1.0 / season, 1.0 / q),
                (ErrorKind::Multiplicative, Season::None) => (q, 0.0),
                (ErrorKind::Multiplicative, Season::Additive) => (mu, mu),
                (ErrorKind::Multiplicative, Season::Multiplicative) => (q, season),
            };
            if !to_level.is_finite() || !to_season.is_finite() {
                return None;
            }
            s.level = q + s.alpha * to_level * e;
            s.trend = s.phi * s.trend + s.beta * to_level * e;
            if self.season != Season::None {
                s.seasonal[at] = season + s.gamma * to_season * e;
            }
            squares += e * e;
            errors.push(e);
            fitted.push(mu);
        }
        let n = y.len() as f64;
        let criterion = if squares > 0.0 {
            n * squares.ln() + 2.0 * logs
        } else {
            -1e300
        };
        criterion.is_finite().then_some(Pass {
            errors,
            fitted,
            criterion,
            last: s,
        })
    }

    /// Starting values of the states: seasonal pattern from a classical
    /// decomposition of the first cycles, level and trend from a line through
    /// the first seasonally adjusted observations.
    fn initial(&self, y: &[f64], m: usize) -> Option<State> {
        let n = y.len();
        let seasonal = match self.season {
            Season::None => Vec::new(),
            kind => {
                let used = &y[..n.min(3 * m)];
                let half = m / 2;
                let mut sum = vec![0.0; m];
                let mut count = vec![0usize; m];
                for t in half..used.len().saturating_sub(half) {
                    let centre = if m.is_multiple_of(2) {
                        let inner: f64 = used[t + 1 - half..t + half].iter().sum();
                        (0.5 * used[t - half] + inner + 0.5 * used[t + half]) / m as f64
                    } else {
                        used[t - half..=t + half].iter().sum::<f64>() / m as f64
                    };
                    sum[t % m] += if kind == Season::Additive {
                        used[t] - centre
                    } else {
                        used[t] / centre
                    };
                    count[t % m] += 1;
                }
                if count.contains(&0) {
                    return None;
                }
                let raw: Vec<f64> = sum.iter().zip(&count).map(|(s, c)| s / *c as f64).collect();
                let mean = raw.iter().sum::<f64>() / m as f64;
                if kind == Season::Additive {
                    raw.iter().map(|v| v - mean).collect()
                } else {
                    raw.iter().map(|v| v / mean).collect()
                }
            }
        };
        let adjusted: Vec<f64> = y
            .iter()
            .enumerate()
            .take(n.min(10.max(2 * m)))
            .map(|(t, v)| match self.season {
                Season::None => *v,
                Season::Additive => v - seasonal[t % m],
                Season::Multiplicative => v / seasonal[t % m],
            })
            .collect();
        let (level, trend) = if self.has_trend() {
            // the line gives the value at the first observation; the state is
            // one step before it
            let (a, b) = crate::linalg::line(&adjusted)?;
            (a - b, b)
        } else {
            (adjusted.iter().sum::<f64>() / adjusted.len() as f64, 0.0)
        };
        Some(State {
            alpha: 0.0,
            beta: 0.0,
            gamma: 0.0,
            phi: 1.0,
            level,
            trend,
            seasonal,
        })
    }

    /// Fits the model and returns everything that was estimated.
    pub fn estimate(&self, y: Series<'_>) -> Option<EtsFit> {
        let m = y.period();
        let spec = self.for_period(m);
        let multiplicative =
            spec.error == ErrorKind::Multiplicative || spec.season == Season::Multiplicative;
        if !y.is_finite() || (multiplicative && !y.is_positive()) {
            return None;
        }
        let n = y.len();
        let k = spec.parameters(m);
        if n < k + 4 || (spec.season != Season::None && n < 2 * m) {
            return None;
        }
        // the models keep their shape under a change of units: work near 1
        let scale = y.values().iter().map(|v| v.abs()).sum::<f64>() / n as f64;
        if scale.is_nan() || scale <= 0.0 {
            return None;
        }
        let v: Vec<f64> = y.values().iter().map(|x| x / scale).collect();
        let base = spec.initial(&v, m)?;
        let seasonal = spec.season != Season::None;
        let free_seasons = if seasonal { m - 1 } else { 0 };

        // unconstrained numbers → parameters inside their bounds and states
        let build = |u: &[f64]| -> State {
            let mut next = u.iter().copied();
            let mut take = || next.next().unwrap_or(0.0);
            let alpha = ALPHA.0 + (ALPHA.1 - ALPHA.0) * logistic(take());
            let beta = if spec.has_trend() {
                BETA_MIN + (alpha - BETA_MIN).max(0.0) * logistic(take())
            } else {
                0.0
            };
            let gamma = if seasonal {
                GAMMA_MIN + (1.0 - alpha - GAMMA_MIN).max(0.0) * logistic(take())
            } else {
                0.0
            };
            let phi = if spec.trend == Trend::Damped {
                PHI.0 + (PHI.1 - PHI.0) * logistic(take())
            } else {
                1.0
            };
            let level = base.level + 0.1 * take();
            let trend = if spec.has_trend() {
                base.trend + 0.02 * take()
            } else {
                0.0
            };
            let mut seasons: Vec<f64> = (0..free_seasons)
                .map(|j| base.seasonal[j] + 0.05 * take())
                .collect();
            if seasonal {
                // the pattern adds up to nothing (or averages one)
                let total: f64 = seasons.iter().sum();
                seasons.push(if spec.season == Season::Additive {
                    -total
                } else {
                    m as f64 - total
                });
            }
            State {
                alpha,
                beta,
                gamma,
                phi,
                level,
                trend,
                seasonal: seasons,
            }
        };
        let objective = |u: &[f64]| -> f64 {
            let state = build(u);
            if spec.season == Season::Multiplicative && state.seasonal.iter().any(|s| *s <= 0.0) {
                return f64::INFINITY;
            }
            spec.pass(&v, &state).map_or(f64::INFINITY, |p| p.criterion)
        };
        let dimension = k - 1;
        let start = |alpha: f64, beta_share: f64, gamma_share: f64| -> Vec<f64> {
            let mut u = vec![logit((alpha - ALPHA.0) / (ALPHA.1 - ALPHA.0))];
            if spec.has_trend() {
                u.push(logit(beta_share));
            }
            if seasonal {
                u.push(logit(gamma_share));
            }
            if spec.trend == Trend::Damped {
                u.push(logit(0.9));
            }
            u.resize(dimension, 0.0);
            u
        };
        let (u, _) = [
            start(0.1, 0.1, 0.05),
            start(0.5, 0.1, 0.2),
            start(0.9, 0.05, 0.5),
        ]
        .iter()
        .map(|s| nelder_mead(&objective, s, 1.0, Effort::THOROUGH))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
        let state = build(&u);
        let pass = spec.pass(&v, &state)?;

        let nf = n as f64;
        // back to the units of the data
        let minus_two_log_likelihood = pass.criterion + 2.0 * nf * scale.ln();
        let kf = k as f64;
        let aic = minus_two_log_likelihood + 2.0 * kf;
        let aicc = if nf - kf - 1.0 > 0.0 {
            aic + 2.0 * kf * (kf + 1.0) / (nf - kf - 1.0)
        } else {
            f64::INFINITY
        };
        let unit = match spec.error {
            ErrorKind::Additive => scale,
            ErrorKind::Multiplicative => 1.0,
        };
        let rescale = |s: &State| State {
            level: s.level * scale,
            trend: s.trend * scale,
            seasonal: if spec.season == Season::Additive {
                s.seasonal.iter().map(|x| x * scale).collect()
            } else {
                s.seasonal.clone()
            },
            ..s.clone()
        };
        let squares: f64 = pass.errors.iter().map(|e| e * e).sum();
        Some(EtsFit {
            spec,
            period: m,
            n,
            alpha: state.alpha,
            beta: spec.has_trend().then_some(state.beta),
            gamma: seasonal.then_some(state.gamma),
            phi: (spec.trend == Trend::Damped).then_some(state.phi),
            initial: rescale(&state),
            last: rescale(&pass.last),
            sigma2: squares * unit * unit / (nf - kf + 1.0).max(1.0),
            log_likelihood: -0.5 * minus_two_log_likelihood,
            aic,
            aicc,
            bic: minus_two_log_likelihood + kf * nf.ln(),
            residuals: pass.errors.iter().map(|e| e * unit).collect(),
            fitted: pass.fitted.iter().map(|f| f * scale).collect(),
        })
    }
}

/// An estimated ETS model.
#[derive(Debug, Clone, PartialEq)]
pub struct EtsFit {
    spec: Ets,
    period: usize,
    n: usize,
    /// Smoothing of the level.
    pub alpha: f64,
    /// Smoothing of the trend.
    pub beta: Option<f64>,
    /// Smoothing of the seasonal pattern.
    pub gamma: Option<f64>,
    /// Damping of the trend.
    pub phi: Option<f64>,
    initial: State,
    last: State,
    /// Variance of the errors (relative errors for multiplicative models).
    pub sigma2: f64,
    /// Log-likelihood up to the usual constant, as reported by R's `ets`.
    pub log_likelihood: f64,
    pub aic: f64,
    pub aicc: f64,
    pub bic: f64,
    /// Errors of the one-step forecasts (relative for multiplicative models).
    pub residuals: Vec<f64>,
    /// One-step forecasts of the history.
    pub fitted: Vec<f64>,
}

impl EtsFit {
    /// The model that was fitted.
    pub fn model(&self) -> Ets {
        self.spec
    }

    /// Level and trend before the first observation.
    pub fn initial_level_and_trend(&self) -> (f64, f64) {
        (self.initial.level, self.initial.trend)
    }

    /// Seasonal pattern before the first observation, by position in the
    /// cycle (position 0 is the first observation).
    pub fn initial_seasonal(&self) -> &[f64] {
        &self.initial.seasonal
    }

    /// Level and trend after the last observation.
    pub fn level_and_trend(&self) -> (f64, f64) {
        (self.last.level, self.last.trend)
    }
}

impl Fitted for EtsFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        let m = self.period.max(1);
        let phi = self.last.phi;
        let (mut damp, mut pow) = (0.0, 1.0);
        (1..=h)
            .map(|k| {
                pow *= phi;
                damp += pow;
                let q = self.last.level + damp * self.last.trend;
                let at = (self.n + k - 1) % m;
                match self.spec.season {
                    Season::None => q,
                    Season::Additive => q + self.last.seasonal[at],
                    Season::Multiplicative => q * self.last.seasonal[at],
                }
            })
            .collect()
    }

    fn params(&self) -> Params {
        let mut p: Params = vec![("alpha".into(), self.alpha)];
        for (name, value) in [
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("phi", self.phi),
        ] {
            if let Some(v) = value {
                p.push((name.into(), v));
            }
        }
        p.push(("initial_level".into(), self.initial.level));
        if self.spec.has_trend() {
            p.push(("initial_trend".into(), self.initial.trend));
        }
        let code = |on: bool| f64::from(u8::from(on));
        p.push((
            "multiplicative_error".into(),
            code(self.spec.error == ErrorKind::Multiplicative),
        ));
        p.push(("trend".into(), code(self.spec.has_trend())));
        p.push(("damped".into(), code(self.spec.trend == Trend::Damped)));
        p.push((
            "seasonal".into(),
            match self.spec.season {
                Season::None => 0.0,
                Season::Additive => 1.0,
                Season::Multiplicative => 2.0,
            },
        ));
        p.push(("sigma2".into(), self.sigma2));
        p.push(("log_likelihood".into(), self.log_likelihood));
        p.push(("aicc".into(), self.aicc));
        p
    }
}

impl Model for Ets {
    fn name(&self) -> String {
        format!("ets_{}", self.code().to_lowercase())
    }

    fn description(&self) -> String {
        let code = self.code();
        let (e, rest) = code.split_at(1);
        let (t, s) = rest.split_at(rest.len() - 1);
        format!("Exponential smoothing ETS({e},{t},{s}) by maximum likelihood")
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        Some(Box::new(self.estimate(y)?))
    }
}

/// The ETS model with the best information criterion among those that suit
/// the series.
///
/// Every combination of error, trend (none, additive, damped) and season
/// (none, additive, multiplicative) is fitted, leaving out the ones with
/// multiplicative parts when the series has values that are not positive, and
/// additive errors with a multiplicative season, whose forecast variance is
/// unbounded.
///
/// As a [`Model`], the choice is made again at every fit, so a backtest judges
/// the whole procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AutoEts {
    criterion: Criterion,
}

impl AutoEts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// The models considered for a series of the given period.
    pub fn candidates(period: usize, positive: bool) -> Vec<Ets> {
        let mut out = Vec::new();
        for error in [ErrorKind::Additive, ErrorKind::Multiplicative] {
            for trend in [Trend::None, Trend::Additive, Trend::Damped] {
                for season in [Season::None, Season::Additive, Season::Multiplicative] {
                    let multiplicative =
                        error == ErrorKind::Multiplicative || season == Season::Multiplicative;
                    if (season != Season::None && !(2..=24).contains(&period))
                        || (multiplicative && !positive)
                        || (error == ErrorKind::Additive && season == Season::Multiplicative)
                    {
                        continue;
                    }
                    out.push(Ets::new(error, trend, season));
                }
            }
        }
        out
    }

    /// Chooses the model and returns it estimated.
    pub fn select(&self, y: Series<'_>) -> Option<EtsFit> {
        let score = |fit: &EtsFit| match self.criterion {
            Criterion::Aicc => fit.aicc,
            Criterion::Aic => fit.aic,
            Criterion::Bic => fit.bic,
        };
        Self::candidates(y.period(), y.is_positive())
            .into_iter()
            .filter_map(|model| model.estimate(y))
            .filter(|fit| score(fit).is_finite())
            .min_by(|a, b| score(a).total_cmp(&score(b)))
    }
}

impl Model for AutoEts {
    fn name(&self) -> String {
        "auto_ets".into()
    }

    fn description(&self) -> String {
        "Exponential smoothing with error, trend and season chosen by the information criterion"
            .into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        Some(Box::new(self.select(y)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    #[test]
    fn codes_go_both_ways() {
        for code in ["ANN", "AAN", "AAdN", "MNM", "MAdM", "AAA", "MNA"] {
            assert_eq!(Ets::from_code(code).unwrap().code(), code);
        }
        for bad in ["", "A", "AN", "XNN", "AXN", "ANX", "ANNN", "AdNN"] {
            assert!(Ets::from_code(bad).is_none(), "{bad}");
        }
        assert_eq!(Ets::from_code("MAdM").unwrap().name(), "ets_madm");
        assert_eq!(
            Ets::from_code("MAdM").unwrap().description(),
            "Exponential smoothing ETS(M,Ad,M) by maximum likelihood"
        );
    }

    #[test]
    fn simple_smoothing_forecasts_a_flat_line_near_the_recent_level() {
        let e = noisy_seasonal_growth(80, 0.2);
        let clean = seasonal_growth(80);
        // a level of 50 that moves to 80 half way
        let y: Vec<f64> = (0..80)
            .map(|t| (if t < 40 { 50.0 } else { 80.0 }) * e[t] / clean[t])
            .collect();
        let fit = Ets::from_code("ANN")
            .unwrap()
            .estimate(Series::non_seasonal(&y))
            .unwrap();
        let p = fit.forecast(3);
        assert!(p[0] == p[1] && p[1] == p[2]);
        assert!((p[0] - 80.0).abs() < 8.0, "{}", p[0]);
        assert!(fit.beta.is_none() && fit.gamma.is_none() && fit.phi.is_none());
        assert_eq!(fit.residuals.len(), 80);
        assert!((fit.fitted[10] + fit.residuals[10] - y[10]).abs() < 1e-9);
    }

    #[test]
    fn multiplicative_holt_winters_follows_trend_and_seasonality() {
        let all = noisy_seasonal_growth(132, 0.04);
        let fit = Ets::from_code("MAM")
            .unwrap()
            .estimate(Series::monthly(&all[..120], 0))
            .unwrap();
        for (k, v) in fit.forecast(12).iter().enumerate() {
            assert!((v / all[120 + k] - 1.0).abs() < 0.06, "h={}", k + 1);
        }
        let pattern = fit.initial_seasonal();
        assert_eq!(pattern.len(), 12);
        assert!((pattern.iter().sum::<f64>() - 12.0).abs() < 1e-9);
        // December is the peak and February the trough, as in the generator
        assert!(
            pattern.iter().all(|v| *v <= pattern[11]) && pattern.iter().all(|v| *v >= pattern[1])
        );
        let (alpha, beta, gamma) = (fit.alpha, fit.beta.unwrap(), fit.gamma.unwrap());
        assert!(beta <= alpha && gamma <= 1.0 - alpha);
    }

    #[test]
    fn a_damped_trend_flattens_out() {
        let y: Vec<f64> = noisy_seasonal_growth(90, 0.02)
            .iter()
            .zip(seasonal_growth(90))
            .enumerate()
            .map(|(t, (a, b))| (100.0 + 2.0 * t as f64) * a / b)
            .collect();
        let series = Series::non_seasonal(&y);
        let damped = Ets::from_code("AAdN").unwrap().estimate(series).unwrap();
        let linear = Ets::from_code("AAN").unwrap().estimate(series).unwrap();
        let (d, l) = (damped.forecast(40), linear.forecast(40));
        assert!(d[39] - d[38] < d[1] - d[0]);
        assert!((l[39] - l[38] - (l[1] - l[0])).abs() < 1e-9);
        let phi = damped.phi.unwrap();
        assert!((0.8..=0.98).contains(&phi));
    }

    #[test]
    fn the_likelihood_does_not_depend_on_the_units() {
        let y = noisy_seasonal_growth(96, 0.05);
        let big: Vec<f64> = y.iter().map(|v| v * 1e6).collect();
        for code in ["AAA", "MAM", "MNN"] {
            let model = Ets::from_code(code).unwrap();
            let a = model.estimate(Series::monthly(&y, 0)).unwrap();
            let b = model.estimate(Series::monthly(&big, 0)).unwrap();
            // the same model in other units: the criterion moves by 2n ln(10⁶)
            let shift = 2.0 * 96.0 * 1e6f64.ln();
            assert!((b.aic - a.aic - shift).abs() < 1e-6, "{code}");
            assert!(
                (b.forecast(1)[0] / a.forecast(1)[0] - 1e6).abs() < 1e-3,
                "{code}"
            );
        }
    }

    #[test]
    fn automatic_choice_respects_what_the_data_allows() {
        assert_eq!(AutoEts::candidates(12, true).len(), 15);
        assert_eq!(AutoEts::candidates(12, false).len(), 6);
        assert_eq!(AutoEts::candidates(1, true).len(), 6);
        assert_eq!(AutoEts::candidates(52, true).len(), 6);
        let y = noisy_seasonal_growth(96, 0.04);
        let fit = AutoEts::new().select(Series::monthly(&y, 0)).unwrap();
        assert_ne!(fit.model().season, Season::None);
        let best = AutoEts::candidates(12, true)
            .into_iter()
            .filter_map(|m| m.estimate(Series::monthly(&y, 0)))
            .map(|f| f.aicc)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(fit.aicc, best);
        // values that are not positive leave only the additive models
        let shifted: Vec<f64> = y.iter().map(|v| v - 150.0).collect();
        let fit = AutoEts::new().select(Series::monthly(&shifted, 0)).unwrap();
        assert_eq!(fit.model().error, ErrorKind::Additive);
        assert!(AutoEts::new()
            .select(Series::non_seasonal(&[1.0, 2.0]))
            .is_none());
    }
}
