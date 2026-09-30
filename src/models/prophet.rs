//! The Prophet model (Taylor & Letham, 2018): a trend that may bend at
//! changepoints, a seasonal pattern and dated events.

use crate::linalg::solve;
use crate::model::{Fitted, Model, Params};
use crate::series::Series;
use std::f64::consts::PI;

/// Piecewise linear trend + Fourier seasonality + events, fitted by maximum a
/// posteriori as in Prophet.
///
/// y(t) = g(t) + s(t) + events(t) + noise, where g is a line whose slope may
/// change at each of a set of candidate changepoints spread over the first
/// part of the history. The changes have a Laplace prior, so most of them come
/// out as exactly zero and the trend bends only where the data insists; the
/// seasonal and event coefficients have normal priors.
///
/// What is and is not here, compared with the original: observations are
/// equally spaced and the seasonal period is the one of the series; growth is
/// linear and the components are additive (for multiplicative behaviour, fit
/// on the log scale with [`Transformed`](crate::transform::Transformed));
/// events are given by position instead of by calendar. The estimate is the
/// exact optimum of the same posterior, found without Stan.
///
/// ```
/// use foresight::{models::Prophet, Model, Series};
///
/// // a trend that changes pace half way, with a yearly pattern
/// let y: Vec<f64> = (0..96)
///     .map(|t| {
///         let trend = if t < 48 { t as f64 } else { 48.0 + 3.0 * (t - 48) as f64 };
///         100.0 + trend + [5.0, -3.0, 0.0, 2.0, -4.0, 1.0, 3.0, -2.0, 0.0, 4.0, -5.0, -1.0][t % 12]
///     })
///     .collect();
/// let forecast = Prophet::new().forecast(Series::monthly(&y, 0), 12).unwrap();
/// // the forecast follows the pace after the change, not the average one
/// assert!(forecast[11] - forecast[0] > 25.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Prophet {
    changepoints: usize,
    changepoint_range: f64,
    changepoint_prior_scale: f64,
    seasonality_prior_scale: f64,
    fourier_order: Option<usize>,
    event_prior_scale: f64,
    events: Vec<Event>,
}

#[derive(Debug, Clone, PartialEq)]
struct Event {
    name: String,
    shape: Shape,
}

#[derive(Debug, Clone, PartialEq)]
enum Shape {
    /// 1 at the given positions, 0 elsewhere.
    Pulse(Vec<usize>),
    /// 0 before the position, 1 from it on.
    Step(usize),
}

impl Shape {
    fn at(&self, index: usize) -> f64 {
        let on = match self {
            Shape::Pulse(positions) => positions.contains(&index),
            Shape::Step(first) => index >= *first,
        };
        f64::from(u8::from(on))
    }
}

impl Default for Prophet {
    fn default() -> Self {
        Prophet {
            changepoints: 25,
            changepoint_range: 0.8,
            changepoint_prior_scale: 0.05,
            seasonality_prior_scale: 10.0,
            fourier_order: None,
            event_prior_scale: 10.0,
            events: Vec::new(),
        }
    }
}

impl Prophet {
    /// The defaults of Prophet: 25 candidate changepoints over the first 80%
    /// of the history with prior scale 0.05, seasonal prior scale 10.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of candidate changepoints (0 for a straight line).
    pub fn changepoints(mut self, n: usize) -> Self {
        self.changepoints = n;
        self
    }

    /// Share of the history, from the start, where changepoints may fall.
    pub fn changepoint_range(mut self, share: f64) -> Self {
        self.changepoint_range = share.clamp(0.0, 1.0);
        self
    }

    /// Flexibility of the trend: larger values let it bend more.
    pub fn changepoint_prior_scale(mut self, scale: f64) -> Self {
        self.changepoint_prior_scale = scale;
        self
    }

    /// Flexibility of the seasonal pattern.
    pub fn seasonality_prior_scale(mut self, scale: f64) -> Self {
        self.seasonality_prior_scale = scale;
        self
    }

    /// Number of harmonics of the seasonal pattern. Default: 10, or half the
    /// period when that is less — which describes any pattern of that period.
    /// 0 turns seasonality off.
    pub fn fourier_order(mut self, order: usize) -> Self {
        self.fourier_order = Some(order);
        self
    }

    /// Flexibility of the effects of events.
    pub fn event_prior_scale(mut self, scale: f64) -> Self {
        self.event_prior_scale = scale;
        self
    }

    /// An event that recurs at the given positions of the ORIGINAL data (see
    /// [`Series::index`]), past and future: one effect is estimated for all
    /// its occurrences. Positions beyond the history are how the forecast
    /// knows the event is coming.
    pub fn event(mut self, name: impl Into<String>, positions: &[usize]) -> Self {
        self.events.push(Event {
            name: name.into(),
            shape: Shape::Pulse(positions.to_vec()),
        });
        self
    }

    /// A lasting change of level from the given position of the original data
    /// on, such as a change in the law.
    pub fn step(mut self, name: impl Into<String>, from: usize) -> Self {
        self.events.push(Event {
            name: name.into(),
            shape: Shape::Step(from),
        });
        self
    }

    /// Harmonics actually used for `n` observations of a series of period
    /// `m`: none before two full cycles, as a pattern seen once cannot be
    /// told from the trend.
    fn harmonics(&self, m: usize, n: usize) -> usize {
        if m < 2 || n < 2 * m {
            return 0;
        }
        self.fourier_order.unwrap_or(10).min(m / 2)
    }

    /// Fits the model and returns everything that was estimated.
    pub fn estimate(&self, y: Series<'_>) -> Option<ProphetFit> {
        let (v, n, m) = (y.values(), y.len(), y.period());
        if n < 4 || !y.is_finite() {
            return None;
        }
        let scales = [
            self.changepoint_prior_scale,
            self.seasonality_prior_scale,
            self.event_prior_scale,
        ];
        if scales.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return None;
        }
        let scale = v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        let scale = if scale > 0.0 { scale } else { 1.0 };
        let target: Vec<f64> = v.iter().map(|x| x / scale).collect();
        let span = (n - 1) as f64;

        // candidate changepoints, evenly spread over the first part of the history
        let history = ((n as f64 * self.changepoint_range).floor() as usize).max(1);
        let count = self.changepoints.min(history - 1);
        let mut at: Vec<usize> = (1..=count)
            .map(|i| (i as f64 * (history - 1) as f64 / count as f64).round_ties_even() as usize)
            .collect();
        at.dedup();

        let layout = Layout {
            span,
            changepoints: at.iter().map(|i| *i as f64 / span).collect(),
            harmonics: self.harmonics(m, n),
            period: m,
            events: self.events.clone(),
        };
        let columns = layout.columns();
        // penalties of each column: ridge (1/variance) and lasso (1/scale)
        let mut ridge = vec![0.0; columns];
        let mut lasso = vec![0.0; columns];
        ridge[0] = 1.0 / 25.0;
        ridge[1] = 1.0 / 25.0;
        for j in layout.range_changepoints() {
            lasso[j] = 1.0 / self.changepoint_prior_scale;
        }
        for j in layout.range_seasonal() {
            ridge[j] = 1.0 / self.seasonality_prior_scale.powi(2);
        }
        for j in layout.range_events() {
            ridge[j] = 1.0 / self.event_prior_scale.powi(2);
        }

        // the design matrix, by column
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| layout.row(i, y.index(i), y.season(i)))
            .collect();
        let design: Vec<Vec<f64>> = (0..columns)
            .map(|j| rows.iter().map(|r| r[j]).collect())
            .collect();
        // everything the search needs from the data: AᵀA and Aᵀy
        let gram: Vec<Vec<f64>> = design
            .iter()
            .map(|a| {
                design
                    .iter()
                    .map(|b| a.iter().zip(b).map(|(x, z)| x * z).sum())
                    .collect()
            })
            .collect();
        let moment: Vec<f64> = design
            .iter()
            .map(|a| a.iter().zip(&target).map(|(x, z)| x * z).sum())
            .collect();

        // start: the line through the first and last observations
        let mut theta = vec![0.0; columns];
        theta[0] = target[n - 1] - target[0];
        theta[1] = target[0];
        let problem = Penalised {
            gram: &gram,
            moment: &moment,
            ridge: &ridge,
            lasso: &lasso,
        };
        let mut sigma = 1.0f64;
        for _ in 0..500 {
            // given the noise level, a lasso with ridge terms
            let weight = 1.0 / (sigma * sigma);
            if !problem.active_set(weight, &mut theta) {
                problem.descend(weight, &mut theta);
            }
            // given the fit, the noise level in closed form
            let rss: f64 = rows
                .iter()
                .zip(&target)
                .map(|(row, observed)| {
                    let fitted: f64 = row.iter().zip(&theta).map(|(x, c)| x * c).sum();
                    (observed - fitted).powi(2)
                })
                .sum();
            let nf = n as f64;
            let next = (((nf * nf + 16.0 * rss).sqrt() - nf) / 8.0)
                .sqrt()
                .max(1e-8);
            let settled = (next - sigma).abs() <= 1e-10 * sigma;
            sigma = next;
            if settled {
                break;
            }
        }
        if theta.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(ProphetFit {
            layout,
            theta,
            sigma,
            scale,
            n,
            start: y.start(),
            first_season: y.season(0),
            at,
        })
    }
}

/// weight/2 · ‖y − Aθ‖² + Σ ridgeⱼ θⱼ²/2 + Σ lassoⱼ |θⱼ|, in terms of AᵀA and Aᵀy.
struct Penalised<'a> {
    gram: &'a [Vec<f64>],
    moment: &'a [f64],
    ridge: &'a [f64],
    lasso: &'a [f64],
}

impl Penalised<'_> {
    /// Exact minimum by an active-set method: solve for the coefficients that
    /// are in play, stop at the first one that would change sign and drop it,
    /// bring in the one the optimality conditions complain most about, until
    /// nothing moves. Returns false if it gives up (singular system or too
    /// many rounds), leaving `theta` at the best point reached.
    fn active_set(&self, weight: f64, theta: &mut [f64]) -> bool {
        let p = theta.len();
        let usable: Vec<bool> = (0..p).map(|j| self.gram[j][j] > 0.0).collect();
        let mut sign: Vec<f64> = (0..p)
            .map(|j| {
                if theta[j] == 0.0 {
                    0.0
                } else {
                    theta[j].signum()
                }
            })
            .collect();
        let mut active: Vec<bool> = (0..p)
            .map(|j| usable[j] && (self.lasso[j] == 0.0 || theta[j] != 0.0))
            .collect();
        for _ in 0..20 * p + 50 {
            let members: Vec<usize> = (0..p).filter(|j| active[*j]).collect();
            let a: Vec<Vec<f64>> = members
                .iter()
                .map(|&i| {
                    members
                        .iter()
                        .map(|&j| {
                            weight * self.gram[i][j] + if i == j { self.ridge[i] } else { 0.0 }
                        })
                        .collect()
                })
                .collect();
            let b: Vec<f64> = members
                .iter()
                .map(|&i| weight * self.moment[i] - self.lasso[i] * sign[i])
                .collect();
            let Some(solution) = solve(a, b) else {
                return false;
            };
            // as far towards the solution as no coefficient changes sign
            let mut step = 1.0;
            let mut stopped = None;
            for (k, &j) in members.iter().enumerate() {
                if self.lasso[j] > 0.0 && solution[k] * sign[j] < 0.0 {
                    let reach = theta[j] / (theta[j] - solution[k]);
                    if reach < step {
                        step = reach;
                        stopped = Some(j);
                    }
                }
            }
            for (k, &j) in members.iter().enumerate() {
                theta[j] += step * (solution[k] - theta[j]);
            }
            if let Some(j) = stopped {
                theta[j] = 0.0;
                sign[j] = 0.0;
                active[j] = false;
                continue;
            }
            // is any coefficient left at zero pulled harder than its penalty?
            let mut worst: Option<(usize, f64, f64)> = None;
            for j in (0..p).filter(|j| usable[*j] && !active[*j]) {
                let fitted: f64 = (0..p).map(|i| self.gram[j][i] * theta[i]).sum();
                let pull = weight * (self.moment[j] - fitted);
                let excess = pull.abs() - self.lasso[j] * (1.0 + 1e-9) - 1e-12;
                if excess > 0.0 && worst.map_or(true, |(_, e, _)| excess > e) {
                    worst = Some((j, excess, pull.signum()));
                }
            }
            match worst {
                Some((j, _, direction)) => {
                    active[j] = true;
                    sign[j] = direction;
                }
                None => return true,
            }
        }
        false
    }

    /// The same minimum by coordinate descent: slower, never stuck.
    fn descend(&self, weight: f64, theta: &mut [f64]) {
        let p = theta.len();
        // Aᵀ(y − Aθ), kept up to date as θ moves
        let mut pull: Vec<f64> = (0..p)
            .map(|j| {
                let fitted: f64 = (0..p).map(|i| self.gram[j][i] * theta[i]).sum();
                self.moment[j] - fitted
            })
            .collect();
        for _ in 0..50_000 {
            let mut moved = 0.0f64;
            for j in 0..p {
                let norm = self.gram[j][j];
                if norm == 0.0 {
                    continue;
                }
                let old = theta[j];
                let force = weight * (pull[j] + norm * old);
                let shrunk = force.signum() * (force.abs() - self.lasso[j]).max(0.0);
                let new = shrunk / (weight * norm + self.ridge[j]);
                if new != old {
                    for (q, g) in pull.iter_mut().zip(&self.gram[j]) {
                        *q -= (new - old) * g;
                    }
                    theta[j] = new;
                    moved = moved.max((new - old).abs() * norm.sqrt());
                }
            }
            if moved < 1e-12 {
                break;
            }
        }
    }
}

/// Where each group of coefficients sits and how to build a row of the design.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    /// n − 1: positions are divided by it, so the history spans [0, 1].
    span: f64,
    /// Changepoints on that scale.
    changepoints: Vec<f64>,
    harmonics: usize,
    period: usize,
    events: Vec<Event>,
}

impl Layout {
    /// Seasonal columns: a sine and a cosine per harmonic, except that the
    /// sine of the harmonic at half the period is zero everywhere.
    fn seasonal(&self) -> usize {
        let full = 2 * self.harmonics;
        if self.harmonics > 0 && 2 * self.harmonics == self.period {
            full - 1
        } else {
            full
        }
    }

    fn columns(&self) -> usize {
        2 + self.changepoints.len() + self.seasonal() + self.events.len()
    }

    fn range_changepoints(&self) -> std::ops::Range<usize> {
        2..2 + self.changepoints.len()
    }

    fn range_seasonal(&self) -> std::ops::Range<usize> {
        let first = 2 + self.changepoints.len();
        first..first + self.seasonal()
    }

    fn range_events(&self) -> std::ops::Range<usize> {
        let first = 2 + self.changepoints.len() + self.seasonal();
        first..first + self.events.len()
    }

    /// Row of the design for the observation at relative position `i`, whose
    /// position in the original data is `index` and whose season is `season`.
    fn row(&self, i: usize, index: usize, season: usize) -> Vec<f64> {
        let t = i as f64 / self.span;
        let mut row = Vec::with_capacity(self.columns());
        row.push(t);
        row.push(1.0);
        row.extend(self.changepoints.iter().map(|s| (t - s).max(0.0)));
        for h in 1..=self.harmonics {
            let angle = 2.0 * PI * (h * season) as f64 / self.period as f64;
            if 2 * h != self.period {
                row.push(angle.sin());
            }
            row.push(angle.cos());
        }
        row.extend(self.events.iter().map(|e| e.shape.at(index)));
        row
    }
}

/// An estimated Prophet model.
#[derive(Debug, Clone, PartialEq)]
pub struct ProphetFit {
    layout: Layout,
    /// Coefficients on the scale of the fit (values divided by `scale`).
    theta: Vec<f64>,
    sigma: f64,
    scale: f64,
    n: usize,
    start: usize,
    first_season: usize,
    at: Vec<usize>,
}

impl ProphetFit {
    fn components(&self, i: usize) -> (f64, f64, f64) {
        let season = (self.first_season + i) % self.layout.period.max(1);
        let row = self.layout.row(i, self.start + i, season);
        let sum = |range: std::ops::Range<usize>| -> f64 {
            range.map(|j| row[j] * self.theta[j]).sum::<f64>() * self.scale
        };
        (
            sum(0..2) + sum(self.layout.range_changepoints()),
            sum(self.layout.range_seasonal()),
            sum(self.layout.range_events()),
        )
    }

    /// Trend at the relative position `i` (0 is the first observation; the
    /// history length and beyond are the future).
    pub fn trend(&self, i: usize) -> f64 {
        self.components(i).0
    }

    /// Seasonal effect at the relative position `i`.
    pub fn seasonal(&self, i: usize) -> f64 {
        self.components(i).1
    }

    /// Effect of the events at the relative position `i`.
    pub fn events(&self, i: usize) -> f64 {
        self.components(i).2
    }

    /// The changepoints where the trend did bend: relative position and the
    /// change in slope, in units of the series per period.
    pub fn changepoints(&self) -> Vec<(usize, f64)> {
        self.at
            .iter()
            .zip(&self.theta[self.layout.range_changepoints()])
            .filter(|(_, delta)| **delta != 0.0)
            .map(|(i, delta)| (*i, delta * self.scale / self.layout.span))
            .collect()
    }

    /// Estimated effect of each event, in units of the series.
    pub fn effects(&self) -> Vec<(String, f64)> {
        self.layout
            .events
            .iter()
            .zip(&self.theta[self.layout.range_events()])
            .map(|(e, c)| (e.name.clone(), c * self.scale))
            .collect()
    }

    /// Values fitted to the history.
    pub fn fitted(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let (trend, seasonal, events) = self.components(i);
                trend + seasonal + events
            })
            .collect()
    }

    /// Standard deviation of the noise, in units of the series.
    pub fn sigma(&self) -> f64 {
        self.sigma * self.scale
    }
}

impl Fitted for ProphetFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        (self.n..self.n + h)
            .map(|i| {
                let (trend, seasonal, events) = self.components(i);
                trend + seasonal + events
            })
            .collect()
    }

    fn params(&self) -> Params {
        let slope = self.scale / self.layout.span;
        let bends = self.changepoints();
        let mut p: Params = vec![
            ("initial_slope".into(), self.theta[0] * slope),
            (
                "final_slope".into(),
                (self.theta[0]
                    + self.theta[self.layout.range_changepoints()]
                        .iter()
                        .sum::<f64>())
                    * slope,
            ),
            ("changepoints".into(), bends.len() as f64),
            ("sigma".into(), self.sigma()),
        ];
        for (name, effect) in self.effects() {
            p.push((format!("event_{name}"), effect));
        }
        p
    }
}

impl Model for Prophet {
    fn name(&self) -> String {
        "prophet".into()
    }

    fn description(&self) -> String {
        "Prophet: piecewise linear trend with changepoints, Fourier seasonality and events".into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        Some(Box::new(self.estimate(y)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::noisy_seasonal_growth;

    const PATTERN: [f64; 12] = [
        5.0, -3.0, 0.0, 2.0, -4.0, 1.0, 3.0, -2.0, 0.0, 4.0, -5.0, 4.0,
    ];

    fn bent(n: usize, noise: f64) -> Vec<f64> {
        let e = noisy_seasonal_growth(n, 1.0);
        let clean = noisy_seasonal_growth(n, 0.0);
        (0..n)
            .map(|t| {
                let trend = if t < 60 {
                    0.5 * t as f64
                } else {
                    30.0 + 2.0 * (t - 60) as f64
                };
                200.0 + trend + PATTERN[t % 12] + noise * (e[t] / clean[t] - 1.0)
            })
            .collect()
    }

    #[test]
    fn follows_a_trend_that_changes_pace() {
        let all = bent(132, 2.0);
        let fit = Prophet::new()
            .estimate(Series::monthly(&all[..120], 0))
            .unwrap();
        for (k, v) in fit.forecast(12).iter().enumerate() {
            assert!((v / all[120 + k] - 1.0).abs() < 0.03, "h={}", k + 1);
        }
        let p = fit.params();
        let value = |name: &str| p.iter().find(|(k, _)| k == name).unwrap().1;
        assert!((value("initial_slope") - 0.5).abs() < 0.3);
        assert!((value("final_slope") - 2.0).abs() < 0.3);
        // most candidate changepoints are exactly zero
        assert!(fit.changepoints().len() < 20, "{:?}", fit.changepoints());
        assert!(fit
            .changepoints()
            .iter()
            .any(|(i, delta)| (50..=70).contains(i) && *delta > 0.2));
    }

    #[test]
    fn the_components_add_up_to_the_fit() {
        let y = bent(96, 2.0);
        let fit = Prophet::new().estimate(Series::monthly(&y, 3)).unwrap();
        let fitted = fit.fitted();
        for i in [0, 17, 95] {
            let sum = fit.trend(i) + fit.seasonal(i) + fit.events(i);
            assert!((sum - fitted[i]).abs() < 1e-9);
        }
        // the seasonal pattern repeats and averages out
        assert!((fit.seasonal(5) - fit.seasonal(17)).abs() < 1e-9);
        let mean: f64 = (0..12).map(|i| fit.seasonal(i)).sum::<f64>() / 12.0;
        assert!(mean.abs() < 0.5);
        assert!(fit.sigma() > 0.0);
    }

    #[test]
    fn events_and_steps_are_estimated_and_carried_into_the_forecast() {
        let e = noisy_seasonal_growth(120, 1.0);
        let clean = noisy_seasonal_growth(120, 0.0);
        let mut y: Vec<f64> = (0..120)
            .map(|t| 200.0 + 0.8 * t as f64 + PATTERN[t % 12] + e[t] / clean[t] - 1.0)
            .collect();
        // a recurring event worth +20 and a lasting drop of 15 from position 80
        let occurrences = [10, 34, 58, 82, 106, 126];
        for (t, v) in y.iter_mut().enumerate() {
            if occurrences.contains(&t) {
                *v += 20.0;
            }
            if t >= 80 {
                *v -= 15.0;
            }
        }
        let model = Prophet::new()
            .changepoints(0)
            .event("campaign", &occurrences)
            .step("new_law", 80);
        let plain = Prophet::new().changepoints(0);
        let series = Series::monthly(&y, 0);
        let fit = model.estimate(series).unwrap();
        let effects = fit.effects();
        assert_eq!(effects[0].0, "campaign");
        assert!((effects[0].1 - 20.0).abs() < 3.0, "{effects:?}");
        assert!((effects[1].1 + 15.0).abs() < 3.0, "{effects:?}");
        // position 126 is the 7th month of the forecast
        let with = fit.forecast(12);
        let without = plain.forecast(series, 12).unwrap();
        assert!(with[6] - with[5] > without[6] - without[5] + 10.0);
        assert!(fit.params().iter().any(|(k, _)| k == "event_new_law"));
    }

    #[test]
    fn both_searches_reach_the_same_minimum() {
        // a small problem: three columns, the last two under a lasso
        let design = [
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        ];
        let y = [1.0, 1.4, 2.1, 3.9, 6.2, 7.8];
        let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, z)| x * z).sum() };
        let gram: Vec<Vec<f64>> = design
            .iter()
            .map(|a| design.iter().map(|b| dot(a, b)).collect())
            .collect();
        let moment: Vec<f64> = design.iter().map(|a| dot(a, &y)).collect();
        for penalty in [0.1, 2.0, 50.0] {
            let problem = Penalised {
                gram: &gram,
                moment: &moment,
                ridge: &[0.04, 0.0, 0.0],
                lasso: &[0.0, penalty, penalty],
            };
            let (mut a, mut b) = (vec![0.0; 3], vec![0.0; 3]);
            assert!(problem.active_set(1.5, &mut a));
            problem.descend(1.5, &mut b);
            for (x, z) in a.iter().zip(&b) {
                assert!((x - z).abs() < 1e-8, "penalty {penalty}: {a:?} × {b:?}");
            }
        }
    }

    #[test]
    fn a_straight_line_stays_straight() {
        let y: Vec<f64> = (0..60).map(|t| 10.0 + 0.7 * t as f64).collect();
        let fit = Prophet::new().estimate(Series::non_seasonal(&y)).unwrap();
        for (k, v) in fit.forecast(6).iter().enumerate() {
            assert!(
                (v - (10.0 + 0.7 * (60 + k) as f64)).abs() < 0.05,
                "h={}",
                k + 1
            );
        }
        assert!(fit.seasonal(3).abs() < 1e-12);
    }

    #[test]
    fn seasonal_columns_describe_any_pattern_of_the_period() {
        let layout = |m: usize, order: usize| Layout {
            span: 1.0,
            changepoints: vec![],
            harmonics: Prophet::new().fourier_order(order).harmonics(m, 2 * m),
            period: m,
            events: vec![],
        };
        assert_eq!(layout(12, 10).seasonal(), 11);
        assert_eq!(layout(12, 3).seasonal(), 6);
        assert_eq!(layout(7, 10).seasonal(), 6);
        assert_eq!(layout(4, 10).seasonal(), 3);
        assert_eq!(layout(1, 10).seasonal(), 0);
        assert_eq!(layout(12, 0).seasonal(), 0);
    }

    #[test]
    fn refuses_short_series_and_bad_settings() {
        assert!(Prophet::new()
            .estimate(Series::non_seasonal(&[1.0, 2.0, 3.0]))
            .is_none());
        assert!(Prophet::new()
            .changepoint_prior_scale(0.0)
            .estimate(Series::non_seasonal(&[1.0, 2.0, 3.0, 4.0, 5.0]))
            .is_none());
        assert!(Prophet::new()
            .estimate(Series::non_seasonal(&[1.0, f64::NAN, 3.0, 4.0]))
            .is_none());
        // constant and all-zero series are fitted without trouble
        let fit = Prophet::new()
            .estimate(Series::non_seasonal(&[0.0; 12]))
            .unwrap();
        assert_eq!(fit.forecast(2), vec![0.0, 0.0]);
    }
}
