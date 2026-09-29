//! TBATS (De Livera, Hyndman & Snyder, 2011): trigonometric seasonality,
//! Box-Cox transformation, ARMA errors, trend and several seasonal periods.

use super::arima::stationary_coefficients;
use super::AutoArima;
use crate::linalg::solve;
use crate::model::{Fitted, Model, Params};
use crate::optimize::{nelder_mead, Effort};
use crate::series::Series;
use crate::transform::BoxCox;
use std::f64::consts::PI;

/// Exponential smoothing with seasonal patterns made of sines and cosines.
///
/// Because a pattern is a handful of harmonics rather than one state per
/// position of the cycle, the period may be long (365 days) or not a whole
/// number (52.18 weeks, 365.25 days), and several periods can be combined.
/// What is left after level, trend and seasonality may follow an ARMA
/// process, and the whole may run on a Box-Cox scale.
///
/// Whatever is not fixed is chosen by AIC: the number of harmonics of each
/// period, trend and damping, the transformation and the ARMA errors. The
/// initial states are found by least squares for each set of parameters,
/// which leaves the search with the few smoothing parameters only.
///
/// ```no_run
/// use foresight::{models::Tbats, Model, Series};
///
/// // daily data with a weekly pattern and one that takes 30.5 days
/// let at = |t: usize| {
///     let x = t as f64;
///     50.0 + 0.05 * x
///         + [4.0, 1.0, 0.0, -1.0, -2.0, -3.0, 1.0][t % 7]
///         + 6.0 * (2.0 * std::f64::consts::PI * x / 30.5).sin()
/// };
/// let y: Vec<f64> = (0..300).map(at).collect();
/// let model = Tbats::new(&[7.0, 30.5]).box_cox(false).arma_errors(false);
/// let forecast = model.forecast(Series::new(&y, 7), 14).unwrap();
/// assert!((forecast[0] - at(300)).abs() < 0.01);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tbats {
    periods: Vec<f64>,
    harmonics: Option<Vec<usize>>,
    box_cox: Option<bool>,
    trend: Option<bool>,
    damped: Option<bool>,
    arma: Option<bool>,
    orders: Option<(usize, usize)>,
}

/// What a model is made of.
#[derive(Debug, Clone, PartialEq)]
struct Shape {
    periods: Vec<f64>,
    harmonics: Vec<usize>,
    trend: bool,
    damped: bool,
    box_cox: bool,
    p: usize,
    q: usize,
}

/// The parameters of a model.
#[derive(Debug, Clone, PartialEq)]
struct Values {
    alpha: f64,
    beta: f64,
    phi: f64,
    /// (γ₁, γ₂) of each period.
    gamma: Vec<(f64, f64)>,
    ar: Vec<f64>,
    ma: Vec<f64>,
    lambda: Option<f64>,
}

fn logistic(u: f64) -> f64 {
    1.0 / (1.0 + (-u).exp())
}

impl Shape {
    fn seasonal_states(&self) -> usize {
        2 * self.harmonics.iter().sum::<usize>()
    }

    /// States estimated from the data: level, trend and the harmonics.
    fn seeds(&self) -> usize {
        1 + usize::from(self.trend) + self.seasonal_states()
    }

    fn states(&self) -> usize {
        self.seeds() + self.p + self.q
    }

    /// Numbers the search moves.
    fn dimension(&self) -> usize {
        1 + usize::from(self.trend)
            + usize::from(self.damped)
            + 2 * self.periods.len()
            + self.p
            + self.q
            + usize::from(self.box_cox)
    }

    /// Parameters and states counted by the information criterion.
    fn counted(&self) -> usize {
        self.dimension() + self.states()
    }

    fn start(&self, lambda: f64) -> Vec<f64> {
        let mut u = vec![0.9];
        if self.trend {
            u.push(1.0);
        }
        if self.damped {
            u.push(3.0);
        }
        u.resize(u.len() + 2 * self.periods.len() + self.p + self.q, 0.0);
        if self.box_cox {
            let l = lambda.clamp(0.02, 0.98);
            u.push((l / (1.0 - l)).ln());
        }
        u
    }

    fn values(&self, u: &[f64]) -> Values {
        let mut next = u.iter().copied();
        let mut take = || next.next().unwrap_or(0.0);
        let alpha = 0.1 * take();
        let beta = if self.trend { 0.05 * take() } else { 0.0 };
        let phi = if self.damped {
            0.8 + 0.2 * logistic(take())
        } else {
            1.0
        };
        let gamma = self
            .periods
            .iter()
            .map(|_| (0.01 * take(), 0.01 * take()))
            .collect();
        let ar: Vec<f64> = (0..self.p).map(|_| take()).collect();
        let ma: Vec<f64> = (0..self.q).map(|_| take()).collect();
        let lambda = self.box_cox.then(|| logistic(take()));
        Values {
            alpha,
            beta,
            phi,
            gamma,
            ar: stationary_coefficients(&ar),
            ma: stationary_coefficients(&ma).iter().map(|c| -c).collect(),
            lambda,
        }
    }
}

/// The model in innovations form: y(t) = w·x(t−1) + ε(t),
/// x(t) = F x(t−1) + g ε(t).
struct Machine<'a> {
    shape: &'a Shape,
    values: &'a Values,
    /// (cos λ, sin λ) of each harmonic, in the order of the states.
    turns: Vec<(f64, f64, usize)>,
    w: Vec<f64>,
    g: Vec<f64>,
}

impl<'a> Machine<'a> {
    fn new(shape: &'a Shape, values: &'a Values) -> Self {
        let n = shape.states();
        let (mut w, mut g) = (vec![0.0; n], vec![0.0; n]);
        w[0] = 1.0;
        g[0] = values.alpha;
        let mut at = 1;
        if shape.trend {
            w[1] = values.phi;
            g[1] = values.beta;
            at = 2;
        }
        let mut turns = Vec::new();
        for (i, (period, k)) in shape.periods.iter().zip(&shape.harmonics).enumerate() {
            for j in 1..=*k {
                let angle = 2.0 * PI * j as f64 / period;
                turns.push((angle.cos(), angle.sin(), i));
                w[at] = 1.0;
                g[at] = values.gamma[i].0;
                g[at + 1] = values.gamma[i].1;
                at += 2;
            }
        }
        for (i, c) in values.ar.iter().enumerate() {
            w[at + i] = *c;
        }
        if shape.p > 0 {
            g[at] = 1.0;
        }
        at += shape.p;
        for (i, c) in values.ma.iter().enumerate() {
            w[at + i] = *c;
        }
        if shape.q > 0 {
            g[at] = 1.0;
        }
        Machine {
            shape,
            values,
            turns,
            w,
            g,
        }
    }

    /// F x, using the structure of F.
    fn advance(&self, x: &[f64]) -> Vec<f64> {
        let s = self.shape;
        let v = self.values;
        let seeds = s.seeds();
        // the part of the error process that is already known
        let known: f64 = (0..s.p + s.q)
            .map(|i| self.w[seeds + i] * x[seeds + i])
            .sum();
        let mut out = vec![0.0; x.len()];
        let mut at = 1;
        out[0] = x[0] + v.alpha * known;
        if s.trend {
            out[0] += v.phi * x[1];
            out[1] = v.phi * x[1] + v.beta * known;
            at = 2;
        }
        for (cos, sin, i) in &self.turns {
            out[at] = cos * x[at] + sin * x[at + 1] + v.gamma[*i].0 * known;
            out[at + 1] = -sin * x[at] + cos * x[at + 1] + v.gamma[*i].1 * known;
            at += 2;
        }
        if s.p > 0 {
            out[at] = known;
            for i in 1..s.p {
                out[at + i] = x[at + i - 1];
            }
        }
        at += s.p;
        for i in 1..s.q {
            out[at + i] = x[at + i - 1];
        }
        out
    }

    fn observe(&self, x: &[f64]) -> f64 {
        self.w.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    /// True when the effect of an error does not grow: the matrix F − g w has
    /// no eigenvalue outside the unit circle, judged by how a vector grows
    /// when the matrix is applied to it many times. Eigenvalues on the circle
    /// are allowed (a seasonal pattern that never changes is one), and with
    /// them the slow growth of a vector that is not the fastest.
    fn is_stable(&self) -> bool {
        let n = self.shape.states();
        let mut x: Vec<f64> = (0..n).map(|i| 1.0 + 0.37 * i as f64).collect();
        let mut log_growth = 0.0;
        const ROUNDS: usize = 400;
        for round in 0..ROUNDS {
            let seen = self.observe(&x);
            let mut next = self.advance(&x);
            for (v, g) in next.iter_mut().zip(&self.g) {
                *v -= g * seen;
            }
            let norm = next.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() {
                return false;
            }
            if norm == 0.0 {
                return true;
            }
            // the first rounds carry the transient, not the long run
            if round >= ROUNDS / 2 {
                log_growth += norm.ln();
            }
            x = next.iter().map(|v| v / norm).collect();
        }
        log_growth / ((ROUNDS / 2) as f64) < 5e-3
    }

    /// Runs the model over `z` from the states `x`; returns the errors and
    /// leaves the final states in `x`.
    fn run(&self, z: &[f64], x: &mut Vec<f64>) -> Vec<f64> {
        z.iter()
            .map(|obs| {
                let e = obs - self.observe(x);
                let mut next = self.advance(x);
                for (v, g) in next.iter_mut().zip(&self.g) {
                    *v += g * e;
                }
                *x = next;
                e
            })
            .collect()
    }

    /// The initial states that minimise the sum of squared errors. The errors
    /// are linear in them: ε(t) = ε̃(t) − r(t)·x₀, where ε̃ are the errors from
    /// zero states and r(t) follows its own recursion.
    fn seed(&self, z: &[f64]) -> Option<Vec<f64>> {
        let seeds = self.shape.seeds();
        let states = self.shape.states();
        let mut x = vec![0.0; states];
        // column j: how the states depend on the j-th initial state
        let mut columns: Vec<Vec<f64>> = (0..seeds)
            .map(|j| {
                let mut c = vec![0.0; states];
                c[j] = 1.0;
                c
            })
            .collect();
        let mut gram = vec![vec![0.0; seeds]; seeds];
        let mut moment = vec![0.0; seeds];
        for obs in z {
            let e = obs - self.observe(&x);
            let r: Vec<f64> = columns.iter().map(|c| self.observe(c)).collect();
            for a in 0..seeds {
                moment[a] += r[a] * e;
                for b in 0..seeds {
                    gram[a][b] += r[a] * r[b];
                }
            }
            let mut next = self.advance(&x);
            for (v, g) in next.iter_mut().zip(&self.g) {
                *v += g * e;
            }
            x = next;
            for (c, rj) in columns.iter_mut().zip(&r) {
                let mut next = self.advance(c);
                for (v, g) in next.iter_mut().zip(&self.g) {
                    *v -= g * rj;
                }
                *c = next;
            }
        }
        // a whisper of ridge keeps the system solvable when two states are
        // nearly indistinguishable over the sample
        let size = (0..seeds).map(|a| gram[a][a]).fold(0.0, f64::max);
        for (a, row) in gram.iter_mut().enumerate() {
            row[a] += 1e-12 * size;
        }
        let mut seed = solve(gram, moment)?;
        seed.resize(states, 0.0);
        Some(seed)
    }
}

/// A fit of one shape.
struct Attempt {
    shape: Shape,
    values: Values,
    seed: Vec<f64>,
    last: Vec<f64>,
    errors: Vec<f64>,
    /// n ln Σε² − 2(λ − 1) Σ ln y
    likelihood: f64,
    aic: f64,
}

fn transform(y: &[f64], lambda: Option<f64>) -> Vec<f64> {
    match lambda {
        Some(l) => {
            let t = BoxCox::new(l);
            y.iter().map(|v| t.apply(*v)).collect()
        }
        None => y.to_vec(),
    }
}

fn attempt(shape: &Shape, y: &[f64], start_lambda: f64, effort: Effort) -> Option<Attempt> {
    let n = y.len();
    if n < shape.counted() + 4 {
        return None;
    }
    let logs: f64 = if shape.box_cox {
        y.iter().map(|v| v.ln()).sum()
    } else {
        0.0
    };
    let evaluate = |u: &[f64]| -> Option<Attempt> {
        let values = shape.values(u);
        let machine = Machine::new(shape, &values);
        if !machine.is_stable() {
            return None;
        }
        let z = transform(y, values.lambda);
        let seed = machine.seed(&z)?;
        let mut x = seed.clone();
        let errors = machine.run(&z, &mut x);
        let squares: f64 = errors.iter().map(|e| e * e).sum();
        if !squares.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let fit = if squares > 0.0 {
            n as f64 * squares.ln()
        } else {
            -1e300
        };
        let likelihood = fit - 2.0 * (values.lambda.unwrap_or(1.0) - 1.0) * logs;
        Some(Attempt {
            shape: shape.clone(),
            aic: likelihood + 2.0 * shape.counted() as f64,
            values,
            seed,
            last: x,
            errors,
            likelihood,
        })
    };
    let objective = |u: &[f64]| evaluate(u).map_or(f64::INFINITY, |a| a.likelihood);
    let (u, _) = nelder_mead(objective, &shape.start(start_lambda), 0.5, effort);
    evaluate(&u)
}

impl Tbats {
    /// With the given seasonal periods, which need not be whole numbers. With
    /// none, the period of the series is used.
    pub fn new(periods: &[f64]) -> Self {
        let mut periods: Vec<f64> = periods
            .iter()
            .copied()
            .filter(|p| p.is_finite() && *p > 1.0)
            .collect();
        periods.sort_by(f64::total_cmp);
        periods.dedup();
        Tbats {
            periods,
            ..Self::default()
        }
    }

    /// Fixes the number of harmonics of each period, in increasing order of
    /// period, instead of choosing them.
    pub fn harmonics(mut self, harmonics: &[usize]) -> Self {
        self.harmonics = Some(harmonics.to_vec());
        self
    }

    /// Fixes whether the model runs on a Box-Cox scale (λ between 0 and 1,
    /// estimated with the other parameters).
    pub fn box_cox(mut self, on: bool) -> Self {
        self.box_cox = Some(on);
        self
    }

    /// Fixes whether there is a trend.
    pub fn trend(mut self, on: bool) -> Self {
        self.trend = Some(on);
        self
    }

    /// Fixes whether the trend is damped.
    pub fn damped(mut self, on: bool) -> Self {
        self.damped = Some(on);
        self
    }

    /// Fixes whether the errors may follow an ARMA process.
    pub fn arma_errors(mut self, on: bool) -> Self {
        self.arma = Some(on);
        self
    }

    /// Fixes the orders (p, q) of the ARMA errors instead of choosing them.
    pub fn arma_orders(mut self, p: usize, q: usize) -> Self {
        (self.arma, self.orders) = (Some(p + q > 0), Some((p, q)));
        self
    }

    /// Most harmonics each period can have: fewer than half the period, and
    /// none that another, shorter period already has.
    fn ceilings(periods: &[f64]) -> Vec<usize> {
        periods
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut most = ((p - 1.0) / 2.0).floor().max(1.0) as usize;
                for shorter in &periods[..i] {
                    let ratio = p / shorter;
                    if (ratio - ratio.round()).abs() < 1e-9 {
                        most = most.min((ratio.round() as usize).saturating_sub(1).max(1));
                    }
                }
                most
            })
            .collect()
    }

    /// Chooses what was left open and returns the estimated model.
    pub fn select(&self, y: Series<'_>) -> Option<TbatsFit> {
        let v = y.values();
        if v.len() < 8 || !y.is_finite() {
            return None;
        }
        let periods: Vec<f64> = if self.periods.is_empty() && y.period() > 1 {
            vec![y.period() as f64]
        } else {
            self.periods.clone()
        };
        let periods: Vec<f64> = periods
            .into_iter()
            .filter(|p| 2.0 * p < v.len() as f64)
            .collect();
        let positive = y.is_positive();
        if self.box_cox == Some(true) && !positive {
            return None;
        }
        let lambda = BoxCox::guerrero(y).map_or(0.5, |t| t.lambda());
        let ceilings = Self::ceilings(&periods);
        let fixed = self.harmonics.as_ref().map(|h| {
            (0..periods.len())
                .map(|i| h.get(i).copied().unwrap_or(1).clamp(1, ceilings[i]))
                .collect::<Vec<usize>>()
        });
        let mut best = attempt(
            &Shape {
                harmonics: fixed.clone().unwrap_or_else(|| vec![1; periods.len()]),
                periods,
                trend: self.trend.unwrap_or(true),
                damped: self.trend.unwrap_or(true) && self.damped.unwrap_or(false),
                box_cox: self.box_cox.unwrap_or(false),
                p: 0,
                q: 0,
            },
            v,
            lambda,
            Effort::QUICK,
        )?;
        let consider = |shape: Shape, best: &mut Attempt| -> bool {
            match attempt(&shape, v, lambda, Effort::QUICK) {
                Some(a) if a.aic < best.aic => {
                    *best = a;
                    true
                }
                _ => false,
            }
        };

        // 1. harmonics of each period, one more for as long as it pays
        if fixed.is_none() {
            for (i, most) in ceilings.iter().enumerate() {
                while best.shape.harmonics[i] < *most {
                    let mut shape = best.shape.clone();
                    shape.harmonics[i] += 1;
                    if !consider(shape, &mut best) {
                        break;
                    }
                }
            }
        }
        // 2. transformation, trend and damping, each with and without ARMA
        //    errors of the orders that suit what it leaves unexplained
        let transformations: Vec<bool> = match self.box_cox {
            Some(on) => vec![on],
            None if positive => vec![false, true],
            None => vec![false],
        };
        let trends: Vec<(bool, bool)> = [(false, false), (true, false), (true, true)]
            .into_iter()
            .filter(|(trend, damped)| {
                self.trend.is_none_or(|t| t == *trend)
                    && (!trend || self.damped.is_none_or(|d| d == *damped))
            })
            .collect();
        let with_arma = |plain: &Attempt| -> Option<Attempt> {
            if !self.arma.unwrap_or(true) {
                return None;
            }
            let (p, q) = match self.orders {
                Some(orders) => orders,
                None => {
                    let (p, _, q) = AutoArima::new()
                        .differences(0, 0)
                        .select(Series::non_seasonal(&plain.errors))?
                        .order();
                    (p, q)
                }
            };
            if p + q == 0 {
                return None;
            }
            let mut shape = plain.shape.clone();
            (shape.p, shape.q) = (p, q);
            attempt(&shape, v, lambda, Effort::THOROUGH)
        };
        let base = best.shape.clone();
        let mut candidates: Vec<Attempt> = Vec::new();
        for box_cox in transformations {
            for &(trend, damped) in &trends {
                let mut shape = base.clone();
                (shape.box_cox, shape.trend, shape.damped) = (box_cox, trend, damped);
                if let Some(plain) = attempt(&shape, v, lambda, Effort::THOROUGH) {
                    candidates.extend(with_arma(&plain));
                    if self.orders.is_none_or(|(p, q)| p + q == 0) {
                        candidates.push(plain);
                    }
                }
            }
        }
        if let Some(winner) = candidates
            .into_iter()
            .min_by(|a, b| a.aic.total_cmp(&b.aic))
        {
            if winner.aic < best.aic || self.orders.is_some() {
                best = winner;
            }
        }
        let n = v.len() as f64;
        let squares: f64 = best.errors.iter().map(|e| e * e).sum();
        Some(TbatsFit {
            sigma2: squares / n,
            attempt: best,
        })
    }
}

/// An estimated TBATS model.
pub struct TbatsFit {
    attempt: Attempt,
    /// Variance of the errors on the scale of the fit.
    pub sigma2: f64,
}

impl TbatsFit {
    /// The seasonal periods and the harmonics of each.
    pub fn seasonal(&self) -> Vec<(f64, usize)> {
        let s = &self.attempt.shape;
        s.periods
            .iter()
            .copied()
            .zip(s.harmonics.iter().copied())
            .collect()
    }

    /// λ of the Box-Cox transformation, if one was used.
    pub fn lambda(&self) -> Option<f64> {
        self.attempt.values.lambda
    }

    /// Whether there is a trend, and its damping (1 for none).
    pub fn trend(&self) -> Option<f64> {
        self.attempt.shape.trend.then_some(self.attempt.values.phi)
    }

    /// Orders (p, q) of the ARMA errors.
    pub fn arma(&self) -> (usize, usize) {
        (self.attempt.shape.p, self.attempt.shape.q)
    }

    /// Smoothing of the level and of the trend.
    pub fn smoothing(&self) -> (f64, f64) {
        (self.attempt.values.alpha, self.attempt.values.beta)
    }

    /// n ln Σε² − 2(λ − 1) Σ ln y: the likelihood as TBATS reports it, the
    /// lower the better.
    pub fn likelihood(&self) -> f64 {
        self.attempt.likelihood
    }

    pub fn aic(&self) -> f64 {
        self.attempt.aic
    }

    /// Errors of the one-step forecasts, on the scale of the fit.
    pub fn residuals(&self) -> &[f64] {
        &self.attempt.errors
    }

    /// The states before the first observation.
    pub fn initial_states(&self) -> &[f64] {
        &self.attempt.seed
    }
}

impl Fitted for TbatsFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        let a = &self.attempt;
        let machine = Machine::new(&a.shape, &a.values);
        let back = a.values.lambda.map(BoxCox::new);
        let mut x = a.last.clone();
        (0..h)
            .map(|_| {
                let z = machine.observe(&x);
                x = machine.advance(&x);
                match back {
                    Some(t) => t.invert(z),
                    None => z,
                }
            })
            .collect()
    }

    fn params(&self) -> Params {
        let a = &self.attempt;
        let mut p: Params = vec![("alpha".into(), a.values.alpha)];
        if a.shape.trend {
            p.push(("beta".into(), a.values.beta));
        }
        if a.shape.damped {
            p.push(("phi".into(), a.values.phi));
        }
        if let Some(l) = a.values.lambda {
            p.push(("lambda".into(), l));
        }
        for (i, ((period, k), (g1, g2))) in a
            .shape
            .periods
            .iter()
            .zip(&a.shape.harmonics)
            .zip(&a.values.gamma)
            .enumerate()
        {
            p.push((format!("period{}", i + 1), *period));
            p.push((format!("harmonics{}", i + 1), *k as f64));
            p.push((format!("gamma1_{}", i + 1), *g1));
            p.push((format!("gamma2_{}", i + 1), *g2));
        }
        for (i, c) in a.values.ar.iter().enumerate() {
            p.push((format!("ar{}", i + 1), *c));
        }
        for (i, c) in a.values.ma.iter().enumerate() {
            p.push((format!("ma{}", i + 1), *c));
        }
        p.push(("sigma2".into(), self.sigma2));
        p.push(("aic".into(), a.aic));
        p
    }
}

impl Model for Tbats {
    fn name(&self) -> String {
        "tbats".into()
    }

    fn description(&self) -> String {
        "TBATS: trigonometric seasonality, Box-Cox, ARMA errors and trend, chosen by AIC".into()
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        Some(Box::new(self.select(y)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    fn shape(periods: &[f64], harmonics: &[usize], trend: bool, p: usize, q: usize) -> Shape {
        Shape {
            periods: periods.to_vec(),
            harmonics: harmonics.to_vec(),
            trend,
            damped: false,
            box_cox: false,
            p,
            q,
        }
    }

    #[test]
    fn states_and_parameters_are_counted() {
        let s = shape(&[7.0, 30.5], &[3, 2], true, 1, 2);
        assert_eq!(s.seeds(), 2 + 2 * 5);
        assert_eq!(s.states(), 12 + 3);
        assert_eq!(s.dimension(), 1 + 1 + 4 + 3);
        assert_eq!(s.start(0.3).len(), s.dimension());
        let v = s.values(&s.start(0.3));
        assert!((v.alpha - 0.09).abs() < 1e-12 && (v.beta - 0.05).abs() < 1e-12);
        assert_eq!(v.phi, 1.0);
        assert_eq!(v.gamma, vec![(0.0, 0.0); 2]);
    }

    #[test]
    fn harmonics_stay_below_half_the_period_and_do_not_repeat() {
        assert_eq!(Tbats::ceilings(&[12.0]), vec![5]);
        assert_eq!(Tbats::ceilings(&[7.0, 365.25]), vec![3, 182]);
        // the 24th harmonic of a week of hours is the first of a day
        assert_eq!(Tbats::ceilings(&[24.0, 168.0]), vec![11, 6]);
        assert_eq!(Tbats::ceilings(&[2.5]), vec![1]);
    }

    #[test]
    fn advancing_the_states_by_structure_agrees_with_the_equations() {
        let s = shape(&[6.0], &[2], true, 1, 1);
        let v = Values {
            alpha: 0.3,
            beta: 0.1,
            phi: 1.0,
            gamma: vec![(0.02, 0.03)],
            ar: vec![0.5],
            ma: vec![0.4],
            lambda: None,
        };
        let m = Machine::new(&s, &v);
        // level, trend, s1, s1*, s2, s2*, d(t), ε(t)
        let x = [10.0, 1.0, 2.0, 0.5, -1.0, 0.3, 0.8, -0.2];
        let known = 0.5 * 0.8 + 0.4 * -0.2;
        assert!((m.observe(&x) - (10.0 + 1.0 + 2.0 - 1.0 + known)).abs() < 1e-12);
        let next = m.advance(&x);
        assert!((next[0] - (11.0 + 0.3 * known)).abs() < 1e-12);
        assert!((next[1] - (1.0 + 0.1 * known)).abs() < 1e-12);
        let (c, sn) = ((2.0 * PI / 6.0).cos(), (2.0 * PI / 6.0).sin());
        assert!((next[2] - (c * 2.0 + sn * 0.5 + 0.02 * known)).abs() < 1e-12);
        assert!((next[3] - (-sn * 2.0 + c * 0.5 + 0.03 * known)).abs() < 1e-12);
        assert!((next[6] - known).abs() < 1e-12 && next[7] == 0.0);
        // a pattern that never changes is on the edge of stability, and allowed
        let steady = Values {
            gamma: vec![(0.0, 0.0)],
            ..v.clone()
        };
        assert!(Machine::new(&s, &steady).is_stable());
        // a level that overreacts is not stable
        let wild = Values { alpha: 2.5, ..v };
        assert!(!Machine::new(&s, &wild).is_stable());
    }

    #[test]
    fn initial_states_by_least_squares_beat_any_other() {
        let y = noisy_seasonal_growth(96, 0.1);
        let s = shape(&[12.0], &[2], true, 0, 0);
        let v = s.values(&s.start(0.5));
        let m = Machine::new(&s, &v);
        let seed = m.seed(&y).unwrap();
        let sse = |x0: &[f64]| -> f64 {
            let mut x = x0.to_vec();
            m.run(&y, &mut x).iter().map(|e| e * e).sum()
        };
        let best = sse(&seed);
        for j in 0..seed.len() {
            for step in [-0.5, 0.5] {
                let mut other = seed.clone();
                other[j] += step;
                assert!(sse(&other) > best);
            }
        }
    }

    #[test]
    fn follows_a_pattern_with_a_period_that_is_not_a_whole_number() {
        let e = noisy_seasonal_growth(420, 1.0);
        let clean = seasonal_growth(420);
        let at = |t: usize| -> f64 {
            let x = t as f64;
            80.0 + 0.04 * x + 5.0 * (2.0 * PI * x / 30.4).sin() + 2.0 * (4.0 * PI * x / 30.4).cos()
        };
        let y: Vec<f64> = (0..400)
            .map(|t| at(t) + 0.5 * (e[t] / clean[t] - 1.0))
            .collect();
        let fit = Tbats::new(&[30.4])
            .box_cox(false)
            .arma_errors(false)
            .select(Series::non_seasonal(&y))
            .unwrap();
        assert_eq!(fit.seasonal()[0].0, 30.4);
        assert!(fit.seasonal()[0].1 >= 2, "{:?}", fit.seasonal());
        for (k, v) in fit.forecast(20).iter().enumerate() {
            assert!((v - at(400 + k)).abs() < 1.0, "h={}", k + 1);
        }
        assert!(fit.lambda().is_none() && fit.arma() == (0, 0));
        assert_eq!(fit.residuals().len(), 400);
    }

    #[test]
    fn multiplicative_monthly_data_is_handled_on_a_box_cox_scale() {
        let all = noisy_seasonal_growth(132, 0.04);
        let fit = Tbats::new(&[])
            .box_cox(true)
            .select(Series::monthly(&all[..120], 0))
            .unwrap();
        let lambda = fit.lambda().unwrap();
        assert!((0.0..=1.0).contains(&lambda));
        for (k, v) in fit.forecast(12).iter().enumerate() {
            assert!((v / all[120 + k] - 1.0).abs() < 0.08, "h={}", k + 1);
        }
        let names: Vec<String> = fit.params().into_iter().map(|(k, _)| k).collect();
        assert!(names.contains(&"lambda".to_string()) && names.contains(&"harmonics1".to_string()));
    }

    #[test]
    fn what_is_fixed_stays_fixed_and_bad_input_is_refused() {
        let y = noisy_seasonal_growth(96, 0.05);
        let fit = Tbats::new(&[12.0])
            .harmonics(&[3])
            .trend(true)
            .damped(true)
            .box_cox(false)
            .arma_errors(false)
            .select(Series::monthly(&y, 0))
            .unwrap();
        assert_eq!(fit.seasonal(), vec![(12.0, 3)]);
        let phi = fit.trend().unwrap();
        assert!((0.8..=1.0).contains(&phi));
        assert!(fit.aic() > fit.likelihood());
        // a period that does not fit twice in the series is left out
        let short = Tbats::new(&[60.0])
            .box_cox(false)
            .arma_errors(false)
            .select(Series::non_seasonal(&y))
            .unwrap();
        assert!(short.seasonal().is_empty());
        let negative: Vec<f64> = y.iter().map(|v| v - 150.0).collect();
        assert!(Tbats::new(&[12.0])
            .box_cox(true)
            .select(Series::monthly(&negative, 0))
            .is_none());
        assert!(Tbats::new(&[12.0])
            .select(Series::monthly(&[1.0, 2.0, 3.0], 0))
            .is_none());
    }
}
