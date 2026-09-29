//! Combinations of models.

use crate::backtest::{map_indices, Candidate};
use crate::model::{Fitted, Model, Params};
use crate::series::Series;

/// How the forecasts of the members are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Weighting {
    /// The plain average.
    Equal,
    /// The median, which a member gone astray cannot drag along.
    Median,
    /// Weights in inverse proportion to the mean squared error each member
    /// had when forecasting the end of the history.
    #[default]
    InverseError,
    /// The weights, none negative and adding up to one, that would have
    /// given the combination with the smallest squared error over the end of
    /// the history. Members that add nothing get exactly zero.
    Stacked,
}

/// A model made of other models.
///
/// The weights are learnt from the series itself: the last `origins` periods
/// are forecast by every member from the data before them, as in a backtest,
/// and the errors decide how much each member counts. Then the members are
/// fitted on the whole series and their forecasts combined. Members that
/// cannot be fitted are left out.
///
/// Being a model, an ensemble can be a candidate in a [`Backtest`], next to
/// its own members: its weights are then learnt again at every origin, from
/// what was known at that point.
///
/// [`Backtest`]: crate::Backtest
///
/// ```
/// use foresight::models::{Drift, Ensemble, Naive, SeasonalNaive, Theta, Weighting};
/// use foresight::{Candidate, Model, Series};
///
/// let y: Vec<f64> = (0..72)
///     .map(|t| 100.0 + t as f64 + [8.0, -3.0, -6.0, 1.0][t % 4] + ((t * 7) % 5) as f64)
///     .collect();
/// let ensemble = Ensemble::new(vec![
///     Candidate::new(Naive),
///     Candidate::new(Drift),
///     Candidate::new(SeasonalNaive::with_growth()),
///     Candidate::new(Theta),
/// ])
/// .weighting(Weighting::Stacked);
/// let fit = ensemble.fit(Series::quarterly(&y, 0)).unwrap();
/// let weights: f64 = fit.params().iter().map(|(_, w)| w).sum();
/// assert!((weights - 1.0).abs() < 1e-9);
/// assert_eq!(fit.forecast(8).len(), 8);
/// ```
pub struct Ensemble {
    members: Vec<Candidate>,
    weighting: Weighting,
    origins: usize,
    horizon: Option<usize>,
    top: Option<usize>,
}

impl Ensemble {
    /// Weights by inverse error over the last 12 periods, forecasting one
    /// seasonal cycle ahead (at most 12 periods).
    pub fn new(members: Vec<Candidate>) -> Self {
        Ensemble {
            members,
            weighting: Weighting::default(),
            origins: 12,
            horizon: None,
            top: None,
        }
    }

    pub fn weighting(mut self, weighting: Weighting) -> Self {
        self.weighting = weighting;
        self
    }

    /// How many of the last periods are forecast to learn the weights.
    pub fn origins(mut self, n: usize) -> Self {
        self.origins = n.max(1);
        self
    }

    /// How far ahead each of those forecasts goes.
    pub fn horizon(mut self, h: usize) -> Self {
        self.horizon = Some(h.max(1));
        self
    }

    /// Keeps only the `k` members with the smallest error.
    pub fn top(mut self, k: usize) -> Self {
        self.top = Some(k.max(1));
        self
    }

    /// For each member, the errors (forecast − actual, in units of the mean
    /// size of the series) over the end of the history; `None` for a member
    /// that could not forecast from every origin, and for all of them when
    /// the series is too short to set the end apart.
    fn errors(&self, y: Series<'_>) -> Vec<Option<Vec<f64>>> {
        let (v, n) = (y.values(), y.len());
        let horizon = self.horizon.unwrap_or_else(|| y.period().clamp(1, 12));
        let shortest = (2 * y.period()).max(8);
        let origins = self.origins.min(n.saturating_sub(shortest));
        if origins == 0 {
            return vec![None; self.members.len()];
        }
        let first = n - origins;
        let scale = v.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
        let scale = if scale > 0.0 { scale } else { 1.0 };
        self.members
            .iter()
            .map(|member| {
                let forecasts: Option<Vec<Vec<f64>>> = map_indices(origins, true, |k| {
                    member
                        .model
                        .forecast(y.head(first + k), horizon)
                        .filter(|p| p.len() == horizon && p.iter().all(|x| x.is_finite()))
                })
                .into_iter()
                .collect();
                let forecasts = forecasts?;
                let mut errors = Vec::new();
                for (k, forecast) in forecasts.iter().enumerate() {
                    for (h, f) in forecast.iter().enumerate() {
                        if let Some(actual) = v.get(first + k + h) {
                            errors.push((f - actual) / scale);
                        }
                    }
                }
                Some(errors)
            })
            .collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, z)| x * z).sum()
}

/// The point of the simplex (no negative coordinate, sum one) closest to `v`.
fn onto_simplex(v: &[f64]) -> Vec<f64> {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (mut running, mut shift) = (0.0, 0.0);
    for (i, x) in sorted.iter().enumerate() {
        running += x;
        let candidate = (running - 1.0) / (i + 1) as f64;
        if x - candidate > 0.0 {
            shift = candidate;
        }
    }
    v.iter().map(|x| (x - shift).max(0.0)).collect()
}

/// Weights on the simplex that minimise the squared length of the weighted
/// sum of the error vectors, by projected gradient with momentum.
fn stacked(errors: &[&Vec<f64>]) -> Vec<f64> {
    let m = errors.len();
    let gram: Vec<Vec<f64>> = errors
        .iter()
        .map(|a| errors.iter().map(|b| dot(a, b)).collect())
        .collect();
    let steepest: f64 = (0..m).map(|i| gram[i][i]).sum::<f64>() * 2.0;
    if steepest <= 0.0 {
        return vec![1.0 / m as f64; m];
    }
    let value = |w: &[f64]| -> f64 { (0..m).map(|i| w[i] * dot(&gram[i], w)).sum() };
    let mut w = vec![1.0 / m as f64; m];
    let mut ahead = w.clone();
    let mut momentum = 1.0f64;
    for _ in 0..20_000 {
        let step: Vec<f64> = (0..m)
            .map(|i| ahead[i] - 2.0 * dot(&gram[i], &ahead) / steepest)
            .collect();
        let next = onto_simplex(&step);
        let moved: f64 = next.iter().zip(&w).map(|(a, b)| (a - b).abs()).sum();
        // momentum that overshoots is dropped
        let (faster, restart) = if value(&next) > value(&w) {
            (1.0, true)
        } else {
            (
                (1.0 + (1.0 + 4.0 * momentum * momentum).sqrt()) / 2.0,
                false,
            )
        };
        ahead = if restart {
            w.clone()
        } else {
            (0..m)
                .map(|i| next[i] + (momentum - 1.0) / faster * (next[i] - w[i]))
                .collect()
        };
        if !restart {
            w = next;
            if moved < 1e-13 {
                break;
            }
        }
        momentum = faster;
    }
    // what is numerically nothing is nothing
    let mut w: Vec<f64> = w.iter().map(|x| if *x < 1e-9 { 0.0 } else { *x }).collect();
    let total: f64 = w.iter().sum();
    w.iter_mut().for_each(|x| *x /= total);
    w
}

/// Name, model fitted on the whole series and errors over its end.
type Member = (String, Box<dyn Fitted>, Option<Vec<f64>>);

struct EnsembleFit {
    /// Name, weight and fitted model of each member that counts.
    members: Vec<(String, f64, Box<dyn Fitted>)>,
    median: bool,
}

impl Fitted for EnsembleFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        let each: Vec<Vec<f64>> = self.members.iter().map(|m| m.2.forecast(h)).collect();
        (0..h)
            .map(|k| {
                if self.median {
                    let column: Vec<f64> = each.iter().map(|f| f[k]).collect();
                    crate::accuracy::quantile(&column, 0.5).unwrap_or(f64::NAN)
                } else {
                    self.members
                        .iter()
                        .zip(&each)
                        .map(|(m, f)| m.1 * f[k])
                        .sum()
                }
            })
            .collect()
    }

    fn params(&self) -> Params {
        self.members
            .iter()
            .map(|(name, weight, _)| (format!("weight_{name}"), *weight))
            .collect()
    }
}

impl Model for Ensemble {
    fn name(&self) -> String {
        match self.weighting {
            Weighting::Equal => "ensemble_equal",
            Weighting::Median => "ensemble_median",
            Weighting::InverseError => "ensemble_inverse_error",
            Weighting::Stacked => "ensemble_stacked",
        }
        .into()
    }

    fn description(&self) -> String {
        let how = match self.weighting {
            Weighting::Equal => "average",
            Weighting::Median => "median",
            Weighting::InverseError => "average weighted by inverse error",
            Weighting::Stacked => "combination with the weights of least error",
        };
        let names: Vec<&str> = self.members.iter().map(|m| m.name.as_str()).collect();
        format!("Ensemble, {how}, of {}", names.join(", "))
    }

    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        let learnt = matches!(self.weighting, Weighting::InverseError | Weighting::Stacked)
            || self.top.is_some();
        let errors = if learnt {
            self.errors(y)
        } else {
            vec![None; self.members.len()]
        };
        // members fitted on the whole series, with their errors if any
        let mut fitted: Vec<Member> = self
            .members
            .iter()
            .zip(errors)
            .filter_map(|(m, e)| Some((m.name.clone(), m.model.fit(y)?, e)))
            .collect();
        if fitted.is_empty() {
            return None;
        }
        // errors can only decide if everyone left has them
        let measured = fitted
            .iter()
            .all(|m| m.2.as_ref().is_some_and(|e| !e.is_empty()));
        if learnt && !measured && fitted.iter().any(|m| m.2.is_some()) {
            fitted.retain(|m| m.2.as_ref().is_some_and(|e| !e.is_empty()));
        }
        let measured = fitted.iter().all(|m| m.2.is_some());
        let mse = |e: &Vec<f64>| dot(e, e) / e.len() as f64;
        if let (Some(k), true) = (self.top, measured) {
            fitted.sort_by(|a, b| {
                let (ea, eb) = (a.2.as_ref().map(mse), b.2.as_ref().map(mse));
                ea.unwrap_or(f64::INFINITY)
                    .total_cmp(&eb.unwrap_or(f64::INFINITY))
            });
            fitted.truncate(k);
        }
        let m = fitted.len();
        let weights: Vec<f64> = match self.weighting {
            Weighting::InverseError if measured => {
                let scores: Vec<f64> = fitted
                    .iter()
                    .filter_map(|f| f.2.as_ref().map(mse))
                    .collect();
                if scores.contains(&0.0) {
                    // whoever made no error takes it all
                    let perfect = scores.iter().filter(|s| **s == 0.0).count() as f64;
                    scores
                        .iter()
                        .map(|s| if *s == 0.0 { 1.0 / perfect } else { 0.0 })
                        .collect()
                } else {
                    let total: f64 = scores.iter().map(|s| 1.0 / s).sum();
                    scores.iter().map(|s| 1.0 / s / total).collect()
                }
            }
            Weighting::Stacked if measured => {
                let vectors: Vec<&Vec<f64>> = fitted.iter().filter_map(|f| f.2.as_ref()).collect();
                stacked(&vectors)
            }
            _ => vec![1.0 / m as f64; m],
        };
        Some(Box::new(EnsembleFit {
            members: fitted
                .into_iter()
                .zip(weights)
                .map(|((name, fit, _), w)| (name, w, fit))
                .collect(),
            median: self.weighting == Weighting::Median,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Drift, HoltWinters, LogLinear, Mean, Naive, SeasonalNaive, Theta};
    use crate::testing::noisy_seasonal_growth;

    fn members() -> Vec<Candidate> {
        vec![
            Candidate::new(Mean),
            Candidate::new(Naive),
            Candidate::new(SeasonalNaive::with_growth()),
            Candidate::new(Theta),
            Candidate::new(HoltWinters),
            Candidate::new(LogLinear::new()),
        ]
    }

    fn weights(fit: &dyn Fitted) -> Vec<(String, f64)> {
        fit.params()
    }

    #[test]
    fn projection_onto_the_simplex() {
        assert_eq!(onto_simplex(&[0.2, 0.3, 0.5]), vec![0.2, 0.3, 0.5]);
        assert_eq!(onto_simplex(&[2.0, 0.0]), vec![1.0, 0.0]);
        let p = onto_simplex(&[0.9, 0.8, -0.5]);
        assert!((p[0] - 0.55).abs() < 1e-12 && (p[1] - 0.45).abs() < 1e-12 && p[2] == 0.0);
    }

    #[test]
    fn stacking_finds_the_combination_that_cancels_the_errors() {
        // the first two err in opposite directions, the third is just bad
        let a = vec![1.0, -2.0, 0.5, 1.5];
        let b = vec![-3.0, 6.0, -1.5, -4.5];
        let c = vec![5.0, 5.0, 5.0, 5.0];
        let w = stacked(&[&a, &b, &c]);
        assert!(
            (w[0] - 0.75).abs() < 1e-6 && (w[1] - 0.25).abs() < 1e-6,
            "{w:?}"
        );
        assert_eq!(w[2], 0.0);
        assert_eq!(stacked(&[&vec![0.0; 3], &vec![0.0; 3]]), vec![0.5, 0.5]);
    }

    #[test]
    fn equal_weights_and_median() {
        let y = noisy_seasonal_growth(96, 0.05);
        let series = Series::monthly(&y, 0);
        let each: Vec<Vec<f64>> = members()
            .iter()
            .map(|m| m.model.forecast(series, 3).unwrap())
            .collect();
        let equal = Ensemble::new(members())
            .weighting(Weighting::Equal)
            .forecast(series, 3)
            .unwrap();
        let median = Ensemble::new(members())
            .weighting(Weighting::Median)
            .forecast(series, 3)
            .unwrap();
        for k in 0..3 {
            let column: Vec<f64> = each.iter().map(|f| f[k]).collect();
            let mean = column.iter().sum::<f64>() / 6.0;
            assert!((equal[k] - mean).abs() < 1e-9);
            let mut sorted = column.clone();
            sorted.sort_by(f64::total_cmp);
            assert!((median[k] - (sorted[2] + sorted[3]) / 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn learnt_weights_favour_the_members_that_forecast_well() {
        let y = noisy_seasonal_growth(120, 0.05);
        let series = Series::monthly(&y, 0);
        for weighting in [Weighting::InverseError, Weighting::Stacked] {
            let fit = Ensemble::new(members())
                .weighting(weighting)
                .fit(series)
                .unwrap();
            let w = weights(fit.as_ref());
            assert_eq!(w.len(), 6);
            assert!((w.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(w.iter().all(|x| x.1 >= 0.0));
            let of = |name: &str| {
                w.iter()
                    .find(|x| x.0 == format!("weight_{name}"))
                    .unwrap()
                    .1
            };
            // the mean of the history is no forecast for a series that grows
            assert!(of("mean") < of("log_linear"), "{weighting:?} {w:?}");
            assert!(of("mean") < 0.05);
        }
        let stacked = Ensemble::new(members()).weighting(Weighting::Stacked);
        assert_eq!(stacked.name(), "ensemble_stacked");
        assert!(stacked.description().contains("holt_winters"));
    }

    #[test]
    fn the_stacked_combination_is_no_worse_than_its_best_member_where_it_learnt() {
        let y = noisy_seasonal_growth(120, 0.08);
        let series = Series::monthly(&y, 0);
        let ensemble = Ensemble::new(members()).weighting(Weighting::Stacked);
        let errors: Vec<Vec<f64>> = ensemble.errors(series).into_iter().flatten().collect();
        let w = stacked(&errors.iter().collect::<Vec<_>>());
        let combined: f64 = (0..errors[0].len())
            .map(|t| (0..6).map(|i| w[i] * errors[i][t]).sum::<f64>().powi(2))
            .sum();
        let best = errors
            .iter()
            .map(|e| dot(e, e))
            .fold(f64::INFINITY, f64::min);
        assert!(combined <= best + 1e-12);
    }

    #[test]
    fn members_that_cannot_be_fitted_are_left_out() {
        // too short for Holt-Winters and the regression, enough for the rest
        let y = noisy_seasonal_growth(30, 0.05);
        let series = Series::monthly(&y, 0);
        let fit = Ensemble::new(members()).fit(series).unwrap();
        let names: Vec<String> = weights(fit.as_ref()).into_iter().map(|x| x.0).collect();
        assert!(!names.contains(&"weight_holt_winters".to_string()));
        assert!(names.contains(&"weight_naive".to_string()));
        // the best two only
        let top = Ensemble::new(members())
            .top(2)
            .fit(Series::monthly(&noisy_seasonal_growth(96, 0.05), 0))
            .unwrap();
        assert_eq!(top.params().len(), 2);
        assert!(Ensemble::new(vec![]).fit(series).is_none());
        assert!(Ensemble::new(vec![Candidate::new(Drift)])
            .fit(Series::non_seasonal(&[1.0]))
            .is_none());
    }
}
