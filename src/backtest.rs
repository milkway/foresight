//! Rolling-origin evaluation, model selection and empirical intervals.
//!
//! 1. **Rolling origin** — for each of the last `origins` periods, every
//!    candidate is fitted ONLY on the data before the origin and forecasts 1 to
//!    `horizon` periods ahead; errors are measured against what happened.
//!    Horizon h has `origins − h + 1` pairs.
//! 2. **Selection** — the candidate with the lowest error averaged over the
//!    horizons, the simple average of the best few models included.
//! 3. **Empirical intervals** — no normality assumed: for each horizon, the
//!    quantiles of the relative error r = actual/forecast − 1 seen in the
//!    backtest; the interval is forecast × (1 + q). With few pairs the wider
//!    levels are close to the worst error observed.
//! 4. **Cumulative intervals** — the same for the ratio of sums over the first
//!    k periods of each origin, because adding up the limits of k intervals
//!    overstates the uncertainty of a total.

use crate::accuracy::{mase_scale, quantile};
use crate::model::{Model, Params};
use crate::series::Series;

/// A model taking part in a backtest, under a name of the caller's choice.
pub struct Candidate {
    pub name: String,
    pub description: String,
    pub model: Box<dyn Model>,
}

impl Candidate {
    pub fn new(model: impl Model + 'static) -> Self {
        Candidate {
            name: model.name(),
            description: model.description(),
            model: Box::new(model),
        }
    }

    /// Replaces the model's own name and description.
    pub fn named(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.name = name.into();
        self.description = description.into();
        self
    }
}

/// Error measure averaged over the horizons to rank the candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    #[default]
    Mape,
    Mae,
    Rmse,
    Mase,
}

/// Configuration of the evaluation. The defaults suit monthly data: 36
/// origins, 12 periods ahead, at least 48 observations to train.
#[derive(Debug, Clone, PartialEq)]
pub struct Backtest {
    /// How many of the last periods are forecast origins.
    pub origins: usize,
    /// Longest horizon forecast and evaluated.
    pub horizon: usize,
    /// Training observations at the first origin. Shorter series get fewer
    /// origins.
    pub min_train: usize,
    /// Train on the last `window` observations only; `None` uses everything
    /// before the origin.
    pub window: Option<usize>,
    /// Also evaluate the simple average of the best `combine` models (fewer
    /// than 2 disables it).
    pub combine: usize,
    /// Coverage of the intervals, e.g. 0.8 for the 10%–90% quantiles.
    pub levels: Vec<f64>,
    /// Measure that ranks the candidates.
    pub metric: Metric,
    /// Fit the origins on all available cores. The result is the same either
    /// way.
    pub parallel: bool,
}

/// `f(0), …, f(n − 1)`, computed on all cores when `parallel`.
pub(crate) fn map_indices<T: Send>(
    n: usize,
    parallel: bool,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let threads = if parallel && cfg!(not(target_family = "wasm")) {
        std::thread::available_parallelism().map_or(1, |t| t.get().min(n))
    } else {
        1
    };
    if threads <= 1 {
        return (0..n).map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done: std::sync::Mutex<Vec<Option<T>>> =
        std::sync::Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let value = f(i);
                done.lock().expect("no thread panics holding the lock")[i] = Some(value);
            });
        }
    });
    done.into_inner()
        .expect("no thread panics holding the lock")
        .into_iter()
        .map(|v| v.expect("every index was computed"))
        .collect()
}

impl Default for Backtest {
    fn default() -> Self {
        Backtest {
            origins: 36,
            horizon: 12,
            min_train: 48,
            window: None,
            combine: 2,
            levels: vec![0.80, 0.95],
            metric: Metric::Mape,
            parallel: true,
        }
    }
}

/// Quantiles of a relative error r (actual/forecast − 1) for one coverage
/// level: the interval is forecast × (1 + lower) to forecast × (1 + upper).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    pub level: f64,
    pub lower: f64,
    pub upper: f64,
}

/// What the backtest measured at one horizon.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HorizonStats {
    /// Forecast/actual pairs evaluated.
    pub n: usize,
    /// Mean absolute percentage error, in percent.
    pub mape: Option<f64>,
    /// Mean of (forecast − actual)/actual, in percent; positive = forecasts
    /// ran high.
    pub bias: Option<f64>,
    pub mae: Option<f64>,
    pub rmse: Option<f64>,
    /// Mean absolute error scaled by the in-sample seasonal naive error of
    /// each training set.
    pub mase: Option<f64>,
    /// Relative error at this horizon, one band per level.
    pub bands: Vec<Band>,
    /// Relative error of the SUM of the first h periods.
    pub cumulative: Vec<Band>,
}

/// An interval around a forecast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub level: f64,
    pub lower: f64,
    pub upper: f64,
}

/// A forecast with its intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Periods ahead (for a cumulative forecast, how many were added up).
    pub horizon: usize,
    pub mean: f64,
    pub intervals: Vec<Interval>,
}

impl Point {
    fn new(horizon: usize, mean: f64, bands: &[Band]) -> Self {
        Point {
            horizon,
            mean,
            intervals: bands
                .iter()
                .map(|b| Interval {
                    level: b.level,
                    lower: mean * (1.0 + b.lower),
                    upper: mean * (1.0 + b.upper),
                })
                .collect(),
        }
    }

    /// The interval with the given coverage, if it was asked for.
    pub fn interval(&self, level: f64) -> Option<Interval> {
        self.intervals
            .iter()
            .copied()
            .find(|i| (i.level - level).abs() < 1e-9)
    }
}

/// Backtest and forecast of one candidate (a model or an average of models).
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateReport {
    pub name: String,
    pub description: String,
    /// Names of the models involved: one, or several for an average.
    pub components: Vec<String>,
    /// Index 0 is horizon 1.
    pub horizons: Vec<HorizonStats>,
    /// The ranking metric averaged over the horizons; infinite when it could
    /// not be computed.
    pub score: f64,
    /// Parameters fitted on the whole series (empty for an average).
    pub params: Params,
    /// Backtest forecasts, `[origin][h − 1]`.
    pub trajectories: Vec<Vec<f64>>,
    /// Forecast from the whole series, with the intervals of the backtest.
    pub forecast: Vec<Point>,
}

impl CandidateReport {
    /// Forecast of the SUM of the first `k` periods, with intervals from the
    /// backtest of sums. `None` when `k` is 0 or beyond the horizon.
    pub fn cumulative(&self, k: usize) -> Option<Point> {
        if k == 0 || k > self.forecast.len() {
            return None;
        }
        let sum = self.forecast[..k].iter().map(|p| p.mean).sum();
        Some(Point::new(k, sum, &self.horizons[k - 1].cumulative))
    }
}

/// Result of a backtest.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// The models in the order given, then the average of the best ones.
    pub candidates: Vec<CandidateReport>,
    /// Index of the chosen candidate.
    pub chosen: usize,
    /// Origins actually used.
    pub origins: usize,
    /// Position in the series of the first period forecast in the backtest.
    pub first_origin: usize,
    pub horizon: usize,
    pub metric: Metric,
}

impl Report {
    pub fn chosen(&self) -> &CandidateReport {
        &self.candidates[self.chosen]
    }

    /// Looks a candidate up by name.
    pub fn candidate(&self, name: &str) -> Option<&CandidateReport> {
        self.candidates.iter().find(|c| c.name == name)
    }
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn bands(errors: &[f64], levels: &[f64]) -> Vec<Band> {
    levels
        .iter()
        .filter_map(|&level| {
            let tail = (1.0 - level) / 2.0;
            Some(Band {
                level,
                lower: quantile(errors, tail)?,
                upper: quantile(errors, 1.0 - tail)?,
            })
        })
        .collect()
}

impl Backtest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn origins(mut self, n: usize) -> Self {
        self.origins = n;
        self
    }

    pub fn horizon(mut self, h: usize) -> Self {
        self.horizon = h;
        self
    }

    pub fn min_train(mut self, n: usize) -> Self {
        self.min_train = n;
        self
    }

    pub fn window(mut self, n: usize) -> Self {
        self.window = Some(n);
        self
    }

    pub fn combine(mut self, k: usize) -> Self {
        self.combine = k;
        self
    }

    pub fn levels(mut self, levels: &[f64]) -> Self {
        self.levels = levels.to_vec();
        self
    }

    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    fn train<'a>(&self, y: Series<'a>, origin: usize) -> Series<'a> {
        match self.window {
            Some(w) => y.slice(origin.saturating_sub(w)..origin),
            None => y.head(origin),
        }
    }

    fn stats(&self, y: Series<'_>, first: usize, trajectories: &[Vec<f64>]) -> Vec<HorizonStats> {
        let (v, n) = (y.values(), y.len());
        let scales: Vec<Option<f64>> = (0..trajectories.len())
            .map(|k| mase_scale(self.train(y, first + k).values(), y.period()))
            .collect();
        (1..=self.horizon)
            .map(|h| {
                let (mut abs_pct, mut pct, mut abs, mut sq) = (vec![], vec![], vec![], vec![]);
                let (mut scaled, mut rel, mut cum) = (vec![], vec![], vec![]);
                for (k, forecast) in trajectories.iter().enumerate() {
                    let origin = first + k; // trained on y[..origin]
                    let target = origin + h - 1;
                    if target >= n {
                        continue;
                    }
                    let (actual, predicted) = (v[target], forecast[h - 1]);
                    abs.push((predicted - actual).abs());
                    sq.push((predicted - actual) * (predicted - actual));
                    if let Some(s) = scales[k] {
                        scaled.push((predicted - actual).abs() / s);
                    }
                    if actual != 0.0 {
                        abs_pct.push((predicted - actual).abs() / actual.abs());
                        pct.push((predicted - actual) / actual);
                    }
                    if predicted != 0.0 {
                        rel.push(actual / predicted - 1.0);
                    }
                    let sum_actual: f64 = v[origin..=target].iter().sum();
                    let sum_predicted: f64 = forecast[..h].iter().sum();
                    if sum_predicted != 0.0 {
                        cum.push(sum_actual / sum_predicted - 1.0);
                    }
                }
                HorizonStats {
                    n: abs.len(),
                    mape: mean(&abs_pct).map(|m| m * 100.0),
                    bias: mean(&pct).map(|m| m * 100.0),
                    mae: mean(&abs),
                    rmse: mean(&sq).map(f64::sqrt),
                    mase: mean(&scaled),
                    bands: bands(&rel, &self.levels),
                    cumulative: bands(&cum, &self.levels),
                }
            })
            .collect()
    }

    fn score(&self, horizons: &[HorizonStats]) -> f64 {
        let values: Vec<f64> = horizons
            .iter()
            .filter_map(|s| match self.metric {
                Metric::Mape => s.mape,
                Metric::Mae => s.mae,
                Metric::Rmse => s.rmse,
                Metric::Mase => s.mase,
            })
            .collect();
        mean(&values)
            .filter(|m| m.is_finite())
            .unwrap_or(f64::INFINITY)
    }

    /// Evaluates the candidates on `y`, chooses one and forecasts `horizon`
    /// periods with every candidate that went through the whole backtest.
    ///
    /// `None` when the series is too short for at least one pair at the
    /// longest horizon, or when no candidate could be fitted at every origin.
    pub fn run(&self, y: Series<'_>, candidates: &[Candidate]) -> Option<Report> {
        let n = y.len();
        let origins = self.origins.min(n.saturating_sub(self.min_train));
        if self.horizon == 0 || origins < self.horizon {
            return None;
        }
        let first = n - origins;

        // 1. forecasts of every model at every origin
        struct Entry<'c> {
            components: Vec<&'c Candidate>,
            trajectories: Vec<Vec<f64>>,
        }
        let mut entries: Vec<Entry> = Vec::new();
        for c in candidates {
            let trajectories: Option<Vec<Vec<f64>>> = map_indices(origins, self.parallel, |k| {
                c.model
                    .forecast(self.train(y, first + k), self.horizon)
                    .filter(|p| p.len() == self.horizon)
            })
            .into_iter()
            .collect();
            if let Some(trajectories) = trajectories {
                entries.push(Entry {
                    components: vec![c],
                    trajectories,
                });
            }
        }
        if entries.is_empty() {
            return None;
        }
        let mut horizons: Vec<Vec<HorizonStats>> = entries
            .iter()
            .map(|e| self.stats(y, first, &e.trajectories))
            .collect();
        let mut scores: Vec<f64> = horizons.iter().map(|h| self.score(h)).collect();

        // 2. simple average of the best models
        let singles = entries.len();
        let mut order: Vec<usize> = (0..singles).collect();
        order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
        if self.combine >= 2 && singles >= self.combine {
            let best = &order[..self.combine];
            let trajectories: Vec<Vec<f64>> = (0..origins)
                .map(|o| {
                    (0..self.horizon)
                        .map(|h| {
                            best.iter()
                                .map(|&i| entries[i].trajectories[o][h])
                                .sum::<f64>()
                                / best.len() as f64
                        })
                        .collect()
                })
                .collect();
            let components = best.iter().map(|&i| entries[i].components[0]).collect();
            let stats = self.stats(y, first, &trajectories);
            scores.push(self.score(&stats));
            horizons.push(stats);
            entries.push(Entry {
                components,
                trajectories,
            });
        }

        // 3. choice by the average error over the horizons
        let chosen = (0..entries.len()).min_by(|&a, &b| scores[a].total_cmp(&scores[b]))?;

        // 4. forecast from the whole series, with each candidate's own bands
        let mut finals: Vec<(Vec<f64>, Params)> = Vec::with_capacity(singles);
        for e in &entries[..singles] {
            let fit = e.components[0].model.fit(y)?;
            finals.push((fit.forecast(self.horizon), fit.params()));
        }
        let position = |c: &Candidate| {
            entries[..singles]
                .iter()
                .position(|e| std::ptr::eq(e.components[0], c))
        };
        let reports = entries
            .iter()
            .zip(horizons)
            .zip(&scores)
            .map(|((e, stats), &score)| {
                let members: Vec<usize> = e.components.iter().filter_map(|c| position(c)).collect();
                let single = members.len() == 1;
                let names: Vec<String> = e.components.iter().map(|c| c.name.clone()).collect();
                let forecast = (0..self.horizon)
                    .map(|h| {
                        let m = members.iter().map(|&i| finals[i].0[h]).sum::<f64>()
                            / members.len() as f64;
                        Point::new(h + 1, m, &stats[h].bands)
                    })
                    .collect();
                CandidateReport {
                    name: if single {
                        names[0].clone()
                    } else {
                        format!("mean({})", names.join("+"))
                    },
                    description: if single {
                        e.components[0].description.clone()
                    } else {
                        format!("Simple average of {}", names.join(" and "))
                    },
                    components: names,
                    horizons: stats,
                    score,
                    params: if single {
                        finals[members[0]].1.clone()
                    } else {
                        Vec::new()
                    },
                    trajectories: e.trajectories.clone(),
                    forecast,
                }
            })
            .collect();

        Some(Report {
            candidates: reports,
            chosen,
            origins,
            first_origin: first,
            horizon: self.horizon,
            metric: self.metric,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{self, HoltWinters, LogLinear, Naive, SeasonalNaive};
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    fn classic() -> Vec<Candidate> {
        vec![
            Candidate::new(SeasonalNaive::with_growth()),
            Candidate::new(HoltWinters),
            Candidate::new(LogLinear::new()),
        ]
    }

    #[test]
    fn uses_36_origins_and_fewer_pairs_at_longer_horizons() {
        let y = seasonal_growth(112);
        let r = Backtest::default()
            .run(Series::monthly(&y, 2), &classic())
            .expect("backtest");
        assert_eq!((r.origins, r.first_origin, r.horizon), (36, 76, 12));
        let hw = r.candidate("holt_winters").unwrap();
        assert_eq!(hw.horizons[0].n, 36);
        assert_eq!(hw.horizons[11].n, 25);
        // without noise the log-linear regression (the right model for
        // geometric growth) is exact and Holt-Winters (additive trend) is close
        assert!(r.candidate("log_linear").unwrap().score < 1e-6);
        assert!(hw.score < 2.0, "Holt-Winters MAPE {}", hw.score);
        assert!(r.chosen().score < 1e-6);
        // the average of the two best was evaluated
        assert!(r.candidates.iter().any(|c| c.components.len() == 2));
        assert_eq!(r.candidates.len(), 4);
    }

    #[test]
    fn chooses_the_lowest_score_and_intervals_are_nested() {
        let y = noisy_seasonal_growth(112, 0.08);
        let r = Backtest::default()
            .run(Series::monthly(&y, 0), &classic())
            .unwrap();
        let min = r
            .candidates
            .iter()
            .map(|c| c.score)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(r.chosen().score, min);
        for p in &r.chosen().forecast {
            let (i80, i95) = (p.interval(0.80).unwrap(), p.interval(0.95).unwrap());
            assert!(
                i95.lower <= i80.lower && i80.lower <= i80.upper && i80.upper <= i95.upper,
                "h={}",
                p.horizon
            );
            assert!(i95.lower <= p.mean * 1.2 && i95.upper >= p.mean * 0.8);
        }
    }

    #[test]
    fn empirical_intervals_cover_about_80_percent_out_of_sample() {
        // long series: backtest on the first part, coverage measured on the 12
        // periods after each of several later origins
        let y = noisy_seasonal_growth(240, 0.10);
        let (mut inside, mut total) = (0, 0);
        for end in (160..=224).step_by(8) {
            let r = Backtest::default()
                .run(Series::monthly(&y[..end], 0), &classic())
                .unwrap();
            for p in &r.chosen().forecast {
                let actual = y[end + p.horizon - 1];
                let i = p.interval(0.80).unwrap();
                total += 1;
                if actual >= i.lower && actual <= i.upper {
                    inside += 1;
                }
            }
        }
        let coverage = f64::from(inside) / f64::from(total);
        assert!(
            (0.65..=0.95).contains(&coverage),
            "coverage of the 80% interval: {coverage:.2}"
        );
    }

    #[test]
    fn cumulative_forecast_adds_up_and_is_tighter_than_adding_limits() {
        let y = noisy_seasonal_growth(112, 0.10);
        let r = Backtest::default()
            .run(Series::monthly(&y, 0), &classic())
            .unwrap();
        let c = r.chosen();
        let total = c.cumulative(6).unwrap();
        let sum: f64 = c.forecast[..6].iter().map(|p| p.mean).sum();
        assert!((total.mean - sum).abs() < 1e-9);
        let naive_width: f64 = c.forecast[..6]
            .iter()
            .map(|p| {
                let i = p.interval(0.80).unwrap();
                i.upper - i.lower
            })
            .sum();
        let i = total.interval(0.80).unwrap();
        assert!(i.upper - i.lower < naive_width);
        assert!(c.cumulative(0).is_none() && c.cumulative(13).is_none());
    }

    #[test]
    fn short_series_and_unfit_models() {
        let y = seasonal_growth(112);
        // 55 observations leave 7 origins: not enough for 12 horizons
        assert!(Backtest::default()
            .run(Series::monthly(&y[..55], 0), &classic())
            .is_none());
        // a model that cannot be fitted at every origin is left out
        let r = Backtest::default()
            .min_train(12)
            .origins(24)
            .run(
                Series::monthly(&y[..48], 0),
                &[Candidate::new(Naive), Candidate::new(HoltWinters)],
            )
            .unwrap();
        assert_eq!(r.candidates.len(), 1);
        assert_eq!(r.chosen().name, "naive");
        assert!(Backtest::default()
            .run(Series::monthly(&y, 0), &[])
            .is_none());
    }

    #[test]
    fn rolling_window_other_metrics_and_custom_names() {
        let y = noisy_seasonal_growth(112, 0.05);
        let candidates = vec![
            Candidate::new(SeasonalNaive::new()).named("same_month", "Same month of last year"),
            Candidate::new(LogLinear::new()),
        ];
        let r = Backtest::new()
            .window(60)
            .metric(Metric::Mase)
            .combine(0)
            .levels(&[0.5])
            .run(Series::monthly(&y, 0), &candidates)
            .unwrap();
        assert_eq!(r.candidates.len(), 2);
        let c = r.candidate("same_month").unwrap();
        assert_eq!(c.description, "Same month of last year");
        assert!(c.horizons[0].mase.is_some());
        assert_eq!(
            c.score,
            mean(&c.horizons.iter().filter_map(|h| h.mase).collect::<Vec<_>>()).unwrap()
        );
        assert!(c.forecast[0].interval(0.5).is_some() && c.forecast[0].interval(0.8).is_none());
    }

    #[test]
    fn parallel_and_sequential_runs_agree() {
        let y = noisy_seasonal_growth(112, 0.08);
        let series = Series::monthly(&y, 0);
        let together = Backtest::default().run(series, &classic()).unwrap();
        let in_turn = Backtest::default()
            .parallel(false)
            .run(series, &classic())
            .unwrap();
        assert_eq!(together, in_turn);
        assert_eq!(map_indices(5, true, |i| i * i), vec![0, 1, 4, 9, 16]);
        assert_eq!(map_indices(0, true, |i| i), Vec::<usize>::new());
    }

    #[test]
    fn default_candidates_run_on_quarterly_data() {
        let idx = [0.9, 1.0, 0.8, 1.3];
        let y: Vec<f64> = noisy_seasonal_growth(80, 0.05)
            .iter()
            .enumerate()
            .map(|(t, v)| v * idx[t % 4])
            .collect();
        let r = Backtest::new()
            .origins(12)
            .horizon(4)
            .min_train(24)
            .run(Series::quarterly(&y, 0), &models::defaults())
            .unwrap();
        assert_eq!(r.candidates.len(), models::defaults().len() + 1);
        assert_eq!(r.chosen().forecast.len(), 4);
    }
}
