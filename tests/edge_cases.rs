//! Inputs at the edges: what used to give wrong answers or no answer at all.

use foresight::clean::outliers;
use foresight::models::{
    self, Arima, AutoArima, Croston, Decomposed, Drift, HoltWinters, Mean, Naive, Prophet,
    SeasonalNaive, Tbats,
};
use foresight::{Backtest, Candidate, Fitted, Model, Regressors, Series, Transformed};

const PATTERN: [f64; 12] = [
    1.05, 0.80, 0.95, 1.00, 0.98, 0.97, 1.02, 1.00, 0.96, 1.01, 0.96, 1.30,
];

fn seasonal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let wobble = 1.0 + 0.02 * ((t * 37 % 11) as f64 / 10.0 - 0.5);
            100.0 * 1.005f64.powi(t as i32) * PATTERN[t % 12] * wobble
        })
        .collect()
}

#[test]
fn a_candidate_that_cannot_give_the_final_forecast_is_left_out() {
    // the last observation is in no training set: multiplicative models go
    // through every origin and then cannot be fitted on the whole series
    let mut y = seasonal(120);
    y[119] = 0.0;
    let candidates = vec![
        Candidate::new(Naive),
        Candidate::new(HoltWinters),
        Candidate::new(Transformed::log(Drift)),
    ];
    let r = Backtest::default()
        .run(Series::monthly(&y, 0), &candidates)
        .expect("the naive forecast is still there");
    assert!(r.candidate("naive").is_some());
    assert!(r.candidate("holt_winters").is_none());
    assert!(r.candidate("log_drift").is_none());
    assert!(Backtest::default()
        .run(Series::monthly(&y, 0), &models::defaults())
        .is_some());
}

#[test]
fn regressors_that_stop_short_of_the_horizon_leave_the_candidate_out() {
    let y = seasonal(120);
    let h = 12;
    let x: Vec<f64> = (0..120 + h - 1).map(|t| (t as f64 * 0.7).sin()).collect();
    let short = Arima::new(1, 0, 0).with_regressors(Regressors::new().with("x", x));
    let candidates = vec![Candidate::new(short.clone()), Candidate::new(Naive)];
    let r = Backtest::default()
        .run(Series::monthly(&y, 0), &candidates)
        .unwrap();
    assert_eq!(r.candidates.len(), 1);
    for c in &r.candidates {
        assert!(c.forecast.iter().all(|p| p.mean.is_finite()));
    }
    // on its own and inside another model, no answer rather than NaN
    let series = Series::monthly(&y, 0);
    assert!(short.forecast(series, h).is_none());
    assert!(Transformed::log(short.clone())
        .forecast(series, h)
        .is_none());
    assert!(Decomposed::new(short).forecast(series, h).is_none());
}

#[test]
fn intervals_of_negative_forecasts_are_in_order() {
    let y: Vec<f64> = seasonal(120).iter().map(|v| -v).collect();
    let r = Backtest::default()
        .run(
            Series::monthly(&y, 0),
            &[Candidate::new(Mean), Candidate::new(Naive)],
        )
        .unwrap();
    // the naive forecast is about right, so it is inside its own interval
    let p = &r.candidate("naive").unwrap().forecast[0];
    let i = p.interval(0.8).unwrap();
    assert!(
        i.lower < p.mean && p.mean < i.upper,
        "{i:?} around {}",
        p.mean
    );
    for c in &r.candidates {
        for p in &c.forecast {
            assert!(p.mean < 0.0);
            for i in &p.intervals {
                assert!(i.lower <= i.upper, "{i:?}");
            }
        }
        let i = c.cumulative(6).unwrap().interval(0.8).unwrap();
        assert!(i.lower <= i.upper, "{i:?}");
    }
}

#[test]
fn levels_have_to_be_between_zero_and_one() {
    let y = seasonal(120);
    for bad in [0.0, 1.0, 1.5, -0.5, f64::NAN] {
        let r = Backtest::default()
            .levels(&[0.8, bad])
            .run(Series::monthly(&y, 0), &[Candidate::new(Naive)]);
        assert!(r.is_none(), "level {bad} accepted");
    }
}

/// Forecasts the number of observations it was fitted on.
struct Length;
struct LengthFit(f64);

impl Fitted for LengthFit {
    fn forecast(&self, h: usize) -> Vec<f64> {
        vec![self.0; h]
    }
}

impl Model for Length {
    fn name(&self) -> String {
        "length".into()
    }
    fn description(&self) -> String {
        "the number of observations".into()
    }
    fn fit(&self, y: Series<'_>) -> Option<Box<dyn Fitted>> {
        Some(Box::new(LengthFit(y.len() as f64)))
    }
}

#[test]
fn the_window_also_holds_for_the_final_forecast() {
    let y = seasonal(120);
    let r = Backtest::default()
        .window(60)
        .run(Series::monthly(&y, 0), &[Candidate::new(Length)])
        .unwrap();
    let c = &r.candidates[0];
    assert!(c.trajectories.iter().all(|t| t[0] == 60.0));
    assert_eq!(c.forecast[0].mean, 60.0);
}

#[test]
fn a_seasonal_pattern_needs_two_cycles_to_be_told_from_the_trend() {
    // eight months between 50 and 60: nothing to say about the seasons
    let y = [52.0, 57.0, 51.0, 59.0, 55.0, 53.0, 58.0, 54.0];
    for model in [
        Box::new(Prophet::new()) as Box<dyn Model>,
        Box::new(Transformed::log(Prophet::new())),
    ] {
        let f = model.forecast(Series::monthly(&y, 0), 12).unwrap();
        assert!(f.iter().all(|v| (30.0..90.0).contains(v)), "{f:?}");
    }
}

#[test]
fn automatic_arima_keeps_an_exact_fit() {
    let constant = [5.0; 60];
    let f = AutoArima::new()
        .forecast(Series::monthly(&constant, 0), 3)
        .unwrap();
    assert!(f.iter().all(|v| (v - 5.0).abs() < 1e-9), "{f:?}");

    let line: Vec<f64> = (1..=60).map(|t| 10.0 + 2.0 * t as f64).collect();
    let f = AutoArima::new()
        .forecast(Series::non_seasonal(&line), 3)
        .unwrap();
    for (k, v) in f.iter().enumerate() {
        assert!((v - (132.0 + 2.0 * k as f64)).abs() < 1e-6, "{f:?}");
    }
    // the criteria of an exact fit are finite
    let fit = Arima::new(0, 1, 0)
        .constant(true)
        .estimate(Series::non_seasonal(&line))
        .unwrap();
    assert!(fit.log_likelihood.is_finite() && fit.aic.is_finite() && fit.bic.is_finite());
}

#[test]
fn a_model_on_the_adjusted_series_keeps_its_place_in_time() {
    // a step of −30 from position 80 on, in a series with a seasonal pattern
    let y: Vec<f64> = (0..120)
        .map(|t| {
            let step = if t >= 80 { -30.0 } else { 0.0 };
            200.0 + 0.5 * t as f64 + 10.0 * (PATTERN[t % 12] - 1.0) * 10.0 + step
        })
        .collect();
    let model = Decomposed::new(Prophet::new().changepoints(0).step("law", 80));
    let whole = model.forecast(Series::monthly(&y, 0), 6).unwrap();
    // from observation 40 on, the step is still at position 80 of the data
    let part = model
        .forecast(Series::monthly(&y, 0).slice(40..120), 6)
        .unwrap();
    for (a, b) in whole.iter().zip(&part) {
        assert!((a - b).abs() < 5.0, "{whole:?} against {part:?}");
    }
}

#[test]
fn exact_data_has_no_outliers() {
    assert!(outliers(&[3.0; 40], 1).unwrap().is_empty());
    assert!(outliers(&[3.0; 40], 12).unwrap().is_empty());
    let line: Vec<f64> = (0..60).map(|t| 10.0 + 0.5 * t as f64).collect();
    assert!(outliers(&line, 1).unwrap().is_empty());
    assert!(outliers(&line, 12).unwrap().is_empty());
    // and one that is there is still found
    let mut spiked = line.clone();
    spiked[30] += 50.0;
    let found = outliers(&spiked, 1).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].index, 30);
    assert!((found[0].replacement - line[30]).abs() < 1.0);
}

#[test]
fn small_things() {
    // growth of a cycle that adds up to a negative number means nothing
    let mut y = seasonal(36);
    for v in &mut y[24..] {
        *v = -*v;
    }
    assert!(SeasonalNaive::with_growth()
        .fit(Series::monthly(&y, 0))
        .is_none());
    // a smoothing that the method does not use is not checked
    let demand = [0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0];
    assert!(Croston::new()
        .beta(2.0)
        .fit(Series::non_seasonal(&demand))
        .is_some());
    // harmonics beyond half the period repeat the earlier ones
    assert_eq!(Regressors::fourier(12.0, 12, 24).width(), 11);
    assert_eq!(Regressors::fourier(12.0, 6, 24).width(), 11);
    assert_eq!(Regressors::fourier(7.0, 5, 24).width(), 6);
    // a forecast that is not finite is no forecast
    let up: Vec<f64> = (1..=60).map(f64::from).collect();
    assert!(Transformed::box_cox(Drift, -1.0)
        .forecast(Series::non_seasonal(&up), 5)
        .is_none());
}

#[test]
fn a_constant_series_does_not_send_tbats_searching() {
    let y = [7.0; 60];
    let started = std::time::Instant::now();
    let fit = Tbats::new(&[]).select(Series::monthly(&y, 0)).unwrap();
    assert!(fit.forecast(3).iter().all(|v| (v - 7.0).abs() < 1e-6));
    assert!(started.elapsed().as_secs() < 20, "{:?}", started.elapsed());
}
