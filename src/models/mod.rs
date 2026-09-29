//! Forecasting methods.
//!
//! | Model | Seasonal | Needs |
//! |---|---|---|
//! | [`Mean`], [`Naive`], [`Drift`] | no | 1–2 observations |
//! | [`SeasonalNaive`] | yes | 1 cycle (2 with growth) |
//! | [`Theta`] | optional | 3 observations |
//! | [`HoltWinters`] | yes | 3 cycles, positive values |
//! | [`LogLinear`] | optional | 3 cycles, positive values |
//! | [`Arima`], [`AutoArima`] | optional | enough observations after differencing |
//!
//! Any model can run on the log or another Box-Cox scale through
//! [`Transformed`].

mod arima;
mod holt_winters;
mod log_linear;
mod naive;
mod theta;

pub use arima::{Arima, ArimaFit, AutoArima, Criterion};
pub use holt_winters::HoltWinters;
pub use log_linear::LogLinear;
pub use naive::{Drift, Mean, Naive, SeasonalNaive};
pub use theta::Theta;

use crate::backtest::Candidate;
use crate::transform::Transformed;

/// A reasonable set of candidates for a backtest: the benchmarks and every
/// model that needs no external data and fits in a moment.
pub fn defaults() -> Vec<Candidate> {
    vec![
        Candidate::new(Naive),
        Candidate::new(Drift),
        Candidate::new(SeasonalNaive::new()),
        Candidate::new(SeasonalNaive::with_growth()),
        Candidate::new(Theta),
        Candidate::new(HoltWinters),
        Candidate::new(LogLinear::new()),
        Candidate::new(Arima::airline()),
        Candidate::new(Transformed::log(Arima::airline())),
    ]
}

/// [`defaults`] plus ARIMA with automatic orders, on the original and on the
/// log scale. The orders are chosen again at every origin of the backtest,
/// which takes seconds rather than milliseconds.
pub fn thorough() -> Vec<Candidate> {
    let mut c = defaults();
    c.push(Candidate::new(AutoArima::new()));
    c.push(Candidate::new(Transformed::log(AutoArima::new())));
    c
}
