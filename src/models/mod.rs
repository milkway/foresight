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
//! | [`Ets`], [`AutoEts`] | optional | 2 cycles when seasonal; positive values for multiplicative parts |
//! | [`Prophet`] | optional | 4 observations |
//! | [`Croston`] | no | values that are not negative |
//! | [`Ensemble`] | as its members | what its members need |
//! | [`Tbats`] | optional, several periods, not necessarily whole | 8 observations |
//!
//! Any model can run on the log or another Box-Cox scale through
//! [`Transformed`], and on the seasonally adjusted series through
//! [`Decomposed`].

mod arima;
mod croston;
mod decomposed;
mod ensemble;
mod ets;
mod holt_winters;
mod log_linear;
mod naive;
mod prophet;
mod tbats;
mod theta;

pub use arima::{Arima, ArimaFit, ArimaX, AutoArima, Criterion};
pub use croston::{Croston, Intermittent};
pub use decomposed::Decomposed;
pub use ensemble::{Ensemble, Weighting};
pub use ets::{AutoEts, ErrorKind, Ets, EtsFit, Season, Trend};
pub use holt_winters::HoltWinters;
pub use log_linear::LogLinear;
pub use naive::{Drift, Mean, Naive, SeasonalNaive};
pub use prophet::{Prophet, ProphetFit};
pub use tbats::{Tbats, TbatsFit};
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
        Candidate::new(Prophet::new()),
        Candidate::new(Transformed::log(Prophet::new())),
    ]
}

/// [`defaults`] plus exponential smoothing chosen automatically and ARIMA
/// with automatic orders, on the original and on the log scale. The choices
/// are made again at every origin of the backtest, which takes seconds rather
/// than milliseconds.
pub fn thorough() -> Vec<Candidate> {
    let mut c = defaults();
    c.push(Candidate::new(
        Ensemble::new(defaults()).weighting(Weighting::InverseError),
    ));
    c.push(Candidate::new(
        Ensemble::new(defaults()).weighting(Weighting::Stacked),
    ));
    c.push(Candidate::new(AutoEts::new()));
    c.push(Candidate::new(Decomposed::new(AutoEts::new())));
    c.push(Candidate::new(Transformed::log(Decomposed::new(
        AutoEts::new(),
    ))));
    c.push(Candidate::new(AutoArima::new()));
    c.push(Candidate::new(Transformed::log(AutoArima::new())));
    c
}
