//! Time series forecasting that picks its model by what would have worked.
//!
//! `foresight` fits several models to a series, replays the past to see how
//! each would have done (rolling-origin backtest), chooses by out-of-sample
//! error and reports intervals taken from the errors actually observed —
//! including intervals for the total of the next k periods. It has no
//! dependencies and every result is deterministic.
//!
//! ```
//! use foresight::{models, Backtest, Series};
//!
//! // eight years of monthly data starting in January
//! let sales: Vec<f64> = (0..96)
//!     .map(|t| 100.0 * 1.004f64.powi(t) * [1.1, 0.8, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.1, 1.3][t as usize % 12])
//!     .collect();
//! let y = Series::monthly(&sales, 0);
//!
//! let report = Backtest::default().run(y, &models::defaults()).unwrap();
//! let best = report.chosen();
//! println!("{} (MAPE {:.1}%)", best.name, best.score);
//! for p in &best.forecast {
//!     let i = p.interval(0.80).unwrap();
//!     println!("h={:2} {:8.1} [{:8.1}, {:8.1}]", p.horizon, p.mean, i.lower, i.upper);
//! }
//! // total of the next six months, with its own interval
//! let half_year = best.cumulative(6).unwrap();
//! assert!(half_year.mean > 0.0);
//! ```
//!
//! A single model can be used on its own through the [`Model`] trait:
//!
//! ```
//! use foresight::{models::Theta, Model, Series};
//!
//! let y = [12.0, 14.0, 13.0, 15.0, 17.0, 16.0, 18.0, 20.0];
//! let fit = Theta.fit(Series::non_seasonal(&y)).unwrap();
//! assert_eq!(fit.forecast(3).len(), 3);
//! ```

pub mod accuracy;
pub mod backtest;
pub mod decompose;
pub mod diagnostics;
mod linalg;
pub mod model;
pub mod models;
mod optimize;
pub mod series;
#[cfg(test)]
mod testing;
pub mod transform;

pub use backtest::{
    Backtest, Band, Candidate, CandidateReport, HorizonStats, Interval, Metric, Point, Report,
};
pub use model::{Fitted, Model, Params};
pub use series::Series;
pub use transform::{BoxCox, Transformed};

/// Version of this crate, to be recorded next to stored forecasts.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
