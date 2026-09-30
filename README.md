# foresight

[![CI](https://github.com/milkway/foresight/actions/workflows/ci.yml/badge.svg)](https://github.com/milkway/foresight/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/foresight.svg)](https://crates.io/crates/foresight)
[![docs.rs](https://img.shields.io/docsrs/foresight)](https://docs.rs/foresight)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](Cargo.toml)
[![Website](https://img.shields.io/badge/website-milkway.github.io%2Fforesight-8A2BE2.svg)](https://milkway.github.io/foresight/)

Time series forecasting in Rust that picks its model by what would have worked.

`foresight` fits several models to a series, replays the past to see how each
would have done, chooses by out-of-sample error and reports intervals taken
from the errors actually observed. It has no dependencies and every result is
deterministic.

**Website:** <https://milkway.github.io/foresight/> ·
**API reference:** <https://milkway.github.io/foresight/api/foresight/> ·
[Português](README.pt-BR.md)

## Install

```bash
cargo add foresight
```

## Use

```rust
use foresight::{models, Backtest, Series};

// monthly data whose first observation is in March
let y = Series::monthly(&values, 2);

// replay the last 36 months, 12 months ahead, with every built-in model
let report = Backtest::default().run(y, &models::defaults()).unwrap();

let best = report.chosen();
println!("{}: MAPE {:.1}%", best.name, best.score);
for p in &best.forecast {
    let i = p.interval(0.80).unwrap();
    println!("{:2} {:.0} [{:.0}, {:.0}]", p.horizon, p.mean, i.lower, i.upper);
}

// the total of the next six months, with its own interval
let half_year = best.cumulative(6).unwrap();
```

One model on its own:

```rust
use foresight::{models::Arima, Model, Series, Transformed};

let model = Transformed::log(Arima::airline()); // ARIMA(0,1,1)(0,1,1) on the log scale
let fit = model.fit(Series::monthly(&values, 0)).unwrap();
let next_year = fit.forecast(12);
```

With a variable that helps, known over the horizon:

```rust
use foresight::{models::Arima, Fitted, Regressors, Series};

// rows for the history and for the periods to forecast
let x = Regressors::new().with("price_index", index);
let fit = Arima::airline().with_regressors(x).estimate(Series::monthly(&values, 0)).unwrap();
println!("{:?}", fit.regression);
let next_year = fit.forecast(12);
```

An ensemble, which can sit in a backtest next to its own members:

```rust
use foresight::models::{self, Ensemble, Weighting};
use foresight::{Backtest, Candidate, Series};

let ensemble = Ensemble::new(models::defaults()).weighting(Weighting::InverseError);
let mut candidates = models::defaults();
candidates.push(Candidate::new(ensemble));
let report = Backtest::default().run(Series::monthly(&values, 0), &candidates).unwrap();
```

Several seasonal patterns at once:

```rust
use foresight::{decompose::Mstl, models::{AutoEts, Decomposed}, Model, Series};

// daily data with a weekly and a yearly pattern
let parts = Mstl::new(&[7, 365]).decompose(&values).unwrap();
println!("weekly strength {:.2}", parts.seasonal_strength(0).unwrap());

let model = Decomposed::new(AutoEts::new()).periods(&[7, 365]);
let next_month = model.forecast(Series::new(&values, 7), 30).unwrap();
```

A trend that bends, with dated events:

```rust
use foresight::{models::Prophet, Fitted, Series};

let model = Prophet::new()
    .event("campaign", &[10, 34, 58, 82, 106, 130]) // positions, future ones included
    .step("new_law", 80);                            // a lasting change of level
let fit = model.estimate(Series::monthly(&values, 0)).unwrap();
println!("{:?} {:?}", fit.changepoints(), fit.effects());
let next_year = fit.forecast(12);
```

Automatic orders, inspected:

```rust
use foresight::{models::AutoArima, Series};

let fit = AutoArima::new().select(Series::monthly(&log_values, 0)).unwrap();
println!("ARIMA{:?}{:?}, AICc {:.1}", fit.order(), fit.seasonal_order(), fit.aicc);
```

## What is in it

| Piece | What it does |
|---|---|
| `Series` | values + seasonal period; slices keep season and position |
| `Model` / `Fitted` | fit once, forecast any horizon, inspect parameters |
| `models` | `Mean`, `Naive`, `Drift`, `SeasonalNaive`, `Theta`, `HoltWinters`, `LogLinear` (optionally deflated by a price index), `Arima` (seasonal, exact maximum likelihood), `AutoArima` (differences by tests, orders by stepwise search), `Prophet` (trend with changepoints, Fourier seasonality, dated events and steps), `Ets` (the exponential smoothing family in state space form), `AutoEts` (error, trend and season by information criterion), `Tbats` (trigonometric seasonality for long, fractional or several periods, Box-Cox, ARMA errors, all chosen by AIC) |
| `decompose` | `Stl` (seasonal-trend decomposition by LOESS, optionally robust) and `Mstl` (several seasonal periods), with strength of trend and of each seasonal pattern |
| `Decomposed` | any model on the seasonally adjusted series, the seasonal patterns added back to its forecasts |
| `Regressors` | external variables aligned with the data, Fourier terms for smooth or long seasonal patterns, seasonal dummies; `Arima::with_regressors` and `AutoArima::regressors` fit a regression with ARIMA errors |
| `Ensemble` | a model made of other models: plain average, median, weights by inverse error, or stacked weights (none negative, adding up to one) learnt on the end of the history |
| `Croston` | intermittent demand: Croston, the Syntetos-Boylan correction and Teunter-Syntetos-Babai |
| `clean` | gaps filled with the season they fall in, outliers found and replaced |
| `Transformed`, `BoxCox` | any model on the log or another Box-Cox scale; λ by Guerrero's method |
| `Backtest` | rolling origin (expanding or fixed window) on all cores; MAPE, MAE, RMSE, MASE and bias by horizon; average of the best models; choice by out-of-sample error |
| intervals | empirical quantiles of the backtest errors, by horizon and for cumulative totals |
| `diagnostics` | KPSS test, number of differences, strength of seasonality, autocorrelations |
| `accuracy` | the measures and quantiles on their own |

## How it differs from the usual toolkits

Most forecasting libraries choose a model by an in-sample information
criterion and derive intervals from distributional assumptions. Here the
choice and the intervals both come from forecasts made without seeing the
future they are judged against. The interval for a total (say, the rest of a
fiscal year) is measured on totals, because adding up monthly limits
overstates its uncertainty.

## Checked against R

Methods are implemented from the published papers and compared with the R
packages `forecast` 9.0.2 and `prophet` 1.1.7 on public data
(`tests/against_r.rs`; the reference values come from the scripts in
`tests/r/`).

| Method | Agreement |
|---|---|
| Seasonal naive, random walk with drift | exact |
| Theta | forecasts within 0.1% |
| ARIMA by maximum likelihood | coefficients within 0.002, forecasts within 0.01% |
| Number of differences (KPSS), seasonal differences | same decisions |
| Box-Cox λ (Guerrero) | within 0.001 |
| Prophet (linear growth, additive seasonality) | forecasts within 0.5% |
| Regression with ARIMA errors (a price index, Fourier terms, levels) | forecasts within 0.001%, same likelihood |
| STL, MSTL (one and two seasonal periods), forecasts by decomposition | exact (10 digits) |
| ETS, 8 models × 3 series | likelihood equal to R's where R reaches the maximum, higher in the other cases |
| Automatic ETS | same model on 2 of 3 series |
| Croston | exact |
| Outliers | the ones put on purpose are found by both; the doubtful points flagged differ (R smooths with Friedman's super smoother, we with the trend of a robust STL) |
| TBATS, 2 structures × 3 series | likelihood better than R's in all 6 |
| Automatic TBATS | AIC lower than R's on the 3 series |
| Automatic ARIMA orders | same model on 2 of 3 series; on the third the two stepwise searches end within one unit of AICc |

Theta and ARIMA differ from R only by the optimiser: the estimates are the
same optimum found by different routes. Prophet is fitted here without Stan:
for a given noise level its posterior is a lasso with ridge terms, solved
exactly by an active-set method, so the changepoints that do not matter come
out as exactly zero.

On ETS, R searches parameters and initial states together with a simplex
method that often stops short of the maximum. Here the search starts from
three points and restarts, and the likelihood found was never lower than R's
in 24 cases; on the airline passengers that changes the automatic choice from
ETS(M,Ad,M) to ETS(M,A,M).

TBATS is the slow one: choosing its structure takes 2 to 12 seconds for a
monthly series of ten years, against about one second in R, so it is not part
of the built-in sets of candidates.

The Prophet here covers equally spaced series, linear growth and additive
components (fit on the log scale for multiplicative behaviour); events are
given by position rather than by calendar.

## Example: public revenue of a Brazilian state

```bash
cargo run --release --example piaui
```

Backtest of 36 origins, 12 months ahead, mean absolute percentage error over
all horizons:

| Series | Best candidate | MAPE | Seasonal naive |
|---|---|---|---|
| ICMS (state sales tax) | average of airline ARIMA on the log and original scales | 3.4% | 11.5% |
| FPE (federal transfers) | average of automatic ARIMA on the log scale and the ensemble weighted by inverse error | 4.8% | 9.5% |

## Data

- `data/piaui_revenue.csv`: monthly ICMS and FPE revenue of the state of
  Piauí, Brazil, March 2017 to June 2026 (Siconfi/STN, RREO Anexo 03), with
  the chained IPCA price index (BCB/SGS 433). Public data.
- `data/air_passengers.csv`: the classic monthly airline passengers series
  (Box & Jenkins, 1976).

## Status

Early: the API may change before 1.0.

Out of scope: neural networks and foundation models, which need heavy
dependencies and trained weights.

## Development

```bash
scripts/dev-check.sh   # fmt, clippy -D warnings, tests, docs
```

## Authors

- André Leite ([ORCID](https://orcid.org/0000-0002-4718-9766))
- Hugo Vasconcelos ([ORCID](https://orcid.org/0000-0001-6249-0920))
- Raydonal Ospina ([ORCID](https://orcid.org/0000-0002-9884-9090))

See [CITATION.cff](CITATION.cff) to cite the software.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
