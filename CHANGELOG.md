# Changelog

## 0.7.3

- `Backtest::parallel(false)` now also keeps ensembles among the candidates on
  the calling thread, as 0.7.2 said it did: they used to start threads of
  their own. Results do not change.

## 0.7.2

Fixes found in a review of the crate and of the packages built on it. Results
on ordinary data do not change.

- **Backtest.** A candidate that cannot give its final forecast (it cannot be
  fitted on the whole series, or a forecast is not finite) is left out of the
  report; it used to make the whole run return `None`. The final forecast
  follows `window`, like every origin. Levels outside (0, 1) are refused.
  Intervals of negative forecasts come in order (lower ≤ upper).
- **Threads.** `set_max_threads` limits the threads of backtests and
  ensembles. An ensemble inside a parallel backtest no longer starts threads
  of its own, and `Backtest::parallel(false)` now holds for its candidates.
- **`Model::forecast`** returns `None` when a forecast is not finite:
  regressors that stop short of the horizon inside `Transformed`,
  `Decomposed` or an ensemble, a Box-Cox scale that cannot be brought back.
- **Prophet** fits seasonal terms from two full cycles on; shorter histories
  gave wild forecasts.
- **ARIMA.** An exact fit has finite criteria (a very large likelihood stands
  for the unbounded one), and `AutoArima` keeps it instead of discarding it:
  a constant series is forecast as that constant, a straight line as the line.
- **`Decomposed`** keeps the position of the series, so models that go by
  position (regressors, events, a deflator) stay aligned on slices and in
  windowed backtests.
- **STL, robust.** On data the fit reproduces exactly, rounding noise no
  longer decides which points count; `clean::outliers` reports nothing on
  exact data and still finds a real outlier in it.
- **TBATS** recognises an exact fit: a constant series takes milliseconds
  instead of a minute.
- `SeasonalNaive::with_growth` refuses a last cycle that does not add up to a
  positive number; `Croston` checks `beta` only for TSB;
  `Regressors::fourier` leaves out harmonics beyond half the period.
- `Fitted` is now `Send + Sync`, like `Model`: a fit can be handed to another
  thread. Every fit of the crate already was; a custom `Fitted` that is not
  will no longer compile.
- `rust-version` is declared: 1.81.
