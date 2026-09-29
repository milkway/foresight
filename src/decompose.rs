//! Seasonal-trend decomposition by LOESS: STL (Cleveland, Cleveland, McRae &
//! Terpenning, 1990) and its extension to several seasonal periods, MSTL
//! (Bandara, Hyndman & Bergmeir, 2021).

/// A series split into trend, one seasonal pattern per period and remainder:
/// the three add up to the series.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    pub periods: Vec<usize>,
    pub trend: Vec<f64>,
    /// One seasonal component per period, in the order of `periods`.
    pub seasonal: Vec<Vec<f64>>,
    pub remainder: Vec<f64>,
}

fn variance(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
}

impl Decomposition {
    /// The series without its seasonal patterns.
    pub fn seasonally_adjusted(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(&self.remainder)
            .map(|(t, r)| t + r)
            .collect()
    }

    /// Strength of the trend in [0, 1]: 1 − Var(remainder) / Var(trend +
    /// remainder) (Wang, Smith & Hyndman, 2006).
    pub fn trend_strength(&self) -> f64 {
        let both = self.seasonally_adjusted();
        let total = variance(&both);
        if total <= 0.0 {
            return 0.0;
        }
        (1.0 - variance(&self.remainder) / total).clamp(0.0, 1.0)
    }

    /// Strength in [0, 1] of the seasonal component of index `i`.
    pub fn seasonal_strength(&self, i: usize) -> Option<f64> {
        let both: Vec<f64> = self
            .seasonal
            .get(i)?
            .iter()
            .zip(&self.remainder)
            .map(|(s, r)| s + r)
            .collect();
        let total = variance(&both);
        if total <= 0.0 {
            return Some(0.0);
        }
        Some((1.0 - variance(&self.remainder) / total).clamp(0.0, 1.0))
    }
}

/// How far the seasonal pattern may change from one cycle to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalWindow {
    /// The same pattern in every cycle.
    Periodic,
    /// LOESS window over the cycles, an odd number of at least 7: the smaller,
    /// the faster the pattern may change.
    Span(usize),
}

/// STL for one seasonal period.
///
/// The defaults are those of `stl` in R: trend window
/// 1.5·period / (1 − 1.5/seasonal window), rounded up to the next odd number,
/// low-pass window the next odd number after the period, local constant
/// fitting for the seasonal pattern and local linear for the rest, and LOESS
/// evaluated at every tenth of each window and interpolated in between.
///
/// ```
/// use foresight::decompose::{SeasonalWindow, Stl};
///
/// let y: Vec<f64> = (0..72)
///     .map(|t| 10.0 + 0.5 * t as f64 + [3.0, -1.0, -2.0, 0.0][t % 4])
///     .collect();
/// let d = Stl::new(4, SeasonalWindow::Periodic).decompose(&y).unwrap();
/// assert!((d.seasonal[0][0] - 3.0).abs() < 0.1);
/// assert!(d.remainder.iter().all(|r| r.abs() < 0.1));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Stl {
    period: usize,
    seasonal: SeasonalWindow,
    trend: Option<usize>,
    low_pass: Option<usize>,
    seasonal_degree: usize,
    trend_degree: usize,
    low_pass_degree: usize,
    robust: bool,
    inner: Option<usize>,
    outer: Option<usize>,
}

fn next_odd(x: usize) -> usize {
    if x.is_multiple_of(2) {
        x + 1
    } else {
        x
    }
}

impl Stl {
    pub fn new(period: usize, seasonal: SeasonalWindow) -> Self {
        Stl {
            period,
            seasonal,
            trend: None,
            low_pass: None,
            seasonal_degree: 0,
            trend_degree: 1,
            low_pass_degree: 1,
            robust: false,
            inner: None,
            outer: None,
        }
    }

    /// LOESS window of the trend.
    pub fn trend_window(mut self, n: usize) -> Self {
        self.trend = Some(n);
        self
    }

    /// LOESS window of the low-pass filter.
    pub fn low_pass_window(mut self, n: usize) -> Self {
        self.low_pass = Some(n);
        self
    }

    /// Degree (0 or 1) of the local fit of the seasonal pattern, of the trend
    /// and of the low-pass filter.
    pub fn degrees(mut self, seasonal: usize, trend: usize, low_pass: usize) -> Self {
        self.seasonal_degree = seasonal.min(1);
        self.trend_degree = trend.min(1);
        self.low_pass_degree = low_pass.min(1);
        self
    }

    /// Down-weights outliers: 15 robustness rounds of one pass each instead of
    /// a single round of two passes.
    ///
    /// Compared with `stl(robust = TRUE)` in R, the result is the same to 12
    /// digits for up to 11 rounds; from the 12th on the two drift apart in the
    /// third digit, R's values no longer settling from one round to the next.
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Passes of the inner loop and robustness rounds of the outer loop.
    pub fn iterations(mut self, inner: usize, outer: usize) -> Self {
        (self.inner, self.outer) = (Some(inner), Some(outer));
        self
    }

    /// Decomposes the values. `None` for a period under 2, fewer than two
    /// full cycles or values that are not finite.
    pub fn decompose(&self, y: &[f64]) -> Option<Decomposition> {
        let (n, np) = (y.len(), self.period);
        if np < 2 || n <= 2 * np || y.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let (periodic, span, seasonal_degree) = match self.seasonal {
            SeasonalWindow::Periodic => (true, 10 * n + 1, 0),
            SeasonalWindow::Span(s) => (false, s, self.seasonal_degree),
        };
        let trend = self.trend.unwrap_or_else(|| {
            next_odd((1.5 * np as f64 / (1.0 - 1.5 / span as f64)).ceil() as usize)
        });
        let low_pass = self.low_pass.unwrap_or_else(|| next_odd(np));
        let jump = |window: usize| window.div_ceil(10);
        let setup = Setup {
            np,
            ns: next_odd(span.max(3)),
            nt: next_odd(trend.max(3)),
            nl: next_odd(low_pass.max(3)),
            seasonal_degree,
            trend_degree: self.trend_degree,
            low_pass_degree: self.low_pass_degree,
            seasonal_jump: jump(span),
            trend_jump: jump(trend),
            low_pass_jump: jump(low_pass),
            inner: self.inner.unwrap_or(if self.robust { 1 } else { 2 }),
        };
        let outer = self.outer.unwrap_or(if self.robust { 15 } else { 0 });

        let mut season = vec![0.0; n];
        let mut trend = vec![0.0; n];
        let mut weights: Option<Vec<f64>> = None;
        for round in 0..=outer {
            setup.pass(y, weights.as_deref(), &mut season, &mut trend);
            if round == outer {
                break;
            }
            let fit: Vec<f64> = trend.iter().zip(&season).map(|(t, s)| t + s).collect();
            weights = Some(robustness_weights(y, &fit));
        }
        if periodic {
            // the same pattern in every cycle: the mean of each position
            let mut sum = vec![0.0; np];
            let mut count = vec![0usize; np];
            for (t, s) in season.iter().enumerate() {
                sum[t % np] += s;
                count[t % np] += 1;
            }
            for (t, s) in season.iter_mut().enumerate() {
                *s = sum[t % np] / count[t % np] as f64;
            }
        }
        let remainder = (0..n).map(|t| y[t] - season[t] - trend[t]).collect();
        Some(Decomposition {
            periods: vec![np],
            trend,
            seasonal: vec![season],
            remainder,
        })
    }
}

/// The windows, degrees and jumps of one decomposition.
struct Setup {
    np: usize,
    ns: usize,
    nt: usize,
    nl: usize,
    seasonal_degree: usize,
    trend_degree: usize,
    low_pass_degree: usize,
    seasonal_jump: usize,
    trend_jump: usize,
    low_pass_jump: usize,
    inner: usize,
}

impl Setup {
    /// The inner loop: detrend, smooth each cycle-subseries, take the
    /// low-frequency part out of the result, deseasonalise, smooth the trend.
    fn pass(&self, y: &[f64], weights: Option<&[f64]>, season: &mut [f64], trend: &mut [f64]) {
        let n = y.len();
        for _ in 0..self.inner {
            let detrended: Vec<f64> = (0..n).map(|t| y[t] - trend[t]).collect();
            let cycles = self.cycle_subseries(&detrended, weights);
            let filtered = moving_average(
                &moving_average(&moving_average(&cycles, self.np), self.np),
                3,
            );
            let low = loess(
                &filtered,
                self.nl,
                self.low_pass_degree,
                self.low_pass_jump,
                None,
            );
            for t in 0..n {
                season[t] = cycles[self.np + t] - low[t];
            }
            let adjusted: Vec<f64> = (0..n).map(|t| y[t] - season[t]).collect();
            let smooth = loess(
                &adjusted,
                self.nt,
                self.trend_degree,
                self.trend_jump,
                weights,
            );
            trend.copy_from_slice(&smooth);
        }
    }

    /// Each cycle-subseries (all the Januaries, all the Februaries…) smoothed
    /// by LOESS and extended by one cycle at each end: n + 2·period values.
    fn cycle_subseries(&self, y: &[f64], weights: Option<&[f64]>) -> Vec<f64> {
        let (n, np) = (y.len(), self.np);
        let mut out = vec![0.0; n + 2 * np];
        for j in 0..np {
            let k = (n - j - 1) / np + 1;
            let sub: Vec<f64> = (0..k).map(|i| y[i * np + j]).collect();
            let sub_weights: Option<Vec<f64>> =
                weights.map(|w| (0..k).map(|i| w[i * np + j]).collect());
            let w = sub_weights.as_deref();
            let smooth = loess(&sub, self.ns, self.seasonal_degree, self.seasonal_jump, w);
            let before = local_fit(
                &sub,
                self.ns,
                self.seasonal_degree,
                0.0,
                1,
                self.ns.min(k),
                w,
            )
            .unwrap_or(smooth[0]);
            let left = (k + 1).saturating_sub(self.ns).max(1);
            let after = local_fit(
                &sub,
                self.ns,
                self.seasonal_degree,
                (k + 1) as f64,
                left,
                k,
                w,
            )
            .unwrap_or(smooth[k - 1]);
            out[j] = before;
            for (i, v) in smooth.iter().enumerate() {
                out[(i + 1) * np + j] = *v;
            }
            out[(k + 1) * np + j] = after;
        }
        out
    }
}

fn moving_average(x: &[f64], len: usize) -> Vec<f64> {
    let mut sum: f64 = x[..len].iter().sum();
    let mut out = Vec::with_capacity(x.len() - len + 1);
    out.push(sum / len as f64);
    for t in len..x.len() {
        sum += x[t] - x[t - len];
        out.push(sum / len as f64);
    }
    out
}

/// Bisquare weights from the size of the residuals: 1 for the small ones, 0
/// beyond six times the median.
fn robustness_weights(y: &[f64], fit: &[f64]) -> Vec<f64> {
    let n = y.len();
    let r: Vec<f64> = y.iter().zip(fit).map(|(a, b)| (a - b).abs()).collect();
    let mut sorted = r.clone();
    sorted.sort_by(f64::total_cmp);
    let middle = n / 2;
    let cmad = 3.0 * (sorted[middle] + sorted[n - middle - 1]);
    let (c9, c1) = (0.999 * cmad, 0.001 * cmad);
    r.iter()
        .map(|v| {
            if *v <= c1 {
                1.0
            } else if *v <= c9 {
                (1.0 - (v / cmad).powi(2)).powi(2)
            } else {
                0.0
            }
        })
        .collect()
}

/// Local fit at the position `at` (the first value is at 1) from the values at
/// positions `left..=right`, with tricube weights. `None` when every weight is
/// zero.
fn local_fit(
    y: &[f64],
    window: usize,
    degree: usize,
    at: f64,
    left: usize,
    right: usize,
    robustness: Option<&[f64]>,
) -> Option<f64> {
    let n = y.len();
    let range = n as f64 - 1.0;
    let mut h = (at - left as f64).max(right as f64 - at);
    if window > n {
        h += ((window - n) / 2) as f64;
    }
    let (h9, h1) = (0.999 * h, 0.001 * h);
    let mut w = vec![0.0; right - left + 1];
    let mut total = 0.0;
    for j in left..=right {
        let r = (j as f64 - at).abs();
        if r <= h9 {
            let mut weight = if r <= h1 {
                1.0
            } else {
                (1.0 - (r / h).powi(3)).powi(3)
            };
            if let Some(rw) = robustness {
                weight *= rw[j - 1];
            }
            w[j - left] = weight;
            total += weight;
        }
    }
    if total <= 0.0 {
        return None;
    }
    for v in &mut w {
        *v /= total;
    }
    if h > 0.0 && degree > 0 {
        let centre: f64 = (left..=right).map(|j| w[j - left] * j as f64).sum();
        let spread: f64 = (left..=right)
            .map(|j| w[j - left] * (j as f64 - centre).powi(2))
            .sum();
        if spread.sqrt() > 0.001 * range {
            let b = (at - centre) / spread;
            for j in left..=right {
                w[j - left] *= b * (j as f64 - centre) + 1.0;
            }
        }
    }
    Some((left..=right).map(|j| w[j - left] * y[j - 1]).sum())
}

/// LOESS over the whole series, evaluated every `jump` positions and
/// interpolated in between.
fn loess(
    y: &[f64],
    window: usize,
    degree: usize,
    jump: usize,
    robustness: Option<&[f64]>,
) -> Vec<f64> {
    let n = y.len();
    if n < 2 {
        return y.to_vec();
    }
    let jump = jump.clamp(1, n - 1);
    let mut out = vec![0.0; n];
    let (mut left, mut right) = (1, n.min(window));
    let half = window.div_ceil(2);
    let at = |i: usize, left: usize, right: usize| -> f64 {
        local_fit(y, window, degree, i as f64, left, right, robustness).unwrap_or(y[i - 1])
    };
    if window >= n {
        (left, right) = (1, n);
        for i in (1..=n).step_by(jump) {
            out[i - 1] = at(i, left, right);
        }
    } else if jump == 1 {
        for i in 1..=n {
            if i > half && right != n {
                left += 1;
                right += 1;
            }
            out[i - 1] = at(i, left, right);
        }
    } else {
        for i in (1..=n).step_by(jump) {
            (left, right) = if i < half {
                (1, window)
            } else if i > n - half {
                (n - window + 1, n)
            } else {
                (i - half + 1, window + i - half)
            };
            out[i - 1] = at(i, left, right);
        }
    }
    if jump != 1 {
        for i in (1..=n - jump).step_by(jump) {
            let slope = (out[i + jump - 1] - out[i - 1]) / jump as f64;
            for j in i + 1..i + jump {
                out[j - 1] = out[i - 1] + slope * (j - i) as f64;
            }
        }
        let last = ((n - 1) / jump) * jump + 1;
        if last != n {
            out[n - 1] = at(n, left, right);
            if last != n - 1 {
                let slope = (out[n - 1] - out[last - 1]) / (n - last) as f64;
                for j in last + 1..n {
                    out[j - 1] = out[last - 1] + slope * (j - last) as f64;
                }
            }
        }
    }
    out
}

/// STL applied in turn to each of several seasonal periods.
///
/// The periods are taken from the shortest to the longest; each pattern is
/// estimated on the series without the others, twice over. Periods longer
/// than half the series are left out. The trend comes from the last fit.
///
/// ```
/// use foresight::decompose::Mstl;
///
/// // a weekly and a monthly pattern in daily data
/// let y: Vec<f64> = (0..400)
///     .map(|t| 100.0 + 0.1 * t as f64 + [5.0, 0.0, -2.0, -3.0, 0.0, 1.0, -1.0][t % 7]
///         + 10.0 * (2.0 * std::f64::consts::PI * t as f64 / 30.0).sin())
///     .collect();
/// let d = Mstl::new(&[7, 30]).decompose(&y).unwrap();
/// assert_eq!(d.periods, vec![7, 30]);
/// assert!(d.seasonal_strength(0).unwrap() > 0.9);
/// assert!(d.seasonal_strength(1).unwrap() > 0.9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Mstl {
    periods: Vec<usize>,
    windows: Vec<usize>,
    iterations: usize,
    robust: bool,
}

impl Mstl {
    /// With the seasonal windows 11, 15, 19… for the first, second, third…
    /// period, and two rounds.
    pub fn new(periods: &[usize]) -> Self {
        let mut periods: Vec<usize> = periods.iter().copied().filter(|p| *p >= 2).collect();
        periods.sort_unstable();
        periods.dedup();
        Mstl {
            periods,
            windows: (1..=6).map(|i| 7 + 4 * i).collect(),
            iterations: 2,
            robust: false,
        }
    }

    /// Seasonal windows, one per period in increasing order of period; the
    /// last one is repeated if there are more periods than windows.
    pub fn seasonal_windows(mut self, windows: &[usize]) -> Self {
        if !windows.is_empty() {
            self.windows = windows.to_vec();
        }
        self
    }

    /// Rounds over the periods.
    pub fn iterations(mut self, n: usize) -> Self {
        self.iterations = n.max(1);
        self
    }

    /// Down-weights outliers in every fit.
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Decomposes the values. `None` when no period fits twice in the series
    /// or the values are not finite.
    pub fn decompose(&self, y: &[f64]) -> Option<Decomposition> {
        let n = y.len();
        if y.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let periods: Vec<usize> = self.periods.iter().copied().filter(|p| 2 * p < n).collect();
        if periods.is_empty() {
            return None;
        }
        let mut seasonal = vec![vec![0.0; n]; periods.len()];
        let mut rest = y.to_vec();
        let mut trend = vec![0.0; n];
        for _ in 0..self.iterations {
            for (i, &period) in periods.iter().enumerate() {
                for t in 0..n {
                    rest[t] += seasonal[i][t];
                }
                let window = self.windows[i.min(self.windows.len() - 1)];
                let fit = Stl::new(period, SeasonalWindow::Span(window))
                    .robust(self.robust)
                    .decompose(&rest)?;
                seasonal[i].clone_from(&fit.seasonal[0]);
                for t in 0..n {
                    rest[t] -= seasonal[i][t];
                }
                trend = fit.trend;
            }
        }
        let remainder = (0..n).map(|t| rest[t] - trend[t]).collect();
        Some(Decomposition {
            periods,
            trend,
            seasonal,
            remainder,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    #[test]
    fn the_parts_add_up_to_the_series() {
        let y: Vec<f64> = noisy_seasonal_growth(120, 0.1)
            .iter()
            .map(|v| v.ln())
            .collect();
        for stl in [
            Stl::new(12, SeasonalWindow::Periodic),
            Stl::new(12, SeasonalWindow::Span(13)),
            Stl::new(12, SeasonalWindow::Span(7)).robust(true),
        ] {
            let d = stl.decompose(&y).unwrap();
            for (t, v) in y.iter().enumerate() {
                let sum = d.trend[t] + d.seasonal[0][t] + d.remainder[t];
                assert!((sum - v).abs() < 1e-12);
            }
            let adjusted = d.seasonally_adjusted();
            assert!((adjusted[5] - (y[5] - d.seasonal[0][5])).abs() < 1e-12);
        }
    }

    #[test]
    fn recovers_a_known_pattern_and_a_smooth_trend() {
        let log: Vec<f64> = seasonal_growth(120).iter().map(|v| v.ln()).collect();
        let d = Stl::new(12, SeasonalWindow::Periodic)
            .decompose(&log)
            .unwrap();
        // the pattern of the generator, on the log scale and centred
        let pattern: Vec<f64> = seasonal_growth(12)
            .iter()
            .enumerate()
            .map(|(t, v)| (v / (100.0 * 1.005f64.powi(t as i32))).ln())
            .collect();
        let mean = pattern.iter().sum::<f64>() / 12.0;
        for t in 0..120 {
            assert!((d.seasonal[0][t] - (pattern[t % 12] - mean)).abs() < 0.01);
            assert!((d.seasonal[0][t] - d.seasonal[0][t % 12]).abs() < 1e-12);
        }
        // the trend grows by log(1.005) a period
        for t in 12..108 {
            assert!((d.trend[t + 1] - d.trend[t] - 1.005f64.ln()).abs() < 1e-3);
        }
        assert!(d.seasonal_strength(0).unwrap() > 0.99);
        assert!(d.trend_strength() > 0.99);
    }

    #[test]
    fn robust_fitting_keeps_an_outlier_out_of_the_pattern() {
        let mut y: Vec<f64> = noisy_seasonal_growth(120, 0.02)
            .iter()
            .map(|v| v.ln())
            .collect();
        let clean = Stl::new(12, SeasonalWindow::Span(13))
            .robust(true)
            .decompose(&y)
            .unwrap();
        y[66] += 1.0;
        let robust = Stl::new(12, SeasonalWindow::Span(13))
            .robust(true)
            .decompose(&y)
            .unwrap();
        let plain = Stl::new(12, SeasonalWindow::Span(13))
            .decompose(&y)
            .unwrap();
        // the outlier ends up in the remainder
        assert!(robust.remainder[66] > 0.9);
        let moved = |d: &Decomposition| (d.seasonal[0][66] - clean.seasonal[0][66]).abs();
        assert!(moved(&robust) < 0.02, "{}", moved(&robust));
        assert!(moved(&plain) > 5.0 * moved(&robust));
    }

    #[test]
    fn interpolation_between_jumps_is_close_to_the_full_fit() {
        let y: Vec<f64> = (0..200)
            .map(|t| (t as f64 / 15.0).sin() * 10.0 + t as f64 * 0.1)
            .collect();
        let full = loess(&y, 31, 1, 1, None);
        let jumped = loess(&y, 31, 1, 4, None);
        for (a, b) in full.iter().zip(&jumped) {
            assert!((a - b).abs() < 0.15);
        }
        // at the evaluated positions they are the same
        assert_eq!(full[0], jumped[0]);
        assert_eq!(full[4], jumped[4]);
        assert_eq!(full[199], jumped[199]);
        assert_eq!(loess(&[3.0], 5, 1, 1, None), vec![3.0]);
        assert_eq!(
            moving_average(&[1.0, 2.0, 3.0, 4.0], 2),
            vec![1.5, 2.5, 3.5]
        );
    }

    #[test]
    fn several_periods_are_told_apart() {
        let weekly = [5.0, 0.0, -2.0, -3.0, 0.0, 1.0, -1.0];
        let e = noisy_seasonal_growth(420, 1.0);
        let clean = seasonal_growth(420);
        let y: Vec<f64> = (0..420)
            .map(|t| {
                100.0
                    + 0.05 * t as f64
                    + weekly[t % 7]
                    + 8.0 * (2.0 * std::f64::consts::PI * t as f64 / 30.0).sin()
                    + (e[t] / clean[t] - 1.0)
            })
            .collect();
        let d = Mstl::new(&[30, 7, 7, 1, 500]).decompose(&y).unwrap();
        assert_eq!(d.periods, vec![7, 30]);
        for (t, v) in y.iter().enumerate().take(300).skip(100) {
            assert!((d.seasonal[0][t] - weekly[t % 7]).abs() < 1.0, "t={t}");
            let monthly = 8.0 * (2.0 * std::f64::consts::PI * t as f64 / 30.0).sin();
            assert!((d.seasonal[1][t] - monthly).abs() < 1.5, "t={t}");
            let sum = d.trend[t] + d.seasonal[0][t] + d.seasonal[1][t] + d.remainder[t];
            assert!((sum - v).abs() < 1e-9);
        }
    }

    #[test]
    fn refuses_what_cannot_be_decomposed() {
        let y = seasonal_growth(30);
        assert!(Stl::new(12, SeasonalWindow::Periodic)
            .decompose(&y[..24])
            .is_none());
        assert!(Stl::new(1, SeasonalWindow::Periodic)
            .decompose(&y)
            .is_none());
        assert!(Stl::new(4, SeasonalWindow::Periodic)
            .decompose(&[1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .is_none());
        assert!(Mstl::new(&[40]).decompose(&y).is_none());
        assert!(Mstl::new(&[]).decompose(&y).is_none());
    }
}
