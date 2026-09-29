//! Tests and measures used to decide how to model a series.

/// Autocorrelations at lags 1..=`max_lag`.
pub fn acf(y: &[f64], max_lag: usize) -> Vec<f64> {
    let n = y.len();
    let mean = y.iter().sum::<f64>() / n as f64;
    let c0: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    (1..=max_lag)
        .map(|k| {
            let ck: f64 = (k..n).map(|t| (y[t] - mean) * (y[t - k] - mean)).sum();
            ck / c0
        })
        .collect()
}

/// `y` differenced once at the given lag (1 for the ordinary difference).
pub fn difference(y: &[f64], lag: usize) -> Vec<f64> {
    (lag..y.len()).map(|t| y[t] - y[t - lag]).collect()
}

fn variance(v: &[f64]) -> Option<f64> {
    let n = v.len();
    if n < 2 {
        return None;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    Some(v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64)
}

/// KPSS statistic for the null hypothesis that the series is stationary
/// around a constant level (Kwiatkowski, Phillips, Schmidt & Shin, 1992),
/// with a Bartlett window of ⌊3√n / 13⌋ lags. Large values reject
/// stationarity; the 5% critical value is [`KPSS_5_PERCENT`].
pub fn kpss(y: &[f64]) -> Option<f64> {
    let n = y.len();
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let mean = y.iter().sum::<f64>() / nf;
    let e: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let mut cumulative = 0.0;
    let mut eta = 0.0;
    for v in &e {
        cumulative += v;
        eta += cumulative * cumulative;
    }
    eta /= nf * nf;
    let lags = (3.0 * nf.sqrt() / 13.0).floor() as usize;
    let mut long_run = e.iter().map(|v| v * v).sum::<f64>() / nf;
    for j in 1..=lags.min(n - 1) {
        let weight = 1.0 - j as f64 / (lags + 1) as f64;
        let cross: f64 = (j..n).map(|t| e[t] * e[t - j]).sum();
        long_run += 2.0 * weight * cross / nf;
    }
    let stat = eta / long_run;
    stat.is_finite().then_some(stat)
}

/// 5% critical value of the KPSS test for level stationarity.
pub const KPSS_5_PERCENT: f64 = 0.463;

/// Number of ordinary differences needed for the series to pass the KPSS test
/// at 5%, at most `max`.
pub fn ndiffs(y: &[f64], max: usize) -> usize {
    let mut current = y.to_vec();
    let mut d = 0;
    while d < max {
        match kpss(&current) {
            Some(stat) if stat > KPSS_5_PERCENT => {
                current = difference(&current, 1);
                d += 1;
            }
            _ => break,
        }
    }
    d
}

/// Strength of seasonality in [0, 1] (Wang, Smith & Hyndman, 2006):
/// 1 − Var(remainder) / Var(seasonal + remainder), from a classical additive
/// decomposition (centred moving average and seasonal means). `None` for
/// non-seasonal or short series (fewer than two cycles plus one observation).
pub fn seasonal_strength(y: &[f64], period: usize) -> Option<f64> {
    let (n, m) = (y.len(), period);
    if m < 2 || n < 2 * m + 1 {
        return None;
    }
    let half = m / 2;
    let mut detrended: Vec<(usize, f64)> = Vec::with_capacity(n);
    for t in half..n - half {
        let trend = if m.is_multiple_of(2) {
            let inner: f64 = y[t + 1 - half..t + half].iter().sum();
            (0.5 * y[t - half] + inner + 0.5 * y[t + half]) / m as f64
        } else {
            y[t - half..=t + half].iter().sum::<f64>() / m as f64
        };
        detrended.push((t % m, y[t] - trend));
    }
    let mut sum = vec![0.0; m];
    let mut count = vec![0usize; m];
    for &(s, v) in &detrended {
        sum[s] += v;
        count[s] += 1;
    }
    if count.contains(&0) {
        return None;
    }
    let mut seasonal: Vec<f64> = sum.iter().zip(&count).map(|(s, c)| s / *c as f64).collect();
    let centre = seasonal.iter().sum::<f64>() / m as f64;
    seasonal.iter_mut().for_each(|s| *s -= centre);
    let remainder: Vec<f64> = detrended.iter().map(|&(s, v)| v - seasonal[s]).collect();
    let both: Vec<f64> = detrended.iter().map(|&(_, v)| v).collect();
    let (vr, vb) = (variance(&remainder)?, variance(&both)?);
    if vb <= 0.0 {
        return Some(0.0);
    }
    Some((1.0 - vr / vb).clamp(0.0, 1.0))
}

/// Seasonal strength above which a seasonal difference is taken.
pub const SEASONAL_STRENGTH_THRESHOLD: f64 = 0.64;

/// Number of seasonal differences (0 or 1) suggested by the strength of the
/// seasonality.
pub fn nsdiffs(y: &[f64], period: usize) -> usize {
    match seasonal_strength(y, period) {
        Some(s) if s > SEASONAL_STRENGTH_THRESHOLD => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{noisy_seasonal_growth, seasonal_growth};

    fn noise(n: usize) -> Vec<f64> {
        noisy_seasonal_growth(n, 1.0)
            .iter()
            .zip(seasonal_growth(n))
            .map(|(a, b)| a / b - 1.0)
            .collect()
    }

    #[test]
    fn white_noise_is_stationary_and_a_random_walk_is_not() {
        let e = noise(200);
        assert!(kpss(&e).unwrap() < KPSS_5_PERCENT);
        assert_eq!(ndiffs(&e, 2), 0);
        let mut level = 0.0;
        let walk: Vec<f64> = e
            .iter()
            .map(|v| {
                level += v;
                level
            })
            .collect();
        assert!(kpss(&walk).unwrap() > KPSS_5_PERCENT);
        assert_eq!(ndiffs(&walk, 2), 1);
    }

    #[test]
    fn seasonal_strength_separates_seasonal_from_noisy_series() {
        let seasonal: Vec<f64> = noisy_seasonal_growth(120, 0.02)
            .iter()
            .map(|v| v.ln())
            .collect();
        assert!(seasonal_strength(&seasonal, 12).unwrap() > 0.9);
        assert_eq!(nsdiffs(&seasonal, 12), 1);
        assert!(seasonal_strength(&noise(120), 12).unwrap() < 0.5);
        assert_eq!(nsdiffs(&noise(120), 12), 0);
        assert_eq!(seasonal_strength(&seasonal, 1), None);
        assert_eq!(seasonal_strength(&seasonal[..20], 12), None);
    }

    #[test]
    fn differences_and_autocorrelations() {
        assert_eq!(difference(&[1.0, 4.0, 9.0, 16.0], 1), vec![3.0, 5.0, 7.0]);
        assert_eq!(difference(&[1.0, 4.0, 9.0, 16.0], 2), vec![8.0, 12.0]);
        let alternating: Vec<f64> = (0..50)
            .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let r = acf(&alternating, 2);
        assert!(r[0] < -0.9 && r[1] > 0.9);
    }
}
