//! Missing values and outliers.
//!
//! The rest of the crate takes series without gaps. Here a gap is a value
//! that is not a number (`f64::NAN`), and the functions return series ready
//! to be modelled.

use crate::accuracy::quantile;
use crate::decompose::{SeasonalWindow, Stl};

fn linear(values: &[f64]) -> Option<Vec<f64>> {
    let known: Vec<usize> = (0..values.len())
        .filter(|i| values[*i].is_finite())
        .collect();
    let (first, last) = (*known.first()?, *known.last()?);
    let mut out = values.to_vec();
    // the ends take the nearest value
    for v in &mut out[..first] {
        *v = values[first];
    }
    for v in &mut out[last + 1..] {
        *v = values[last];
    }
    for pair in known.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        for (i, v) in out.iter_mut().enumerate().take(b).skip(a + 1) {
            let share = (i - a) as f64 / (b - a) as f64;
            *v = values[a] + share * (values[b] - values[a]);
        }
    }
    Some(out)
}

/// Fills the gaps. Without seasonality (period under 2, or fewer than two
/// full cycles) by straight lines between the neighbours; with it, the
/// straight lines are drawn on the seasonally adjusted series and the
/// seasonal pattern is put back, so a gap in December is filled with a
/// December. `None` when no value is known.
pub fn interpolate(values: &[f64], period: usize) -> Option<Vec<f64>> {
    let first = linear(values)?;
    if values.iter().all(|v| v.is_finite()) {
        return Some(first);
    }
    let Some(parts) = Stl::new(period, SeasonalWindow::Periodic)
        .robust(true)
        .decompose(&first)
    else {
        return Some(first);
    };
    let seasonal = &parts.seasonal[0];
    let adjusted: Vec<f64> = values.iter().zip(seasonal).map(|(v, s)| v - s).collect();
    let filled = linear(&adjusted)?;
    Some(filled.iter().zip(seasonal).map(|(a, s)| a + s).collect())
}

/// An observation that does not fit with the others.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outlier {
    /// Position in the series.
    pub index: usize,
    pub value: f64,
    /// What the neighbours and the season suggest instead.
    pub replacement: f64,
}

/// Strength of the seasonal pattern measured with interquartile ranges in
/// place of variances, so that the outliers being looked for do not pass for
/// a weak pattern.
fn robust_strength(seasonal: &[f64], remainder: &[f64]) -> f64 {
    let spread = |v: &[f64]| -> f64 {
        let (q1, q3) = (quantile(v, 0.25), quantile(v, 0.75));
        q1.zip(q3).map_or(0.0, |(a, b)| (b - a) * (b - a))
    };
    let both: Vec<f64> = seasonal.iter().zip(remainder).map(|(s, r)| s + r).collect();
    let total = spread(&both);
    if total <= 0.0 {
        return 0.0;
    }
    (1.0 - spread(remainder) / total).clamp(0.0, 1.0)
}

/// Finds the outliers: observations whose distance from trend and seasonality
/// is more than three interquartile ranges beyond the quartiles of those
/// distances.
///
/// Trend and seasonality come from a robust STL decomposition; the seasonal
/// pattern is only taken out when it is strong (strength of at least 0.6,
/// measured with interquartile ranges), as a weak one would be mostly noise. Series without seasonality get a robust
/// local linear trend. Gaps are filled before looking. `None` when the series
/// has fewer than 8 known values.
pub fn outliers(values: &[f64], period: usize) -> Option<Vec<Outlier>> {
    let n = values.len();
    if values.iter().filter(|v| v.is_finite()).count() < 8 {
        return None;
    }
    let filled = interpolate(values, period)?;
    let parts = Stl::new(period, SeasonalWindow::Span(11))
        .robust(true)
        .decompose(&filled);
    let smooth: Vec<f64> = match parts {
        Some(d) if robust_strength(&d.seasonal[0], &d.remainder) >= 0.6 => d
            .trend
            .iter()
            .zip(&d.seasonal[0])
            .map(|(t, s)| t + s)
            .collect(),
        Some(d) => d.trend,
        None => trend_without_seasonality(&filled)?,
    };
    let distance: Vec<f64> = filled.iter().zip(&smooth).map(|(v, s)| v - s).collect();
    let (q1, q3) = (quantile(&distance, 0.25)?, quantile(&distance, 0.75)?);
    let reach = 3.0 * (q3 - q1);
    let flagged: Vec<usize> = (0..n)
        .filter(|i| values[*i].is_finite())
        .filter(|i| distance[*i] < q1 - reach || distance[*i] > q3 + reach)
        .collect();
    if flagged.is_empty() {
        return Some(Vec::new());
    }
    let mut holes = values.to_vec();
    for i in &flagged {
        holes[*i] = f64::NAN;
    }
    let replaced = interpolate(&holes, period)?;
    Some(
        flagged
            .into_iter()
            .map(|index| Outlier {
                index,
                value: values[index],
                replacement: replaced[index],
            })
            .collect(),
    )
}

/// A robust trend for a series without seasonality: the trend of an STL
/// decomposition whose "cycle" of two periods carries no pattern to speak of.
fn trend_without_seasonality(values: &[f64]) -> Option<Vec<f64>> {
    let span = (values.len() / 5).max(7);
    let d = Stl::new(2, SeasonalWindow::Periodic)
        .trend_window(span)
        .robust(true)
        .decompose(values)?;
    Some(d.trend)
}

/// The series with the gaps filled and the outliers replaced. `None` when
/// fewer than 8 values are known.
pub fn clean(values: &[f64], period: usize) -> Option<Vec<f64>> {
    let mut out = interpolate(values, period)?;
    for o in outliers(values, period)? {
        out[o.index] = o.replacement;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::noisy_seasonal_growth;

    fn log_series(n: usize) -> Vec<f64> {
        noisy_seasonal_growth(n, 0.04)
            .iter()
            .map(|v| v.ln())
            .collect()
    }

    #[test]
    fn straight_lines_between_neighbours() {
        let v = [f64::NAN, 1.0, f64::NAN, f64::NAN, 4.0, f64::NAN];
        assert_eq!(linear(&v).unwrap(), vec![1.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
        assert_eq!(
            interpolate(&v, 1).unwrap(),
            vec![1.0, 1.0, 2.0, 3.0, 4.0, 4.0]
        );
        assert!(interpolate(&[f64::NAN; 4], 1).is_none());
        assert_eq!(interpolate(&[2.0, 3.0], 12).unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    fn a_gap_in_a_seasonal_series_is_filled_with_its_season() {
        let full = log_series(120);
        let mut v = full.clone();
        // December of the sixth year and February of the eighth
        for i in [71, 85] {
            v[i] = f64::NAN;
        }
        let filled = interpolate(&v, 12).unwrap();
        let straight = linear(&v).unwrap();
        for i in [71, 85] {
            assert!((filled[i] - full[i]).abs() < 0.05, "{i}");
            assert!((filled[i] - full[i]).abs() < 0.3 * (straight[i] - full[i]).abs());
        }
        for i in (0..120).filter(|i| ![71, 85].contains(i)) {
            assert_eq!(filled[i], full[i]);
        }
    }

    #[test]
    fn outliers_are_found_and_replaced_by_something_plausible() {
        let full = log_series(120);
        let mut v = full.clone();
        v[40] += 1.0;
        v[97] -= 0.8;
        let found = outliers(&v, 12).unwrap();
        let at: Vec<usize> = found.iter().map(|o| o.index).collect();
        assert_eq!(at, vec![40, 97]);
        for o in &found {
            assert_eq!(o.value, v[o.index]);
            assert!((o.replacement - full[o.index]).abs() < 0.06);
        }
        let cleaned = clean(&v, 12).unwrap();
        assert!((cleaned[40] - full[40]).abs() < 0.06);
        assert_eq!(cleaned[10], v[10]);
        // nothing to find in the series as it was
        assert!(outliers(&full, 12).unwrap().is_empty());
    }

    #[test]
    fn works_without_seasonality_and_with_gaps() {
        let noisy = noisy_seasonal_growth(80, 1.0);
        let clean_series = noisy_seasonal_growth(80, 0.0);
        let mut v: Vec<f64> = (0..80)
            .map(|t| 10.0 + 0.2 * t as f64 + 2.0 * (noisy[t] / clean_series[t] - 1.0))
            .collect();
        v[30] += 25.0;
        v[50] = f64::NAN;
        let found = outliers(&v, 1).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].index, 30);
        let cleaned = clean(&v, 1).unwrap();
        assert!(cleaned.iter().all(|x| x.is_finite()));
        assert!((cleaned[30] - (10.0 + 0.2 * 30.0)).abs() < 3.0);
        assert!(outliers(&[1.0, 2.0, 3.0], 1).is_none());
    }
}
