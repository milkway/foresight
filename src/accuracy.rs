//! Accuracy measures and quantiles.
//!
//! Percentage measures are returned in percent, as in most forecasting
//! software. Pairs whose actual value is zero are left out of them.

/// Quantile with linear interpolation (type 7, the default in R). `v` does
/// not need to be sorted.
pub fn quantile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = p.clamp(0.0, 1.0) * (s.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos.fract());
    Some(if i + 1 < s.len() {
        s[i] + (s[i + 1] - s[i]) * f
    } else {
        s[i]
    })
}

fn mean(v: impl Iterator<Item = f64>) -> Option<f64> {
    let (mut s, mut n) = (0.0, 0usize);
    for x in v {
        s += x;
        n += 1;
    }
    (n > 0).then(|| s / n as f64)
}

fn pairs<'a>(actual: &'a [f64], forecast: &'a [f64]) -> impl Iterator<Item = (f64, f64)> + 'a {
    actual.iter().copied().zip(forecast.iter().copied())
}

/// Mean absolute percentage error, in percent.
pub fn mape(actual: &[f64], forecast: &[f64]) -> Option<f64> {
    mean(
        pairs(actual, forecast)
            .filter(|(a, _)| *a != 0.0)
            .map(|(a, f)| (f - a).abs() / a.abs()),
    )
    .map(|m| m * 100.0)
}

/// Mean of (forecast − actual) / actual, in percent: positive when the
/// forecasts run above what happened.
pub fn bias(actual: &[f64], forecast: &[f64]) -> Option<f64> {
    mean(
        pairs(actual, forecast)
            .filter(|(a, _)| *a != 0.0)
            .map(|(a, f)| (f - a) / a),
    )
    .map(|m| m * 100.0)
}

/// Mean absolute error.
pub fn mae(actual: &[f64], forecast: &[f64]) -> Option<f64> {
    mean(pairs(actual, forecast).map(|(a, f)| (f - a).abs()))
}

/// Root mean squared error.
pub fn rmse(actual: &[f64], forecast: &[f64]) -> Option<f64> {
    mean(pairs(actual, forecast).map(|(a, f)| (f - a) * (f - a))).map(f64::sqrt)
}

/// Scale of the mean absolute scaled error: the in-sample mean absolute error
/// of the seasonal naive forecast (naive when `period` is 1). `None` when the
/// training data is shorter than one period or the scale is zero.
pub fn mase_scale(train: &[f64], period: usize) -> Option<f64> {
    let m = period.max(1);
    if train.len() <= m {
        return None;
    }
    mean((m..train.len()).map(|t| (train[t] - train[t - m]).abs())).filter(|s| *s > 0.0)
}

/// Mean absolute scaled error given the scale from [`mase_scale`].
pub fn mase(actual: &[f64], forecast: &[f64], scale: f64) -> Option<f64> {
    mae(actual, forecast).map(|e| e / scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_interpolates_like_r_type_7() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(quantile(&v, 0.0), Some(1.0));
        assert_eq!(quantile(&v, 0.5), Some(3.0));
        assert_eq!(quantile(&v, 1.0), Some(5.0));
        assert!((quantile(&v, 0.1).unwrap() - 1.4).abs() < 1e-12);
        assert_eq!(quantile(&[], 0.5), None);
    }

    #[test]
    fn measures_on_a_known_case() {
        let actual = [100.0, 200.0, 0.0];
        let forecast = [110.0, 180.0, 5.0];
        // the zero actual is left out of the percentage measures
        assert!((mape(&actual, &forecast).unwrap() - 10.0).abs() < 1e-12);
        assert!(bias(&actual, &forecast).unwrap().abs() < 1e-12);
        assert!((mae(&actual, &forecast).unwrap() - 35.0 / 3.0).abs() < 1e-12);
        assert!((rmse(&actual, &forecast).unwrap() - (525.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(mape(&[], &[]), None);
    }

    #[test]
    fn mase_uses_the_seasonal_naive_error_of_the_training_data() {
        let train = [1.0, 2.0, 3.0, 5.0, 4.0, 7.0];
        // |3-1|, |5-2|, |4-3|, |7-5| → mean 2
        assert_eq!(mase_scale(&train, 2), Some(2.0));
        assert_eq!(mase(&[10.0], &[13.0], 2.0), Some(1.5));
        assert_eq!(mase_scale(&[1.0, 1.0, 1.0], 1), None);
        assert_eq!(mase_scale(&[1.0, 2.0], 4), None);
    }
}
