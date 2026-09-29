//! External variables that help explain a series.

use std::f64::consts::PI;

/// Named columns of values aligned with the ORIGINAL data: row `i` belongs to
/// the observation whose position is `i` (see
/// [`Series::index`](crate::Series::index)). The rows go on past the end of
/// the history, because a forecast needs the values of the regressors over the
/// horizon.
///
/// ```
/// use foresight::Regressors;
///
/// // two harmonics of a yearly cycle in weekly data, and a variable of our own
/// let price = vec![1.0; 120];
/// let x = Regressors::fourier(52.18, 2, 120).with("price", price);
/// assert_eq!(x.names(), ["sin1_52.18", "cos1_52.18", "sin2_52.18", "cos2_52.18", "price"]);
/// assert_eq!(x.rows(), 120);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Regressors {
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
}

impl Regressors {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a column.
    pub fn with(mut self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.names.push(name.into());
        self.columns.push(values);
        self
    }

    /// Adds the columns of another set.
    pub fn and(mut self, other: Regressors) -> Self {
        self.names.extend(other.names);
        self.columns.extend(other.columns);
        self
    }

    /// Sines and cosines of the first `order` harmonics of a cycle of the
    /// given period, which need not be a whole number, for `rows` positions:
    /// a smooth seasonal pattern with few coefficients. The sine that is zero
    /// everywhere (the harmonic at half a whole period) is left out.
    pub fn fourier(period: f64, order: usize, rows: usize) -> Self {
        let mut out = Regressors::new();
        for k in 1..=order {
            let angle = |t: usize| 2.0 * PI * k as f64 * t as f64 / period;
            if (2.0 * k as f64 - period).abs() > 1e-9 {
                out = out.with(
                    format!("sin{k}_{period}"),
                    (0..rows).map(|t| angle(t).sin()).collect(),
                );
            }
            out = out.with(
                format!("cos{k}_{period}"),
                (0..rows).map(|t| angle(t).cos()).collect(),
            );
        }
        out
    }

    /// One indicator for each position of the cycle but the first.
    pub fn seasonal_dummies(period: usize, rows: usize) -> Self {
        let mut out = Regressors::new();
        for s in 1..period {
            out = out.with(
                format!("season{}", s + 1),
                (0..rows)
                    .map(|t| f64::from(u8::from(t % period == s)))
                    .collect(),
            );
        }
        out
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn columns(&self) -> &[Vec<f64>] {
        &self.columns
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Number of rows that every column has.
    pub fn rows(&self) -> usize {
        self.columns.iter().map(Vec::len).min().unwrap_or(0)
    }

    /// True when there are rows for the positions `0..end` and all of them
    /// are finite.
    pub fn covers(&self, end: usize) -> bool {
        self.columns
            .iter()
            .all(|c| c.len() >= end && c[..end].iter().all(|v| v.is_finite()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourier_terms_repeat_with_the_period() {
        let x = Regressors::fourier(12.0, 6, 36);
        // the sixth sine is zero on whole positions and is left out
        assert_eq!(x.width(), 11);
        for c in x.columns() {
            for t in 0..24 {
                assert!((c[t] - c[t + 12]).abs() < 1e-12);
            }
        }
        assert_eq!(x.names()[0], "sin1_12");
        assert_eq!(Regressors::fourier(7.0, 3, 10).width(), 6);
    }

    #[test]
    fn dummies_mark_each_position_but_the_first() {
        let x = Regressors::seasonal_dummies(4, 9);
        assert_eq!(x.width(), 3);
        assert_eq!(
            x.columns()[0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(x.names(), ["season2", "season3", "season4"]);
    }

    #[test]
    fn coverage_is_the_shortest_column_without_gaps() {
        let x = Regressors::new()
            .with("a", vec![1.0, 2.0, 3.0])
            .and(Regressors::new().with("b", vec![1.0, f64::NAN, 3.0, 4.0]));
        assert_eq!((x.width(), x.rows()), (2, 3));
        assert!(x.covers(1) && !x.covers(2) && !x.covers(4));
        assert!(Regressors::new().covers(10));
        assert_eq!(Regressors::new().rows(), 0);
    }
}
