//! A borrowed, regularly spaced time series with a seasonal period.

use std::ops::Range;

/// A view over contiguous observations, oldest first.
///
/// Besides the values, a series knows its seasonal `period` (12 for monthly
/// data with a yearly cycle, 4 for quarterly, 1 for none) and where it sits in
/// the original data: slicing keeps the absolute position, so the season of
/// each observation and the alignment with external data (a price index, a
/// regressor) survive any [`head`](Series::head), [`tail`](Series::tail) or
/// [`slice`](Series::slice).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Series<'a> {
    values: &'a [f64],
    period: usize,
    /// Season of the observation at absolute index 0.
    phase: usize,
    /// Absolute index of `values[0]`.
    start: usize,
}

impl<'a> Series<'a> {
    /// A series with the given seasonal period (0 is read as 1: no seasonality).
    /// The first observation is taken to be at season 0; see
    /// [`with_phase`](Series::with_phase).
    pub fn new(values: &'a [f64], period: usize) -> Self {
        Series {
            values,
            period: period.max(1),
            phase: 0,
            start: 0,
        }
    }

    /// A series without seasonality.
    pub fn non_seasonal(values: &'a [f64]) -> Self {
        Self::new(values, 1)
    }

    /// Monthly data with a yearly cycle; `first_month` is the calendar month
    /// of the first observation, 0 = January.
    pub fn monthly(values: &'a [f64], first_month: usize) -> Self {
        Self::new(values, 12).with_phase(first_month)
    }

    /// Quarterly data with a yearly cycle; `first_quarter` is 0 for Q1.
    pub fn quarterly(values: &'a [f64], first_quarter: usize) -> Self {
        Self::new(values, 4).with_phase(first_quarter)
    }

    /// Sets the season of the first observation of this view.
    pub fn with_phase(mut self, phase: usize) -> Self {
        let p = self.period;
        self.phase = (phase % p + p - self.start % p) % p;
        self
    }

    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn period(&self) -> usize {
        self.period
    }

    /// Absolute index, in the original data, of the first observation.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Absolute index of observation `i`. `i` may be past the end: `len()` is
    /// the first period to be forecast.
    pub fn index(&self, i: usize) -> usize {
        self.start + i
    }

    /// Season (0..period) of observation `i`; like [`index`](Series::index),
    /// `i` may be past the end.
    pub fn season(&self, i: usize) -> usize {
        (self.phase + self.start + i) % self.period
    }

    /// The first `n` observations (all of them if `n` is larger).
    pub fn head(&self, n: usize) -> Self {
        self.slice(0..n)
    }

    /// The last `n` observations (all of them if `n` is larger).
    pub fn tail(&self, n: usize) -> Self {
        self.slice(self.len().saturating_sub(n)..self.len())
    }

    /// Observations in `range`, clamped to the series.
    pub fn slice(&self, range: Range<usize>) -> Self {
        let end = range.end.min(self.len());
        let begin = range.start.min(end);
        Series {
            values: &self.values[begin..end],
            period: self.period,
            phase: self.phase,
            start: self.start + begin,
        }
    }

    /// The same series (period, season and position) over other values, such
    /// as a transformation of the original ones. `None` if the lengths differ.
    pub fn with_values<'b>(&self, values: &'b [f64]) -> Option<Series<'b>> {
        (values.len() == self.len()).then_some(Series {
            values,
            period: self.period,
            phase: self.phase,
            start: self.start,
        })
    }

    /// True when every value is finite.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// True when every value is finite and greater than zero (what
    /// multiplicative and logarithmic models need).
    pub fn is_positive(&self) -> bool {
        self.values.iter().all(|v| v.is_finite() && *v > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slicing_keeps_season_and_absolute_index() {
        let v: Vec<f64> = (0..30).map(f64::from).collect();
        let y = Series::monthly(&v, 2); // starts in March
        assert_eq!(y.season(0), 2);
        assert_eq!(y.season(10), 0); // January of the next year
        let s = y.slice(10..20);
        assert_eq!((s.len(), s.start(), s.index(3)), (10, 10, 13));
        assert_eq!(s.season(0), 0);
        assert_eq!(s.tail(4).season(0), 6);
        assert_eq!(s.tail(4).values(), &v[16..20]);
        // forecasting positions
        assert_eq!(y.head(12).season(12), 2);
    }

    #[test]
    fn with_phase_refers_to_the_first_observation_of_the_view() {
        let v = [1.0; 20];
        let s = Series::new(&v, 4).slice(5..20).with_phase(3);
        assert_eq!(s.season(0), 3);
        assert_eq!(s.season(1), 0);
    }

    #[test]
    fn out_of_range_slices_are_clamped() {
        let v = [1.0, 2.0, 3.0];
        let y = Series::non_seasonal(&v);
        assert_eq!(y.head(10).len(), 3);
        assert!(y.slice(5..9).is_empty());
        assert_eq!(y.period(), 1);
        assert!(y.is_positive());
        assert!(!Series::non_seasonal(&[1.0, 0.0]).is_positive());
    }
}
