//! Derivative-free minimisation.

/// How hard [`nelder_mead`] tries.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Effort {
    /// Searches in a row, each from the best point of the one before.
    pub runs: usize,
    /// Relative spread of the values on the simplex at which a search stops.
    pub tolerance: f64,
}

impl Effort {
    /// Enough to rank alternatives.
    pub const QUICK: Effort = Effort {
        runs: 1,
        tolerance: 1e-8,
    };
    /// For the estimates that are reported.
    pub const THOROUGH: Effort = Effort {
        runs: 3,
        tolerance: 1e-11,
    };
}

/// Nelder-Mead simplex search from `start`, restarted from the best point
/// with a smaller simplex while a restart still helps. `f` may return
/// infinity for points it cannot evaluate.
pub(crate) fn nelder_mead<F: FnMut(&[f64]) -> f64>(
    mut f: F,
    start: &[f64],
    step: f64,
    effort: Effort,
) -> (Vec<f64>, f64) {
    let mut best = start.to_vec();
    let mut value = f(&best);
    if start.is_empty() {
        return (best, value);
    }
    let mut step = step;
    for _ in 0..effort.runs {
        let (x, v) = search(&mut f, &best, value, step, effort.tolerance);
        let gain = value - v;
        if v < value {
            best = x;
            value = v;
        }
        if gain.is_nan() || gain <= 1e-9 * (1.0 + value.abs()) {
            break;
        }
        step *= 0.5;
    }
    (best, value)
}

fn search<F: FnMut(&[f64]) -> f64>(
    f: &mut F,
    start: &[f64],
    start_value: f64,
    step: f64,
    tolerance: f64,
) -> (Vec<f64>, f64) {
    let n = start.len();
    let mut points: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    points.push((start.to_vec(), start_value));
    for i in 0..n {
        let mut x = start.to_vec();
        x[i] += step;
        let v = f(&x);
        points.push((x, v));
    }
    let along = |centre: &[f64], worst: &[f64], t: f64| -> Vec<f64> {
        centre
            .iter()
            .zip(worst)
            .map(|(c, w)| c + t * (w - c))
            .collect()
    };
    for _ in 0..300 * n {
        points.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (low, high) = (points[0].1, points[n].1);
        if (high - low).abs() <= tolerance * (low.abs() + high.abs()) + 1e-300 {
            break;
        }
        let mut centre = vec![0.0; n];
        for (x, _) in &points[..n] {
            for (c, v) in centre.iter_mut().zip(x) {
                *c += v / n as f64;
            }
        }
        let worst = points[n].0.clone();
        let second = points[n - 1].1;
        let reflected = along(&centre, &worst, -1.0);
        let fr = f(&reflected);
        if fr < low {
            let expanded = along(&centre, &worst, -2.0);
            let fe = f(&expanded);
            points[n] = if fe < fr {
                (expanded, fe)
            } else {
                (reflected, fr)
            };
        } else if fr < second {
            points[n] = (reflected, fr);
        } else {
            let t = if fr < high { -0.5 } else { 0.5 };
            let contracted = along(&centre, &worst, t);
            let fc = f(&contracted);
            if fc < fr.min(high) {
                points[n] = (contracted, fc);
            } else {
                let best = points[0].0.clone();
                for (x, v) in points.iter_mut().skip(1) {
                    for (xi, b) in x.iter_mut().zip(&best) {
                        *xi = b + 0.5 * (*xi - b);
                    }
                    *v = f(x);
                }
            }
        }
    }
    points.sort_by(|a, b| a.1.total_cmp(&b.1));
    points.swap_remove(0)
}

/// Minimum of a function of one variable on [lo, hi] by golden section.
pub(crate) fn golden<F: FnMut(f64) -> f64>(mut f: F, mut lo: f64, mut hi: f64) -> f64 {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut fa, mut fb) = (f(a), f(b));
    for _ in 0..80 {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - ratio * (hi - lo);
            fa = f(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + ratio * (hi - lo);
            fb = f(b);
        }
    }
    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_minimum_of_the_rosenbrock_function() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let (x, v) = nelder_mead(f, &[-1.2, 1.0], 0.5, Effort::THOROUGH);
        assert!(v < 1e-10, "value {v}");
        assert!((x[0] - 1.0).abs() < 1e-4 && (x[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn copes_with_regions_it_cannot_evaluate() {
        let f = |x: &[f64]| {
            if x[0] < 0.0 {
                f64::INFINITY
            } else {
                (x[0] - 2.0).powi(2)
            }
        };
        let (x, _) = nelder_mead(f, &[0.5], 1.0, Effort::THOROUGH);
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert_eq!(nelder_mead(|_| 3.0, &[], 1.0, Effort::QUICK), (vec![], 3.0));
    }

    #[test]
    fn golden_section_finds_a_one_dimensional_minimum() {
        let x = golden(|x| (x - 0.3).powi(2), -1.0, 2.0);
        assert!((x - 0.3).abs() < 1e-8);
    }
}
