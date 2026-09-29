//! The little linear algebra the models need.

/// Solves `a·x = b` by Gaussian elimination with partial pivoting. `None` if
/// the matrix is singular or the solution is not finite.
pub(crate) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let k = b.len();
    if a.len() != k || a.iter().any(|row| row.len() != k) {
        return None;
    }
    for col in 0..k {
        let piv = (col..k).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let pivot = a[col].clone();
        for row in col + 1..k {
            let f = a[row][col] / pivot[col];
            if f != 0.0 {
                for (c, v) in a[row].iter_mut().enumerate().skip(col) {
                    *v -= f * pivot[c];
                }
                b[row] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let s: f64 = (row + 1..k).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Least squares fit of `y = a + b·t`, t = 0, 1, …; returns `(a, b)`.
pub(crate) fn line(y: &[f64]) -> Option<(f64, f64)> {
    let n = y.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let t_mean = (nf - 1.0) / 2.0;
    let y_mean = y.iter().sum::<f64>() / nf;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (t, v) in y.iter().enumerate() {
        let d = t as f64 - t_mean;
        sxy += d * (v - y_mean);
        sxx += d * d;
    }
    let b = sxy / sxx;
    (b.is_finite()).then_some((y_mean - b * t_mean, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_system_and_rejects_a_singular_one() {
        let x = solve(vec![vec![2.0, 1.0], vec![1.0, 3.0]], vec![5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);
        assert!(solve(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 2.0]).is_none());
    }

    #[test]
    fn fits_a_line() {
        let y: Vec<f64> = (0..10).map(|t| 3.0 + 0.5 * t as f64).collect();
        let (a, b) = line(&y).unwrap();
        assert!((a - 3.0).abs() < 1e-12 && (b - 0.5).abs() < 1e-12);
    }
}
